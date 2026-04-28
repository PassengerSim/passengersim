from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import altair as alt

    from passengersim import Simulation, SimulationTables
    from passengersim.contrast import Contrast


import pandas as pd

from .generic import GenericTracer
from .welford import Welford


class _ForecastTracer(GenericTracer):
    """
    A per-sample tracer that records the forecasts for specified items.

    This will generally be called as a begin-sample callback, so that
    it records the forecasts at the start of each sample.

    Subclasses must override the __call__ method to collect data.
    """

    # name: str = "selected_leg_forecasts"
    _forecasts: dict[int, Welford]
    _n_total_samples: int
    _on: str

    def __init__(self, _ids: list[int]):
        self._ids = _ids
        self.reset()

    def reset(self) -> None:
        self._forecasts = {leg_id: Welford() for leg_id in self._ids}
        self._n_total_samples = 0

    def fresh(self) -> _ForecastTracer:
        return type(self)(self._ids)

    def attach(self, sim: Simulation) -> None:
        sim.begin_sample_callback(self.fresh())

    def finalize(self) -> pd.DataFrame:
        df = pd.concat(
            {_id: welford.mean for _id, welford in self._forecasts.items()},
            names=[self._on],
        )
        df.attrs["n_samples"] = self._n_total_samples
        df.attrs["aggregation_process"] = "wgt_by_samples"
        return df


class PathForecastTracer(_ForecastTracer):
    """
    A per-sample tracer that records the forecasts for specified paths.

    This will generally be called as a begin-sample callback, so that
    it records the forecasts at the start of each sample.
    """

    name: str = "selected_path_forecasts"
    _on: str = "path_id"

    def __init__(self, path_ids: list[int]):
        super().__init__(path_ids)

    def __call__(self, sim: Simulation):
        if sim.eng.sample < sim.eng.burn_samples:
            return
        for path_id in self._ids:
            fd = sim.eng.paths.select(path_id=path_id).get_forecast_data()
            self._forecasts[path_id].update(fd.to_dataframe())
            self._n_total_samples += 1


class LegForecastTracer(_ForecastTracer):
    """
    A per-sample tracer that records the forecasts for specified legs.

    This will generally be called as a begin-sample callback, so that
    it records the forecasts at the start of each sample.
    """

    name: str = "selected_leg_forecasts"
    _on: str = "leg_id"

    def __init__(self, leg_ids: list[int]):
        super().__init__(leg_ids)

    def __call__(self, sim: Simulation):
        if sim.eng.sample < sim.eng.burn_samples:
            return
        for leg_id in self._ids:
            fd = sim.eng.legs.select(leg_id=leg_id).get_forecast_data()
            self._forecasts[leg_id].update(fd.to_dataframe())
            self._n_total_samples += 1


def _iterate_subfigures(fig: alt.Chart):
    if "vconcat" in fig._kwds:
        for i in fig._kwds["vconcat"]:
            yield from _iterate_subfigures(i)
    elif "hconcat" in fig._kwds:
        for i in fig._kwds["hconcat"]:
            yield from _iterate_subfigures(i)
    else:
        yield fig


def _collect_data_by_panel(fig: alt.Chart) -> dict[str, pd.DataFrame]:
    collected = {}
    for subfig in _iterate_subfigures(fig):
        try:
            y_title = subfig._kwds["encoding"].y["title"]
        except (KeyError, AttributeError) as e:
            raise ValueError("Could not find y title in subfigure") from e
        collected[y_title] = subfig._kwds["data"]
    return collected


def _collect_multiple_data_by_panel(figs) -> dict[str, dict[str, pd.DataFrame]]:
    """
    Collect data from multiple dashboard figures.

    Result is nested dict first by source, then by panel title.
    """
    data = {}
    for k, fig in figs.items():
        data[k] = _collect_data_by_panel(fig)
    return data


def _combine_data_by_panel(d: dict[str, dict[str, pd.DataFrame]]):
    # by_panel = list(zip(*(_iterate_nested_list(d[z]) for z in top_keys)))

    from collections import defaultdict

    combined = {}
    pending = defaultdict(dict)
    for source_name, content in d.items():
        for panel_name, df in content.items():
            pending[panel_name][source_name] = df

    for panel_name, sources in pending.items():
        try:
            c = pd.concat(sources, names=["source"]).reset_index("source")
        except:
            import sys

            print(f"panel_name: {panel_name}", file=sys.stderr)
            print(f"sources: {sources.keys()}", file=sys.stderr)
            print("----", file=sys.stderr)
            for s in sources.values():
                print(s, file=sys.stderr)
                print("----", file=sys.stderr)
            raise
        combined[panel_name] = c
    return combined


def _base_dashboard_figures(
    summaries: dict[str, SimulationTables],
    path_id: int,
    include: tuple[str] | None = None,
):
    if include is not None:
        summaries = {k: v for k, v in summaries.items() if k in include}
    figures = {k: fig_path_forecast_dashboard(v, path_id=path_id) for k, v in summaries.items()}
    return figures


def fig_path_forecast_dashboard(
    summary: SimulationTables | Contrast,
    path_id: int,
    *,
    include: tuple[str] | None = None,
) -> alt.Chart:
    """
    Create a dashboard of path forecasts.

    Parameters
    ----------
    summary : SimulationTables | Contrast
        The summary data.
    path_id : int
        The path_id to display.
    include : tuple[str], optional
        If provided, only include the specified sources.  This is only used
        when the summary is a Contrast, and it is ignored otherwise, as
        there is only one source in that case.

    Returns
    -------
    alt.Chart
        The dashboard figure.
    """
    from passengersim_core.forecast_tools import ForecastData

    from passengersim.contrast import Contrast  # prevent circular import

    if isinstance(summary, Contrast):
        figs = _base_dashboard_figures(summary, path_id, include=include)
        figd_panel = _collect_multiple_data_by_panel(figs)
        dfs_p = _combine_data_by_panel(figd_panel)

        # get a single summary
        singleton = next(iter(summary.values()))

        fd = ForecastData(
            **{k.replace(" ", "_"): v for (k, v) in dfs_p.items()},
            label=singleton.path_identifier(path_id),
            stdev_in_timeframe=pd.DataFrame(),
        )
        return fd.dashboard()

    fig = ForecastData.from_dataframe(
        summary.callback_data.selected_path_forecasts.query(f"path_id=={int(path_id)}").droplevel("path_id")
    ).dashboard()

    return fig


def fig_leg_forecast_dashboard(
    summary: SimulationTables | Contrast,
    leg_id: int,
    *,
    include: tuple[str] | None = None,
) -> alt.Chart:
    """
    Create a dashboard of leg forecasts.

    Parameters
    ----------
    summary : SimulationTables | Contrast
        The summary data.
    leg_id : int
        The leg_id to display.
    include : tuple[str], optional
        If provided, only include the specified sources.  This is only used
        when the summary is a Contrast, and it is ignored otherwise, as
        there is only one source in that case.

    Returns
    -------
    alt.Chart
        The dashboard figure.
    """
    from passengersim_core.forecast_tools import ForecastData

    from passengersim.contrast import Contrast  # prevent circular import

    if isinstance(summary, Contrast):

        def _base_dashboard_figures_leg(summaries, leg_id, include=None):
            if include is not None:
                summaries = {k: v for k, v in summaries.items() if k in include}
            figures = {k: fig_leg_forecast_dashboard(v, leg_id=leg_id) for k, v in summaries.items()}
            return figures

        figs = _base_dashboard_figures_leg(summary, leg_id, include=include)
        figd_panel = _collect_multiple_data_by_panel(figs)
        dfs_p = _combine_data_by_panel(figd_panel)
        # singleton = next(iter(summary.values()))
        fd = ForecastData(
            **{k.replace(" ", "_"): v for (k, v) in dfs_p.items()},
            label=str(leg_id),
            stdev_in_timeframe=pd.DataFrame(),
        )
        return fd.dashboard()

    fig = ForecastData.from_dataframe(
        summary.callback_data.selected_leg_forecasts.query(f"leg_id=={int(leg_id)}").droplevel("leg_id")
    ).dashboard()
    return fig
