from __future__ import annotations

import warnings
from collections.abc import Callable, Collection
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from passengersim import Simulation
    from passengersim.config import Config
    from passengersim.database import Database


class MissingDataError(KeyError):
    """Exception raised when data is missing from a summary table."""

    pass


class SimulationTableItem:
    def __init__(
        self,
        aggregation_func: Callable[[list[SimulationTables]], pd.DataFrame | None],
        extraction_func: Callable[[Simulation], pd.DataFrame] = None,
        computed_fields: dict[str, Any] = None,
        doc: str | None = None,
    ):
        self._doc = doc
        self._aggregation_func = aggregation_func
        self._extraction_func = extraction_func
        self._computed_fields = computed_fields or {}

    def __set_name__(self, owner, name):
        print(f"Setting {name} on {owner}")
        self.name = name
        owner._std_agg[name] = self._aggregation_func
        owner._std_extract[name] = self._extraction_func

    def __get__(self, instance, owner):
        if instance is None:
            return self
        try:
            df = instance._data[self.name]
            for field, func in self._computed_fields.items():
                if field in df:
                    continue
                try:
                    df[field] = df.eval(func)
                except Exception as e:
                    warnings.warn(
                        f"Error computing {field} for {self.name}: {e}", stacklevel=2
                    )
            return df
        except KeyError:
            raise MissingDataError(self.name) from None

    def __set__(self, instance, value):
        instance._data[self.name] = value

    @property
    def __doc__(self):
        return self._doc

    def _get_raw(self, instance):
        df = instance._data.get(self.name, None)
        if df is not None:
            df = df.drop(columns=self._computed_fields.keys(), errors="ignore")
        return df


class SimulationTables:
    __slots__ = ("_data", "config", "cnx", "sim", "n_total_samples", "meta_summaries")

    def __init__(
        self,
        data: dict[str, pd.DataFrame] = None,
        *,
        config: Config | None = None,
        cnx: Database | None = None,
        sim: Simulation | None = None,
        n_total_samples: int = 0,
    ):
        self._data = data or {}
        """Dataframes that summarize a Simulation run."""

        self.config = config
        """Configuration for the Simulation run."""

        self.cnx = cnx
        """Database connection for the Simulation run."""

        self.sim = sim
        """Simulation object for the Simulation run."""

        self.n_total_samples = n_total_samples
        """Total number of sample departures simulated to create these summaries.

        This excludes any burn samples.
        """

        self.meta_summaries = []
        """Summaries that were aggregated to create this summary."""

    _std_agg: dict[str, Callable[[list[SimulationTables]], pd.DataFrame | None]] = {}
    _std_extract: dict[str, Callable[[Simulation], pd.DataFrame]] = {}

    @classmethod
    def extract(cls, sim: Simulation, items: Collection[str] = ()) -> SimulationTables:
        """Extract summary data from a Simulation."""
        eng = sim.sim
        num_samples = eng.num_trials_completed * (eng.num_samples - eng.burn_samples)
        if num_samples <= 0:
            raise ValueError(
                "insufficient number of samples outside burn period for reporting"
                f"\n- num_trials = {eng.num_trials}"
                f"\n- num_samples = {eng.num_samples}"
                f"\n- burn_samples = {eng.burn_samples}"
            )

        data = {}
        items = set(items) or cls._std_extract.keys()
        for name, func in cls._std_extract.items():
            if name in items:
                if func is not None:
                    data[name] = func(sim)
        return cls(
            data, sim=sim, config=sim.config, cnx=sim.cnx, n_total_samples=num_samples
        )

    @classmethod
    def aggregate(cls, summaries: Collection[SimulationTables]):
        """Aggregate multiple summary tables."""
        if not summaries:
            return None

        result = cls({})
        for name, func in cls._std_agg.items():
            if func is not None:
                result._data[name] = func(summaries)
        result.meta_summaries = summaries
        result.n_total_samples = sum(s.n_total_samples for s in summaries)
        return result


def SimulationTable_add_item(name: str, *args, **kwargs):
    item = SimulationTableItem(*args, **kwargs)
    setattr(SimulationTables, name, item)
    item.__set_name__(SimulationTables, name)
    setattr(SimulationTables, "_raw_" + name, property(item._get_raw))
