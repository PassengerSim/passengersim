from __future__ import annotations

import glob
import os
import pathlib
import pickle
import time
import warnings
from collections.abc import Callable, Collection
from typing import TYPE_CHECKING, Any

import pandas as pd

from passengersim.reporting import report_figure

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
        aggregation_func: Callable[
            [list[_GenericSimulationTables]], pd.DataFrame | None
        ],
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


class _GenericSimulationTables:
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

    __writable_attrs = {
        "_data",
        "config",
        "cnx",
        "sim",
        "n_total_samples",
        "meta_summaries",
        "_preserve_meta_summaries",
        "_preserve_config",
    }

    def __setattr__(self, item, value):
        """Deny setting of attributes that are not in __writable_attrs."""
        if item in self.__writable_attrs:
            # writable attributes are handled normally
            super().__setattr__(item, value)
        else:
            raise AttributeError(f"Cannot set attribute {item!r}")

    _std_agg: dict[
        str, Callable[[list[_GenericSimulationTables]], pd.DataFrame | None]
    ] = {}
    _std_extract: dict[str, Callable[[Simulation], pd.DataFrame]] = {}

    @classmethod
    def extract(
        cls, sim: Simulation, items: Collection[str] = ()
    ) -> _GenericSimulationTables:
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
    def aggregate(cls, summaries: Collection[_GenericSimulationTables]):
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

    def __getstate__(self):
        state = self.__dict__.copy()
        if "cnx" in state:
            del state["cnx"]
        if "sim" in state:
            del state["sim"]
        if "config" in state:
            if state.get("_preserve_config", False):
                state["_config_yaml"] = state["config"].to_yaml()
            del state["config"]
        if "meta_trials" in state and not state.get("_preserve_meta_trials", True):
            del state["meta_trials"]
        if "_preserve_meta_trials" in state:
            del state["_preserve_meta_trials"]
        if "_preserve_config" in state:
            del state["_preserve_config"]
        return state

    def __setstate__(self, state):
        if "_config_yaml" in state:
            state["config"] = Config.from_raw_yaml(state.pop("_config_yaml"))
        self.__dict__.update(state)

    def to_pickle(
        self,
        filename: str | pathlib.Path,
        add_timestamp_ext: bool = True,
        *,
        preserve_meta_summaries: bool = False,
        preserve_config: bool = False,
    ):
        """Save to a pickle file.

        This method uses lz4 compression if the lz4.frame module is available.

        Parameters
        ----------
        filename : str or Path-like
            The filename to save the object to.  An extension map be added or
            modified, to optionally add a time stamp and/or compression flag.
        add_timestamp_ext : bool, default True
            Add a timestamp extension to the filename.
        preserve_meta_summaries : bool, default False
            Preserve the meta_summaries attribute in the saved object.
        preserve_config : bool, default False
            Preserve the config attribute in the saved object.  This includes
            the entire network, and can potentially be a lot of data.
        """
        if add_timestamp_ext:
            filename = pathlib.Path(filename)
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = filename.with_suffix(f".{timestamp}.pkl")
        else:
            filename = pathlib.Path(filename)

        try:
            import lz4.frame
        except ImportError:
            with open(filename, "wb") as f:
                self._preserve_meta_summaries = preserve_meta_summaries
                self._preserve_config = preserve_config
                pickle.dump(self, f)
                del self._preserve_meta_summaries
                del self._preserve_config
        else:
            with lz4.frame.open(
                filename.with_suffix(filename.suffix + ".lz4"), "wb"
            ) as f:
                self._preserve_meta_summaries = preserve_meta_summaries
                self._preserve_config = preserve_config
                pickle.dump(self, f)
                del self._preserve_meta_summaries
                del self._preserve_config

    @classmethod
    def from_pickle(cls, filename: str | pathlib.Path, read_latest: bool = True):
        """Load the object from a pickle file.

        Parameters
        ----------
        filename : str or Path-like
            The filename to load the object from.
        read_latest : bool, default True
            If True, read the latest file matching the pattern.
        """
        try:
            import lz4.frame
        except ImportError:
            pass
        else:
            # first try lz4 compressed files if available
            try:
                if read_latest:
                    filename_glob = pathlib.Path(filename).with_suffix(".*.pkl.lz4")
                    files = sorted(glob.glob(str(filename_glob)))
                    if not files:
                        if not os.path.exists(filename):
                            raise FileNotFoundError(filename)
                    else:
                        filename = files[-1]

                with lz4.frame.open(filename, "rb") as f:
                    result = pickle.load(f)
                    if result.__class__.__name__ != cls.__name__:
                        raise TypeError(f"Expected {cls}, got {type(result)}")
                    return result
            except FileNotFoundError:
                pass

        if read_latest:
            filename_glob = pathlib.Path(filename).with_suffix(".*.pkl")
            files = sorted(glob.glob(str(filename_glob)))
            if not files:
                if not os.path.exists(filename):
                    raise FileNotFoundError(filename)
            else:
                filename = files[-1]

        with open(filename, "rb") as f:
            result = pickle.load(f)
            if result.__class__.__name__ != cls.__name__:
                raise TypeError(f"Expected {cls}, got {type(result)}")
            return result


def SimulationTable_add_item(name: str, *args, **kwargs):
    item = SimulationTableItem(*args, **kwargs)
    setattr(_GenericSimulationTables, name, item)
    item.__set_name__(_GenericSimulationTables, name)
    setattr(_GenericSimulationTables, "_raw_" + name, property(item._get_raw))


def simulation_table_figure(func):
    """Decorator for figures generated on a _GenericSimulationTables object."""

    wrapped = report_figure(func)

    setattr(_GenericSimulationTables, func.__name__, wrapped)

    # @wraps(func)
    # @report_figure
    # def wrapper(self, *args, **kwargs):
    #     fig = func(self, *args, **kwargs)
    #     self.add_figure(fig)
    #     return fig

    return wrapped
