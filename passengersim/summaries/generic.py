from __future__ import annotations

import glob
import inspect
import os
import pathlib
import pickle
import time
import warnings
from collections.abc import Callable, Collection
from typing import TYPE_CHECKING, Any, ClassVar, Self

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
        aggregation_func: Callable[
            [list[GenericSimulationTables]], pd.DataFrame | None
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
        self.name = name
        owner._std_agg[name] = self._aggregation_func
        owner._std_extract[name] = self._extraction_func
        setattr(owner, "_raw_" + name, property(self._get_raw))

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


class DatabaseTableItem:
    def __init__(
        self,
        query_func: Callable[[Database], pd.DataFrame],
        aggregation_func: Callable[
            [list[GenericSimulationTables]], pd.DataFrame | None
        ],
        doc: str | None = None,
    ):
        self._doc = doc
        self._aggregation_func = aggregation_func
        self._query_func = query_func

    def __set_name__(self, owner, name):
        self.name = name
        owner._std_agg[name] = self._aggregation_func
        owner._std_query[name] = self._query_func

    def __get__(self, instance, owner):
        if instance is None:
            return self
        if self.name not in instance._data:
            try:
                instance.run_queries(items=[self.name])
            except Exception as e:
                warnings.warn(f"Error querying {self.name}: {e}", stacklevel=2)
        try:
            return instance._data[self.name]
        except KeyError:
            raise MissingDataError(self.name) from None

    def __set__(self, instance, value):
        instance._data[self.name] = value

    @property
    def __doc__(self):
        return self._doc

    def _get_raw(self, instance):
        df = instance._data.get(self.name, None)
        return df


class GenericSimulationTables:
    _subclasses: ClassVar[set[type[GenericSimulationTables]]] = set()

    def __init_subclass__(cls, **kwargs):
        """Capture a set of all concrete subclasses"""
        super().__init_subclass__(**kwargs)
        if inspect.isabstract(cls):
            return  # do not include intermediate abstract base classes
        cls._subclasses.add(cls)

    def __init__(
        self,
        data: dict[str, pd.DataFrame] = None,
        *,
        config: Config | None = None,
        cnx: Database | None = None,
        sim: Simulation | None = None,
        n_total_samples: int = 0,
        items: Collection[str] = (),
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

        self._items = items
        """Collection of items that should extracted to create this summary.

        If empty, all items will be extracted."""

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
        "_items",
    }

    def __setattr__(self, item, value):
        """Deny setting of attributes that are not in __writable_attrs."""
        if item in self.__writable_attrs:
            # writable attributes are handled normally
            super().__setattr__(item, value)
        else:
            raise AttributeError(f"Cannot set attribute {item!r}")

    _std_agg: dict[
        str, Callable[[list[GenericSimulationTables]], pd.DataFrame | None]
    ] = {}
    _std_extract: dict[str, Callable[[Simulation], pd.DataFrame]] = {}
    _std_query: dict[str, Callable[..., pd.DataFrame]] = {}

    @classmethod
    def extract(cls, sim: Simulation, items: Collection[str] = ()) -> Self:
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

    def _extract(self: Self, sim: Simulation) -> Self:
        """Extract summary data from a Simulation."""
        return self.__class__.extract(sim, self._items)

    def run_queries(
        self,
        cnx: Database = None,
        items: Collection[str] = (),
        *,
        scenario: str = None,
        burn_samples: int = 100,
    ) -> Self:
        """Query summary data from a Database."""
        if cnx is None:
            cnx = self.cnx
        items = set(items) or self._std_query.keys()
        if burn_samples is None:
            if self.config is not None:
                burn_samples = self.config.simulation_controls.burn_samples
            elif self.sim is not None:
                burn_samples = self.sim.config.simulation_controls.burn_samples
        burn_samples = burn_samples or 100
        for name, query in self._std_query.items():
            if name in items:
                self._data[name] = query(
                    cnx, scenario=scenario, burn_samples=burn_samples
                )
        return self

    @classmethod
    def aggregate(cls, summaries: Collection[GenericSimulationTables]):
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
        mkdir: bool = False,
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
        mkdir : bool, default False
            If True, create the parent directory for the pickle file if it does
            not already exist.  If the directory is created, it will be created
            with a `.gitignore` file to prevent accidental inclusion of pickled
            output in Git repositories.
        """
        if add_timestamp_ext:
            filename = pathlib.Path(filename)
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = filename.with_suffix(f".{timestamp}.pkl")
        else:
            filename = pathlib.Path(filename)
        if mkdir:
            if not filename.parent.exists():
                filename.parent.mkdir(parents=True, exist_ok=True)
                with open(filename.parent / ".gitignore", "w") as f:
                    f.write("*.pkl\n")
                    f.write("*.pkl.lz4\n")

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
