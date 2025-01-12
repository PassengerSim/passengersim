from __future__ import annotations

import glob
import inspect
import multiprocessing
import os
import pathlib
import pickle
import platform
import warnings
from collections.abc import Callable, Collection
from datetime import datetime, timezone
from functools import partialmethod
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Self

import pandas as pd

from passengersim.config import Config
from passengersim.utils.filenaming import filename_with_timestamp

if TYPE_CHECKING:
    from passengersim import Simulation
    from passengersim.database import Database


class MissingDataError(KeyError):
    """Exception raised when data is missing from a summary table."""

    pass


def initialize_metadata() -> dict[str, Any]:
    """Initialize metadata for the summary."""
    from passengersim_core import __version__ as core_version

    from passengersim import __version__ as version

    metadata = {}
    metadata["time.created"] = datetime.now(timezone.utc).isoformat()
    metadata["machine.system"] = platform.system()
    metadata["machine.release"] = platform.release()
    metadata["machine.version"] = platform.version()
    metadata["machine.machine"] = platform.machine()
    metadata["machine.processor"] = platform.processor()
    metadata["machine.architecture"] = platform.architecture()
    metadata["machine.node"] = platform.node()
    metadata["machine.platform"] = platform.platform()
    metadata["machine.python_version"] = platform.python_version()
    metadata["machine.cpu_count"] = multiprocessing.cpu_count()
    metadata["version.passengersim"] = version
    metadata["version.passengersim_core"] = core_version
    return metadata


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
            if isinstance(df, Exception):
                raise df
            if df is not None:
                engine = "python" if len(df) < 10000 else "numexpr"
                for field, func in self._computed_fields.items():
                    if field in df:
                        continue
                    try:
                        df[field] = df.eval(func, engine=engine)
                    except Exception as e:
                        warnings.warn(
                            f"Error computing {field} for {self.name}: {e}",
                            stacklevel=2,
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
        setattr(owner, "_raw_" + name, property(self._get_raw))
        setattr(
            owner,
            "_requery_" + name,
            partialmethod(
                lambda instance, *arg, **kwarg: self._requery(instance, *arg, **kwarg)
            ),
        )

    def __get__(self, instance, owner):
        if instance is None:
            return self
        if self.name not in instance._data:
            try:
                instance.run_queries(items=[self.name])
            except Exception as e:
                warnings.warn(f"Error querying {self.name}: {e}", stacklevel=2)
        try:
            df = instance._data[self.name]
            if isinstance(df, Exception):
                raise df
            return df
        except KeyError:
            raise MissingDataError(self.name) from None

    def __set__(self, instance, value):
        instance._data[self.name] = value

    @property
    def __doc__(self):
        return self._doc

    def _get_raw(self, instance: GenericSimulationTables):
        df = instance._data.get(self.name, None)
        return df

    def _requery(
        self, instance: GenericSimulationTables, cnx: Database = None, **kwargs
    ):
        instance.run_queries(cnx=cnx, items=[self.name], **kwargs)
        return self._get_raw(instance)


class GenericSimulationTables:
    __subclasses: ClassVar[set[type[GenericSimulationTables]]] = set()

    @classmethod
    def subclasses(cls) -> list[type[GenericSimulationTables]]:
        """Return a list of all concrete subclasses.

        User defined subclasses (those not in the passengersim package)
        are at the front of the list, so they come first in MRO and
        thus can override native subclasses.
        """
        subs = []
        for sub in cls.__subclasses:
            if getattr(sub, "__final__", False):
                # do not include classes marked as final
                continue
            if sub.__module__.startswith("passengersim.summaries"):
                # these are native subclasses
                subs.append(sub)
            else:
                subs.insert(0, sub)
        subs.append(GenericSimulationTables)
        return subs

    def __init_subclass__(cls, **kwargs):
        """Capture a set of all concrete subclasses"""
        super().__init_subclass__(**kwargs)
        if inspect.isabstract(cls):
            return  # do not include intermediate abstract base classes
        cls.__subclasses.add(cls)

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

        self._metadata = initialize_metadata()
        """Metadata for the summary."""

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
        "_metadata",
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
                f"\n- num_trials_completed = {eng.num_trials_completed}"
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
        items: Collection[str] | None = None,
        *,
        scenario: str = None,
        burn_samples: int | None = None,
    ) -> Self:
        """Query summary data from a Database.

        The requested items will be queried from the database and stored in this
        summary object.  If the item is not available, an exception will be raised.

        Parameters
        ----------
        cnx : Database, optional
            Database connection to use for querying.
        items : Collection[str], optional
            The items to query.  If None, or if only "*" is given, then all
            available items will be queried.
        scenario : str, optional
            The scenario to use for querying.
        burn_samples : int, optional
            The number of burn samples to use for querying. If explicitly `None`,
            the burn_samples value from the configuration will be used if available,
            otherwise the default value of 100 will be used.
        """
        if cnx is None:
            cnx = self.cnx
        if items is None or len(items) == 1 and "*" in items:
            items = self._std_query.keys()
        else:
            items = set(items)
        if burn_samples is None:
            if self.config is not None:
                burn_samples = self.config.simulation_controls.burn_samples
            elif self.sim is not None:
                burn_samples = self.sim.config.simulation_controls.burn_samples
        if burn_samples is None:
            burn_samples = 100
        for name, query in self._std_query.items():
            if name in items:
                if cnx is None:
                    warnings.warn(
                        f"no database connection available for {name}", stacklevel=2
                    )
                    self._data[name] = ValueError(
                        f"no database connection available for {name}"
                    )
                else:
                    try:
                        self._data[name] = query(
                            cnx, scenario=scenario, burn_samples=burn_samples
                        )
                    except Exception as e:
                        warnings.warn(f"error in query for {name}: {e}", stacklevel=2)
                        self._data[name] = e
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
            content = state.pop("_config_yaml")
            try:
                state["config"] = Config.from_raw_yaml(content)
            except Exception:
                try:
                    if isinstance(content, bytes):
                        content_lines = content.decode("utf-8").split("\n")
                    else:
                        content_lines = content.split("\n")
                    for i, line in enumerate(content_lines):
                        print(f"{i+1:>4} | {line}")
                except Exception:
                    pass
                raise
        self.__dict__.update(state)
        if "cnx" not in self.__dict__:
            self.cnx = None
        if "sim" not in self.__dict__:
            self.sim = None
        if "config" not in self.__dict__:
            self.config = None
        if "n_total_samples" not in self.__dict__:
            self.n_total_samples = 0
        if "_metadata" not in self.__dict__:
            self._metadata = {}

    def to_pickle(
        self,
        filename: str | pathlib.Path,
        add_timestamp_ext: bool = True,
        *,
        preserve_meta_summaries: bool = False,
        preserve_config: bool = True,
        make_dirs: Literal[True, False, "git"] = True,
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
        make_dirs : bool or "git", default True
            If True, create the parent directory for the pickle file if it does
            not already exist.  If the directory is created, it will be created
            with a `.gitignore` file to prevent accidental inclusion of pickled
            output in Git repositories, unless the value is "git", in which case
            no `.gitignore` file is created and the results will be eligible for
            inclusion in Git.
        """
        if add_timestamp_ext:
            filename = filename_with_timestamp(filename, suffix=".pkl")
        else:
            filename = pathlib.Path(filename)
        if make_dirs:
            if not filename.parent.exists():
                filename.parent.mkdir(parents=True, exist_ok=True)
                if make_dirs != "git":
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

                try:
                    with lz4.frame.open(filename, "rb") as f:
                        try:
                            result = pickle.load(f)
                        except Exception as e:
                            raise RuntimeError(f"Error loading {filename}: {e}") from e
                        # if result.__class__.__name__ != cls.__name__:
                        #     raise TypeError(f"Expected {cls}, got {type(result)}")
                        if hasattr(result, "_metadata"):
                            result._metadata["loaded.filename"] = filename
                            result._metadata["loaded.time"] = datetime.now(
                                timezone.utc
                            ).isoformat()
                        return result
                except RuntimeError as err:
                    if "LZ4F_decompress failed" in str(err):
                        # lz4 frame error, try uncompressed file
                        with open(filename, "rb") as f:
                            result = pickle.load(f)
                            # if result.__class__.__name__ != cls.__name__:
                            #     raise TypeError(f"Expected {cls}, got {type(result)}")
                            if hasattr(result, "_metadata"):
                                result._metadata["loaded.filename"] = filename
                                result._metadata["loaded.time"] = datetime.now(
                                    timezone.utc
                                ).isoformat()
                            return result
                    raise
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
            if hasattr(result, "_metadata"):
                result._metadata["loaded.filename"] = filename
                result._metadata["loaded.time"] = datetime.now(timezone.utc).isoformat()
            return result

    def to_xlsx(self, filename: str | pathlib.Path) -> None:
        """Write simulation tables to excel.

        Parameters
        ----------
        filename : Path-like
            The excel file to write.
        """
        if isinstance(filename, str):
            filename = pathlib.Path(filename)
        filename.parent.mkdir(exist_ok=True, parents=True)
        with pd.ExcelWriter(filename) as writer:
            for k, v in self._data.items():
                if isinstance(v, pd.DataFrame):
                    v.to_excel(writer, sheet_name=k)

    def to_html(
        self,
        filename: str | pathlib.Path,
        *,
        cfg: Config | None = None,
        make_dirs: bool = True,
    ) -> None:
        """Write simulation tables report summary to html.

        Parameters
        ----------
        filename : Path-like, optional
            The html file to write.
        cfg : Config, optional
            The configuration to use for the report.  If None, the configuration
            from the simulation object will be used.
        make_dirs : bool, default True
            If True, create any necessary directories.
        """
        from passengersim.reporting.html import to_html

        to_html(self, filename, cfg=cfg, make_dirs=make_dirs)

    def metadata(self, key: str):
        """Return a metadata value."""
        if key in self._metadata:
            return self._metadata[key]
        if "." in key:
            # dotted keys are always exact matches
            raise KeyError(key)
        matches = {}
        for k in self._metadata:
            subkeys = k.split(".")
            subkey = subkeys[0]
            others = ".".join(subkeys[1:])
            if subkey == key:
                matches[others] = self._metadata[k]
        if not matches:
            raise KeyError(key)
        return matches
