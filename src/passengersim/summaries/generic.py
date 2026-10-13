from __future__ import annotations

import glob
import inspect
import multiprocessing
import os
import pathlib
import platform
import sqlite3
import stat
import tempfile
import time
import traceback
import warnings
from collections.abc import Callable, Collection
from datetime import UTC, datetime
from functools import partialmethod
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Self

import dill as pickle
import pandas as pd

from passengersim.callbacks import CallbackData
from passengersim.config import Config
from passengersim.utils._database_compat import deprecated_cnx_keyword
from passengersim.utils.filenaming import (
    filename_with_timestamp,
    filenames_with_timestamp,
    make_parent_directory,
)
from passengersim.utils.kvstore import KVStore

if TYPE_CHECKING:
    from passengersim import Simulation
    from passengersim.database import Database


def _restore_file_config(value: Config | dict[str, Any] | None) -> Config | None:
    """Adapt retired archive fields without relaxing validation of new input."""
    if not isinstance(value, dict):
        return value
    from copy import deepcopy

    value = deepcopy(value)
    controls = value.get("simulation_controls", {})
    old = controls.get("pax_type_k_factor")
    if old and old == controls.get("simple_k_factor"):
        controls.pop("pax_type_k_factor")
    for demand in value.get("demands", []):
        if demand.get("group_sizes", False) is None:
            demand.pop("group_sizes")
    return Config.model_validate(value, context={"pxsim_archive": True})


class MissingDataError(KeyError):
    """Exception raised when data is missing from a summary table."""

    pass


def initialize_metadata() -> dict[str, Any]:
    """Initialize metadata for the summary."""
    from passengersim_core import __version__ as core_version

    from passengersim import __version__ as version

    metadata = {}
    metadata["time.created"] = datetime.now(UTC).isoformat()
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
        aggregation_func: Callable[[list[GenericSimulationTables]], pd.DataFrame | None],
        extraction_func: Callable[[Simulation], pd.DataFrame | None] = None,
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
            # if we don't have the data we want but we have a file_store,
            # try to load it from there
            if self.name not in instance._data and instance._file_store is not None:
                try:
                    instance._data[self.name] = instance._file_store[self.name]
                except KeyError:
                    pass
            # now work with what we have
            df = instance._data[self.name]
            # make n_total_samples available for computed fields, if it is present in the instance
            # to use in, write `@n_total_samples` in the expression
            local_dict = {}
            n_total_samples = getattr(instance, "n_total_samples", None)
            if n_total_samples is not None:
                local_dict["n_total_samples"] = n_total_samples
            if isinstance(df, Exception):
                raise df
            if df is not None:
                engine = "python" if len(df) < 10000 else "numexpr"
                for field, func in self._computed_fields.items():
                    if field in df:
                        continue
                    try:
                        df[field] = df.eval(func, engine=engine, local_dict=local_dict)
                    except Exception as e:
                        warnings.warn(
                            f"Error computing {field} for {self.name}: {e}",
                            stacklevel=2,
                        )
            return df
        except KeyError:
            raise MissingDataError(self.name) from None

    def __set__(self, instance, value):
        if isinstance(value, pd.DataFrame):
            instance._data[self.name] = value
        else:
            raise TypeError(f"expected DataFrame, got {type(value)}")

    @property
    def __doc__(self):
        return self._doc

    def _get_raw(self, instance):
        df = instance._data.get(self.name, None)
        if df is not None:
            df = df.drop(columns=self._computed_fields.keys(), errors="ignore")
        return df

    def _get_export(self, instance: GenericSimulationTables) -> pd.DataFrame | None:
        """Return stored data unless a table supplies a specialized public view."""
        return instance._data[self.name]


class DatabaseTableItem:
    def __init__(
        self,
        query_func: Callable[[Database], pd.DataFrame],
        aggregation_func: Callable[[list[GenericSimulationTables]], pd.DataFrame | None],
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
            partialmethod(lambda instance, *arg, **kwarg: self._requery(instance, *arg, **kwarg)),
        )

    def __get__(
        self, instance: GenericSimulationTables | None, owner: type[GenericSimulationTables]
    ) -> pd.DataFrame | DatabaseTableItem:
        """Return the descriptor or load its saved table before querying the database."""
        if instance is None:
            return self
        if self.name not in instance._data and instance._file_store is not None and self.name in instance._file_store:
            instance._data[self.name] = instance._file_store[self.name]
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

    # Skip the forwarding lambda used by the generated requery method.
    @deprecated_cnx_keyword(stacklevel=3)
    def _requery(self, instance: GenericSimulationTables, db: Database = None, **kwargs):
        instance.run_queries(db=db, items=[self.name], **kwargs)
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

    @deprecated_cnx_keyword()
    def __init__(
        self,
        data: dict[str, pd.DataFrame] = None,
        *,
        config: Config | None = None,
        db: Database | None = None,
        sim: Simulation | None = None,
        n_total_samples: int = 0,
        items: Collection[str] = (),
        callback_data: CallbackData | None = None,
    ):
        self._file_store = None
        """File store for loading data when it is otherwise missing."""

        self._data = data or {}
        """Dataframes that summarize a Simulation run."""

        self.config = config
        """Configuration for the Simulation run."""

        self.db = db
        """PassengerSim database wrapper for the Simulation run."""

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

        self.callback_data = callback_data or CallbackData()
        """Data collected during callbacks."""

    __writable_attrs = {
        "_file_store",
        "_data",
        "config",
        "_config",
        "db",
        "cnx",
        "_trial_database_store",
        "sim",
        "n_total_samples",
        "meta_summaries",
        "_preserve_meta_summaries",
        "_preserve_config",
        "_items",
        "_metadata",
        "callback_data",
        "_callback_data",
        "_config_loaded",
        "_callback_data_loaded",
    }

    def __setattr__(self, item, value):
        """Deny setting of attributes that are not in __writable_attrs."""
        if item in self.__writable_attrs:
            # writable attributes are handled normally
            super().__setattr__(item, value)
        else:
            raise AttributeError(f"Cannot set attribute {item!r}")

    _std_agg: dict[str, Callable[[list[GenericSimulationTables]], pd.DataFrame | None]] = {}
    _std_extract: dict[str, Callable[[Simulation], pd.DataFrame]] = {}
    _std_query: dict[str, Callable[..., pd.DataFrame]] = {}

    @classmethod
    def extract(cls, sim: Simulation, items: Collection[str] = ()) -> Self:
        """Extract summary data from a Simulation."""
        eng = sim.eng
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
                    try:
                        data[name] = func(sim)
                    except Exception as err:
                        # do not let an exception here take down the whole report
                        warnings.warn(
                            f"Exception {type(err).__name__} in {name}: {err}\n{traceback.format_exc()}",
                            stacklevel=2,
                        )
        return cls(
            data,
            sim=sim,
            config=sim.config,
            db=sim.db,
            n_total_samples=num_samples,
            callback_data=sim.callback_data,
        )

    def _extract(self: Self, sim: Simulation) -> Self:
        """Extract summary data from a Simulation."""
        return self.__class__.extract(sim, self._items)

    @deprecated_cnx_keyword()
    def run_queries(
        self,
        db: Database = None,
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
        db : Database, optional
            PassengerSim database wrapper to use for querying.
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
        if db is None:
            db = self.db
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
                if db is None:
                    warnings.warn(f"no database connection available for {name}", stacklevel=2)
                    self._data[name] = ValueError(f"no database connection available for {name}")
                else:
                    try:
                        self._data[name] = query(db, scenario=scenario, burn_samples=burn_samples)
                    except Exception as e:
                        warnings.warn(f"error in query for {name}: {e}", stacklevel=2)
                        self._data[name] = e
        return self

    @classmethod
    def aggregate(cls, summaries: Collection[GenericSimulationTables]) -> Self:
        """Aggregate multiple summary tables."""
        if not summaries:
            warnings.warn("no summaries available", stacklevel=2)
            return cls({})

        result = cls({})
        for name, func in cls._std_agg.items():
            if func is not None:
                result._data[name] = func(summaries)
        result.meta_summaries = summaries
        result.n_total_samples = sum(s.n_total_samples for s in summaries)
        result.callback_data = sum(s.callback_data for s in summaries)
        return result

    def close(self):
        """Close retained storage and invalidate all shared database aliases.

        Persistent output files are preserved. Closing an active run or closing
        from a different thread/process raises without removing its files.
        """
        store = self.__dict__.get("_trial_database_store")
        if store is not None:
            store.close()
        elif self.db is not None:
            self.db.close()
        if self._file_store is not None:
            self._file_store.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    @property
    def db(self):
        """The shared retained database, opened on first access by the caller."""
        store = self.__dict__.get("_trial_database_store")
        return store.database if store is not None else self.__dict__.get("db")

    @db.setter
    def db(self, value):
        self.__dict__["db"] = value

    @property
    def cnx(self):
        """Deprecated alias for the PassengerSim database wrapper."""
        warnings.warn(".cnx is deprecated; use .db instead.", DeprecationWarning, stacklevel=2)
        return self.db

    @cnx.setter
    def cnx(self, value):
        # Assignment passes through this class's attribute whitelist first.
        warnings.warn(".cnx is deprecated; use .db instead.", DeprecationWarning, stacklevel=3)
        self.db = value

    @property
    def database_status(self):
        """Trial completion and failure status for retained multiprocess results."""
        store = self.__dict__.get("_trial_database_store")
        return store.status if store is not None else None

    @property
    def trial_databases(self):
        """Completed trial files owned by this live result, if available."""
        store = getattr(self, "_trial_database_store", None)
        return dict(store.completed) if store is not None else {}

    def __getstate__(self):
        state = self.__dict__.copy()
        # A local temporary-directory owner cannot survive a disk-cache/pickle
        # round trip. Saved summaries never own or restore a detail database.
        state.pop("_trial_database_store", None)
        state.pop("db", None)
        state.pop("cnx", None)
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
        # Legacy state may contain the old name; restoration is not deprecated API use.
        state = state.copy()
        legacy_db = state.pop("cnx", None)
        state.setdefault("db", legacy_db)
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
                        print(f"{i + 1:>4} | {line}")
                except Exception:
                    pass
                raise
        self.__dict__.update(state)
        if "db" not in self.__dict__:
            self.db = None
        if "sim" not in self.__dict__:
            self.sim = None
        if "_config" not in self.__dict__:
            self._config = None
        if "n_total_samples" not in self.__dict__:
            self.n_total_samples = 0
        if "_metadata" not in self.__dict__:
            self._metadata = {}
        if "_callback_data" not in self.__dict__:
            self._callback_data = CallbackData()

    def to_pickle(
        self,
        filename: str | pathlib.Path,
        add_timestamp_ext: bool = True,
        *,
        preserve_meta_summaries: bool = False,
        preserve_config: bool = True,
        make_dirs: Literal[True, False, "git"] = True,
    ) -> pathlib.Path:
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

        Returns
        -------
        Path-like
            The resolved filename for the saved outputs.
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
                        f.write(".gitignore\n")  # ignore this file itself
                        f.write("*.pkl\n")  # ignore pickles
                        f.write("*.pkl.lz4\n")  # ignore compressed pickles

        try:
            import lz4.frame
        except ImportError:
            with open(filename, "wb") as f:
                self._preserve_meta_summaries = preserve_meta_summaries
                self._preserve_config = preserve_config
                pickle.dump(self, f)
                del self._preserve_meta_summaries
                del self._preserve_config
            return filename
        else:
            use_filename = filename.with_suffix(filename.suffix + ".lz4")
            with lz4.frame.open(use_filename, "wb") as f:
                self._preserve_meta_summaries = preserve_meta_summaries
                self._preserve_config = preserve_config
                pickle.dump(self, f)
                del self._preserve_meta_summaries
                del self._preserve_config
            return use_filename

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
                            result._metadata["loaded.time"] = datetime.now(UTC).isoformat()
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
                                result._metadata["loaded.time"] = datetime.now(UTC).isoformat()
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
                result._metadata["loaded.time"] = datetime.now(UTC).isoformat()
            return result

    @property
    def database_path(self):
        """Absolute main-database path, or None when no file backs the result."""
        store = self.__dict__.get("_trial_database_store")
        if store is not None:
            return store.consolidated_path
        database = self.__dict__.get("db")
        return database.database_path if database is not None else None

    @property
    def database_available(self):
        """Whether this live result has retained detail data, without opening it."""
        store = self.__dict__.get("_trial_database_store")
        if store is not None:
            return bool(store.available)
        database = self.__dict__.get("db")
        return database is not None and database.connection is not None

    def to_file(
        self,
        filename: str | pathlib.Path,
        add_timestamp_ext: bool = True,
        *,
        preserve_config: bool = True,
        make_dirs: Literal[True, False, "git"] = True,
    ) -> pathlib.Path:
        """Atomically write simulation tables to a chunked PXSIM archive.

        Parameters
        ----------
        filename : Path-like
            The file to write.
        add_timestamp_ext : bool, default True
            Add a timestamp extension to the filename.
        preserve_config : bool, default True
            Preserve the config attribute in the saved object.  This includes
            the entire network, and can potentially be a lot of data.
        make_dirs : bool or "git", default True
            If True, create the parent directory for the file if it does
            not already exist.  If the directory is created, it will be created
            with a `.gitignore` file to prevent accidental inclusion of output
            in Git repositories, unless the value is "git", in which case
            no `.gitignore` file is created and the results will be eligible for
            inclusion in Git.

        Returns
        -------
        pathlib.Path
            The output filename, with timestamp when requested.

        Raises
        ------
        ValueError
            Serialization fails or the target is a hard-link alias of the source.
        OSError
            File creation or atomic publication fails.

        Notes
        -----
        Unloaded entries are copied without unpickling when possible. Loaded
        values are serialized again to preserve in-place edits. Publication
        failure leaves the existing target intact. Older readers that support
        only the legacy layout cannot read newly written version 2 files.
        Detailed simulation data stays in a separate SQLite file and is never
        embedded or modified by this export. New files honor the process umask;
        replacing an existing file preserves its permission bits.
        """
        if add_timestamp_ext:
            filename = filename_with_timestamp(filename, suffix=".pxsim")
        else:
            filename = pathlib.Path(filename)
        if make_dirs:
            git_ignores = ["*.pxsim", "*.pkl.lz4", "*.pkl"] if make_dirs != "git" else False
            make_parent_directory(filename, git_ignores)
        if filename.is_symlink():
            filename = filename.resolve()
        if self._file_store is not None and filename.exists():
            source_path = pathlib.Path(self._file_store.path)
            if (
                source_path.exists()
                and source_path.resolve() != filename.resolve()
                and os.path.samefile(source_path, filename)
            ):
                raise ValueError("Cannot export over a hard-link alias of the open source; use its original path")
        # A private sibling directory keeps unfinished data inaccessible while
        # ordinary file creation applies the umask without changing global state.
        destination_mode = stat.S_IMODE(filename.stat().st_mode) if filename.exists() else None
        temporary_directory = tempfile.TemporaryDirectory(prefix=".pxsim-", dir=filename.parent)
        temporary = pathlib.Path(temporary_directory.name) / "archive.pxsim"
        kvs = None
        reopen = False
        source_read_only = self._file_store.read_only if self._file_store is not None else False
        try:
            with temporary.open("xb"):
                pass
            kvs = KVStore(temporary, mode="w", format_version=2)
            with kvs.transaction():
                keys = set(self._data)
                if self._file_store is not None:
                    keys.update(k for k in self._file_store if not (k.startswith("_") and k.endswith("_")))
                for k in keys:
                    if preserve_config and k.endswith("_defs"):
                        continue
                    if k.endswith("_") and not k.startswith("_"):
                        continue
                    if k in self._data:
                        kvs[k] = self._data[k]
                    else:
                        kvs.copy_entry_from(self._file_store, k)
                if preserve_config:
                    if not getattr(self, "_config_loaded", True) and self._file_store is not None:
                        if "_config_" in self._file_store:
                            kvs.copy_entry_from(self._file_store, "_config_")
                    elif self.config is not None:
                        kvs["_config_"] = self.config.model_dump(serialize_as_any=True)
                kvs["_metadata_"] = self._metadata
                kvs["_n_total_samples_"] = self.n_total_samples
                if not getattr(self, "_callback_data_loaded", True) and self._file_store is not None:
                    if "_callback_data_" in self._file_store:
                        kvs.copy_entry_from(self._file_store, "_callback_data_")
                elif self.callback_data:
                    kvs["_callback_data_"] = self.callback_data
            kvs.close()
            kvs = None
            if destination_mode is not None:
                temporary.chmod(destination_mode)
            source = self._metadata.get("store.filename")
            if self._file_store is not None and source and pathlib.Path(source).resolve() == filename.resolve():
                self._file_store.close()
                reopen = True
            # Replacing only after all summary data are complete prevents
            # failed exports from destroying a previous good cache.
            os.replace(temporary, filename)
        finally:
            try:
                if kvs is not None:
                    kvs.close()
                if reopen:
                    self._file_store = KVStore(filename, mode="r" if source_read_only else "rw")
            finally:
                temporary_directory.cleanup()
        return filename

    @classmethod
    def from_file(
        cls,
        filename: str | pathlib.Path,
        read_latest: bool = True,
        lazy: bool = True,
        *,
        read_only: bool = False,
    ) -> Self:
        """Load summary tables, configuration, and metadata from a saved result.

        Parameters
        ----------
        filename : str or pathlib.Path
            PXSIM or legacy pickle file to read.
        read_latest : bool, default True
            Select the latest timestamped file when present.
        lazy : bool, default True
            Load summary tables on demand.
        read_only : bool, default False
            Open the PXSIM archive read-only. Otherwise, attempt writable access
            and fall back to read-only access if needed. This option does not
            affect legacy pickles.

        Returns
        -------
        Self
            Loaded result. Close it, or use it as a context manager, to release
            archive connections. Detailed simulation databases are not restored.

        Raises
        ------
        FileNotFoundError
            The selected file does not exist.
        ValueError
            Archive format, integrity, or configuration validation fails.

        Notes
        -----
        Existing legacy PXSIM files are read without schema migration.
        Only load trusted files: summary values use Python pickle.
        """
        if str(filename).endswith((".pkl", ".pkl.lz4")):
            result = cls.from_pickle(filename, read_latest=read_latest)
            return result
        if read_latest:
            files = sorted(glob.glob(str(pathlib.Path(filename).with_suffix(".*.pxsim"))))
            if files:
                filename = files[-1]
        if not pathlib.Path(filename).exists():
            raise FileNotFoundError(filename)
        result = cls()
        try:
            try:
                result._file_store = KVStore(filename, mode="r" if read_only else "rw")
            except (PermissionError, sqlite3.OperationalError) as error:
                # Loading is a read operation even when optional later edits
                # are permitted. Explicit KVStore writable opens remain strict.
                if read_only or (
                    isinstance(error, sqlite3.OperationalError)
                    and getattr(error, "sqlite_errorcode", None) != sqlite3.SQLITE_CANTOPEN
                ):
                    raise
                result._file_store = KVStore(filename, mode="r")
            result._config_loaded = False
            result._callback_data_loaded = False
            result._metadata = result._file_store["_metadata_"]
            result.n_total_samples = result._file_store["_n_total_samples_"]
            result._metadata["store.filename"] = filename
            result._metadata["database.available"] = False
            if not lazy:
                for k in result._file_store:
                    if not (k.startswith("_") and k.endswith("_")):
                        result._data[k] = result._file_store[k]
                    if k == "_callback_data_":
                        result.callback_data = result._file_store[k]
                    if k == "_config_":
                        temp = result._file_store[k]
                        result.config = _restore_file_config(temp)
            return result
        except BaseException:
            result.close()
            raise

    def file_info(self):
        """Return information about the file store."""
        from passengersim.utils.si import si_units

        if self._file_store is None:
            return "No file store"
        info = f"File store: {self.metadata('store.filename')}"
        info += f"\n  {'Data Element':30} Disk Usage"
        sizes = pd.Series(self._file_store.get_sizes()).sort_values(ascending=False)
        for k, v in sizes.items():
            info += f"\n* {k:30} {si_units(v, 'B')}"
        return info

    def remove_data(self, keys: Collection[str] | str) -> Self:
        """Remove data from the summary tables.

        This can be used to reduce the size of the summary tables when saving
        to a file, or to remove sensitive data before sharing the summary tables.

        Parameters
        ----------
        keys : Collection[str] or str
            The key(s) of the data to remove.

        Returns
        -------
        Self
            The summary tables object, with the specified data removed.

        Raises
        ------
        PermissionError
            The backing archive is read-only; no in-memory data is removed.

        Warns
        -----
        UserWarning
            Deletion succeeded but reclaiming unused disk space failed.

        Notes
        -----
        Missing keys are ignored. Backing-file deletions are atomic and
        complete before in-memory entries are removed.
        """
        if isinstance(keys, str):
            keys = [keys]
        if self._file_store is not None:
            with self._file_store.transaction():
                for key in keys:
                    if key in self._file_store:
                        del self._file_store[key]
        for key in keys:
            self._data.pop(key, None)
        if self._file_store is not None:
            try:
                self._file_store.vacuum()
            except Exception as error:
                warnings.warn(f"Data removed, but PXSIM compaction failed: {error}", stacklevel=2)
        return self

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
                    descriptor = getattr(type(self), k, None)
                    if isinstance(descriptor, SimulationTableItem):
                        v = descriptor._get_export(self)
                    v.to_excel(writer, sheet_name=k)

    def to_html(
        self,
        filename: str | pathlib.Path,
        *,
        cfg: Config | None = None,
        make_dirs: bool = True,
        extra: tuple = (),
        add_timestamp: bool = True,
    ) -> pathlib.Path:
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
        extra : tuple, optional
            Additional data to include in the report.  Each item in the tuple should
            either a section or subsection title, or a tuple of (title, func), or
            just a function.  If a function is provided, it should take the summary
            as its only argument and return a figure (altair.Chart or xmle.Elem) or
            table (pandas.DataFrame).  The function will be called with the summary
            as its only argument. To use a function that requires other arguments,
            use `functools.partial` provide the other arguments.
        add_timestamp : bool, default True
            If True, append a timestamp to the filename.  This ensures that each
            report is unique and does not overwrite previous reports.  If False,
            the filename will be used as-is. Set this to False if you want to
            overwrite previous reports with the same filename, or if you are
            already setting the timestamp yourself.

        Returns
        -------
        Path-like
            The resolved filename for the saved outputs.
        """
        from passengersim.reporting.html import to_html

        return to_html(self, filename, cfg=cfg, make_dirs=make_dirs, extra=extra, add_timestamp=add_timestamp)

    def save(
        self,
        filename: str | pathlib.Path,
        *,
        timestamp: float | time.struct_time | datetime | None = None,
        make_dirs: Literal[True, False, "git"] = True,
        cfg: Config | None = None,
        extra_html: tuple = (),
    ) -> dict[str, pathlib.Path]:
        """Save the object to a set of files.

        This method will write both an HTML report on this simulation tables
        object and a ".pxsim" file allowing the content to be restored.

        Parameters
        ----------
        filename : Path-like
            The file stem to use for writing files.
        timestamp : float or time.struct_time or datetime, optional
            The timestamp to use for the filenames.  If not provided, the current
            time will be used.
        make_dirs : bool or "git", default True
            If True, create the parent directory for the files if it does not
            already exist.  If the directory is created, it will be created with
            a `.gitignore` file to prevent accidental inclusion of output in Git
            repositories, unless the value is "git", in which case no `.gitignore`
            file is created and the results will be eligible for inclusion in Git.
        cfg : Config, optional
            The configuration to use for the HTML report.  If None, the configuration
            from the simulation object will be used if available.
        extra_html : tuple, optional
            Additional data to include in the HTML report. This argument is passed
            to `to_html`, see that function for more details.

        Returns
        -------
        dict
            A dictionary of filenames written, including the timestamp added.
        """
        if make_dirs:
            make_parent_directory(
                filename,
                git_ignore_things=False if make_dirs == "git" else ["*.pxsim", "*.html"],
            )
        filename = pathlib.Path(filename)
        filenames = filenames_with_timestamp(filename, timestamp=timestamp, suffix=[".pxsim", ".html"])
        self.to_html(filenames[".html"], cfg=cfg, make_dirs=False, extra=extra_html)
        self.to_file(filenames[".pxsim"], make_dirs=False, add_timestamp_ext=False)
        return filenames

    def metadata(self, key: str = ""):
        """Return a metadata value."""
        if key in self._metadata:
            return self._metadata[key]
        if "." in key:
            # dotted keys are always exact matches
            raise KeyError(key)
        if key == "":
            return self._metadata.copy()
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

    @property
    def config(self) -> Config | None:
        """Saved simulation configuration, loaded on first access, or None."""
        if not getattr(self, "_config_loaded", self._config is not None) and self._file_store is not None:
            try:
                temp = self._file_store["_config_"]
            except KeyError:
                self._config_loaded = True
            else:
                self.config = _restore_file_config(temp)
        return self._config

    @config.setter
    def config(self, value: Config | None) -> None:
        """Replace the configuration and mark its lazy value as loaded."""
        self._config = value
        self._config_loaded = True

    @config.deleter
    def config(self) -> None:
        """Clear the configuration without reloading it from the archive."""
        self.config = None

    @property
    def callback_data(self) -> CallbackData:
        """Saved callback outputs, loaded on first access."""
        if not getattr(self, "_callback_data_loaded", bool(self._callback_data)) and self._file_store is not None:
            try:
                self.callback_data = self._file_store["_callback_data_"]
            except KeyError:
                self._callback_data_loaded = True
        return self._callback_data

    @callback_data.setter
    def callback_data(self, value: CallbackData) -> None:
        """Replace callback data and mark its lazy value as loaded."""
        self._callback_data = value
        self._callback_data_loaded = True

    @callback_data.deleter
    def callback_data(self) -> None:
        """Clear callback data without reloading it from the archive."""
        self.callback_data = CallbackData()
