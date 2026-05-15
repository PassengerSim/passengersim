# TITLE: Config
# DOC-NAME: 00-configs
from __future__ import annotations

import io
import logging
import os
import pathlib
import time
import typing
import warnings
from datetime import UTC, datetime
from typing import Annotated, Any, ClassVar
from urllib.request import urlopen
from zoneinfo import ZoneInfo

import addicty
import pandas as pd
import yaml
from pydantic import (
    AfterValidator,
    Field,
    SerializerFunctionWrapHandler,
    ValidationError,
    field_serializer,
    field_validator,
    model_validator,
)

from passengersim.pseudonym import random_label
from passengersim.rm.systems import (
    check_registered_rm_system,
    describe_rm_systems,
    list_registered_rm_systems,
    reload_rm_systems,
)
from passengersim.utils.airport_lookup import lookup_airport
from passengersim.utils.close_matching import did_you_mean
from passengersim.utils.compression import (
    deserialize_from_file,
    serialize_to_file,
    smart_open,
)
from passengersim.utils.file_freshness import check_modification_times, preprocess_filenames

from ._csv import load_from_csv
from ._preprocess import preprocess_config
from .blf_curves import BlfCurve
from .booking_curves import BookingCurve
from .carriers import Carrier
from .choice_model import ChoiceModel
from .circuity_rules import CircuityRule
from .database import DatabaseConfig
from .dataframes import _DataFrameAccessor
from .demands import Demand
from .fares import Fare
from .frat5_curves import Frat5Curve
from .legs import Leg
from .load_factor_curves import LoadFactorCurve
from .markets import Market
from .named import DictAttr, DictOfNamed, ListOfNamed
from .outputs import OutputConfig
from .paths import Path
from .places import Place, great_circle
from .pretty import PrettyModel, repr_dict_with_indent
from .rm_steps import RmStepBase
from .rm_systems import RmSystem
from .simulation_controls import SimulationSettings
from .snapshot_filter import SnapshotFilter
from .speed_limits import get_speed_limits
from .todd_curves import ToddCurve

PathLike = str | os.PathLike[str]

if typing.TYPE_CHECKING:
    from typing import Literal, Self

    from pydantic.main import IncEx

logger = logging.getLogger("passengersim.config")

_warn_skips = (os.path.dirname(__file__),)

TConfig = typing.TypeVar("TConfig", bound="YamlConfig")

_TABULAR_COMPATIBLE_KEYS = {"demands", "fares", "legs"}


def web_opener(x):
    return urlopen(x.parts[0] + "//" + "/".join(x.parts[1:]))


def _path_to_str(x):
    """Convert paths in a nested structure to strings."""
    if isinstance(x, dict):
        return {k: _path_to_str(v) for k, v in x.items()}
    if isinstance(x, list):
        return list(_path_to_str(i) for i in x)
    if isinstance(x, tuple):
        return list(_path_to_str(i) for i in x)
    if isinstance(x, pathlib.Path):
        return str(x)
    else:
        return x


class OptionalPath(pathlib.Path):
    """A pathlib.Path that, if missing, is ignored by the Yaml loader."""

    pass


class YamlConfig(PrettyModel):
    @classmethod
    def _load_unformatted_yaml(
        cls: type[TConfig],
        filenames: str | pathlib.Path | list[str] | list[pathlib.Path],
    ) -> addicty.Dict:
        """
        Read from YAML to an unvalidated addicty.Dict.

        Parameters
        ----------
        filenames : path-like or list[path-like]
            If multiple filenames are provided, they are loaded in order
            and values with matching keys defined in later files will overwrite
            the ones found in earlier files.

        Returns
        -------
        addicty.Dict
        """
        if isinstance(filenames, str | bytes | os.PathLike):
            filenames = [filenames]
        raw_config = addicty.Dict()
        for filename in filenames:
            t = time.time()
            if isinstance(filename, str) and "\n" in filename:
                # explicit YAML content cannot have include statements
                content = addicty.Dict.load(filename, freeze=False, Loader=yaml.CSafeLoader)
                raw_config.update(content)
                continue
            if not isinstance(filename, pathlib.Path):
                filename = pathlib.Path(filename)
            if filename.suffix in (".pem", ".crt", ".cert"):
                # license certificate
                with open(filename, "rb") as f:
                    raw_config.raw_license_certificate = f.read()
            else:
                opener = smart_open
                if filename.parts[0] in {"https:", "http:", "s3:"}:
                    opener = web_opener
                    if filename.suffix == ".gz" or filename.suffix == ".lz4":
                        raise NotImplementedError("cannot load compressed files from web yet")
                if isinstance(filename, OptionalPath) and not filename.exists():
                    continue
                with opener(filename) as f:
                    content = addicty.Dict.load(f, freeze=False, Loader=yaml.CSafeLoader)
                    if content is None:
                        warnings.warn(
                            f"Empty file {filename}",
                            skip_file_prefixes=_warn_skips,
                            stacklevel=1,
                        )
                        continue
                    include = content.pop("include", None)
                    if include is not None:
                        if isinstance(include, str):
                            filename.parent.joinpath(include)
                            inclusions = [filename.parent.joinpath(include)]
                        else:
                            inclusions = [filename.parent.joinpath(i) for i in include]
                        raw_config.update(cls._load_unformatted_yaml(inclusions))

                    # look for selected keys that are lists, but can be given as a CSV table
                    for tab_key in _TABULAR_COMPATIBLE_KEYS:
                        if tab_key in content and isinstance(content[tab_key], str | pathlib.Path):
                            content[tab_key] = _path_to_str(content[tab_key])
                            content[tab_key] = load_from_csv(filename.parent.joinpath(content[tab_key]))
                    raw_config.update(content)
            logger.info("loaded config from %s in %.2f secs", filename, time.time() - t)
        return raw_config

    @classmethod
    def from_yaml(
        cls: type[TConfig],
        filenames: PathLike | list[PathLike],
        *,
        cache_file: PathLike | None = None,
        on_validation_error: Literal["raise", "warn"] = "raise",
    ) -> TConfig:
        """
        Read from YAML.

        Parameters
        ----------
        filenames : path-like or list[path-like]
            If multiple filenames are provided, they are loaded in order
            and values with matching keys defined in later files will overwrite
            the ones found in earlier files.
        cache_file : path-like, optional
            If provided, the validated config will be cached to this file in
            binary format using pickle.  If the cache file exists and is
            newer than the YAML files, the cached config will be loaded
            instead of reloading and revalidating the YAML files, which can be
            considerably faster.
        on_validation_error : {'raise', 'warn'}, default 'raise'
            Whether to raise an exception or log a warning when a validation
            error is encountered. If 'warn', the error is logged and the
            unvalidated raw loaded yaml content (not a Config object) is returned.
            This can be useful for debugging YAML files that are failing validation,
            but it is recommended to leave this as 'raise' in production to avoid
            accidentally using an unvalidated config.

        Returns
        -------
        Config
        """
        filenames = preprocess_filenames(filenames, expand_includes=False)
        cache_is_outdated = True
        if cache_file:
            cache_is_outdated = check_modification_times(filenames, cache_file)
            if cache_is_outdated:
                logger.info(f"cache file is {cache_is_outdated}, will reload YAML files")
        if not cache_file or cache_is_outdated:
            raw_config = cls._load_unformatted_yaml(filenames)
            t = time.time()
            try:
                result = cls.model_validate(raw_config.to_dict())
            except ValidationError as e:
                if on_validation_error == "raise":
                    raise
                warnings.warn(str(e), stacklevel=2)
                return raw_config
            logger.info("validated config in %.2f secs", time.time() - t)
            if cache_file:
                t = time.time()
                serialize_to_file(cache_file, result)
                logger.info("cached config in %.2f secs", time.time() - t)
            return result
        else:
            t = time.time()
            result = deserialize_from_file(cache_file)
            logger.info("loaded config from cache in %.2f secs", time.time() - t)
            return result

    @classmethod
    def from_raw_yaml(cls, content: str | bytes) -> Self:
        """
        Read from raw YAML content.

        Parameters
        ----------
        content : str or bytes
            The YAML content to parse.

        Returns
        -------
        Config
        """
        if isinstance(content, bytes):
            content = content.decode("utf8")
        raw_config = addicty.Dict.load(content, freeze=False, Loader=yaml.CSafeLoader)
        return cls.model_validate(raw_config.to_dict())

    tags: dict[str, Any] = {}
    """Tags that can be used in format strings in the config."""

    @model_validator(mode="before")
    @classmethod
    def _parse_format_tags(cls, data: Any) -> Any:
        """Parse format tags in the config."""
        tags = {}
        if "tags" in data:
            tags.update(data["tags"])
        if "scenario" in data:
            tags["scenario"] = data["scenario"]
        tags["date_Ymd"] = time.strftime("%Y-%m-%d")
        tags["time_HM"] = time.strftime("%H%M")

        def apply_tags(x):
            if isinstance(x, dict):
                return {k: apply_tags(v) for k, v in x.items()}
            if isinstance(x, list):
                return [apply_tags(i) for i in x]
            if isinstance(x, str):
                return x.format(**tags)
            return x

        return apply_tags(data)

    def to_yaml(
        self,
        stream: os.PathLike | io.FileIO | None = None,
        *,
        include: IncEx = None,
        exclude: IncEx = None,
        exclude_unset: bool = False,
        exclude_defaults: bool = False,
        exclude_none: bool = False,
        warnings: bool = True,
        human_readable: bool = True,
    ) -> None | bytes:
        """
        Write a config to YAML format.

        Parameters
        ----------
        stream : Path-like or File-like, optional
            Write the results here.  If given as a path, a new file is written
            at this location, or give a File-like object open for writing.
        include : list[int | str]
            A list of fields to include in the output.
        exclude : list[int | str]
            A list of fields to exclude from the output.
        exclude_unset : bool, default False
            Whether to exclude from the output fields that were unset when
            the Config was created.  There are a variety of cleaning and updating
            functions that may have been run on this Config that would set things
            that were originally not set (e.g. path building), so you probably
            want to leave this as `False` unless you know what you are doing.
        exclude_defaults : bool, default False
            Whether to exclude fields that are set to their default value from
            the output.
        exclude_none : bool, default False
            Whether to exclude fields that have a value of `None` from the output.
        warnings : bool, default True
            Whether to log warnings when invalid fields are encountered.
        human_readable : bool, default True
            Whether to write the YAML files in a human-readable format.  This will
            reformat inputs back to normal human-readable values (e.g. leg departure
            times as 'HH:MN' in local time, not an integer giving unix time.

        Returns
        -------
        bytes or None
            When no stream is given, the YAML content is returned as bytes,
            otherwise this method returns nothing.
        """

        y = _path_to_str(
            self.model_dump(
                include=include,
                exclude=exclude,
                exclude_unset=exclude_unset,
                exclude_defaults=exclude_defaults,
                exclude_none=exclude_none,
                warnings=warnings,
                context={"human_readable": True} if human_readable else {},
            )
        )
        b = yaml.dump(y, encoding="utf8", Dumper=yaml.CSafeDumper)
        if isinstance(stream, str):
            stream = pathlib.Path(stream)
        if isinstance(stream, pathlib.Path):
            if stream.suffix == ".lz4":
                with smart_open(stream, "wb") as f:
                    f.write(b)
            else:
                stream.write_bytes(b)
        elif isinstance(stream, io.RawIOBase):
            stream.write(b)
        elif isinstance(stream, io.TextIOBase):
            stream.write(b.decode())
        else:
            return b

    def to_yaml_parts(
        self,
        directory: pathlib.Path | str,
        *,
        include: IncEx = None,
        exclude: IncEx = None,
        exclude_unset: bool = False,
        exclude_defaults: bool = False,
        exclude_none: bool = False,
        warnings: bool = True,
        general_config: tuple[str] = ("simulation_controls", "tags"),
        human_readable: bool = True,
        compress_large_files: bool = True,
    ) -> None:
        """Write to a set of YAML files, one for each top level key.

        Parameters
        ----------
        directory : pathlib.Path or str
            The directory to write the YAML files to. If it does not exist,
            it will be created.
        include : list[int | str], optional
            A list of fields to include in the output.
        exclude : list[int | str], optional
            A list of fields to exclude from the output.
        exclude_unset : bool, default False
            Whether to exclude from the output fields that were unset when
            the Config was created.  There are a variety of cleaning and updating
            functions that may have been run on this Config that would set things
            that were originally not set (e.g. path building), so you probably
            want to leave this as `False` unless you know what you are doing.
        exclude_defaults : bool, default False
            Whether to exclude fields that are set to their default value from
            the output.
        exclude_none : bool, default False
            Whether to exclude fields that have a value of `None` from the output.
        warnings : bool, default True
            Whether to log warnings when invalid fields are encountered.
        general_config : tuple[str], default ("simulation_controls", "tags")
            The top-level keys to write to the general config file (general.yaml)
            instead of their own files.
        human_readable : bool, default True
            Whether to write the YAML files in a human-readable format.  This will
            reformat inputs back to normal human-readable values (e.g. leg departure
            times as 'HH:MN' in local time, not an integer giving unix time.
        compress_large_files : bool, default True
            Whether to compress files larger than 1MB with lz4.  This can be helpful
            to reduce file sizes when saving large networks.
        """
        if isinstance(directory, str):
            directory = pathlib.Path(directory)
        if not directory.exists():
            directory.mkdir(parents=True, exist_ok=True)
        y = _path_to_str(
            self.model_dump(
                include=include,
                exclude=exclude,
                exclude_unset=exclude_unset,
                exclude_defaults=exclude_defaults,
                exclude_none=exclude_none,
                warnings=warnings,
                context={"human_readable": True} if human_readable else {},
            )
        )
        remainder = {"rm_systems_setup": describe_rm_systems(self)}
        files_written = []
        tables_written = {}
        for key, value in y.items():
            if key == "raw_license_certificate":
                # do not write the license certificate to a file
                continue
            if key in _TABULAR_COMPATIBLE_KEYS:
                filename = directory / f"{key}.csv"
                pd.DataFrame(value).to_csv(filename, index=False)
                if compress_large_files and filename.stat().st_size > 1024 * 1024:
                    from passengersim.utils.compression import compress_file

                    compressed_filename = compress_file(filename, rm_original=True)
                    tables_written[key] = pathlib.Path(compressed_filename).name
                else:
                    tables_written[key] = filename.name
            elif isinstance(value, dict | list) and key not in general_config:
                filename = directory / f"{key}.yaml"
                # when the value is empty, we don't write a file for it
                if value:
                    with open(filename, "w", encoding="utf8") as f:
                        yaml.dump({key: value}, f, Dumper=yaml.CSafeDumper, sort_keys=False)
                    # if the file just written is bigger than 1MB, compress it with lz4
                    if compress_large_files and filename.stat().st_size > 1024 * 1024:
                        from passengersim.utils.compression import compress_file

                        compress_file(filename, rm_original=True)
                        files_written.append(f"{key}.yaml.lz4")
                    else:
                        files_written.append(f"{key}.yaml")
            else:
                remainder[key] = value
        if remainder:
            # write the general config to a separate file
            filename = directory / "general.yaml"
            with open(filename, "w", encoding="utf8") as f:
                yaml.dump(remainder, f, Dumper=yaml.CSafeDumper, sort_keys=False)
            files_written.append("general.yaml")
        if files_written:
            filename = directory / "__main__.yaml"
            with open(filename, "w", encoding="utf8") as f:
                yaml.dump({"include": files_written, **tables_written}, f, Dumper=yaml.CSafeDumper, sort_keys=False)


def find_differences(left, right):
    """Find the differences between two nested dictionaries."""
    if isinstance(left, dict) and isinstance(right, dict):
        diff = {}
        for key in left.keys() | right.keys():
            if key not in right:
                diff[key] = "missing in right"
            elif key not in left:
                diff[key] = "missing in left"
            else:
                sub_diff = find_differences(left[key], right[key])
                if sub_diff:
                    diff[key] = sub_diff
        return diff
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return "different lengths"
        diff = {}
        for i, (left_item, right_item) in enumerate(zip(left, right)):
            sub_diff = find_differences(left_item, right_item)
            if sub_diff:
                diff[i] = sub_diff
        return diff
    if isinstance(left, dict) and not isinstance(right, dict):
        return "left is dict and right is not"
    if isinstance(right, dict) and not isinstance(left, dict):
        return "right is dict and left is not"
    if isinstance(left, list) and not isinstance(right, list):
        return "left is list and right is not"
    if isinstance(right, list) and not isinstance(left, list):
        return "right is list and left is not"
    if left == right:
        return {}
    return f"{left} != {right}"


class filterable_list(list):
    def set_filters(self, **kwargs):
        """Filter items in this list based on matching attributes."""
        queue = self.__class__()
        for i in self:
            for k, v in kwargs.items():
                try:
                    if getattr(i, k) != v:
                        break
                except AttributeError:
                    if isinstance(i, dict):
                        if i.get(k, None) != v:
                            break
                    else:
                        raise
            else:
                queue.append(i)
        return queue

    def select(self, **kwargs):
        """Select one item from the list matching attributes."""
        for i in self:
            for k, v in kwargs.items():
                try:
                    if getattr(i, k) != v:
                        break
                except AttributeError:
                    if isinstance(i, dict):
                        if i.get(k, None) != v:
                            break
                    else:
                        raise
            else:
                return i
        raise ValueError(f"no items matching {kwargs!r}")

    def model_dump(self, *args, **kwargs):
        """Call model_dump on all objects."""
        return [i.model_dump(*args, **kwargs) for i in self]


# Validator that wraps a standard list into your subclass
def make_filterable(v: list) -> filterable_list:
    return filterable_list(v)


# Define the reusable annotated type
FilterableList = Annotated[list, AfterValidator(make_filterable)]


class Config(YamlConfig, extra="forbid"):
    dataframes: ClassVar = _DataFrameAccessor()
    """Accessor for getting selected configuration settings as a Pandas DataFrame."""

    scenario: str = Field(default_factory=random_label)
    """Name for this scenario.

    The scenario name is helpful when writing different simulations to the same
    database so you can uniquely identify and query results for a particular
    scenario."""

    simulation_controls: SimulationSettings = SimulationSettings()
    """
    Controls that apply broadly to the overall simulation.

    See [SimulationSettings][passengersim.config.SimulationSettings] for detailed
    documentation.
    """

    db: DatabaseConfig = DatabaseConfig()
    """
    See [passengersim.config.DatabaseConfig][] for detailed documentation.
    """

    @model_validator(mode="before")
    @classmethod
    def _db_is_none(cls, data: Any) -> Any:
        """Setting the database to none creates a null database."""
        if isinstance(data, dict):
            db = data.get("db", None)
            if db is None:
                db = DatabaseConfig(filename=None, write_items=set())
            data["db"] = db
        return data

    @field_serializer("db", mode="wrap")
    @classmethod
    def _db_to_none(cls, v: DatabaseConfig | None, nxt: SerializerFunctionWrapHandler) -> dict | None:
        """Serialize the database to None if it is a null database."""
        if v is None:
            return None
        if v.filename is None and not v.write_items:
            return None
        return nxt(v)

    outputs: OutputConfig = OutputConfig()
    """
    See [passengersim.config.OutputConfig][] for detailed documentation.
    """

    rm_systems: DictOfNamed[RmSystem] = {}
    """
    The revenue management systems used by the carriers in this simulation.

    See [RM Systems][rm-systems] for details.
    """

    @model_validator(mode="before")
    @classmethod
    def _rm_systems_removed_before(cls, raw: Any) -> Any:
        """RM Systems are now defined on each carrier."""
        if isinstance(raw, dict) and "rm_systems" in raw and len(raw["rm_systems"]):
            warnings.warn(
                "rm_systems has been deprecated; set rm_system on each carrier instead",
                stacklevel=2,
                category=DeprecationWarning,
            )
            raw["rm_systems"] = {}
        return raw

    @field_validator("rm_systems")
    @classmethod
    def _rm_systems_removed(cls, v: DictOfNamed[RmSystem]) -> DictOfNamed[RmSystem]:
        """RM Systems are now defined on each carrier."""
        if len(v):
            warnings.warn(
                "rm_systems has been deprecated; set rm_system on each carrier instead",
                stacklevel=2,
                category=DeprecationWarning,
            )
        return DictAttr()

    @model_validator(mode="before")
    @classmethod
    def _rm_systems_setup(cls, raw: Any) -> Any:
        """Tool to set up RM systems based on instructions in the config."""
        if isinstance(raw, dict) and "rm_systems_setup" in raw:
            ss = raw.pop("rm_systems_setup")
            reload_rm_systems(ss)
        return raw

    blf_curves: DictOfNamed[BlfCurve] = {}
    """ Booked Load Factor curves"""

    frat5_curves: DictOfNamed[Frat5Curve] = {}
    """ FRAT5 curves are used to model sellup rates in Q-forecasting"""

    def get_frat5_curve(self, name: str, deep_copy: bool = False) -> Frat5Curve:
        """Get a Frat5 curve from this configuration (if defined) or from the standard set.

        PassengerSim has a number of "standard" Frat5 curves available, which do not need
        to be explicitly defined in your config.  These standard curves run over a 63 day
        booking curve and follow those used historically in PODS.

        Parameters
        ----------
        name : str
            The name of the Frat5 curve to retrieve.  If this curve is not explicitly defined
            in the config, it will be loaded from the standard set of curves.
        deep_copy : bool, default False
            Whether to return a deep copy of the curve.  This can be useful if you want to
            modify the curve for a particular Carrier without affecting other Carriers
            that use the same curve.

        Returns
        -------
        Frat5Curve

        Raises
        ------
        KeyError
            If the name is not a defined Frat5 curve in this configuration, nor a standard
            name.
        """
        if name not in self.frat5_curves:
            self._load_std_frat5(name)
        curve = self.frat5_curves[name]
        if deep_copy:
            curve = curve.model_copy(deep=True)
        return curve

    load_factor_curves: DictOfNamed[LoadFactorCurve] = {}
    """ FRAT5 curves are used to model sellup rates in Q-forecasting"""

    todd_curves: DictOfNamed[ToddCurve] = {}
    """ Time of Day curves"""

    choice_models: DictOfNamed[ChoiceModel] = {}
    """Several choice models are programmed behind the scenes.

    The choice_models option allows the user to set the parameters used in the
    utility model for a particular choice model. There are two choice models
    currently programmed.
    1. PODS-like
    2. MNL, using the Lurkin et. al. paper (needs more testing and pdating)

    Need to explaining more here"""

    carriers: DictOfNamed[Carrier] = {}
    """A list of carriers.

    One convention is to use Airline1, Airline2, ... to list the carriers in the
    network.  Another convention is to use IATA industry-standard two-letter airline
    codes.  See the
    [IATA code search](https://www.iata.org/en/publications/directories/code-search/)
    for more information."""

    places: DictOfNamed[Place] = {}
    """A list of places (airports, vertiports, other stations)."""

    def get_place(self, name: str, *, error_if_missing: bool = True) -> Place | None:
        """Get a place from the config, or look it up if not defined.

        Parameters
        ----------
        name : str
            The name of the place to retrieve.  If this place is not defined, it is assumed
            the name is an IATA code, and it will be loaded from the standard set of places.
        error_if_missing : bool, default True
            Whether to raise an error if the place is not found in the config or as a standard
            place.  If False, this method will return None instead of raising an error when
            the place is not found.

        Returns
        -------
        Place

        Raises
        ------
        KeyError
            If the name is not a defined place in this configuration, and cannot be found
            as an IATA code in the standard set of places.
        """
        if name not in self.places:
            try:
                self.places[name] = lookup_airport(name)
            except KeyError:
                if error_if_missing:
                    raise
                else:
                    return None
        return self.places[name]

    circuity_rules: ListOfNamed[CircuityRule] = []
    """Specifies exceptions and the default rule"""

    classes: list[str] = []
    """A list of fare classes.

    One convention is to use Y0, Y1, ... to label fare classes from the highest
    fare (Y0) to the lowest fare (Yn).  You can also use Y, B, M, H,... etc.
    An example of classes is below.

    Example
    -------
    ```{yaml}
    classes:
      - Y0
      - Y1
      - Y2
      - Y3
      - Y4
      - Y5
    ```
    """

    @model_validator(mode="before")
    @classmethod
    def _classes_are_for_carriers(cls, data: Any) -> Any:
        """Any carrier that doesn't have its own classes gets the global ones."""
        if isinstance(data, dict):
            carriers = data.get("carriers", {})
            if isinstance(carriers, list):
                for carrier in carriers:
                    if isinstance(carrier, dict) and "classes" not in carrier:
                        carrier["classes"] = data.get("classes", [])
            else:
                for carrier in carriers.values():
                    if isinstance(carrier, dict) and "classes" not in carrier:
                        carrier["classes"] = data.get("classes", [])
            data["carriers"] = carriers
        return data

    dcps: list[int] = []
    """A list of DCPs (data collection points).

    The DCPs are given as integers, which represent the number of days
    before departure.   An example of data collection points is given below.
    Note that typically as you get closer to day of departure (DCP=0) the number
    of days between two consecutive DCP periods decreases.  The DCP intervals are
    shorter because as you get closer to departure, customer arrival rates tend
    to increase, and it is advantageous to forecast changes in demand for shorter
    intervals.

    Example
    -------
    ```{yaml}
    dcps: [63, 56, 49, 42, 35, 31, 28, 24, 21, 17, 14, 10, 7, 5, 3, 1]
    ```
    """

    booking_curves: DictOfNamed[BookingCurve] = {}
    """Booking curves

    The booking curve points typically line up with the DCPs.

    Example
    -------
    ```{yaml}
    booking_curves:
      - name: c1
        curve:
          63: 0.06
          56: 0.11
          49: 0.15
          42: 0.2
          35: 0.23
          31: 0.25
          28: 0.28
          24: 0.31
          21: 0.35
          17: 0.4
          14: 0.5
          10: 0.62
          7: 0.7
          5: 0.78
          3: 0.95
          1: 1.0
    """

    legs: Annotated[list[Leg], AfterValidator(make_filterable)] = []
    demands: Annotated[list[Demand], AfterValidator(make_filterable)] = []
    fares: Annotated[list[Fare], AfterValidator(make_filterable)] = []
    paths: Annotated[list[Path], AfterValidator(make_filterable)] = []
    markets: Annotated[list[Market], AfterValidator(make_filterable)] = []

    other_controls: dict[str, Any] = {}

    @property
    def markets_dict(self):
        result = {}
        for m in self.markets:
            if isinstance(m, dict):
                m = Market(**m)
            ident = f"{m.orig}~{m.dest}"
            if ident in result:
                raise ValueError(f"Duplicate market {ident}")
            result[ident] = m
        return result

    @model_validator(mode="after")
    def _all_demands_have_markets(self) -> Self:
        """All demands must have a market that matches one of the markets defined in the config.

        It is valid to have markets not defined in the raw input, in which case a default-initialized
        market will be created for any market that appears in the demands but not in the markets list.
        This allows users to only specify markets that they want to customize, and not have to define
        a market for every single OD pair in the network.
        """
        market_idents = {f"{m.orig}~{m.dest}" for m in self.markets}
        for d in self.demands:
            ident = f"{d.orig}~{d.dest}"
            if ident not in market_idents:
                mkt = Market(orig=d.orig, dest=d.dest)
                self.markets.append(mkt)
                market_idents.add(ident)
        return self

    @model_validator(mode="after")
    def _fare_ap_restrictions_must_match_dcps(self) -> Self:
        """AP restrictions on fare can only be invoked at the DCPs."""
        for f in self.fares:
            if f.advance_purchase != 0 and f.advance_purchase not in self.dcps:
                raise ValueError(f"Advance purchase restriction not aligned with DCP for Fare {f}")
        return self

    @field_validator("markets")
    @classmethod
    def _no_duplicate_markets(cls, v: list[Market]) -> list[Market]:
        """Check for duplicate markets."""
        seen = set()
        for mkt in v:
            ident = f"{mkt.orig}~{mkt.dest}"
            if ident in seen:
                raise ValueError(f"Duplicate market {ident}")
            seen.add(ident)
        return v

    snapshot_filters: list[SnapshotFilter] = []

    @field_validator("snapshot_filters", mode="before")
    @classmethod
    def _handle_no_snapshot_filters(cls, v):
        if v is None:
            v = []
        return v

    dwm_tolerance: list[dict] = []
    """Each item is a dictionary of {min_distance, max_distance, business, leisure}
       The segments are named, so you can add more in the future, and they will be
       matched agains the Demand segment name by the loader"""

    raw_license_certificate: bytes | None | bool = None

    @field_validator("raw_license_certificate", mode="before")
    def _handle_license_certificate(cls, v):
        if isinstance(v, str) and v.startswith("-----BEGIN CERTIFICATE-----"):
            v = v.encode("utf8")
        return v

    @property
    def license_certificate(self):
        from cryptography.x509 import load_pem_x509_certificate

        if isinstance(self.raw_license_certificate, bytes):
            return load_pem_x509_certificate(self.raw_license_certificate)
        elif self.raw_license_certificate is False:
            return False
        elif self.raw_license_certificate is None:
            return None
        raise ValueError("invalid license certificate")

    @model_validator(mode="after")
    def _burn_samples(self) -> Self:
        """burn_samples must be strictly less than samples"""
        if self.simulation_controls.burn_samples >= self.simulation_controls.num_samples:
            raise ValueError(
                "burn_samples must be strictly less than samples. It will default to 100 if you haven't set a value"
            )
        if self.simulation_controls.burn_samples == 0 and self.simulation_controls.num_samples > 10:
            raise ValueError("to ensure meaningful results, burn_samples may not be zero when num_samples > 10.")
        return self

    @model_validator(mode="after")
    def _manual_paths(self) -> Self:
        """Check that paths are provided if they are required, not not if they are prohibited."""
        if self.simulation_controls.connection_builder.existing_paths == "required" and len(self.paths) == 0:
            raise ValueError(
                "`connection_builder.existing_paths` is set to 'required', but no paths found in the input config"
            )
        if self.simulation_controls.connection_builder.existing_paths == "none" and len(self.paths) > 0:
            raise ValueError(
                "`connection_builder.existing_paths` is set to 'none', but some paths were found in the input config"
            )
        return self

    def _load_std_rm_system(self, std_name: str):
        """Load a standard RM system from the standard RM systems file.

        Parameters
        ----------
        std_name : str
            The name of the standard RM system to load.
        """
        from .standards import standard_rm_systems_raw

        raw_rm_systems = standard_rm_systems_raw()
        if std_name in raw_rm_systems:
            rm_sys = Config.model_validate({"rm_systems": {std_name: raw_rm_systems[std_name]}})
            self.rm_systems[std_name] = rm_sys.rm_systems[std_name]
        else:
            raise KeyError(f"Unknown standard RM system {std_name}")

    @model_validator(mode="after")
    def _carriers_have_rm_systems(self) -> Self:
        """Check that all carriers have RM systems that have been defined."""
        for carrier in self.carriers.values():
            assert isinstance(carrier, Carrier)
            # an RM system name must be defined somewhere for each carrier
            if not carrier.rm_system and not isinstance(carrier.rm_system_options, dict):
                raise ValueError(f"Carrier {carrier.name} has no RM system name defined")
            # that RM system must be stored in the config at `carrier.rm_system`
            # if it's not but it's defined in carrier.rm_system_options, that fixable
            if not carrier.rm_system:
                carrier.rm_system = carrier.rm_system_options.pop("name", None)
            # if the name is defined in rm_system_options, it must match carrier.rm_system
            if carrier.rm_system_options is not None and carrier.rm_system_options is not False:
                if carrier.rm_system_options.get("name", carrier.rm_system) != carrier.rm_system:
                    raise ValueError(f"rm_system {carrier.rm_system} does not match name in rm_system_options")
            # if the named system is supposed to be a callback-style RM system, check that it's registered
            if carrier.rm_system_options:
                # it's definitely supposed to be a callback-style RM system
                if not check_registered_rm_system(carrier.rm_system):
                    raise ValueError(f"Carrier {carrier.name} has unregistered RM system {carrier.rm_system}")
            elif not check_registered_rm_system(carrier.rm_system) or carrier.rm_system_options is False:
                # the named system is not a registered callback-style system, or the user has explicitly
                # disabled callback-style RM systems for this carrier
                # so it must be defined in rm_systems, or be an old-style standard RM system we can load
                raise ValueError(
                    did_you_mean(
                        f"Carrier {carrier.name} has unknown RM system {carrier.rm_system}",
                        carrier.rm_system,
                        list_registered_rm_systems(),
                    )
                ) from None
        return self

    def _load_std_frat5(self, std_name: str):
        """Load a standard Frat5 curve from the standard Frat5 file.

        Parameters
        ----------
        std_name : str
            The name of the standard Frat5 curve to load.
        """
        from passengersim import demo_network

        std_cfg = Config.from_yaml(demo_network("standard-frat5.yaml"))
        if std_name in std_cfg.frat5_curves:
            self.frat5_curves[std_name] = std_cfg.frat5_curves[std_name]
        else:
            raise KeyError(f"Unknown standard Frat5 curve {std_name}")

    @model_validator(mode="after")
    def _carriers_have_frat5(self) -> Self:
        """Check that all carriers have defined or null Frat5 curves."""
        for carrier in self.carriers.values():
            if carrier.frat5 and carrier.frat5 not in self.frat5_curves:
                try:
                    self._load_std_frat5(carrier.frat5)
                except KeyError:
                    raise ValueError(f"Carrier {carrier.name} has unknown Frat5 curve {carrier.frat5}") from None
        return self

    @model_validator(mode="after")
    def _legs_have_carriers(self) -> Self:
        """Check that all legs have a carrier that has been defined."""
        for leg in self.legs:
            if leg.carrier not in self.carriers:
                raise ValueError(f"Carrier for leg {leg.carrier} {leg.fltno} is not defined")
        return self

    @model_validator(mode="after")
    def _legs_have_unique_leg_ids(self) -> Self:
        """Check that all legs have a unique leg_id."""
        all_leg_ids = set()
        # First, scan for duplicate assigned leg_ids, which is an error.
        for leg in self.legs:
            if leg.leg_id is not None:
                if leg.leg_id in all_leg_ids:
                    raise ValueError(f"Leg ID {leg.leg_id} is duplicated")
                else:
                    all_leg_ids.add(leg.leg_id)
        # Then, for all legs without a leg_id but with a flt_no,
        # if the flt_no is available make it the leg_id
        for leg in self.legs:
            if leg.leg_id is None and leg.fltno is not None and leg.fltno not in all_leg_ids:
                leg.leg_id = leg.fltno
                all_leg_ids.add(leg.leg_id)
        # Finally, for all remaining legs, assign them a new unique leg_id
        next_available_id = 1
        for leg in self.legs:
            if leg.leg_id is None:
                while next_available_id in all_leg_ids:
                    next_available_id += 1
                leg.leg_id = next_available_id
                all_leg_ids.add(leg.leg_id)
        return self

    @model_validator(mode="after")
    def _choice_model_todd_curves_exist(self) -> Self:
        """Check that any TODD curves referenced in Demand objects have been defined."""
        for name, cm in self.choice_models.items():
            if cm.todd_curve is not None and cm.todd_curve not in self.todd_curves:
                raise ValueError(f"ChoiceModel {name} has unknown TOD Curve {cm.todd_curve}")
        return self

    @model_validator(mode="after")
    def _choice_model_airline_preferences(self) -> Self:
        """Check that only one way of inputting airline preference was specified."""
        for _name, cm in self.choice_models.items():
            a1 = 1 if cm.airline_pref_pods is not None else 0
            a2 = 1 if cm.airline_pref_hhi is not None else 0
            a3 = 1 if cm.airline_pref_seat_share is not None else 0
            if a1 + a2 + a3 > 1:
                raise ValueError(f"ChoiceModel '{cm.name}' has more than one airline preference model specified")
        return self

    @model_validator(mode="after")
    def _choice_model_curve_s_vs_replanning(self) -> Self:
        """Check that only one way of inputting airline preference was specified."""
        for _name, cm in self.choice_models.items():
            if cm.replanning is not None and (cm.early_dep is not None or cm.late_arr is not None):
                raise ValueError(
                    f"ChoiceModel '{cm.name}' has replanning and early_dep / late_arr "
                    f"specified, pick one or the other but not both !!!"
                )
        return self

    @model_validator(mode="after")
    def _choice_set_sampling(self) -> Self:
        """Ensure there is a limit on the number of observations in a choice set.
        Don't allow a choice set to be created without a specified limit of observations
        as unlimited sampling will run out of storage very quickly"""
        if (
            len(self.simulation_controls.capture_choice_set_file) > 0
            and self.simulation_controls.capture_choice_set_obs is None
        ):
            self.simulation_controls.capture_choice_set_obs = 10000
            warnings.warn(
                "capture_choice_set_obs not specified, has been set to 10000",
                stacklevel=2,
            )
        return self

    @model_validator(mode="after")
    def _demand_todd_curves_exist(self) -> Self:
        """Check that any TODD curves referenced in Demand objects have been defined."""
        for dmd in self.demands:
            if dmd.todd_curve is not None and dmd.todd_curve not in self.todd_curves:
                raise ValueError(f"Demand {dmd.orig}-{dmd.dest}:{dmd.segment} has unknown TOD Curve {dmd.todd_curve}")
        return self

    @model_validator(mode="after")
    def _demand_booking_curves_exist(self) -> Self:
        """Check that any booking curves referenced in Demand objects have been defined."""
        for dmd in self.demands:
            if dmd.curve is not None and dmd.curve not in self.booking_curves:
                raise ValueError(
                    f"Demand {dmd.orig}-{dmd.dest}:{dmd.segment} has unknown customer arrival curve {dmd.curve!r}"
                )
        return self

    @model_validator(mode="after")
    def _reference_price_variation_vs_multiplier(self) -> Self:
        """Prevent combining per-OD reference price variation with a non-trivial
        choice-model ``reference_price_multiplier``.

        At most one of the following may be true:

        1. Within any (orig, dest) grouping of demands, the ``reference_price``
           values vary (i.e., not all equal).
        2. Any choice model has an attribute ``reference_price_multiplier`` set
           to a non-None value other than 1.0.

        Having both simultaneously leads to ambiguous/compounding reference
        price semantics, so this is disallowed.
        """
        # Condition 1: variation of reference_price within any (orig, dest) group.
        od_prices: dict[tuple[str, str], set[float | None]] = {}
        for dmd in self.demands:
            od_prices.setdefault((dmd.orig, dmd.dest), set()).add(dmd.reference_price)
        varying_od = [od for od, prices in od_prices.items() if len(prices) > 1]
        cond1 = bool(varying_od)

        # Condition 2: any choice model has reference_price_multiplier set to
        # a non-None value other than 1.0. Use getattr so this applies to any
        # future choice model class that defines a similarly named attribute.
        offending_cms: list[str] = []
        for name, cm in self.choice_models.items():
            mult = getattr(cm, "reference_price_multiplier", None)
            if mult is not None and mult != 1.0:
                offending_cms.append(name)
        cond2 = bool(offending_cms)

        if cond1 and cond2:
            raise ValueError(
                "Config has both per-OD variation in Demand.reference_price "
                f"(e.g., OD groups {varying_od[:3]}) and choice model(s) with a "
                f"non-trivial reference_price_multiplier (e.g., {offending_cms[:3]}). "
                "At most one of these may be used; please pick exactly one mechanism."
            )
        return self

    @model_validator(mode="after")
    def _booking_curves_match_dcps(self) -> Self:
        """Check that all booking curves are complete and valid."""
        sorted_dcps = reversed(sorted(self.dcps))
        for curve in self.booking_curves.values():
            i = 0
            for dcp in sorted_dcps:
                assert dcp in curve.curve, f"booking curve {curve.name} is missing dcp {dcp}"
                assert curve.curve[dcp] >= i, f"booking curve {curve.name} moves backwards at dcp {dcp}"
                i = curve.curve[dcp]
        return self

    @model_validator(mode="after")
    def _requested_summaries_have_data(self) -> Self:
        """Check that requested summary outputs will have the data needed."""
        if "local_and_flow_yields" in self.outputs.reports:
            if not self.db.write_items & {"pathclass_final", "pathclass"}:
                raise ValueError(
                    "the `local_and_flow_yields` report requires recording "
                    "at least `pathclass_final` details in the database"
                )
        if "bid_price_history" in self.outputs.reports:
            if "leg" not in self.db.write_items:
                raise ValueError("the `bid_price_history` report requires recording `leg` details in the database")
            if not self.db.store_leg_bid_prices:
                raise ValueError("the `bid_price_history` report requires recording `store_leg_bid_prices` to be True")
        if "demand_to_come" in self.outputs.reports:
            if "demand" not in self.db.write_items:
                raise ValueError("the `demand_to_come` report requires recording `demand` details in the database")
        if "demand_to_come_summary" in self.outputs.reports:
            if "demand" not in self.db.write_items:
                raise ValueError(
                    "the `demand_to_come_summary` report requires recording `demand` details in the database"
                )
        if "path_forecasts" in self.outputs.reports:
            if "pathclass" not in self.db.write_items:
                raise ValueError("the `path_forecasts` report requires recording `pathclass` details in the database")
        if "leg_forecasts" in self.outputs.reports:
            if "bucket" not in self.db.write_items:
                raise ValueError("the `leg_forecasts` report requires recording `bucket` details in the database")
        if "bookings_by_timeframe" in self.outputs.reports:
            if not self.db.write_items & {"bookings", "fare"}:
                raise ValueError(
                    "the `bookings_by_timeframe` report requires recording `fare` or `bookings` details in the database"
                )
        if "total_demand" in self.outputs.reports:
            if not self.db.write_items & {"demand", "demand_final"}:
                raise ValueError(
                    "the `total_demand` report requires recording at least `demand_final` details in the database"
                )
        if "fare_class_mix" in self.outputs.reports:
            if not self.db.write_items & {"fare", "fare_final"}:
                raise ValueError(
                    "the `fare_class_mix` report requires recording at least `fare_final` details in the database"
                )
        if "load_factor_distribution" in self.outputs.reports:
            if not self.db.write_items & {"leg", "leg_final"}:
                raise ValueError(
                    "the `load_factor_distribution` report requires recording "
                    "at least `leg_final` details in the database"
                )
        if "edgar" in self.outputs.reports:
            if not self.db.write_items & {"edgar"}:
                raise ValueError(
                    "the 'edgar' forecast accuray report requires recording 'edgar' details in the database"
                )
        return self

    @model_validator(mode="after")
    def _bp_controls_are_expected_but_not_set(self) -> Self:
        """Warn if bid price controls are expected but not set."""
        for rm_system in self.rm_systems.values():
            if "dcp" in rm_system.processes:
                for step in rm_system.processes["dcp"]:
                    if isinstance(step, RmStepBase):
                        try:
                            req = step.require_availability_control
                        except AttributeError:
                            req = None
                        if req is not None and rm_system.availability_control not in req:
                            raise ValueError(
                                f"RM System {rm_system.name} requires availability control {req} for step {step.name}"
                            )
        return self

    __rm_steps_loaded: ClassVar[set[type[RmStepBase]]] = RmStepBase._get_subclasses()

    def model_revalidate(
        self,
    ) -> typing.Self:
        """Revalidate the passengersim Config instance."""
        return self.model_validate(self.model_dump(serialize_as_any=True))

    def find_differences(
        self,
        other: Config,
        *,
        include: IncEx = None,
        exclude: IncEx = None,
    ) -> dict:
        """Find the differences between two Config objects."""
        if exclude is None:
            exclude = {
                "raw_license_certificate": True,
                "outputs": {"pickle", "excel", "html", "log_reports"},
            }
        return find_differences(
            self.model_dump(include=include, exclude=exclude),
            other.model_dump(include=include, exclude=exclude),
        )

    def add_output_prefix(self, prefix: pathlib.Path, spool_format: str = "%Y%m%d-%H%M"):
        """
        Add a prefix directory to all simulation output files.
        """
        if not isinstance(prefix, pathlib.Path):
            prefix = pathlib.Path(prefix)
        if spool_format:
            proposal = prefix.joinpath(time.strftime(spool_format))
            n = 0
            while proposal.exists():
                n += 1
                proposal = prefix.joinpath(time.strftime(spool_format) + f".{n}")
            prefix = proposal
        prefix.mkdir(parents=True)

        if self.db.filename:
            self.db.filename = prefix.joinpath(self.db.filename)
        if self.outputs.excel:
            self.outputs.excel = prefix.joinpath(self.outputs.excel)
        for sf in self.snapshot_filters:
            if sf.directory:
                sf.directory = prefix.joinpath(sf.directory)
        return prefix

    @model_validator(mode="after")
    def _attach_distance_to_things_without_it(self):
        """Attach distance in nautical miles to legs that are missing distance."""
        for leg in self.legs:
            if leg.distance is None:
                place_o = self.get_place(leg.orig)
                place_d = self.get_place(leg.dest)
                if place_o is not None and place_d is not None:
                    leg.distance = round(great_circle(place_o.lat, place_o.lon, place_d.lat, place_d.lon), 3)
                if place_o is None:
                    warnings.warn(f"No defined place for {leg.orig}", stacklevel=2)
                if place_d is None:
                    warnings.warn(f"No defined place for {leg.dest}", stacklevel=2)
        for dmd in self.demands:
            if not dmd.distance:
                place_o = self.get_place(dmd.orig)
                place_d = self.get_place(dmd.dest)
                if place_o is not None and place_d is not None:
                    dmd.distance = round(great_circle(place_o.lat, place_o.lon, place_d.lat, place_d.lon), 3)
                if place_o is None:
                    warnings.warn(f"No defined place for {dmd.orig}", stacklevel=2)
                if place_d is None:
                    warnings.warn(f"No defined place for {dmd.dest}", stacklevel=2)
        return self

    @model_validator(mode="after")
    def _adjust_times_for_time_zones(self):
        """Adjust arrival/departure times to local time from UTC."""

        negative_duration_legs = []
        unreasonable_leg_speeds = []

        def adjust_time_zone(t, place, explicit_tz: str | None = None):
            if place is not None:
                if (
                    explicit_tz is not None
                    and place.time_zone_info is not None
                    and str(explicit_tz) != str(place.time_zone_info)
                ):
                    warnings.warn(
                        f"Explicit time zone {explicit_tz} does not match place "
                        f"time zone {place.time_zone_info}, using explicit time zone",
                        stacklevel=2,
                    )
                tz = explicit_tz or place.time_zone_info
                # explicit_tz arrives as a plain string when it was stored
                # in a previous serialization (e.g. after a human-readable
                # round-trip via to_yaml_parts).  datetime() requires a
                # tzinfo subclass, so coerce strings to ZoneInfo objects.
                if isinstance(tz, str):
                    tz = ZoneInfo(tz)
                if tz is not None:
                    # Alan's approach
                    # It was converted as a local time, so unpack it and
                    #   create a new datetime in the given TZ
                    dt = datetime.fromtimestamp(t, tz=UTC)
                    dt2 = datetime(
                        dt.year,
                        dt.month,
                        dt.day,
                        dt.hour,
                        dt.minute,
                        0,
                        0,
                        tzinfo=tz,
                    )
                    new_ts = int(dt2.timestamp())
                    return new_ts, t - new_ts
            return t, 0

        for leg in self.legs:
            # the nominal time is local time but so far got stored as UTC,
            # so we need to add the time zone offset to be actually local time

            if not leg.time_adjusted:
                place_o = self.get_place(leg.orig)
                leg.dep_time, leg.dep_time_offset = adjust_time_zone(leg.dep_time, place_o, leg.orig_timezone)
                if not leg.orig_timezone:
                    leg.orig_timezone = str(place_o.time_zone_info) if place_o else None
                place_d = self.get_place(leg.dest)
                leg.arr_time, leg.arr_time_offset = adjust_time_zone(leg.arr_time, place_d, leg.dest_timezone)
                if not leg.dest_timezone:
                    leg.dest_timezone = str(place_d.time_zone_info) if place_d else None
                if place_o is None:
                    warnings.warn(f"No defined place for {leg.orig}", stacklevel=2)
                if place_d is None:
                    warnings.warn(f"No defined place for {leg.dest}", stacklevel=2)
                leg.time_adjusted = True

            # after any time zone adjustments have been made, check that the
            # leg departure time is earlier than the leg arrival time
            duration_minutes = (leg.arr_time - leg.dep_time) / 60.0
            if duration_minutes <= 0:
                if leg.leg_id:
                    msg = f"- Leg ID {leg.leg_id},"
                else:
                    msg = "- "
                msg += f" {leg.carrier}{leg.fltno}:{leg.orig}-{leg.dest} ({duration_minutes} minutes)"
                negative_duration_legs.append(msg)

            # also check that leg speeds are reasonable
            if leg.distance:
                haul, lowspeed, highspeed = get_speed_limits(leg.distance, self.simulation_controls.speed_limits)
                leg_speed = leg.distance / (duration_minutes / 60.0)
                if leg_speed < lowspeed or leg_speed > highspeed:
                    if leg.leg_id:
                        msg = f"- Leg ID {leg.leg_id},"
                    else:
                        msg = "- "
                    msg += (
                        f" {leg.carrier}{leg.fltno}:{leg.orig}-{leg.dest} "
                        f"({leg_speed:.1f} mph on {haul.lower()} haul, should be {lowspeed:.0f}-{highspeed:.0f} mph)"
                    )
                    unreasonable_leg_speeds.append(msg)

        if negative_duration_legs:
            raise ValueError(
                "Some legs have non-positive duration, check if you need to set `arr_day` to +1:\n"
                + "\n".join(negative_duration_legs)
            )
        if unreasonable_leg_speeds:
            raise ValueError("Some legs have unreasonable speeds:\n" + "\n".join(unreasonable_leg_speeds))

        return self

    @model_validator(mode="after")
    def _places_exist_for_circuity(self) -> Self:
        """Circuity rules can only refer to airports in the places data.
        The core code will not crash if the places are missing, but the rules
        may not work as expected and that'll be a PITA to debug !!!"""
        for rule in self.circuity_rules:
            if rule.carrier != "" and rule.carrier is not None and rule.carrier not in self.carriers:
                raise ValueError(f"Circuity rule '{rule.name}' refers to a carrier that isn't specified in carriers")
            if rule.orig_airport != "" and rule.orig_airport is not None and rule.orig_airport not in self.places:
                raise ValueError(
                    f"Circuity rule '{rule.name}' refers to an orig airport that isn't specified in places"
                )
            if (
                rule.connect_airport != ""
                and rule.connect_airport is not None
                and rule.connect_airport not in self.places
            ):
                raise ValueError(
                    f"Circuity rule '{rule.name}' refers to a connecting airport that isn't specified in places"
                )
            if rule.dest_airport != "" and rule.dest_airport is not None and rule.dest_airport not in self.places:
                raise ValueError(f"Circuity rule '{rule.name}' refers to a dest airport that isn't specified in places")

            # Now we check state codes
        return self

    def preprocess(self) -> Self:
        """Conduct cleaning and preprocessing on this Config.

        This will run the following steps:

        - connection builder
        - compute delta-t for all markets as needed
        - assign standard TODD curves to all demands if `simulation_controls.use_standard_todd_curves`
        """
        preprocess_config(self)
        return self

    def __repr__(self):
        indent = 2
        x = []
        i = " " * indent
        for k, v in self:
            if k in {"legs", "paths", "fares", "demands"}:
                val = f"<list of {len(v)} {k}>"
            elif k == "carriers":
                carrier_lines = []
                for carrier_name, carrier_cfg in v.items():
                    carrier_lines.append(f"  {carrier_name}: <carrier config, rm_system={carrier_cfg.rm_system}>")
                val = "\n".join(carrier_lines)
            elif k in {"booking_curves", "places", "carriers", "choice_models", "frat5_curves"}:
                val = f"<dict of {len(v)} {k}>"
            elif k in {"outputs"}:
                val = "<outputs config>"
            elif k in {"db"}:
                val = f"<database config, engine={v.engine}, filename={v.filename}>"
            elif k in {"simulation_controls"}:
                if v.num_trials == 1:
                    val = f"<simulation controls with {v.num_samples} samples, {v.burn_samples} burn>"
                else:
                    val = (
                        f"<simulation controls with {v.num_trials} trials, "
                        f"{v.num_samples} samples, {v.burn_samples} burn>"
                    )
            elif k in {"raw_license_certificate", "license_certificate"}:
                continue
            elif isinstance(v, dict):
                val = repr_dict_with_indent(v, indent)
            else:
                try:
                    val = v.__repr_with_indent__(indent)
                except AttributeError:
                    val = repr(v)
            if "\n" in val:
                val_lines = val.split("\n")
                val = "\n  " + "\n  ".join(val_lines)
            x.append(f"{i}{k}: {val}")
        return "passengersim.Config:\n" + "\n".join(x)

    def __getstate__(self):
        state = super().__getstate__()
        # do not save a user's license certificate into the state
        if "raw_license_certificate" in state:
            state["raw_license_certificate"] = None
        if "__dict__" in state:
            if "raw_license_certificate" in state["__dict__"]:
                state["__dict__"]["raw_license_certificate"] = None
        return state

    def __getattr__(self, name):
        """Allow accessing figure methods."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name.startswith("fig_") or name.startswith("plotly_"):
                import passengersim.config._figures

                if name in passengersim.config._figures.__all__:
                    func = getattr(passengersim.config._figures, name)

                    def wrapper(*args, **kwargs):
                        return func(self, *args, **kwargs)

                    return wrapper
            # otherwise, re-raise
            raise

    def __dir__(self):
        import passengersim.config._figures

        extras = [i for i in passengersim.config._figures.__all__ if i.startswith("fig_") or i.startswith("plotly_")]
        result = super().__dir__()
        result.extend(extras)
        return result
