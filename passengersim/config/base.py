# TITLE: Config
# DOC-NAME: 00-configs
from __future__ import annotations

import importlib
import io
import logging
import os
import pathlib
import sys
import time
import typing
import warnings
from datetime import datetime
from typing import Any, ClassVar
from urllib.request import urlopen

import addicty
import yaml
from pydantic import (
    Field,
    SerializerFunctionWrapHandler,
    ValidationError,
    field_serializer,
    field_validator,
    model_validator,
)

from passengersim.pseudonym import random_label
from passengersim.utils.compression import (
    deserialize_from_file,
    serialize_to_file,
    smart_open,
)
from passengersim.utils.file_freshness import check_modification_times

from .blf_curves import BlfCurve
from .booking_curves import BookingCurve
from .carriers import Carrier
from .choice_model import ChoiceModel
from .circuity_rules import CircuityRule
from .database import DatabaseConfig
from .demands import Demand
from .fares import Fare
from .frat5_curves import Frat5Curve
from .legs import Leg
from .load_factor_curves import LoadFactorCurve
from .markets import Market
from .named import DictOfNamed, ListOfNamed
from .outputs import OutputConfig
from .paths import Path
from .places import Place, great_circle
from .pretty import PrettyModel, repr_dict_with_indent
from .rm_steps import RmStepBase
from .rm_systems import RmSystem
from .simulation_controls import SimulationSettings
from .snapshot_filter import SnapshotFilter
from .todd_curves import ToddCurve

if typing.TYPE_CHECKING:
    from typing import Literal, Self

    from pydantic.main import IncEx

logger = logging.getLogger("passengersim.config")

_warn_skips = (os.path.dirname(__file__),)

TConfig = typing.TypeVar("TConfig", bound="YamlConfig")


def web_opener(x):
    return urlopen(x.parts[0] + "//" + "/".join(x.parts[1:]))


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
                content = addicty.Dict.load(
                    filename, freeze=False, Loader=yaml.CSafeLoader
                )
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
                        raise NotImplementedError(
                            "cannot load compressed files from web yet"
                        )
                if isinstance(filename, OptionalPath) and not filename.exists():
                    continue
                with opener(filename) as f:
                    content = addicty.Dict.load(
                        f, freeze=False, Loader=yaml.CSafeLoader
                    )
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
                    raw_config.update(content)
            logger.info("loaded config from %s in %.2f secs", filename, time.time() - t)
        return raw_config

    @classmethod
    def from_yaml(
        cls: type[TConfig],
        filenames: pathlib.Path | list[pathlib.Path],
        *,
        cache_file: pathlib.Path | None = None,
        on_validation_error: Literal["raise", "warn"] = "raise",
    ) -> TConfig | addicty.Dict:
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

        Returns
        -------
        Config or addicty.Dict
        """
        cache_is_outdated = True
        if cache_file:
            cache_is_outdated = check_modification_times(filenames, cache_file)
            if cache_is_outdated:
                logger.info(
                    f"cache file is {cache_is_outdated}, will reload YAML files"
                )
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
            Whether to exclude fields that are unset or None from the output.
        exclude_defaults : bool, default False
            Whether to exclude fields that are set to their default value from
            the output.
        exclude_none : bool, default False
            Whether to exclude fields that have a value of `None` from the output.
        warnings : bool, default True
            Whether to log warnings when invalid fields are encountered.

        Returns
        -------
        bytes or None
            When no stream is given, the YAML content is returned as bytes,
            otherwise this method returns nothing.
        """

        def path_to_str(x):
            if isinstance(x, dict):
                return {k: path_to_str(v) for k, v in x.items()}
            if isinstance(x, list):
                return list(path_to_str(i) for i in x)
            if isinstance(x, tuple):
                return list(path_to_str(i) for i in x)
            if isinstance(x, pathlib.Path):
                return str(x)
            else:
                return x

        y = path_to_str(
            self.model_dump(
                include=include,
                exclude=exclude,
                exclude_unset=exclude_unset,
                exclude_defaults=exclude_defaults,
                exclude_none=exclude_none,
                warnings=warnings,
            )
        )
        b = yaml.dump(y, encoding="utf8", Dumper=yaml.CSafeDumper)
        if isinstance(stream, str):
            stream = pathlib.Path(stream)
        if isinstance(stream, pathlib.Path):
            stream.write_bytes(b)
        elif isinstance(stream, io.RawIOBase):
            stream.write(b)
        elif isinstance(stream, io.TextIOBase):
            stream.write(b.decode())
        else:
            return b


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


class Config(YamlConfig, extra="forbid"):
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
    def _db_to_none(
        cls, v: DatabaseConfig | None, nxt: SerializerFunctionWrapHandler
    ) -> dict | None:
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

    blf_curves: DictOfNamed[BlfCurve] = {}
    """ Booked Load Factor curves"""

    frat5_curves: DictOfNamed[Frat5Curve] = {}
    """ FRAT5 curves are used to model sellup rates in Q-forecasting"""

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

    legs: list[Leg] = []
    demands: list[Demand] = []
    fares: list[Fare] = []
    paths: list[Path] = []
    markets: list[Market] = []

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
    def _fare_ap_restrictions_must_match_dcps(cls, m: Config):
        """AP restrictions on fare can only be invoked at the DCPs."""
        for f in m.fares:
            if f.advance_purchase != 0 and f.advance_purchase not in m.dcps:
                raise ValueError(
                    f"Advance purchase restriction not aligned with DCP for Fare {f}"
                )
        return m

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
    def _burn_samples(cls, m: Config):
        """burn_samples must be strictly less than samples"""
        if m.simulation_controls.burn_samples >= m.simulation_controls.num_samples:
            raise ValueError(
                "burn_samples must be strictly less than samples. "
                "It will default to 100 if you haven't set a value"
            )
        if (
            m.simulation_controls.burn_samples == 0
            and m.simulation_controls.num_samples > 10
        ):
            raise ValueError(
                "to ensure meaningful results, burn_samples may not "
                "be zero when num_samples > 10."
            )
        return m

    @model_validator(mode="after")
    def _manual_paths(cls, m: Config):
        """If manual_paths is true, there must be Path items
        if it's set to false, then there shouldn't be any path items"""
        if m.simulation_controls.manual_paths and len(m.paths) == 0:
            raise ValueError(
                "manual_paths is set to true, but no paths found in the input config"
            )
        if not m.simulation_controls.manual_paths and len(m.paths) > 0:
            raise ValueError(
                "manual_paths is set to false, "
                "but paths were specified in the input config"
            )
        return m

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
            rm_sys = Config.model_validate(
                {"rm_systems": {std_name: raw_rm_systems[std_name]}}
            )
            self.rm_systems[std_name] = rm_sys.rm_systems[std_name]
        else:
            raise KeyError(f"Unknown standard RM system {std_name}")

    @classmethod
    def _carriers_have_rm_systems(cls, m: Config):
        """Check that all carriers have RM systems that have been defined."""
        for carrier in m.carriers.values():
            if carrier.rm_system not in m.rm_systems:
                try:
                    m._load_std_rm_system(carrier.rm_system)
                except KeyError:
                    raise ValueError(
                        f"Carrier {carrier.name} has unknown "
                        f"RM system {carrier.rm_system}"
                    ) from None
        return m

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
    def _carriers_have_frat5(cls, m: Config):
        """Check that all carriers have defined or null Frat5 curves."""
        cls._carriers_have_rm_systems(m)
        for carrier in m.carriers.values():
            # first, if the carrier has no Frat5 curve, see if the RM system has one
            if not carrier.frat5:
                rm_system = m.rm_systems[carrier.rm_system]
                if rm_system.frat5:
                    carrier.frat5 = rm_system.frat5
            if carrier.frat5 and carrier.frat5 not in m.frat5_curves:
                try:
                    m._load_std_frat5(carrier.frat5)
                except KeyError:
                    raise ValueError(
                        f"Carrier {carrier.name} has unknown "
                        f"Frat5 curve {carrier.frat5}"
                    ) from None
        return m

    @model_validator(mode="after")
    def _legs_have_carriers(cls, m: Config):
        """Check that all legs have a carrier that has been defined."""
        for leg in m.legs:
            if leg.carrier not in m.carriers:
                raise ValueError(
                    f"Carrier for leg {leg.carrier} {leg.fltno} is not defined"
                )
        return m

    @model_validator(mode="after")
    def _choice_model_todd_curves_exist(cls, m: Config):
        """Check that any TODD curves referenced in Demand objects have been defined."""
        for name, cm in m.choice_models.items():
            if cm.todd_curve is not None and cm.todd_curve not in m.todd_curves:
                raise ValueError(
                    f"ChoiceModel {name} has unknown TOD Curve {cm.todd_curve}"
                )
        return m

    @model_validator(mode="after")
    def _choice_model_airline_preferences(cls, m: Config):
        """Check that only one way of inputting airline preference was specified."""
        for _name, cm in m.choice_models.items():
            a1 = 1 if cm.airline_pref_pods is not None else 0
            a2 = 1 if cm.airline_pref_hhi is not None else 0
            a3 = 1 if cm.airline_pref_seat_share is not None else 0
            if a1 + a2 + a3 > 1:
                raise ValueError(
                    f"ChoiceModel '{cm.name}' has more than one "
                    f"airline preference model specified"
                )
        return m

    @model_validator(mode="after")
    def _choice_model_curve_s_vs_replanning(cls, m: Config):
        """Check that only one way of inputting airline preference was specified."""
        for _name, cm in m.choice_models.items():
            if cm.replanning is not None and (
                cm.early_dep is not None or cm.late_arr is not None
            ):
                raise ValueError(
                    f"ChoiceModel '{cm.name}' has replanning and early_dep / late_arr "
                    f"specified, pick one or the other but not both !!!"
                )
        return m

    @model_validator(mode="after")
    def _choice_set_sampling(cls, m: Config):
        """Ensure there is a limit on the number of observations in a choice set.
        Don't allow a choice set to be created without a specified limit of observations
        as unlimited sampling will run out of storage very quickly"""
        if (
            len(m.simulation_controls.capture_choice_set_file) > 0
            and m.simulation_controls.capture_choice_set_obs is None
        ):
            m.simulation_controls.capture_choice_set_obs = 10000
            warnings.warn(
                "capture_choice_set_obs not specified, has been set to 10000",
                stacklevel=2,
            )
        return m

    @model_validator(mode="after")
    def _demand_todd_curves_exist(cls, m: Config):
        """Check that any TODD curves referenced in Demand objects have been defined."""
        for dmd in m.demands:
            if dmd.todd_curve is not None and dmd.todd_curve not in m.todd_curves:
                raise ValueError(
                    f"Demand {dmd.orig}-{dmd.dest}:{dmd.segment} has "
                    f"unknown TOD Curve {dmd.todd_curve}"
                )
        return m

    @model_validator(mode="after")
    def _booking_curves_match_dcps(cls, m: Config):
        """Check that all booking curves are complete and valid."""
        sorted_dcps = reversed(sorted(m.dcps))
        for curve in m.booking_curves.values():
            i = 0
            for dcp in sorted_dcps:
                assert (
                    dcp in curve.curve
                ), f"booking curve {curve.name} is missing dcp {dcp}"
                assert (
                    curve.curve[dcp] >= i
                ), f"booking curve {curve.name} moves backwards at dcp {dcp}"
                i = curve.curve[dcp]
        return m

    @model_validator(mode="after")
    def _requested_summaries_have_data(cls, m: Config):
        """Check that requested summary outputs will have the data needed."""
        if "local_and_flow_yields" in m.outputs.reports:
            if not m.db.write_items & {"pathclass_final", "pathclass"}:
                raise ValueError(
                    "the `local_and_flow_yields` report requires recording "
                    "at least `pathclass_final` details in the database"
                )
        if "bid_price_history" in m.outputs.reports:
            if "leg" not in m.db.write_items:
                raise ValueError(
                    "the `bid_price_history` report requires recording "
                    "`leg` details in the database"
                )
            if not m.db.store_leg_bid_prices:
                raise ValueError(
                    "the `bid_price_history` report requires recording "
                    "`store_leg_bid_prices` to be True"
                )
        if "demand_to_come" in m.outputs.reports:
            if "demand" not in m.db.write_items:
                raise ValueError(
                    "the `demand_to_come` report requires recording "
                    "`demand` details in the database"
                )
        if "demand_to_come_summary" in m.outputs.reports:
            if "demand" not in m.db.write_items:
                raise ValueError(
                    "the `demand_to_come_summary` report requires recording "
                    "`demand` details in the database"
                )
        if "path_forecasts" in m.outputs.reports:
            if "pathclass" not in m.db.write_items:
                raise ValueError(
                    "the `path_forecasts` report requires recording "
                    "`pathclass` details in the database"
                )
        if "leg_forecasts" in m.outputs.reports:
            if "bucket" not in m.db.write_items:
                raise ValueError(
                    "the `leg_forecasts` report requires recording "
                    "`bucket` details in the database"
                )
        if "bookings_by_timeframe" in m.outputs.reports:
            if not m.db.write_items & {"bookings", "fare"}:
                raise ValueError(
                    "the `bookings_by_timeframe` report requires recording "
                    "`fare` or `bookings` details in the database"
                )
        if "total_demand" in m.outputs.reports:
            if not m.db.write_items & {"demand", "demand_final"}:
                raise ValueError(
                    "the `total_demand` report requires recording "
                    "at least `demand_final` details in the database"
                )
        if "fare_class_mix" in m.outputs.reports:
            if not m.db.write_items & {"fare", "fare_final"}:
                raise ValueError(
                    "the `fare_class_mix` report requires recording "
                    "at least `fare_final` details in the database"
                )
        if "load_factor_distribution" in m.outputs.reports:
            if not m.db.write_items & {"leg", "leg_final"}:
                raise ValueError(
                    "the `load_factor_distribution` report requires recording "
                    "at least `leg_final` details in the database"
                )
        if "edgar" in m.outputs.reports:
            if not m.db.write_items & {"edgar"}:
                raise ValueError(
                    "the 'edgar' forecast accuray report requires recording "
                    "'edgar' details in the database"
                )
        return m

    @model_validator(mode="after")
    def _bp_controls_are_expected_but_not_set(cls, m: Config):
        """Warn if bid price controls are expected but not set."""
        for rm_system in m.rm_systems.values():
            if "dcp" in rm_system.processes:
                for step in rm_system.processes["dcp"]:
                    if isinstance(step, RmStepBase):
                        try:
                            req = step.require_availability_control
                        except AttributeError:
                            req = None
                        if (
                            req is not None
                            and rm_system.availability_control not in req
                        ):
                            raise ValueError(
                                f"RM System {rm_system.name} requires "
                                f"availability control {req} for step {step.name}"
                            )
        return m

    __rm_steps_loaded: ClassVar[set[type[RmStepBase]]] = RmStepBase._get_subclasses()

    @classmethod
    def model_validate(
        cls,
        *args,
        **kwargs,
    ) -> typing.Any:
        """Validate the passengersim Config inputs.

        This method reloads the Config class to ensure all imported
        RmSteps are properly registered before validation.

        Parameters
        ----------
        obj
            The object to validate.
        strict : bool
            Whether to raise an exception on invalid fields.
        from_attributes
            Whether to extract data from object attributes.
        context
            Additional context to pass to the validator.

        Raises
        ------
        ValidationError
            If the object could not be validated.

        Returns
        -------
        Config
            The validated model instance.
        """
        # detect if there are any new RmSteps and reload the Config class
        # to ensure they are properly registered
        reloaded_class = cls
        for k in RmStepBase._get_subclasses():
            if k not in cls.__rm_steps_loaded:
                # reload these to refresh for any newly defined RmSteps
                module_parent = ".".join(__name__.split(".")[:-1])
                importlib.reload(sys.modules.get(f"{module_parent}.rm_systems"))
                importlib.reload(sys.modules.get(__name__))
                module = importlib.reload(sys.modules.get(module_parent))
                reloaded_class = getattr(module, cls.__name__)
        # `__tracebackhide__` tells pytest and some other tools to omit this
        # function from tracebacks
        __tracebackhide__ = True
        return reloaded_class.__pydantic_validator__.validate_python(*args, **kwargs)

    def model_revalidate(
        self,
    ) -> typing.Self:
        """Revalidate the passengersim Config instance."""
        return self.as_reloaded.model_validate(self.model_dump(serialize_as_any=True))

    @classmethod
    @property
    def as_reloaded(cls) -> type[Config]:
        """Get the Config class, as most recently reloaded."""
        module_parent = ".".join(__name__.split(".")[:-1])
        module = sys.modules.get(module_parent)
        reloaded_class = getattr(module, cls.__name__)
        return reloaded_class

    @classmethod
    def instance_check(cls, obj) -> bool:
        """Check if an object is an instance of the Config class."""
        # module_parent = ".".join(__name__.split(".")[:-1])
        # module = sys.modules.get(module_parent)
        # reloaded_class = getattr(module, cls.__name__)
        return isinstance(obj, cls.as_reloaded)

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

    def add_output_prefix(
        self, prefix: pathlib.Path, spool_format: str = "%Y%m%d-%H%M"
    ):
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
                place_o = self.places.get(leg.orig, None)
                place_d = self.places.get(leg.dest, None)
                if place_o is not None and place_d is not None:
                    leg.distance = great_circle(place_o, place_d)
                if place_o is None:
                    warnings.warn(f"No defined place for {leg.orig}", stacklevel=2)
                if place_d is None:
                    warnings.warn(f"No defined place for {leg.dest}", stacklevel=2)
        for dmd in self.demands:
            if not dmd.distance:
                place_o = self.places.get(dmd.orig, None)
                place_d = self.places.get(dmd.dest, None)
                if place_o is not None and place_d is not None:
                    dmd.distance = great_circle(place_o, place_d)
                if place_o is None:
                    warnings.warn(f"No defined place for {dmd.orig}", stacklevel=2)
                if place_d is None:
                    warnings.warn(f"No defined place for {dmd.dest}", stacklevel=2)
        return self

    @model_validator(mode="after")
    def _adjust_times_for_time_zones(self):
        """Adjust arrival/departure times to local time from UTC."""
        for leg in self.legs:
            # the nominal time is local time but so far got stored as UTC,
            # so we need to add the time zone offset to be actually local time

            def adjust_time_zone(t, place):
                if place is not None:
                    tz = place.time_zone_info
                    if tz is not None:
                        # Alan's approach
                        # It was converted as a local time, so unpack it and
                        #   create a new datetime in the given TZ
                        dt = datetime.fromtimestamp(t)  # , tz=timezone.utc)
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

            if not leg.time_adjusted:
                # if leg.orig == "DFW" and leg.dest == "CLE":
                #     pass
                place_o = self.places.get(leg.orig, None)
                leg.dep_time, leg.dep_time_offset = adjust_time_zone(
                    leg.dep_time, place_o
                )
                leg.orig_timezone = str(place_o.time_zone_info) if place_o else None
                place_d = self.places.get(leg.dest, None)
                leg.arr_time, leg.arr_time_offset = adjust_time_zone(
                    leg.arr_time, place_d
                )
                leg.dest_timezone = str(place_d.time_zone_info) if place_d else None
                if place_o is None:
                    warnings.warn(f"No defined place for {leg.orig}", stacklevel=2)
                if place_d is None:
                    warnings.warn(f"No defined place for {leg.dest}", stacklevel=2)
                leg.time_adjusted = True
        return self

    @model_validator(mode="after")
    def _places_exist_for_circuity(cls, cfg: Config):
        """Circuity rules can only refer to airports in the places data.
        The core code will not crash if the places are missing, but the rules
        may not work as expected and that'll be a PITA to debug !!!"""
        for rule in cfg.circuity_rules:
            if (
                rule.carrier != ""
                and rule.carrier is not None
                and rule.carrier not in cfg.carriers
            ):
                raise ValueError(
                    f"Circuity rule '{rule.name}' refers to a "
                    f"carrier that isn't specified in carriers"
                )
            if (
                rule.orig_airport != ""
                and rule.orig_airport is not None
                and rule.orig_airport not in cfg.places
            ):
                raise ValueError(
                    f"Circuity rule '{rule.name}' refers to an "
                    f"orig airport that isn't specified in places"
                )
            if (
                rule.connect_airport != ""
                and rule.connect_airport is not None
                and rule.connect_airport not in cfg.places
            ):
                raise ValueError(
                    f"Circuity rule '{rule.name}' refers to a "
                    f"connecting airport that isn't specified in places"
                )
            if (
                rule.dest_airport != ""
                and rule.dest_airport is not None
                and rule.dest_airport not in cfg.places
            ):
                raise ValueError(
                    f"Circuity rule '{rule.name}' refers to a "
                    f"dest airport that isn't specified in places"
                )

            # Now we check state codes
        return cfg

    def __repr__(self):
        indent = 2
        x = []
        i = " " * indent
        for k, v in self:
            if k in {"legs", "paths", "fares", "demands"}:
                val = f"<list of {len(v)} {k}>"
            elif k in {"booking_curves"}:
                val = f"<dict of {len(v)} {k}>"
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
