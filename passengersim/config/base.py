# TITLE: Config
# DOC-NAME: 00-configs
from __future__ import annotations

import gzip
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
from typing import Any
from urllib.request import urlopen

import addicty
import yaml
from pydantic import Field, field_validator, model_validator

from passengersim.pseudonym import random_label

from .airlines import Airline
from .booking_curves import BookingCurve
from .choice_model import ChoiceModel
from .database import DatabaseConfig
from .demands import Demand
from .fares import Fare
from .frat5_curves import Frat5Curve
from .legs import Leg
from .load_factor_curves import LoadFactorCurve
from .named import DictOfNamed
from .outputs import OutputConfig
from .paths import Path
from .places import Place, great_circle
from .pretty import PrettyModel, repr_dict_with_indent
from .rm_systems import RmSystem
from .simulation_controls import SimulationSettings
from .snapshot_filter import SnapshotFilter
from .todd_curves import ToddCurve

if typing.TYPE_CHECKING:
    from pydantic.main import IncEx

logger = logging.getLogger("passengersim.config")


TConfig = typing.TypeVar("TConfig", bound="YamlConfig")


def web_opener(x):
    return urlopen(x.parts[0] + "//" + "/".join(x.parts[1:]))


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
        if isinstance(filenames, str | pathlib.Path):
            filenames = [filenames]
        raw_config = addicty.Dict()
        for filename in filenames:
            if isinstance(filename, str) and "\n" in filename:
                # explicit YAML content cannot have include statements
                content = addicty.Dict.load(filename, freeze=False)
                raw_config.update(content)
                continue
            filename = pathlib.Path(filename)
            if filename.suffix in (".pem", ".crt", ".cert"):
                # license certificate
                with open(filename, "rb") as f:
                    raw_config.raw_license_certificate = f.read()
            else:
                opener = gzip.open if filename.suffix == ".gz" else open
                if filename.parts[0] in {"https:", "http:", "s3:"}:
                    opener = web_opener
                    if filename.suffix == ".gz":
                        raise NotImplementedError(
                            "cannot load compressed files from web yet"
                        )
                with opener(filename) as f:
                    content = addicty.Dict.load(f, freeze=False)
                    include = content.pop("include", None)
                    if include is not None:
                        if isinstance(include, str):
                            filename.parent.joinpath(include)
                            inclusions = [filename.parent.joinpath(include)]
                        else:
                            inclusions = [filename.parent.joinpath(i) for i in include]
                        raw_config.update(cls._load_unformatted_yaml(inclusions))
                    raw_config.update(content)
            logger.info("loaded config from %s", filename)
        return raw_config

    @classmethod
    def from_yaml(
        cls: type[TConfig],
        filenames: pathlib.Path | list[pathlib.Path],
    ) -> TConfig:
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
        Config
        """
        raw_config = cls._load_unformatted_yaml(filenames)
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
        b = yaml.dump(y, encoding="utf8", Dumper=yaml.SafeDumper)
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

    outputs: OutputConfig = OutputConfig()
    """
    See [passengersim.config.OutputConfig][] for detailed documentation.
    """

    rm_systems: DictOfNamed[RmSystem] = {}
    """
    The revenue management systems used by the carriers in this simulation.

    See [RM Systems][rm-systems] for details.
    """

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

    airlines: DictOfNamed[Airline] = {}
    """A list of airlines.

    One convention is to use Airline1, Airline2, ... to list the airlines in the
    network.  Another convention is to use IATA industry-standard two-letter airline
    codes.  See the
    [IATA code search](https://www.iata.org/en/publications/directories/code-search/)
    for more information."""

    places: DictOfNamed[Place] = {}
    """A list of places (airports, vertiports, other stations)."""

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
            carriers = data.get("airlines", {})
            if isinstance(carriers, list):
                for carrier in carriers:
                    if isinstance(carrier, dict) and "classes" not in carrier:
                        carrier["classes"] = data.get("classes", [])
            else:
                for carrier in carriers.values():
                    if isinstance(carrier, dict) and "classes" not in carrier:
                        carrier["classes"] = data.get("classes", [])
            data["airlines"] = carriers
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

    snapshot_filters: list[SnapshotFilter] = []

    @field_validator("snapshot_filters", mode="before")
    def _handle_no_snapshot_filters(cls, v):
        if v is None:
            v = []
        return v

    raw_license_certificate: bytes | None = None

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

    @model_validator(mode="after")
    def _airlines_have_rm_systems(cls, m: Config):
        """Check that all airlines have RM systems that have been defined."""
        for airline in m.airlines.values():
            if airline.rm_system not in m.rm_systems:
                raise ValueError(
                    f"Airline {airline.name} has unknown RM system {airline.rm_system}"
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
        return m

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
    def _attach_distance_to_legs_without_it(self):
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
                        # construct a datetime object as if it is UTC
                        # t = datetime.utcfromtimestamp(t)
                        # naively inject the desired timezone
                        # t = t.replace(tzinfo=tz)
                        # convert "back" to UTC
                        # t = t.astimezone(timezone.utc)

                        # Alan's approach
                        # It was converted as a local time, so unpack it and
                        #   create a new datetime in the given TZ
                        dt = datetime.fromtimestamp(t)
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
                        return int(dt2.timestamp())
                return t

            place_o = self.places.get(leg.orig, None)
            leg.dep_time = adjust_time_zone(leg.dep_time, place_o)
            leg.orig_timezone = str(place_o.time_zone_info) if place_o else None
            place_d = self.places.get(leg.dest, None)
            leg.arr_time = adjust_time_zone(leg.arr_time, place_d)
            leg.dest_timezone = str(place_d.time_zone_info) if place_d else None
            if place_o is None:
                warnings.warn(f"No defined place for {leg.orig}", stacklevel=2)
            if place_d is None:
                warnings.warn(f"No defined place for {leg.dest}", stacklevel=2)
        return self

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
