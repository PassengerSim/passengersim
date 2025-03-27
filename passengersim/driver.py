#
# Driver program to load a simulation from YAML, run it and return results
# (c) PassengerSim LLC
#

from __future__ import annotations

import contextlib
import logging
import os
import pathlib
import sqlite3
import sys  # noqa: F401
import time
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Mapping
from datetime import datetime, timezone
from math import sqrt
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
import pandas as pd
import psutil  # noqa: F401
from passengersim_core import Ancillary, DbWriter
from passengersim_core.utils.airsim_utils import get_mileage
from rich.progress import Progress
from scipy.stats import gamma

import passengersim.config.rm_systems
import passengersim.core
from passengersim.config import Config
from passengersim.config.manipulate import revalidate
from passengersim.config.snapshot_filter import SnapshotFilter
from passengersim.core import (
    Airport,
    DecisionWindow,
    Event,
    Frat5,
    Market,
    PathClass,
    SimulationEngine,
)
from passengersim.summaries import SimulationTables
from passengersim.summaries.generic import GenericSimulationTables
from passengersim.summary import SummaryTables
from passengersim.tracers.generic import GenericTracer
from passengersim.utils.nested_dict import from_nested_dict  # noqa: F401
from passengersim.utils.si import si_units  # noqa: F401

from . import database
from .callbacks import CallbackData, CallbackMixin
from .progressbar import DummyProgressBar, ProgressBar

if TYPE_CHECKING:
    from passengersim.config.rm_systems import RmSystem as RmSystemConfig
    from passengersim.core import ChoiceModel

logger = logging.getLogger("passengersim")

SimulationTablesT = TypeVar("SimulationTablesT", bound=GenericSimulationTables)

_warn_skips = (os.path.dirname(__file__), os.path.dirname(contextlib.__file__))


def memory_log(tag):
    pass
    # p = psutil.Process()
    # mem_info = p.memory_info()
    # print(
    #     f"\nPSUTIL {tag}: rss={si_units(mem_info.rss, kind='B')} "
    #     f"vmw={si_units(mem_info.vms, kind='B')}",
    #     file=sys.stderr,
    # )


class BaseSimulation(ABC):
    @classmethod
    def from_yaml(
        cls,
        filenames: pathlib.Path | list[pathlib.Path],
        output_dir: pathlib.Path | None = None,
    ):
        """
        Create a Simulation object from a YAML file.

        Parameters
        ----------
        filenames : pathlib.Path | list[pathlib.Path]
        output_dir : pathlib.Path | None, optional

        Returns
        -------
        Simulation
        """
        config = passengersim.config.Config.from_yaml(filenames)
        return cls(config, output_dir)

    def __init__(
        self,
        config: Config,
        output_dir: pathlib.Path | None = None,
    ):
        if output_dir is None:
            import tempfile

            self._tempdir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
            output_dir = os.path.join(self._tempdir.name, "test1")
        self.cnx = None
        self.output_dir = output_dir

    @property
    @abstractmethod
    def _sim(self) -> SimulationEngine:
        raise NotImplementedError

    def path_names(self):
        result = {}
        for p in self._sim.paths:
            result[p.path_id] = str(p)
        return result

    @property
    def markets(self) -> Mapping[str, Market]:
        """Access markets in the simulation."""
        return self._sim.markets

    @property
    def paths(self):
        """Generator of all paths in the simulation."""
        return self._sim.paths

    @property
    def pathclasses(self):
        """Generator of all path classes in the simulation."""
        for path in self._sim.paths:
            yield from path.pathclasses

    def pathclasses_for_carrier(self, carrier: str):
        """Generator of all path classes for a given carrier."""
        for path in self._sim.paths:
            if path.carrier_name == carrier:
                yield from path.pathclasses

    @property
    def demands(self):
        """Generator of all demands in the simulation."""
        from .iterators.demand import DemandIterator

        return DemandIterator(self._sim)

    @property
    def fares(self):
        """Generator of all fares in the simulation."""
        from .iterators.fare import FareIterator

        return FareIterator(self._sim)


class Simulation(BaseSimulation, CallbackMixin):
    def __init__(
        self,
        config: Config,
        output_dir: pathlib.Path | None = None,
    ):
        revalidate(config)
        super().__init__(config, output_dir)
        if config.simulation_controls.write_raw_files:
            try:
                from passengersim_core.utils import FileWriter
            except ImportError:
                self.file_writer = None
            else:
                self.file_writer = FileWriter.FileWriter(output_dir)
        else:
            self.file_writer = None
        self.db_writer = None
        self.dcp_list = [63, 56, 49, 42, 35, 31, 28, 24, 21, 17, 14, 10, 7, 5, 3, 1, 0]
        self.classes = []
        self.fare_sales_by_dcp = defaultdict(int)
        self.fare_sales_by_carrier_dcp = defaultdict(int)
        self.fare_details_sold = defaultdict(int)
        self.fare_details_sold_business = defaultdict(int)
        self.fare_details_revenue = defaultdict(float)
        self.demand_multiplier = 1.0
        self.capacity_multiplier = 1.0
        self.airports = {}
        self.choice_models = {}
        self.frat5curves = {}
        self.load_factor_curves = {}
        self.todd_curves = {}
        self.debug = False
        self.update_frequency = None
        self.random_generator = passengersim.core.Generator(42)
        self.sample_done_callback = lambda n, n_total: None
        self.choice_set_file = None
        self.choice_set_obs = 0
        self.segmentation_data_by_timeframe: dict[int, pd.DataFrame] = {}
        """Bookings and revenue segmentation by timeframe.

        The key is the trial number, and the value is a DataFrame with a
        breakdown of bookings and revenue by timeframe, customer segment,
        carrier, and booking class.
        """

        self.bid_price_traces: dict[int, Any] = {}
        """Bid price traces for each carrier.

        The key is the trial number, and the value is a dictionary with
        carrier names as keys and bid price traces as values."""

        self.displacement_traces: dict[int, Any] = {}
        """Displacement cost traces for each carrier.

        The key is the trial number, and the value is a dictionary with
        carrier names as keys and displacement cost traces as values."""

        self._fare_restriction_mapping = {}
        """Mapping of fare restriction names to restriction numbers."""

        self._fare_restriction_list = []
        """List of fare restriction names in the order they were added."""

        self._initialize(config)
        if not config.db:
            self.cnx = database.Database()
        else:
            self.cnx = database.Database(
                engine=config.db.engine,
                filename=config.db.filename,
                pragmas=config.db.pragmas,
                commit_count_delay=config.db.commit_count_delay,
            )
        if self.cnx.is_open:
            database.tables.create_table_leg_defs(self.cnx._connection, self.sim.legs)
            database.tables.create_table_fare_defs(self.cnx._connection, self.sim.fares)
            database.tables.create_table_fare_restriction_defs(
                self.cnx._connection, self._fare_restriction_list
            )
            database.tables.create_table_path_defs(self.cnx._connection, self.sim.paths)
            if config.db != ":memory:":
                self.cnx.save_configs(config)

        self.callback_data = CallbackData()
        """Data stored from callbacks.

        This allows a user to store arbitrary data during a simulation using callbacks,
        and access it later.
        """

    @property
    def _sim(self) -> SimulationEngine:
        return self.sim

    @property
    def base_time(self) -> int:
        """
        The base time for the simulation, in seconds since the epoch.
        """
        return self.sim.base_time

    @property
    def snapshot_filters(self) -> list[SnapshotFilter] | None:
        try:
            sim = self.sim
        except AttributeError:
            return None
        return sim.snapshot_filters

    @snapshot_filters.setter
    def snapshot_filters(self, x: list[SnapshotFilter]):
        try:
            sim = self.sim
        except AttributeError as err:
            raise ValueError(
                "sim not initialized, cannot set snapshot_filters"
            ) from err
        sim.snapshot_filters = x

    def _initialize(self, config: Config):
        self._init_sim_and_parms(config)
        self._init_circuity(config)
        self._init_rm_systems(config)
        self._init_todd_curves(config)
        self._init_choice_models(config)
        self._init_frat5_curves(config)
        self._init_blf_curves(config)
        self._init_load_factor_curves(config)
        self._init_carriers(config)
        self._init_booking_curves(config)
        self._init_airports(config)
        self._init_demands(config)
        self._init_fares(config)
        logger.info("Connecting markets")
        self.sim.connect_markets()
        self.db_writer = DbWriter("db", config, self.sim)

    def _init_sim_and_parms(self, config):
        logger.info("Initializing simulation engine parameters")
        self.sim = passengersim.core.SimulationEngine(name=config.scenario)
        self.sim.config = config
        self.sim.random_generator = self.random_generator
        self.sim.snapshot_filters = config.snapshot_filters
        for pname, pvalue in config.simulation_controls:
            if pname == "demand_multiplier":
                self.demand_multiplier = pvalue
            elif pname == "capacity_multiplier":
                self.capacity_multiplier = pvalue
            elif pname == "write_raw_files":
                self.write_raw_files = pvalue
            elif pname == "random_seed":
                self.random_generator.seed(pvalue)
            elif pname == "update_frequency":
                self.update_frequency = pvalue
            elif pname == "capture_choice_set_file":
                if len(pvalue) > 0:
                    self.sim.set_parm("capture_choice_set", 1)
                    self.choice_set_file = open(pvalue, "w")
                    cols = self.sim.choice_set_columns()
                    tmp = ",".join(cols)
                    print(tmp, file=self.choice_set_file)
            elif pname == "capture_choice_set_obs":
                self.choice_set_obs = pvalue

            # These parameters are not used directly in the core, but leave them listed
            # for now to not break config files reading
            elif pname in [
                "base_date",
                "capture_competitor_data",
                "dcp_hour",
                "double_capacity_until",
                "dwm_lite",
                "show_progress_bar",
                "simple_k_factor",
                "timeframe_demand_allocation",
                "tot_z_factor",
                "allow_unused_restrictions",
                "additional_settings",
            ]:
                pass
            else:
                self.sim.set_parm(pname, float(pvalue))
        for pname, pvalue in config.simulation_controls.model_extra.items():
            print(f"extra simulation setting: {pname} = ", float(pvalue))
            self.sim.set_parm(pname, float(pvalue))
        if config.simulation_controls.additional_settings:
            self.sim.additional_settings(
                **config.simulation_controls.additional_settings
            )

        # There is a default array of DCPs, we'll override it with the data from the
        # input file (if available)
        if len(config.dcps) > 0:
            self.dcp_list = []
            for dcp_index, days_prior in enumerate(config.dcps):
                self.sim.add_dcp(dcp_index, days_prior)
                self.dcp_list.append(days_prior)
            # We need to add the last DCP, which is always 0, if not already in the list
            if self.dcp_list[-1] != 0:
                self.sim.add_dcp(len(self.dcp_list), 0)
                self.dcp_list.append(0)

    def _init_circuity(self, config):
        logger.info("Initializing circuity rules")
        for rule in config.circuity_rules:
            # Flatten the object into a dictionary,
            # SimulationEngine will iterate over it
            self.sim.add_circuity_rule(dict(rule))

    def _init_rm_system(self, rm_name: str, rm_system: RmSystemConfig, config: Config):
        from passengersim_core.carrier.rm_system import Rm_System

        logger.info("Initializing RM system %s", rm_name)
        x = self.rm_systems[rm_name] = Rm_System(rm_name)
        x.availability_control = rm_system.availability_control
        for process_name, process in rm_system.processes.items():
            step_list = [s._factory() for s in process]
            for s in step_list:
                s.use_config(config)
            x.add_process(process_name, step_list)

    def _init_rm_systems(self, config):
        self.rm_systems = {}
        for rm_name, rm_system in config.rm_systems.items():
            self._init_rm_system(rm_name, rm_system, config)

    def _init_todd_curves(self, config):
        logger.info("Initializing TODD curves")
        for todd_name, todd in config.todd_curves.items():
            dwm = DecisionWindow(todd_name)
            if todd.k_factor:
                dwm.k_factor = todd.k_factor
            if todd.min_distance:
                dwm.min_distance = todd.min_distance
            if todd.probabilities:
                dwm.dwm_tod = list(todd.probabilities.values())
            self.todd_curves[todd_name] = dwm

    def _get_fare_restriction_num(
        self, restriction_name: str, *, ignore_when_missing: bool = False
    ):
        r = str(restriction_name).casefold()
        if r not in self._fare_restriction_mapping:
            if ignore_when_missing:
                return None
            self._fare_restriction_mapping[r] = len(self._fare_restriction_mapping) + 1
            self._fare_restriction_list.append(r)
        return self._fare_restriction_mapping[r]

    def parse_restriction_flags(self, restriction_flags: int) -> list[str]:
        """Convert restriction flags to a tuple of restriction names."""
        result = []
        rest_num = 1
        rest_names = self._fare_restriction_list
        while restriction_flags:
            if restriction_flags & 1:
                result.append(rest_names[rest_num - 1])
            rest_num += 1
            restriction_flags >>= 1
        return result

    def get_restriction_name(self, restriction_num: int) -> str:
        """Convert restriction number to a restriction name."""
        if restriction_num < 1:
            raise IndexError(restriction_num)
        return self._fare_restriction_list[restriction_num - 1]

    def _init_choice_models(self, config):
        logger.info("Initializing choice models")
        for cm_name, cm in config.choice_models.items():
            x = passengersim.core.ChoiceModel(
                cm_name, cm.kind, random_generator=self.random_generator
            )
            for pname, pvalue in cm:
                if pname in ("kind", "name") or pvalue is None:
                    continue

                if pname == "todd_curve":
                    tmp_dwm = self.todd_curves[pvalue]
                    x.add_dwm(tmp_dwm)
                elif pname == "early_dep" and pvalue is not None:
                    x.early_dep_offset = pvalue["offset"]
                    x.early_dep_slope = pvalue["slope"]
                    x.early_dep_beta = pvalue["beta"]
                elif pname == "late_arr" and pvalue is not None:
                    x.late_arr_offset = pvalue["offset"]
                    x.late_arr_slope = pvalue["slope"]
                    x.late_arr_beta = pvalue["beta"]
                elif pname == "replanning" and pvalue is not None:
                    x.replanning_alpha = pvalue[0]
                    x.replanning_beta = pvalue[1]
                elif pname == "restrictions":
                    for rname, rvalue in pvalue.items():
                        restriction_num = self._get_fare_restriction_num(rname)
                        if isinstance(rvalue, list | tuple):
                            x.add_restriction(restriction_num, *rvalue)
                        else:
                            x.add_restriction(restriction_num, rvalue)
                elif isinstance(pvalue, list | tuple):
                    x.add_parm(pname, *pvalue)
                else:
                    x.add_parm(pname, pvalue)
            self.choice_models[cm_name] = x

    def _init_frat5_curves(self, config):
        logger.info("Initializing Frat5 curves")
        for f5_name, f5_data in config.frat5_curves.items():
            f5 = Frat5(f5_name)
            # ensure that the curve is sorted in descending order by days prior
            sorted_days_prior = reversed(sorted(f5_data.curve.keys()))
            for days_prior in sorted_days_prior:
                val = f5_data.curve[days_prior]
                f5.add_vals(val)
            f5.max_cap = f5_data.max_cap
            self.sim.add_frat5(f5)
            self.frat5curves[f5_name] = f5

    def _init_blf_curves(self, config):
        """These are currently grabbed by the RmStep"""
        pass

    def _init_load_factor_curves(self, config):
        logger.info("Initializing load factor curves")
        for lf_name, lf_curve in config.load_factor_curves.items():
            self.load_factor_curves[lf_name] = lf_curve

    def _init_carriers(self, config: Config):
        logger.info("Initializing carriers")
        self.carriers_dict = {}
        for carrier_name, carrier_config in config.carriers.items():
            try:
                rm_sys = self.rm_systems[carrier_config.rm_system]
            except KeyError:
                config._load_std_rm_system(carrier_config.rm_system)
                self._init_rm_system(
                    carrier_config.rm_system,
                    config.rm_systems[carrier_config.rm_system],
                    config,
                )
                rm_sys = self.rm_systems[carrier_config.rm_system]
            availability_control = rm_sys.availability_control
            carrier = passengersim.core.Carrier(carrier_name, availability_control)
            self.carriers_dict[carrier_name] = carrier
            carrier.rm_system = self.rm_systems[carrier_config.rm_system]
            carrier.truncation_rule = carrier_config.truncation_rule
            carrier.history_length = carrier_config.history_length
            carrier.cp_algorithm = carrier_config.cp_algorithm
            carrier.cp_quantize = carrier_config.cp_quantize
            carrier.cp_scale = carrier_config.cp_scale
            carrier.cp_record = carrier_config.cp_record
            if carrier_config.cp_elasticity is not None:
                carrier.cp_elasticity = carrier_config.cp_elasticity
            frat5_name = carrier_config.frat5
            if not frat5_name:
                frat5_name = config.rm_systems[carrier_config.rm_system].frat5
            if frat5_name is not None and frat5_name != "":
                # We want a deep copy of the Frat5 curve,
                # in case two carriers are using the same curve,
                # and we want to adjust one of them using ML
                try:
                    f5_data = config.frat5_curves[frat5_name]
                except KeyError:
                    config._load_std_frat5(frat5_name)
                    f5_data = config.frat5_curves[frat5_name]
                f5 = Frat5(f5_data.name)
                for _dcp, val in f5_data.curve.items():
                    f5.add_vals(val)
                if carrier_config.fare_adjustment_scale is not None:
                    f5.fare_adjustment_scale = carrier_config.fare_adjustment_scale
                carrier.frat5 = f5
            if (
                carrier_config.load_factor_curve is not None
                and carrier_config.load_factor_curve != ""
            ):
                lfc = self.load_factor_curves[carrier_config.load_factor_curve]
                carrier.load_factor_curve = lfc

            # Frat5 curve by market - experimental code !!!
            for k, name in carrier_config.frat5_map.items():
                a = k.split("-")  # orig-dest
                f5 = self.frat5curves[name]
                try:
                    carrier.add_frat5_mkt(a[0], a[1], f5)
                except Exception as e:
                    print(e)
                    print("values =", a[0], a[1], f5)
            for anc_code, anc_price in carrier_config.ancillaries.items():
                anc = Ancillary(anc_code, anc_price, 0)
                carrier.add_ancillary(anc)

            self.sim.add_carrier(carrier)

        self.classes = config.classes
        self.init_rm = {}  # TODO

    def _init_airports(self, config: Config):
        logger.info("Initializing airports")
        # Load the places into Airport objects.  We use lat/lon to get
        # great circle distance, and this also has the MCT data
        for code, p in config.places.items():
            assert isinstance(p, passengersim.config.Place)
            a = Airport(code, p.label)
            a.latitude, a.longitude = p.lat, p.lon
            if p.country is not None:
                a.country = p.country
            if p.state is not None:
                a.state = p.state
            if p.mct is not None:
                assert isinstance(p.mct, passengersim.config.MinConnectTime)
                a.mct_dd = p.mct.domestic_domestic
                a.mct_di = p.mct.domestic_international
                a.mct_id = p.mct.international_domestic
                a.mct_ii = p.mct.international_international
            self.airports[code] = a
            self.sim.add_airport(a)

    def _init_booking_curves(self, config):
        logger.info("Initializing booking curves")
        self.curves = {}
        for curve_name, curve_config in config.booking_curves.items():
            bc = passengersim.core.BookingCurve(curve_name)
            bc.random_generator = self.random_generator
            # ensure that the curve is sorted in descending order by days prior
            sorted_days_prior = reversed(sorted(curve_config.curve.keys()))
            for days_prior in sorted_days_prior:
                pct = curve_config.curve[days_prior]
                bc.add_dcp(days_prior, pct)
            self.curves[curve_name] = bc

        # It got more complex with cabins and buckets, so now it's in a separate method
        self._initialize_leg_cabin_bucket(config)

    def _init_demands(self, config):
        logger.info("Initializing demands")
        markets = {}
        market_multipliers = {}
        for mkt_config in config.markets:
            market_multipliers[f"{mkt_config.orig}~{mkt_config.dest}"] = (
                mkt_config.demand_multiplier
            )
        # This simulates PODS' favored carrier logic.  The CALP values are
        # all set to 1.0 in all their networks, so hard-coded for now but
        # we can load from YAML in the future if we need to
        if len(config.carriers) > 0:
            prob = 1.0 / len(config.carriers)
            calp = {cxr_name: prob for cxr_name in config.carriers.keys()}
        else:
            calp = {}

        for dmd_config in config.demands:
            mkt_ident = f"{dmd_config.orig}~{dmd_config.dest}"
            if mkt_ident not in markets:
                mkt = passengersim.core.Market(dmd_config.orig, dmd_config.dest)
                markets[mkt_ident] = mkt
            else:
                mkt = markets[mkt_ident]
            dmd = passengersim.core.Demand(segment=dmd_config.segment, market=mkt)
            dmd.base_demand = (
                dmd_config.base_demand
                * self.demand_multiplier
                * market_multipliers.get(mkt_ident, 1.0)
            )
            dmd.price = dmd_config.reference_fare
            dmd.reference_fare = dmd_config.reference_fare
            if dmd_config.distance > 0.01:
                dmd.distance = dmd_config.distance
            elif dmd.orig in self.airports and dmd.dest in self.airports:
                dmd.distance = get_mileage(self.airports, dmd.orig, dmd.dest)

            # Get the choice model name to use for this demand.
            model_name = dmd_config.choice_model or dmd_config.segment
            cm = self.choice_models.get(model_name, None)
            if cm is not None:
                dmd.add_choice_model(cm)
            else:
                raise ValueError(
                    f"Choice model {model_name} not found for demand {dmd}"
                )
            if dmd_config.curve:
                curve_name = str(dmd_config.curve).strip()
                curve = self.curves[curve_name]
                dmd.add_curve(curve)
            if dmd_config.todd_curve in self.todd_curves:
                dmd.add_dwm(self.todd_curves[dmd_config.todd_curve])
            if dmd_config.group_sizes is not None:
                dmd.add_group_sizes(dmd_config.group_sizes)
            dmd.prob_saturday_night = dmd_config.prob_saturday_night
            dmd.prob_num_days = dmd_config.prob_num_days
            dmd.prob_favored_carrier = calp

            if dmd_config.dwm_tolerance > 0.0:
                dmd.dwm_tolerance = dmd_config.dwm_tolerance
            elif len(self.config.dwm_tolerance) > 0:
                for tolerance in self.config.dwm_tolerance:
                    if tolerance["min_dist"] <= dmd.distance <= tolerance["max_dist"]:
                        if dmd.segment in tolerance:
                            dmd.dwm_tolerance = tolerance[dmd.segment]
                        else:
                            raise Exception(f"DWM tolerance data is missing segment '{dmd.segment}'")

            self.sim.add_demand(dmd)
            if self.debug:
                print(f"Added demand: {dmd}, base_demand = {dmd.base_demand}")
        # Hold PyObjects for markets in a dictionary in order to avoid duplicates
        self._markets = {k: v for k, v in self.markets.items()}

    def _init_fares(self, config: Config):
        logger.info("Initializing fares")
        # self.fares = []
        disable_ap = config.simulation_controls.disable_ap

        discovered_restrictions = set()

        for fare_config in config.fares:
            fare = passengersim.core.Fare(
                self.carriers_dict[fare_config.carrier],
                fare_config.orig,
                fare_config.dest,
                fare_config.booking_class,
                fare_config.price,
            )
            if not disable_ap:
                fare.adv_purch = fare_config.advance_purchase
            for rest_code in fare_config.restrictions:
                rest_num = self._get_fare_restriction_num(
                    rest_code, ignore_when_missing=True
                )
                if rest_num:
                    fare.add_restriction(rest_num)
                    discovered_restrictions.add(str(rest_code).casefold())
                else:
                    if config.simulation_controls.allow_unused_restrictions:
                        warnings.warn(
                            f"Restriction {rest_code!r} found in fares "
                            f"but not used in any choice model",
                            skip_file_prefixes=_warn_skips,
                            stacklevel=1,
                        )
                    else:
                        raise ValueError(
                            f"Restriction {rest_code!r} found in fares but not "
                            f"used in any choice model"
                        )
            self.sim.add_fare(fare)
            if self.debug:
                print(f"Added fare: {fare}")
            # self.fares.append(fare)

        # check that all restrictions used in choice models are present in fares
        for r in self._fare_restriction_list:
            if r not in discovered_restrictions:
                if config.simulation_controls.allow_unused_restrictions:
                    warnings.warn(
                        f"Restriction {r!r} used in choice models but "
                        f"not found in fares",
                        skip_file_prefixes=_warn_skips,
                        stacklevel=1,
                    )
                else:
                    raise ValueError(
                        f"Restriction {r!r} used in choice models but not found "
                        f"in fares"
                    )

        carriers = {cxr.name: cxr for cxr in self.sim.carriers}
        for path_config in config.paths:
            p = passengersim.core.Path(path_config.orig, path_config.dest, 0.0)
            p.path_quality_index = path_config.path_quality_index
            leg_index1 = path_config.legs[0]
            tmp_leg = self.legs[leg_index1]
            assert (
                tmp_leg.orig == path_config.orig
            ), "Path statement is corrupted, orig doesn't match"
            assert tmp_leg.flt_no == leg_index1
            p.add_leg(tmp_leg)
            if len(path_config.legs) >= 2:
                leg_index2 = path_config.legs[1]
                if leg_index2 > 0:
                    tmp_leg = self.legs[leg_index2]
                    p.add_leg(self.legs[leg_index2])
            assert (
                tmp_leg.dest == path_config.dest
            ), "Path statement is corrupted, dest doesn't match"
            path_carrier_name = tmp_leg.carrier_name
            if path_carrier_name not in carriers:
                raise ValueError(f"Carrier {path_carrier_name} not found")
            p.add_carrier(carriers[path_carrier_name])
            self.sim.add_path(p)

        # Go through and make sure things are linked correctly
        fares_dict = defaultdict(list)
        lowest_fare_dict = defaultdict(lambda: 9e9)
        highest_fare_dict = defaultdict(float)
        for f in self.sim.fares:
            od_key = (f.orig, f.dest)
            fares_dict[od_key].append(f)
            lowest_fare_dict[od_key] = min(lowest_fare_dict[od_key], f.price)
            highest_fare_dict[od_key] = max(highest_fare_dict[od_key], f.price)
        for dmd in self.sim.demands:
            tmp_fares = fares_dict[(dmd.orig, dmd.dest)]
            tmp_fares = sorted(tmp_fares, reverse=True, key=lambda p: p.price)
            for fare in tmp_fares:
                dmd.add_fare(fare)

            # Now set upper and lower bounds, these are used in continuous pricing
            # CP can never go lower than the lowest published fare
            lowest_published = lowest_fare_dict[(dmd.orig, dmd.dest)]
            highest_published = highest_fare_dict[(dmd.orig, dmd.dest)]
            for cxr in self.sim.carriers:
                cp_bounds = self.config.carriers[cxr.name].cp_bounds
                prev_fare = None
                for fare in tmp_fares:
                    if fare.carrier_name != cxr.name:
                        continue
                    if prev_fare is not None:
                        diff = prev_fare.price - fare.price
                        prev_fare.price_lower_bound = max(
                            prev_fare.price - diff * cp_bounds, lowest_published
                        )
                        fare.price_upper_bound = min(
                            fare.price + diff * cp_bounds, highest_published
                        )
                        # This provides a price floor, but will be overwritten
                        # each time through the loop EXCEPT for the lowest fare
                        fare.price_lower_bound = max(
                            fare.price - diff * cp_bounds, lowest_published
                        )
                    else:
                        fare.price_upper_bound = min(fare.price, highest_published)
                    prev_fare = fare

        logger.info("Initializing bucket decision fares")
        for leg in self.sim.legs:
            try:
                leg_market = self.sim.markets[f"{leg.orig}~{leg.dest}"]
            except KeyError:
                # no market for this leg, so no fares, that's ok
                continue
            assert len(leg_market.fares) > 0, f"No fares found for market {leg_market}"
            for fare in leg_market.fares:
                if fare.carrier_name == leg.carrier_name:
                    leg.set_bucket_blank_value(fare.booking_class, fare.price)

        self.sim.base_time = config.simulation_controls.reference_epoch()

    def _initialize_leg_cabin_bucket(self, config: Config):
        logger.info("Initializing legs, cabins, and buckets")
        self.legs = {}
        carriers = {}
        for carrier in self.sim.carriers:
            carriers[carrier.name] = carrier
        next_leg_id = 1
        for leg_config in config.legs:
            # if no leg_id is provided, we'll use the fltno if it's not already in use
            if (
                leg_config.leg_id is None
                and leg_config.fltno is not None
                and not self.sim.leg_id_exists(leg_config.fltno)
            ):
                leg_config.leg_id = leg_config.fltno
            # if still no leg_id, we'll use the next available
            if leg_config.leg_id is None:
                while self.sim.leg_id_exists(next_leg_id):
                    next_leg_id += 1
                leg_config.leg_id = next_leg_id
            leg = passengersim.core.Leg(
                leg_config.leg_id,
                carriers[leg_config.carrier],
                leg_config.fltno,
                leg_config.orig,
                leg_config.dest,
            )
            leg.dep_time = leg_config.dep_time
            leg.arr_time = leg_config.arr_time
            leg.dep_time_offset = leg_config.dep_time_offset
            leg.arr_time_offset = leg_config.arr_time_offset
            if leg_config.distance:
                leg.distance = leg_config.distance
            elif len(self.airports) > 0:
                leg.distance = get_mileage(self.airports, leg.orig, leg.dest)
            self.sim.add_leg(leg)

            # Now we do the cabins and buckets
            if isinstance(leg_config.capacity, int):
                cap = int(leg_config.capacity * self.capacity_multiplier)
                leg.capacity = cap
                cabin = passengersim.core.Cabin("", cap)
                leg.add_cabin(cabin)
                self.set_classes(leg, cabin)
            else:
                tot_cap = 0
                for cabin_code, tmp_cap in leg_config.capacity.items():
                    cap = int(tmp_cap * self.capacity_multiplier)
                    tot_cap += cap
                    cabin = passengersim.core.Cabin(cabin_code, cap)
                    leg.add_cabin(cabin)
                leg.capacity = tot_cap
                self.set_classes(leg, cabin)
            if self.debug:
                print(f"Added leg: {leg}, dist = {leg.distance}")
            self.legs[leg.leg_id] = leg

    def set_classes(self, leg: passengersim.core.Leg, _cabin, debug=False):
        leg_classes = self.config.carriers[leg.carrier.name].classes
        if len(leg_classes) == 0:
            return
        cap = float(leg.capacity)
        if debug:
            print(leg, "Capacity = ", cap)
        history_def = leg.carrier.get_history_def()
        for bkg_class in leg_classes:
            # Input as a percentage
            auth = int(cap * self.init_rm.get(bkg_class, 100.0) / 100.0)
            if isinstance(bkg_class, tuple):
                # We are likely using multi-cabin, so unpack it
                (bkg_class, cabin_code) = bkg_class
            b = passengersim.core.Bucket(bkg_class, alloc=auth, history=history_def)
            leg.add_bucket(b)
            if debug:
                print("    Bucket", bkg_class, auth)

    def setup_scenario(self) -> None:
        """
        Set up the scenario for the simulation.

        This will delete any existing data in the database under the same simulation
        name, build the connections if needed, and then call the vn_initial_mapping
        method to set up the initial mapping for the carriers using virtual nesting.
        """
        self.cnx.delete_experiment(self.sim.name)
        logger.debug("building connections")
        num_paths = self.sim.build_connections()
        self.sim.compute_hhi()
        if num_paths and self.cnx.is_open:
            database.tables.create_table_path_defs(self.cnx._connection, self.sim.paths)
        logger.debug(f"Connections done, num_paths = {num_paths}")
        self.sim.initialize_bucket_ap_rules()

        # start with default number of timeframes
        num_timeframes_default = len(self.config.dcps)
        if len(self.config.dcps) and self.config.dcps[-1] == 0:
            num_timeframes_default -= 1

        # initialize pathclasses for each carrier, using settings from the carrier
        # to size the history buffers
        for carrier in self.sim.carriers:
            self.sim.initialize_pathclasses(carrier.get_history_def(), carrier.name)

        # TODO: only initialize nonstop linkage when needed?
        self.sim.initialize_nonstop_path_linkage()

        # Airlines using Q-forecasting need to have pathclasses set up for all paths
        # so Q-demand can be forecasted by pathclass even in the absence of bookings
        for carrier in self.sim.carriers:
            if carrier.frat5:
                logger.info(
                    f"Setting up path classes for carrier {carrier.name}, "
                    "which is using a Frat5 curve"
                )
                for pth in self.sim.paths:
                    if pth.carrier_name != carrier.name:
                        continue
                    mkt = self.sim.markets[f"{pth.orig}~{pth.dest}"]
                    for fare in mkt.fares:
                        if fare.carrier_name == pth.carrier_name:
                            pthcls = pth.add_booking_class(
                                fare.booking_class, if_not_found=True
                            )
                            if pthcls is not None:
                                pthcls.add_fare(fare)
            self.vn_initial_mapping2(carrier.name)

        # This will save approximately the number of choice sets requested
        if self.choice_set_file is not None and self.choice_set_obs > 0:
            tot_dmd = 0
            for d in self.config.demands:
                tot_dmd += d.base_demand
            total_choice_sets = (
                tot_dmd
                * self.sim.num_trials
                * (self.sim.num_samples - self.sim.burn_samples)
            )
            prob = (
                self.choice_set_obs / total_choice_sets if total_choice_sets > 0 else 0
            )
            self.sim.choice_set_sampling_probability = prob

    def vn_initial_mapping(self):
        vn_carriers = []
        for carrier in self.sim.carriers:
            if carrier.control == "vn":
                vn_carriers.append(carrier.name)
        for path in self.sim.paths:
            if path.get_leg_carrier(0) in vn_carriers:
                for bc in self.classes:
                    pc = PathClass(bc)
                    index = int(bc[1])
                    pc.set_index(0, index)
                    path.add_path_class(pc)

    def vn_initial_mapping2(self, carrier_code):
        for path in self.sim.paths:
            if path.get_leg_carrier(0) == carrier_code:
                for i, pc in enumerate(path.pathclasses):
                    pc.set_index(0, i)

    def begin_sample(self, sample: int | None = None):
        """Beginning of sample processing."""
        if sample is None:
            # when sample is None, we simply increment the current sample number
            self.sim.sample += 1
        else:
            # otherwise, we set the sample number to the given value
            self.sim.sample = sample
        if self.sim.config.simulation_controls.random_seed is not None:
            self.reseed(
                [
                    self.sim.config.simulation_controls.random_seed,
                    self.sim.trial,
                    self.sim.sample,
                ]
            )
        self.sim.reset_counters()
        self.generate_demands()

    def end_sample(self):
        """End of sample processing."""

        # Record the departure statistics to carrier-level counters in the simulation
        self.sim.record_departure_statistics()

        # Roll histories to next sample
        self.sim.next_departure()

        # Commit data to the database
        if self.cnx:
            try:
                self.cnx.commit()
            except AttributeError:
                pass

        # Are we capturing choice-set data?
        if self.choice_set_file is not None:
            if self.sim.sample > self.sim.burn_samples:
                cs = self.sim.get_choice_set()
                for line in cs:
                    tmp = [str(z) for z in line]
                    tmp2 = ",".join(tmp)
                    print(tmp2, file=self.choice_set_file)
            self.sim.clear_choice_set()

        # Market share computation (MIDT-lite), might move to C++ in a future version
        alpha = 0.15
        for m in self.sim.markets.values():
            sold = float(m.sold)
            for a in self.sim.carriers:
                carrier_sold = m.get_carrier_sold(a.name)
                share = carrier_sold / sold if sold > 0 else 0
                if self.sim.sample > 1:
                    try:
                        old_share = m.get_carrier_share(a.name)
                    except KeyError:
                        old_share = 0.0
                    new_share = alpha * share + (1.0 - alpha) * old_share
                    m.set_carrier_share(a.name, new_share)
                else:
                    m.set_carrier_share(a.name, share)

    def begin_trial(self, trial: int):
        """Beginning of trial processing.

        Parameters
        ----------
        trial : int
            The trial number.
        """
        self.sim.trial = trial
        self.sim.reset_trial_counters()

        for carrier in self.sim.carriers:
            # Initialize the histories all the various things that need them.
            # This is by-carrier, as the carriers may eventually have different
            # data requirements (sizes) for their history arrays.
            self.sim.initialize_histories(
                carrier,
                num_departures=26,  # TODO make this a parameter
                num_timeframes=len(self.dcp_list) - 1,
                truncation_rule=carrier.truncation_rule,
                store_priceable=bool(carrier.frat5),
                floating_closures=False,
                wipe_existing=True,
            )

    def end_trial(self):
        """End of trial processing."""
        self.extract_segmentation_by_timeframe()
        self.extract_and_reset_bid_price_traces()
        if self.cnx.is_open:
            self.db_writer.final_write_to_sqlite(self.cnx._connection)
            # self.cnx.save_final(self.sim)

    def extract_and_reset_bid_price_traces(self):
        self.bid_price_traces[self.sim.trial] = {
            carrier.name: carrier.raw_bid_price_trace() for carrier in self.sim.carriers
        }
        self.displacement_traces[self.sim.trial] = {
            carrier.name: carrier.raw_displacement_cost_trace()
            for carrier in self.sim.carriers
        }
        for carrier in self.sim.carriers:
            carrier.reset_bid_price_trace()
            carrier.reset_displacement_cost_trace()

    def extract_segmentation_by_timeframe(
        self,
    ):
        # this should be run, if desired, at the end of each trial
        num_samples = self.sim.num_samples - self.sim.burn_samples
        top_level = {}
        for k in ("bookings", "revenue"):
            data = {}
            for carrier in self.sim.carriers:
                carrier_data = {}
                for segment, values in getattr(
                    carrier, f"raw_{k}_by_segment_fare_dcp"
                )().items():
                    carrier_data[segment] = (
                        pd.DataFrame.from_dict(values, "columns")
                        .rename_axis(index="days_prior", columns="booking_class")
                        .stack()
                    )
                if carrier_data:
                    data[carrier.name] = (
                        pd.concat(carrier_data, axis=1, names=["segment"]).fillna(0)
                        / num_samples
                    )
            if len(data) == 0:
                return None
            top_level[k] = pd.concat(data, axis=0, names=["carrier"])
        df = pd.concat(top_level, axis=1, names=["metric"])
        self.segmentation_data_by_timeframe[self.sim.trial] = df
        return df

    @contextlib.contextmanager
    def run_single_sample(self) -> int:
        """Context manager to run the next sample in the current trial.

        On entry, the sample number is run through to departure, so all
        sales have happened, but per-sample wrap up (e.g. rolling history
        forward, resetting counters) is deferred until exit.  This is useful
        for running a single sample in a testing framework.

        Yields
        ------
        int
            The sample number just completed.
        """
        if self.sim.trial < 0:
            warnings.warn(
                "Trial must be started before running a sample, "
                "implicitly starting Trial 0",
                skip_file_prefixes=_warn_skips,
                stacklevel=1,
            )
            self.begin_trial(0)
        self.begin_sample()
        while True:
            event = self.sim.go()
            self.run_carrier_models(event)
            if event is None or str(event) == "Done" or (event[0] == "Done"):
                assert (
                    self.sim.num_events() == 0
                ), f"Event queue still has {self.sim.num_events()} events"
                break
        yield self.sim.sample
        self.end_sample()

    def _run_single_trial(
        self,
        trial: int,
        n_samples_done: int = 0,
        n_samples_total: int = 0,
        progress: ProgressBar | None = None,
        update_freq: int | None = None,
    ):
        """Run a single trial of the simulation."""
        memory_log(f"begin _run_single_trial {trial}")
        if not n_samples_total:
            n_samples_total = self.sim.num_trials * self.sim.num_samples

        self.begin_trial(trial)
        for sample in range(self.sim.num_samples):
            t = time.time()
            if self.sim.config.simulation_controls.double_capacity_until:
                # Just trying this, PODS has something similar during burn phase
                if sample == 0:
                    for leg in self.sim.legs:
                        leg.capacity = leg.capacity * 2.0
                elif (
                    sample == self.sim.config.simulation_controls.double_capacity_until
                ):
                    for leg in self.sim.legs:
                        leg.capacity = leg.capacity / 2.0

            self.begin_sample(sample)
            if update_freq is not None and self.sim.sample % update_freq == 0:
                total_rev, n = 0.0, 0
                carrier_info = ""
                for cxr in self.sim.carriers:
                    total_rev += cxr.revenue
                    n += 1
                    carrier_info += (
                        f"{', ' if n > 0 else ''}{cxr.name}=${cxr.revenue:8.0f}"
                    )
                dmd_b, dmd_l = 0, 0
                for dmd in self.sim.demands:
                    if dmd.business:
                        dmd_b += dmd.scenario_demand
                    else:
                        dmd_l += dmd.scenario_demand
                d_info = f", {int(dmd_b)}, {int(dmd_l)}"
                logger.info(
                    f"Trial={self.sim.trial}, "
                    f"Sample={self.sim.sample}{carrier_info}{d_info}"
                )

            # Loop on passengers
            while True:
                event = self.sim.go()
                memory_log(f"pre-run_carrier_models {event}")
                self.run_carrier_models(event)
                memory_log(f"post-run_carrier_models {event}")
                if event is None or str(event) == "Done" or (event[0] == "Done"):
                    assert (
                        self.sim.num_events() == 0
                    ), f"Event queue still has {self.sim.num_events()} events"
                    break

            n_samples_done += 1
            self.sample_done_callback(n_samples_done, n_samples_total)
            self.end_sample()
            if progress is not None:
                progress.tick(refresh=(sample == 0))
            logger.info("completed sample %i in %.2f secs", sample, time.time() - t)

        self.sim.num_trials_completed += 1
        self.end_trial()

    def _run_sim(self, rich_progress: ProgressBar | None = None):
        update_freq = self.update_frequency
        logger.debug(
            f"run_sim, num_trials = {self.sim.num_trials}, "
            f"num_samples = {self.sim.num_samples}"
        )
        self.db_writer.update_db_write_flags()
        n_samples_total = self.sim.num_trials * self.sim.num_samples
        n_samples_done = 0
        self.sample_done_callback(n_samples_done, n_samples_total)
        if rich_progress is None:
            if self.sim.config.simulation_controls.show_progress_bar:
                progress = ProgressBar(total=n_samples_total)
            else:
                progress = DummyProgressBar()
        elif isinstance(rich_progress, Progress):
            if self.sim.config.simulation_controls.show_progress_bar:
                # if an external Progress object is provided, generate a
                # ProgressBar object from it
                progress = ProgressBar(
                    total=n_samples_total, external_progress=rich_progress
                )
            else:
                progress = DummyProgressBar()
        else:
            raise TypeError("rich_progress must be a Progress object")
        with progress:
            for trial in range(self.sim.num_trials):
                self._run_single_trial(
                    trial,
                    n_samples_done,
                    n_samples_total,
                    progress,
                    update_freq,
                )

    def _run_sim_single_trial(
        self, trial: int, *, rich_progress: Progress | None = None
    ):
        update_freq = self.update_frequency
        self.db_writer.update_db_write_flags()
        n_samples_total = self.sim.num_samples
        n_samples_done = 0
        self.sample_done_callback(n_samples_done, n_samples_total)
        if rich_progress is None:
            progress = DummyProgressBar()
        elif isinstance(rich_progress, Progress):
            progress = ProgressBar(
                total=n_samples_total, external_progress=rich_progress
            )
        else:
            raise TypeError("rich_progress must be a Progress object")
        with progress:
            self._run_single_trial(
                trial,
                n_samples_done,
                n_samples_total,
                progress,
                update_freq,
            )

    def run_carrier_models(self, info: Any = None, departed: bool = False, debug=False):
        what_had_happened_was = []
        try:
            event_type = info[0]

            if event_type.startswith("callback_"):
                # This is a callback function, not a string event type
                # so, call it with the remaining arguments
                callback_t = event_type[9:]
                callback_f = info[1]
                result = callback_f(self, *info[2:])
                if isinstance(result, dict):
                    self.callback_data.update_data(
                        callback_t, self.sim.trial, self.sim.sample, *info[2:], **result
                    )
                return

            recording_day = info[1]  # could in theory be non-integer for fractional days
            dcp_index = info[2]
            if dcp_index == -1:
                dcp_index = len(self.dcp_list) - 1

            if event_type.lower() in {"dcp", "done"}:
                self.sim.last_dcp = recording_day
                self.sim.last_dcp_index = dcp_index
                # self.capture_dcp_data(dcp_index)
                # self.capture_competitor_data()  # Simulates Infare / QL2

            # Run the specified process(es) for the carriers
            for carrier in self.sim.carriers:
                if event_type.lower() == "dcp":
                    # Regular Data Collection Points (pre-departure)
                    what_had_happened_was.append(f"run {carrier.name} DCP")
                    carrier.rm_system.run(
                        self.sim,
                        carrier.name,
                        dcp_index,
                        recording_day,
                        event_type="dcp",
                    )
                elif event_type.lower() == "daily":
                    # Daily report, every day prior to departure EXCEPT specified DCPs
                    what_had_happened_was.append(f"run {carrier.name} daily")
                    carrier.rm_system.run(
                        self.sim,
                        carrier.name,
                        dcp_index,
                        recording_day,
                        event_type="daily",
                    )
                elif event_type.lower() == "done":
                    # Post departure processing
                    what_had_happened_was.append(f"run {carrier.name} done")
                    carrier.rm_system.run(
                        self.sim,
                        carrier.name,
                        dcp_index,
                        recording_day,
                        event_type="dcp",
                    )
                    carrier.rm_system.run(
                        self.sim,
                        carrier.name,
                        dcp_index,
                        recording_day,
                        event_type="departure",
                    )
                    if self.sim.sample % 7 == 0:
                        # Can be used less frequently,
                        # such as ML steps on accumulated data
                        carrier.rm_system.run(
                            self.sim,
                            carrier.name,
                            dcp_index,
                            recording_day,
                            event_type="weekly",
                        )

            # Internal simulation data capture that is normally done by RM systems
            if event_type.lower() in {"dcp", "done"}:
                self.sim.last_dcp = recording_day
                self.sim.last_dcp_index = dcp_index
                self.capture_dcp_data(dcp_index)
                what_had_happened_was.append("capture_dcp_close_data")
                if self.sim.config.simulation_controls.capture_competitor_data:
                    self.capture_competitor_data()  # Simulates Infare / QL2

            # Database capture
            if event_type.lower() == "daily":
                if (
                    self.cnx.is_open
                    and self.sim.save_timeframe_details
                    and recording_day > 0
                ):
                    # if self.sim.sample == 101:
                    #     print("write_to_sqlite DAILY")
                    what_had_happened_was.append("write_to_sqlite daily")
                    _internal_log = self.db_writer.write_to_sqlite(
                        self.cnx._connection,
                        recording_day,
                        store_bid_prices=self.sim.config.db.store_leg_bid_prices,
                        intermediate_day=True,
                        store_displacements=self.sim.config.db.store_displacements,
                    )
            elif event_type.lower() in {"dcp", "done"}:
                if (
                    event_type.lower() == "done"
                    and "forecast_accuracy" in self.config.outputs.reports
                ):
                    self.sim.capture_forecast_accuracy()
                if self.cnx.is_open:
                    self.cnx.save_details(self.db_writer, self.sim, recording_day)
                if self.file_writer is not None:
                    self.file_writer.save_details(self.sim, recording_day)

            # simulation statistics record
            if event_type.lower() in {"dcp", "done"}:
                self.sim.record_dcp_statistics(recording_day)
            self.sim.record_daily_statistics(recording_day)

        except Exception as e:
            print(e)
            print("Error in run_carrier_models")
            print(f"{info=}")
            print("what_had_happened_was=", what_had_happened_was)
            raise

    def capture_competitor_data(self):
        for mkt in self.sim.markets.values():
            lowest = self.sim.shop(mkt.orig, mkt.dest)
            for cxr, price in lowest:
                mkt.set_competitor_price(cxr, price)

    def capture_dcp_data(self, dcp_index, closures_only=False):
        for leg in self.sim.legs:
            leg.capture_dcp(dcp_index)
        for path in self.sim.paths:
            path.capture_dcp(dcp_index, closures_only=closures_only)
        for carrier in self.sim.carriers:
            if dcp_index > 0:
                carrier.current_tf_index += 1

    def _accum_by_tf(self, dcp_index):
        # This is now replaced by C++ native counters ...
        if dcp_index > 0:
            prev_dcp = self.dcp_list[dcp_index - 1]
            for f in self.sim.fares:
                curr_business = self.fare_sales_by_dcp.get(("business", prev_dcp), 0)
                curr_leisure = self.fare_sales_by_dcp.get(("leisure", prev_dcp), 0)
                inc_leisure = curr_leisure + (f.sold - f.sold_business)
                inc_business = curr_business + f.sold_business
                self.fare_sales_by_dcp[("business", prev_dcp)] = inc_business
                self.fare_sales_by_dcp[("leisure", prev_dcp)] = inc_leisure

                key2 = (f.carrier_name, prev_dcp)
                curr_carrier = self.fare_sales_by_carrier_dcp[key2]
                self.fare_sales_by_carrier_dcp[key2] = curr_carrier + f.sold

                key3 = (f.carrier_name, f.booking_class, prev_dcp)
                self.fare_details_sold[key3] += f.sold
                self.fare_details_sold_business[key3] += f.sold_business
                self.fare_details_revenue[key3] += f.price * f.sold

    def generate_dcp_rm_events(self, debug=False):
        """Pushes an event per reading day (DCP) onto the queue.
        Also adds events for daily reoptimzation"""
        dcp_hour = self.sim.config.simulation_controls.dcp_hour
        if debug:
            tmp = datetime.fromtimestamp(self.sim.base_time, tz=timezone.utc)
            print(f"Base Time is {tmp.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        for dcp_index, dcp in enumerate(self.dcp_list):
            if dcp == 0:
                continue
            event_time = int(self.sim.base_time - dcp * 86400 + 3600 * dcp_hour)
            if debug:
                tmp = datetime.fromtimestamp(event_time, tz=timezone.utc)
                print(f"Added DCP {dcp} at {tmp.strftime('%Y-%m-%d %H:%M:%S %Z')}")
            info = ("DCP", dcp, dcp_index)
            rm_event = Event(info, event_time)
            self.sim.add_event(rm_event)

        # Now add the events for daily reoptimization
        max_days_prior = max(self.dcp_list)
        dcp_idx = 0
        for days_prior in reversed(range(max_days_prior)):
            if days_prior not in self.dcp_list:
                info = ("daily", days_prior, dcp_idx)
                event_time = int(
                    self.sim.base_time - days_prior * 86400 + 3600 * dcp_hour
                )
                rm_event = Event(info, event_time)
                self.sim.add_event(rm_event)
            else:
                dcp_idx += 1

        # add events for begin and end sample callbacks
        self.add_callback_events()

    def generate_demands(self, system_rn=None, debug=False):
        """Generate demands, following the procedure used in PODS
        The biggest difference is that we can put all the timeframe (DCP) demands
        into the event queue before any processing.
        For large models, I might rewrite this into the C++ core in the future"""
        self.generate_dcp_rm_events()
        total_events = 0
        system_rn = (
            self.random_generator.get_normal() if system_rn is None else system_rn
        )

        # We don't have an O&D object, but we use this to get a market random number
        # per market
        mrn_ref = {}

        # Need to have leisure / business split for PODS
        trn_ref = {
            "business": self.random_generator.get_normal(),
            "leisure": self.random_generator.get_normal(),
        }

        def get_or_make_random(grouping, key):
            if key not in grouping:
                grouping[key] = self.random_generator.get_normal()
            return grouping[key]

        end_time = self.base_time

        for dmd in self.sim.demands:
            base = dmd.base_demand

            # Get the random numbers we're going to use to perturb demand
            trn = get_or_make_random(trn_ref, (dmd.orig, dmd.dest, dmd.segment))
            mrn = get_or_make_random(mrn_ref, (dmd.orig, dmd.dest))
            if self.sim.config.simulation_controls.simple_k_factor:
                urn = (
                    self.random_generator.get_normal()
                    * self.sim.config.simulation_controls.simple_k_factor
                )
            else:
                urn = 0

            mu = base * (
                1.0
                + system_rn * self.sim.sys_k_factor
                + mrn * self.sim.mkt_k_factor
                + trn * self.sim.pax_type_k_factor
                + urn
            )
            mu = max(mu, 0.0)
            sigma = sqrt(
                mu * self.sim.config.simulation_controls.tot_z_factor
            )  # Correct?
            n = mu + sigma * self.random_generator.get_normal()
            dmd.scenario_demand = max(n, 0)

            if debug:
                logger.debug(
                    f"DMD,{self.sim.sample},{dmd.orig},{dmd.dest},"
                    f"{dmd.segment},{dmd.base_demand},"
                    f"{round(mu,2)},{round(sigma,2)},{round(n,2)}"
                )

            # Now we split it up over timeframes and add it to the simulation
            num_pax = int(dmd.scenario_demand + 0.5)  # rounding
            if (
                self.sim.config.simulation_controls.timeframe_demand_allocation
                == "pods"
            ):
                num_events_by_tf = self.sim.allocate_demand_to_tf_pods(
                    dmd, num_pax, self.sim.tf_k_factor, int(end_time)
                )
            else:
                num_events_by_tf = self.sim.allocate_demand_to_tf(
                    dmd, num_pax, self.sim.tf_k_factor, int(end_time)
                )
            num_events = sum(num_events_by_tf)
            total_events += num_events
            if num_events != round(num_pax):
                raise ValueError(
                    f"Generate demand function, num_pax={num_pax}, "
                    f"num_events={num_events}"
                )

        return total_events

    def generate_demands_gamma(self, system_rn=None, debug=False):
        """Using this as a quick test"""
        self.generate_dcp_rm_events()
        end_time = self.base_time
        cv100 = 0.3
        for dmd in self.sim.demands:
            mu = dmd.base_demand
            std_dev = cv100 * sqrt(mu) * 10.0
            # std_dev = mu * 0.3
            var = std_dev**2
            shape_a = mu**2 / var
            scale_b = var / mu
            loc = 0.0
            r = gamma.rvs(shape_a, loc, scale_b, size=1)
            num_pax = int(r[0] + 0.5)
            dmd.scenario_demand = num_pax
            self.sim.allocate_demand_to_tf_pods(
                dmd, num_pax, self.sim.tf_k_factor, int(end_time)
            )
        total_events = 0
        return total_events

    def compute_reports(
        self,
        sim: SimulationEngine,
        to_log=True,
        to_db: bool | database.Database = True,
        additional=(
            "fare_class_mix",
            "load_factors",
            # "bookings_by_timeframe",
            "total_demand",
        ),
    ) -> SummaryTables:
        num_samples = sim.num_trials_completed * (sim.num_samples - sim.burn_samples)
        if num_samples <= 0:
            raise ValueError(
                "insufficient number of samples outside burn period for reporting"
                f"\n- num_trials = {sim.num_trials}"
                f"\n- num_samples = {sim.num_samples}"
                f"\n- burn_samples = {sim.burn_samples}"
            )

        if to_db is True:
            to_db = self.cnx
        class_dist_df = self.compute_class_dist(sim, to_log, to_db)
        dmd_df = self.compute_demand_report(sim, to_log, to_db)
        fare_df = self.compute_fare_report(sim, to_log, to_db)
        leg_df = self.compute_leg_report(sim, to_log, to_db)
        path_df = self.compute_path_report(sim, to_log, to_db)
        path_classes_df = self.compute_path_class_report(sim, to_log, to_db)
        carrier_df = self.compute_carrier_report(sim, to_log, to_db)
        segmentation_df = self.compute_segmentation_by_timeframe()
        raw_load_factor_dist_df = self.compute_raw_load_factor_distribution(
            sim, to_log, to_db
        )
        leg_avg_load_factor_dist_df = self.compute_leg_avg_load_factor_distribution(
            sim, to_log, to_db
        )
        fare_class_dist_df = self.compute_raw_fare_class_mix(sim, to_log, to_db)
        bid_price_history_df = self.compute_bid_price_history(sim, to_log, to_db)
        displacement_df = self.compute_displacement_history(sim, to_log, to_db)
        demand_to_come_df = self.compute_demand_to_come_summary(sim, to_log, to_db)
        local_fraction_dist_df = self.compute_leg_local_fraction_distribution(
            sim, to_log, to_db
        )
        local_fraction_by_place = self.compute_local_fraction_by_place(
            sim, to_log, to_db
        )

        summary = SummaryTables(
            name=sim.name,
            class_dist=class_dist_df,
            config=sim.config,
            demands=dmd_df,
            fares=fare_df,
            legs=leg_df,
            paths=path_df,
            path_classes=path_classes_df,
            carriers=carrier_df,
            bid_price_history=bid_price_history_df,
            raw_load_factor_distribution=raw_load_factor_dist_df,
            leg_avg_load_factor_distribution=leg_avg_load_factor_dist_df,
            raw_fare_class_mix=fare_class_dist_df,
            leg_local_fraction_distribution=local_fraction_dist_df,
            local_fraction_by_place=local_fraction_by_place,
            n_total_samples=num_samples,
            segmentation_by_timeframe=segmentation_df,
            displacement_history=displacement_df,
            demand_to_come_summary=demand_to_come_df,
        )
        summary.load_additional_tables(self.cnx, sim.name, sim.burn_samples, additional)
        summary.cnx = self.cnx
        return summary

    def compute_demand_report(
        self, sim: SimulationEngine, to_log=True, to_db: database.Database | None = None
    ):
        dmd_df = []
        for m in sim.demands:
            avg_price = m.revenue / m.sold if m.sold > 0 else 0
            dmd_df.append(
                dict(
                    orig=m.orig,
                    dest=m.dest,
                    segment=m.segment,
                    sold=m.sold,
                    revenue=m.revenue,
                    avg_fare=m.revenue / m.sold if m.sold > 0 else 0,
                    gt_demand=m.gt_demand,
                    gt_sold=m.gt_sold,
                    gt_revenue=m.gt_revenue,
                )
            )
            if to_log:
                logger.info(
                    f"   Dmd: {m.orig}-{m.dest}:{m.segment}"
                    f"  Sold = {m.sold},  "
                    f"Rev = {m.revenue}, "
                    f"AvgFare = {avg_price:.2f}"
                )
        dmd_df = pd.DataFrame(dmd_df)
        if to_db and to_db.is_open:
            to_db.save_dataframe("demand_summary", dmd_df)
        return dmd_df

    def compute_class_dist(
        self, sim: SimulationEngine, to_log=True, to_db: database.Database | None = None
    ):
        # Get unique segments
        segs = set([dmd.segment for dmd in sim.demands])
        dist = defaultdict(int)
        for f in sim.fares:
            for seg in segs:
                k = (f.booking_class, seg)
                try:
                    dist[k] += f.get_sales_by_segment(seg)
                except Exception:
                    # If the segment isn't found, just ignore it.
                    # i.e. basic economy won't book Y0
                    pass

        class_dist_df = []
        for (cls, seg), sold in dist.items():
            class_dist_df.append(dict(booking_class=cls, segment=seg, sold=sold))
        class_dist_df = pd.DataFrame(class_dist_df)
        return class_dist_df

    def compute_fare_report(
        self, sim: SimulationEngine, to_log=True, to_db: database.Database | None = None
    ):
        fare_df = []
        for f in sim.fares:
            for dcp_index, days_prior in enumerate(self.dcp_list):
                fare_df.append(
                    dict(
                        carrier=f.carrier.name,
                        orig=f.orig,
                        dest=f.dest,
                        booking_class=f.booking_class,
                        dcp_index=dcp_index,
                        price=f.price,
                        sold=f.get_sales_by_dcp2(days_prior),
                        gt_sold=f.gt_sold,
                        avg_adjusted_price=f.get_adjusted_by_dcp(dcp_index),
                    )
                )
                if to_log:
                    logger.info(
                        f"   Fare: {f.carrier} {f.orig}-{f.dest}:{f.booking_class}"
                        # f"AvgAdjFare = {avg_adj_price:.2f},"
                        f"  Sold = {f.sold},  "
                        f"Price = {f.price}"
                    )
        fare_df = pd.DataFrame(fare_df)
        #        if to_db and to_db.is_open:
        #            to_db.save_dataframe("fare_summary", fare_df)
        return fare_df

    def compute_leg_report(
        self, sim: SimulationEngine, to_log=True, to_db: database.Database | None = None
    ):
        num_samples = sim.num_trials_completed * (sim.num_samples - sim.burn_samples)
        leg_df = []
        for leg in sim.legs:
            # Checking consistency while I debug the cabin code
            sum_b1, sum_b2 = 0, 0
            for b in leg.buckets:
                sum_b1 += b.sold
            for c in leg.cabins:
                for b in c.buckets:
                    sum_b2 += b.sold
            if sum_b1 != sum_b2:
                print("Oh, crap!")
            avg_sold = leg.gt_sold / num_samples
            avg_rev = leg.gt_revenue / num_samples
            lf = 100.0 * leg.gt_sold / (leg.capacity * num_samples)
            if to_log:
                logger.info(
                    f"    Leg: {leg.carrier}:{leg.flt_no} {leg.orig}-{leg.dest}: "
                    f" AvgSold = {avg_sold:6.2f},  AvgRev = ${avg_rev:,.2f}, "
                    f"LF = {lf:,.2f}%"
                )
            leg_df.append(
                dict(
                    leg_id=leg.leg_id,
                    carrier=leg.carrier_name,
                    flt_no=leg.flt_no,
                    orig=leg.orig,
                    dest=leg.dest,
                    avg_sold=avg_sold,
                    avg_rev=avg_rev,
                    lf=lf,
                )
            )
        leg_df = pd.DataFrame(leg_df)
        if to_db and to_db.is_open:
            to_db.save_dataframe("leg_summary", leg_df)
        return leg_df

    def compute_path_report(
        self, sim: SimulationEngine, to_log=True, to_db: database.Database | None = None
    ):
        num_samples = sim.num_trials_completed * (sim.num_samples - sim.burn_samples)
        avg_lf, n = 0.0, 0
        for leg in sim.legs:
            lf = 100.0 * leg.gt_sold / (leg.capacity * num_samples)
            avg_lf += lf
            n += 1

        tot_rev = 0.0
        for m in sim.demands:
            tot_rev += m.revenue

        avg_lf = avg_lf / n if n > 0 else 0
        if to_log:
            logger.info(f"    LF:  {avg_lf:6.2f}%, Total revenue = ${tot_rev:,.2f}")

        path_df = []
        for path in sim.paths:
            avg_sold = path.gt_sold / num_samples
            avg_sold_priceable = path.gt_sold_priceable / num_samples
            avg_rev = path.gt_revenue / num_samples
            if to_log:
                logger.info(
                    f"{path}, avg_sold={avg_sold:6.2f}, avg_rev=${avg_rev:10,.2f}"
                )
            data = dict(
                orig=path.orig,
                dest=path.dest,
                carrier1=path.get_leg_carrier(0),
                leg_id1=path.get_leg_id(0),
                carrier2=None,
                leg_id2=None,
                carrier3=None,
                leg_id3=None,
                avg_sold=avg_sold,
                avg_sold_priceable=avg_sold_priceable,
                avg_rev=avg_rev,
            )
            if path.num_legs() == 1:
                path_df.append(data)
            elif path.num_legs() == 2:
                data["carrier2"] = path.get_leg_carrier(1)
                data["leg_id2"] = path.get_leg_id(1)
                path_df.append(data)
            elif path.num_legs() == 3:
                data["carrier2"] = path.get_leg_carrier(1)
                data["leg_id2"] = path.get_leg_id(1)
                data["carrier3"] = path.get_leg_carrier(2)
                data["leg_id3"] = path.get_leg_id(2)
                path_df.append(data)
            else:
                raise NotImplementedError("path with more than 3 legs")
        path_df = pd.DataFrame(path_df)
        if to_db and to_db.is_open:
            to_db.save_dataframe("path_summary", path_df)
        return path_df

    def compute_path_class_report(
        self, sim: SimulationEngine, to_log=True, to_db: database.Database | None = None
    ):
        num_samples = sim.num_trials_completed * (sim.num_samples - sim.burn_samples)

        path_class_df = []
        for path in sim.paths:
            for pc in path.pathclasses:
                avg_sold = pc.gt_sold / num_samples
                avg_sold_priceable = pc.gt_sold_priceable / num_samples
                avg_rev = pc.gt_revenue / num_samples
                if to_log:
                    logger.info(
                        f"{pc}, avg_sold={avg_sold:6.2f}, avg_rev=${avg_rev:10,.2f}"
                    )
                data = dict(
                    orig=path.orig,
                    dest=path.dest,
                    carrier1=path.get_leg_carrier(0),
                    leg_id1=path.get_leg_id(0),
                    carrier2=None,
                    leg_id2=None,
                    carrier3=None,
                    leg_id3=None,
                    booking_class=pc.booking_class,
                    avg_sold=avg_sold,
                    avg_sold_priceable=avg_sold_priceable,
                    avg_rev=avg_rev,
                )
                if path.num_legs() == 1:
                    path_class_df.append(data)
                elif path.num_legs() == 2:
                    data["carrier2"] = path.get_leg_carrier(1)
                    data["leg_id2"] = path.get_leg_id(1)
                    path_class_df.append(data)
                elif path.num_legs() == 3:
                    data["carrier2"] = path.get_leg_carrier(1)
                    data["leg_id2"] = path.get_leg_id(1)
                    data["carrier3"] = path.get_leg_carrier(2)
                    data["leg_id3"] = path.get_leg_id(2)
                    path_class_df.append(data)
                else:
                    raise NotImplementedError("path with more than 3 legs")
        path_class_df = pd.DataFrame(path_class_df)
        if not path_class_df.empty:
            path_class_df.sort_values(
                by=["orig", "dest", "carrier1", "leg_id1", "booking_class"]
            )
            #        if to_db and to_db.is_open:
            #            to_db.save_dataframe("path_class_summary", path_class_df)
        return path_class_df

    def compute_carrier_report(
        self,
        sim: SimulationEngine,
        to_log: bool = True,
        to_db: database.Database | None = None,
    ) -> pd.DataFrame:
        """
        Compute a carrier summary table.

        The resulting table has one row per simulated carrier, and the following
        columns:

        - name
        - avg_sold
        - load_factor
        - avg_rev
        - asm (available seat miles)
        - rpm (revenue passenger miles)
        """
        num_samples = sim.num_trials_completed * (sim.num_samples - sim.burn_samples)
        carrier_df = []

        carrier_asm = defaultdict(float)
        carrier_rpm = defaultdict(float)
        carrier_leg_lf = defaultdict(float)
        carrier_leg_count = defaultdict(float)
        for leg in sim.legs:
            carrier_name = (
                leg.carrier_name if hasattr(leg, "carrier_name") else leg.carrier
            )  # TODO: remove hasattr
            carrier_asm[carrier_name] += leg.distance * leg.capacity * num_samples
            carrier_rpm[carrier_name] += leg.distance * leg.gt_sold
            carrier_leg_lf[carrier_name] += leg.gt_sold / (leg.capacity * num_samples)
            carrier_leg_count[carrier_name] += 1

        for cxr in sim.carriers:
            avg_sold = cxr.gt_sold / num_samples
            avg_rev = cxr.gt_revenue / num_samples
            asm = carrier_asm[cxr.name] / num_samples
            rpm = carrier_rpm[cxr.name] / num_samples
            # sys_lf = 100.0 * cxr.gt_revenue_passenger_miles / asm if asm > 0 else 0.0
            denom = carrier_asm[cxr.name]
            sys_lf = (100.0 * carrier_rpm[cxr.name] / denom) if denom > 0 else 0
            if to_log:
                logger.info(
                    f"Carrier: {cxr.name}, AvgSold: {round(avg_sold, 2)}, "
                    f"LF {sys_lf:.2f}%,  AvgRev ${avg_rev:10,.2f}"
                )

            # Add up total ancillaries
            tot_anc_rev = 0.0
            for anc in cxr.ancillaries:
                print(str(anc))
                tot_anc_rev += anc.price * anc.sold

            carrier_df.append(
                {
                    "carrier": cxr.name,
                    "sold": avg_sold,
                    "sys_lf": sys_lf,
                    "avg_leg_lf": 100
                    * carrier_leg_lf[cxr.name]
                    / max(carrier_leg_count[cxr.name], 1),
                    "avg_rev": avg_rev,
                    "avg_price": avg_rev / avg_sold if avg_sold > 0 else 0,
                    "asm": asm,
                    "rpm": rpm,
                    "yield": np.nan if rpm == 0 else avg_rev / rpm,
                    "ancillary_rev": tot_anc_rev,
                }
            )
        carrier_df = pd.DataFrame(carrier_df)
        if to_db and to_db.is_open:
            to_db.save_dataframe("carrier_summary", carrier_df)
        return carrier_df

    def compute_segmentation_by_timeframe(self) -> pd.DataFrame | None:
        if self.segmentation_data_by_timeframe:
            df = (
                pd.concat(self.segmentation_data_by_timeframe, axis=0, names=["trial"])
                .reorder_levels(["trial", "carrier", "booking_class", "days_prior"])
                .sort_index()
            )
            # df["Total"] = df.sum(axis=1)
            return df

    @staticmethod
    def compute_raw_load_factor_distribution(
        sim: SimulationEngine,
        to_log: bool = True,
        to_db: database.Database | None = None,
    ) -> pd.DataFrame:
        """
        Compute a load factor distribution report.

        This report is a dataframe, with integer index values from 0 to 100,
        and column for each carrier in the simulation. The values are the
        frequency of each leg load factor observed during the simulation
        (excluding any burn period).  The values for leg load factors are
        rounded down, so that a leg load factor of 99.9% is counted as 99,
        and only actually sold-out flights are in the 100% bin.
        """
        result = {}
        for carrier in sim.carriers:
            lf = pd.Series(
                carrier.raw_load_factor_distribution(),
                index=pd.RangeIndex(101, name="leg_load_factor"),
                name="frequency",
            )
            result[carrier.name] = lf
        if result:
            df = pd.concat(result, axis=1, names=["carrier"])
        else:
            df = pd.DataFrame(
                index=pd.RangeIndex(101, name="leg_load_factor"), columns=[]
            )
        if to_db and to_db.is_open:
            to_db.save_dataframe("raw_load_factor_distribution", df)
        return df

    @staticmethod
    def compute_leg_avg_load_factor_distribution(
        sim: SimulationEngine,
        to_log: bool = True,
        to_db: database.Database | None = None,
    ) -> pd.DataFrame:
        """
        Compute a leg average load factor distribution report.

        This report is a dataframe, with integer index values from 0 to 100,
        and column for each carrier in the simulation. The values are the
        frequency of each leg average load factor observed over the simulation
        (excluding any burn period).  The values for leg average load factors
        are rounded down, so that a leg average load factor of 99.9% is counted
        as 99, and only always sold-out flights are in the 100% bin.

        This is different from the raw load factor distribution, which is the
        distribution of load factors across sample days.  The number of
        observations in the leg average load factor (this distribution) is
        equal to the number of legs, while the raw load factor distribution
        has one observation per leg per sample day.  The variance of this
        distribution is much lower than the raw load factor distribution.
        """
        idx = pd.RangeIndex(101, name="leg_load_factor")
        result = {
            carrier.name: pd.Series(np.zeros(101, dtype=np.int32), index=idx)
            for carrier in sim.carriers
        }
        for leg in sim.legs:
            try:
                lf = int(np.floor(leg.avg_load_factor()))
            except TypeError:
                # TODO: remove this
                lf = int(np.floor(leg.avg_load_factor))
            if lf > 100:
                lf = 100
            if lf < 0:
                lf = 0
            # TODO remove hasattr
            result[
                leg.carrier_name if hasattr(leg, "carrier_name") else leg.carrier
            ].iloc[lf] += 1
        if result:
            df = pd.concat(result, axis=1, names=["carrier"])
        else:
            df = pd.DataFrame(
                index=pd.RangeIndex(101, name="leg_load_factor"), columns=[]
            )
        if to_db and to_db.is_open:
            to_db.save_dataframe("leg_avg_load_factor_distribution", df)
        return df

    def compute_raw_fare_class_mix(
        self,
        sim: SimulationEngine,
        to_log: bool = True,
        to_db: database.Database | None = None,
    ) -> pd.DataFrame:
        """
        Compute a fare class distribution report.

        This report is a dataframe, with index values giving the fare class,
        and column for each carrier in the simulation. The values are the
        number of passengers for each fare class observed during the simulation
        (excluding any burn period). This is a count of passengers not legs, so
        a passenger on a connecting itinerary only counts once.
        """
        result = {}
        for carrier in sim.carriers:
            fc = carrier.raw_fare_class_distribution()
            fc_sold = pd.Series(
                {k: v["sold"] for k, v in fc.items()},
                name="frequency",
            )
            fc_rev = pd.Series(
                {k: v["revenue"] for k, v in fc.items()},
                name="frequency",
            )
            result[carrier.name] = pd.concat(
                [fc_sold, fc_rev], axis=1, keys=["sold", "revenue"]
            ).rename_axis(index="booking_class")
        if result:
            df = pd.concat(result, axis=0, names=["carrier"])
        else:
            df = pd.DataFrame(
                columns=["sold", "revenue"],
                index=pd.MultiIndex(
                    [[], []], [[], []], names=["carrier", "booking_class"]
                ),
            )
        df = df.fillna(0)
        df["sold"] = df["sold"].astype(int)
        if to_db and to_db.is_open:
            to_db.save_dataframe("fare_class_distribution", df)
        return df

    @staticmethod
    def compute_bid_price_history(
        sim: SimulationEngine,
        to_log: bool = True,
        to_db: database.Database | None = None,
    ) -> pd.DataFrame:
        """Compute the average bid price history for each carrier."""
        result = {}
        for carrier in sim.carriers:
            bp = carrier.raw_bid_price_trace()
            result[carrier.name] = (
                pd.DataFrame.from_dict(bp, orient="index")
                .sort_index(ascending=False)
                .rename_axis(index="days_prior")
            )
        if result:
            df = pd.concat(result, axis=0, names=["carrier"])
        else:
            df = pd.DataFrame(
                columns=[
                    "bid_price_mean",
                    "bid_price_stdev",
                    "some_cap_bid_price_mean",
                    "some_cap_bid_price_stdev",
                    "fraction_some_cap",
                    "fraction_zero_cap",
                ],
                index=pd.MultiIndex(
                    [[], []], [[], []], names=["carrier", "days_prior"]
                ),
            )
        df = df.fillna(0)
        if to_db and to_db.is_open:
            to_db.save_dataframe("bid_price_history", df)
        return df

    @staticmethod
    def compute_displacement_history(
        sim: SimulationEngine,
        to_log: bool = True,
        to_db: database.Database | None = None,
    ) -> pd.DataFrame:
        """Compute the average displacement cost history for each carrier."""
        result = {}
        for carrier in sim.carriers:
            bp = carrier.raw_displacement_cost_trace()
            result[carrier.name] = (
                pd.DataFrame.from_dict(bp, orient="index")
                .sort_index(ascending=False)
                .rename_axis(index="days_prior")
            )
        if result:
            df = pd.concat(result, axis=0, names=["carrier"])
        else:
            df = pd.DataFrame(
                columns=[
                    "displacement_mean",
                    "displacement_stdev",
                ],
                index=pd.MultiIndex(
                    [[], []], [[], []], names=["carrier", "days_prior"]
                ),
            )
        df = df.fillna(0)
        if to_db and to_db.is_open:
            to_db.save_dataframe("displacement_history", df)
        return df

    @staticmethod
    def compute_demand_to_come_summary(
        sim: SimulationEngine,
        to_log: bool = True,
        to_db: database.Database | None = None,
    ) -> pd.DataFrame:
        raw = sim.summary_demand_to_come()
        df = (
            from_nested_dict(raw, ["segment", "days_prior", "metric"])
            .sort_index(ascending=[True, False])
            .rename(
                columns={"mean": "mean_future_demand", "stdev": "stdev_future_demand"}
            )
        )
        return df

    def compute_leg_local_fraction_distribution(
        self,
        sim: SimulationEngine,
        to_log: bool = True,
        to_db: database.Database | None = None,
    ) -> pd.DataFrame:
        """
        Compute a report on the fraction of leg passengers who are local.

        This report is a dataframe, with integer index values from 0 to 100,
        and column for each carrier in the simulation. The values are the
        frequency of the local leg-passenger fraction on each leg observed
        over the simulation (excluding any burn period).  The values are
        rounded down, so that a leg local fraction of 99.9% is counted
        as 99, and only always-local flights are in the 100% bin.
        """
        result = {}
        for carrier in sim.carriers:
            lf = pd.Series(
                sim.distribution_local_leg_passengers(carrier),
                index=pd.RangeIndex(101, name="local_fraction"),
                name="frequency",
            )
            result[carrier.name] = lf
        if result:
            df = pd.concat(result, axis=1, names=["carrier"])
        else:
            df = pd.DataFrame(
                index=pd.RangeIndex(101, name="local_fraction"), columns=[]
            )
        if to_db and to_db.is_open:
            to_db.save_dataframe("leg_local_fraction_distribution", df)
        return df

    def compute_local_fraction_by_place(
        self,
        sim: SimulationEngine,
        to_log: bool = True,
        to_db: database.Database | None = None,
    ) -> pd.DataFrame:
        """
        Compute a report on the fraction of leg passengers who are local.

        Parameters
        ----------
        sim
        to_log
        to_db

        Returns
        -------
        pd.DataFrame
        """
        result = {}
        for carrier in sim.carriers:
            df = pd.Series(
                sim.fraction_local_by_carrier_and_place(carrier.name),
                name=carrier.name,
            )
            result[carrier.name] = df
        if result:
            df = pd.concat(result, axis=1, names=["carrier"])
        else:
            df = pd.DataFrame(index=[], columns=[])
        if to_db and to_db.is_open:
            to_db.save_dataframe("local_fraction_by_place", df)
        return df

    def reseed(self, seed: int | list[int] | None = 42):
        logger.debug("reseeding random_generator: %s", seed)
        self.sim.random_generator.seed(seed)

    def _user_certificate(self, certificate_filename=None):
        if certificate_filename:
            from cryptography.x509 import load_pem_x509_certificate

            certificate_filename = pathlib.Path(certificate_filename)
            with certificate_filename.open("rb") as f:
                user_cert = load_pem_x509_certificate(f.read())
        else:
            user_cert = self.sim.config.license_certificate
        return user_cert

    def validate_license(self, certificate_filename=None, future: int = 0):
        user_cert = self._user_certificate(certificate_filename)
        return self.sim.validate_license(user_cert, future=future)

    def license_info(self, certificate_filename=None):
        user_cert = self._user_certificate(certificate_filename)
        return self.sim.license_info(user_cert)

    @property
    def config(self) -> Config:
        """The configuration used for this Simulation."""
        return self.sim.config

    def run(
        self,
        log_reports: bool = False,
        *,
        single_trial: int | None = None,
        summarizer: type[SimulationTablesT]
        | SimulationTablesT
        | None = SimulationTables,
        rich_progress: Progress | None = None,
    ) -> SummaryTables | SimulationTablesT:
        """
        Run the simulation and compute reports.

        Parameters
        ----------
        log_reports : bool
        single_trial : int, optional
            Run only a single trial, with the given trial number (to get
            the correct fixed random seed, for example).
        summarizer : type[SimulationTables] | SimulationTables | None
            Use this summarizer to compute the reports.  If None, the
            reports are computed in the SummaryTables object; this option
            is deprecated and will eventually be removed.
        rich_progress : Progress, optional
            A rich Progress object to use for displaying progress.  If not
            provided, a new Progress object will be created unless the
            simulation configuration specifies not to show progress.

        Returns
        -------
        SimulationTables or SummaryTables
        """
        if summarizer is None:
            warnings.warn(
                "Using SummaryTables to compute reports is deprecated, "
                "prefer SimulationTables in new code.",
                DeprecationWarning,
                stacklevel=2,
            )

        start_time = time.time()
        self.setup_scenario()
        if single_trial is not None:
            self._run_sim_single_trial(single_trial, rich_progress=rich_progress)
        else:
            self._run_sim(rich_progress=rich_progress)
        if self.choice_set_file is not None:
            self.choice_set_file.close()
        logger.info("Computing reports")
        if summarizer is None:
            summary = self.compute_reports(
                self.sim,
                to_log=log_reports or self.sim.config.outputs.log_reports,
                additional=self.sim.config.outputs.reports,
            )
            logger.info("Saving reports")
            if self.sim.config.outputs.excel:
                summary.to_xlsx(self.sim.config.outputs.excel)
        else:
            if isinstance(summarizer, GenericSimulationTables):
                summary = summarizer._extract(self)
            elif issubclass(summarizer, GenericSimulationTables):
                summary = summarizer.extract(self)
            else:
                raise TypeError(
                    "summarizer must be an instance or subclass of "
                    "GenericSimulationTables"
                )

        # check all callbacks for tracers, and if any are found, write their
        # finalized data to callback_data
        for cb_group in [
            "daily_callbacks",
            "begin_sample_callbacks",
            "end_sample_callbacks",
        ]:
            for cb in getattr(self, cb_group, []):
                if isinstance(cb, GenericTracer):
                    summary.callback_data[cb.name] = cb.finalize()

        # write output files if designated
        if isinstance(summary, GenericSimulationTables):
            if self.config.outputs.html and (
                self.config.outputs.disk is True
                or self.config.outputs.html.filename == self.config.outputs.disk
            ):
                # this will ensure the html and disk files have the same timestamp
                filenames = summary.save(self.config.outputs.html.filename)
                summary._metadata["outputs.html_filename"] = filenames[".html"]
                summary._metadata["outputs.disk_filename"] = filenames[".pxsim"]
            else:
                if self.config.outputs.html:
                    out_filename = summary.to_html(self.config.outputs.html.filename)
                    summary._metadata["outputs.html_filename"] = out_filename
                if isinstance(self.config.outputs.disk, str | pathlib.Path):
                    out_filename = summary.to_file(self.config.outputs.disk)
                    summary._metadata["outputs.disk_filename"] = out_filename
            if self.config.outputs.pickle:
                pkl_filename = summary.to_pickle(self.config.outputs.pickle)
                summary._metadata["outputs.pickle_filename"] = pkl_filename
            if self.config.outputs.excel:
                summary.to_xlsx(self.config.outputs.excel)

        logger.info(
            f"Th' th' that's all folks !!!    "
            f"(Elapsed time = {round(time.time() - start_time, 2)})"
        )
        return summary

    def run_trial(self, trial: int, log_reports: bool = False) -> SummaryTables:
        self.setup_scenario()
        self.sim.trial = trial

        update_freq = self.update_frequency
        logger.debug(
            f"run_sim, num_trials = {self.sim.num_trials}, "
            f"num_samples = {self.sim.num_samples}"
        )
        self.db_writer.update_db_write_flags()
        n_samples_total = self.sim.num_samples
        n_samples_done = 0
        self.sample_done_callback(n_samples_done, n_samples_total)
        if self.sim.config.simulation_controls.show_progress_bar:
            progress = ProgressBar(total=n_samples_total)
        else:
            progress = DummyProgressBar()
        with progress:
            self._run_single_trial(
                trial,
                n_samples_done,
                n_samples_total,
                progress,
                update_freq,
            )
        summary = self.compute_reports(
            self.sim,
            to_log=log_reports or self.sim.config.outputs.log_reports,
            additional=self.sim.config.outputs.reports,
        )
        return summary

    def backup_db(self, dst: pathlib.Path | str | sqlite3.Connection):
        """Back up this database to another copy.

        Parameters
        ----------
        dst : Path-like or sqlite3.Connection
        """
        return self.cnx.backup(dst)

    def get_choice_parameters(self, choicemodel: str | ChoiceModel):
        """Get the parameters for a choice model."""
        if isinstance(choicemodel, str):
            choicemodel = self.choice_models[choicemodel]
        raw = choicemodel.get_parameters()
        r = raw.pop("restrictions", ())
        rsigma = raw.pop("restriction_sigmas", ())
        for rname, rval, rsig in zip(self._fare_restriction_list, r, rsigma):
            raw[f"restrictions_{rname}"] = rval
            raw[f"restrictions_{rname}_sigma"] = rsig
        return raw

    def set_choice_parameters(
        self, choicemodel: str | ChoiceModel, values: dict[str, float]
    ):
        """Set the parameters for a choice model."""
        if isinstance(choicemodel, str):
            choicemodel = self.choice_models[choicemodel]
        raw = choicemodel.get_parameters()
        for k, v in values.items():
            if k.startswith("restrictions_"):
                if k.endswith("_sigma"):
                    kr = k[13:-6]
                else:
                    kr = k[13:]
                position = self._fare_restriction_mapping[kr] - 1
                if k.endswith("_sigma"):
                    raw["restriction_sigmas"][position] = v
                else:
                    raw["restrictions"][position] = v
            else:
                raw[k] = v
        choicemodel.set_parameters(raw)
