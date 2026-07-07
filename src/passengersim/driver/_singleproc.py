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
from collections import defaultdict
from datetime import UTC, datetime
from math import sqrt
from typing import TYPE_CHECKING, Any, TypeVar

import pandas as pd
from passengersim_core import Ancillary, ContextualOptimizer, CustomerModel
from rich.progress import Progress
from scipy.stats import gamma

import passengersim.config.rm_systems
import passengersim.core
from passengersim import database
from passengersim.callbacks import CallbackData, CallbackMixin
from passengersim.config import Config
from passengersim.config.manipulate import revalidate
from passengersim.config.places import get_mileage
from passengersim.config.snapshot_filter import SnapshotFilter
from passengersim.core import (
    Airport,
    DecisionWindow,
    Event,
    Frat5,
    SimulationEngine,
)
from passengersim.progressbar import DummyProgressBar, ProgressBar
from passengersim.rm.systems import get_registered_rm_system
from passengersim.summaries import SimulationTables
from passengersim.summaries.generic import GenericSimulationTables
from passengersim.tracers.generic import GenericTracer
from passengersim.utils.nested_dict import from_nested_dict  # noqa: F401
from passengersim.utils.si import si_units  # noqa: F401
from passengersim.utils.string_counting import StringTracker
from passengersim.utils.tempdir import MaybeTemporaryDirectory  # noqa: F401

from ._base_sim import BaseSimulation
from ._constructors import make_core_choice_model, make_core_leg
from ._demand_gen import allocate_sample_demands, generate_sample_demands
from ._firehose import Firehose

if TYPE_CHECKING:
    from passengersim.core import ChoiceModel

logger = logging.getLogger("passengersim")

SimulationTablesT = TypeVar("SimulationTablesT", bound=GenericSimulationTables)

_warn_skips = (os.path.dirname(__file__), os.path.dirname(contextlib.__file__))


def memory_log(tag):
    """
    Log memory usage information for debugging purposes.

    Parameters
    ----------
    tag : str
        A label to identify the memory logging point.

    Notes
    -----
    This function is currently disabled (pass statement) but can be
    used to log RSS (Resident Set Size) and VMS (Virtual Memory Size)
    information using psutil when enabled.
    """
    pass
    # import psutil  # noqa: F401
    #
    # p = psutil.Process()
    # mem_info = p.memory_info()
    # print(
    #     f"\nPSUTIL {tag}: rss={si_units(mem_info.rss, kind='B')} "
    #     f"vmw={si_units(mem_info.vms, kind='B')}",
    #     file=sys.stderr,
    # )


_DEFAULT_SUMMARIZER = SimulationTables


def get_default_summarizer() -> type[SimulationTablesT]:
    return _DEFAULT_SUMMARIZER


def set_default_summarizer[SimulationTablesT: GenericSimulationTables](summarizer: type[SimulationTablesT]):
    global _DEFAULT_SUMMARIZER
    _DEFAULT_SUMMARIZER = summarizer


def check_summarizer[SimulationTablesT: GenericSimulationTables](
    summarizer: type[SimulationTablesT] | SimulationTablesT | None,
) -> type[SimulationTablesT]:
    if summarizer is None:
        summarizer = get_default_summarizer()
    if not isinstance(summarizer, GenericSimulationTables) and not issubclass(summarizer, GenericSimulationTables):
        raise TypeError("summarizer must be an instance or subclass of GenericSimulationTables")
    return summarizer


class Simulation(BaseSimulation, CallbackMixin, Firehose):
    def __init__(
        self,
        config: Config,
        output_dir: pathlib.Path | None = None,
    ):
        """
        Initialize a Simulation instance.

        Parameters
        ----------
        config : Config
            The simulation configuration object. Will be revalidated during
            initialization.
        output_dir : pathlib.Path or None, optional
            Directory for output files. If None, a temporary directory
            will be created automatically.

        Notes
        -----
        This initializes the simulation with default parameters including
        DCP lists, choice models, and various data structures for tracking
        simulation results.
        """
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
        self.sample_done_callback = lambda n, n_total: None
        self.choice_set_file = None
        self.choice_set_obs = 0
        self.choice_set_mkts = []
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

        self._fare_restriction_mapping = StringTracker(start_from=1, case_sensitive=False)
        """Mapping of fare restriction names to restriction numbers."""

        self._rm_data: dict[tuple[str, str], Any] = {}
        """A collection of RM data, by carrier and data type.

        This can contain forecasts, optimizers, and/or other data or cached objects that can
        be shared across RM actions for a given carrier.  The key is a tuple of (carrier_name, data_type).
        """

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
            self._initialize_db_tables(config)

        self.callback_data = CallbackData()
        """Data stored from callbacks.

        This allows a user to store arbitrary data during a simulation using callbacks,
        and access it later.
        """

        self._firehose_buffers = {}

    def _initialize_db_tables(self, config: Config):
        with self.cnx:
            database.tables.create_table_leg_defs(self.cnx._connection, self.eng.legs)
            database.tables.create_table_fare_defs(self.cnx._connection, self.eng.fares)
            database.tables.create_table_fare_restriction_defs(
                self.cnx._connection, self._fare_restriction_mapping.list_all()
            )
            database.tables.create_table_path_defs(self.cnx._connection, self.eng.paths)
            if config.db != ":memory:":
                self.cnx.save_configs(config)
            self.cnx._commit_raw()

    @property
    def _eng(self) -> SimulationEngine:
        """
        Access to the underlying simulation engine.

        Returns
        -------
        SimulationEngine
            The core simulation engine instance.
        """
        return self.eng

    @property
    def random_generator(self) -> passengersim.core.Generator:
        """
        Access to the random generator for the simulation.

        Returns
        -------
        passengersim.core.Generator
            The random generator instance used for stochastic processes in the simulation.
        """
        return self.eng.random_generator

    @property
    def base_time(self) -> int:
        """
        The base time for the simulation.

        Returns
        -------
        int
            The base time in seconds since the epoch.
        """
        return self.eng.base_time

    @property
    def snapshot_filters(self) -> list[SnapshotFilter] | None:
        """
        Get the snapshot filters for the simulation.

        Returns
        -------
        list[SnapshotFilter] or None
            List of snapshot filter objects, or None if simulation
            is not initialized.
        """
        try:
            sim = self.eng
        except AttributeError:
            return None
        return sim.snapshot_filters

    @snapshot_filters.setter
    def snapshot_filters(self, x: list[SnapshotFilter]):
        """
        Set the snapshot filters for the simulation.

        Parameters
        ----------
        x : list[SnapshotFilter]
            List of snapshot filter objects to set.

        Raises
        ------
        ValueError
            If the simulation is not initialized.
        """
        try:
            sim = self.eng
        except AttributeError as err:
            raise ValueError("sim not initialized, cannot set snapshot_filters") from err
        sim.snapshot_filters = x

    def _initialize(self, config: Config):
        """
        Initialize all simulation components.

        Parameters
        ----------
        config : Config
            The simulation configuration object containing all settings
            and parameters for initialization.

        Notes
        -----
        This method orchestrates the initialization of all simulation
        components in the correct order, including the simulation engine,
        parameters, carriers, airports, demands, fares, and various curves.
        """
        self._init_sim_and_parms(config)
        self._init_circuity(config)
        self._init_todd_curves(config)
        self._init_choice_models(config)
        self._init_frat5_curves(config)
        self._init_blf_curves(config)
        self._init_load_factor_curves(config)
        self._init_carriers(config)
        self._init_booking_curves(config)
        self._init_airports(config)
        self._initialize_leg_cabin_bucket(config)
        self._init_demands(config)
        self._init_fares(config)
        logger.info("Connecting markets")
        self.eng.connect_markets()

        # For each carrier, cycle through all `RmAction`s in its `RmSys` and call the `init` for each.
        for carrier in self.eng.carriers:
            for action in carrier.rm_sys.action_queue:
                action.init(self)

    @property
    def db_writer(self):
        self.cnx.connect_to_simulation_engine(self.eng)
        return self.cnx._db_writer

    def _init_sim_and_parms(self, config):
        """
        Initialize the simulation engine and parameters.

        Parameters
        ----------
        config : Config
            Configuration object containing simulation parameters and settings.

        Notes
        -----
        This method creates the core simulation engine instance and configures
        it with parameters from the config, including demand/capacity multipliers,
        random seed, DCP settings, and choice set capture options.
        """
        logger.info("Initializing simulation engine parameters")
        self.eng = passengersim.core.SimulationEngine(name=config.scenario)
        self.eng.config = config
        self.eng.snapshot_filters = config.snapshot_filters
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
                    self.eng.set_parm("capture_choice_set", 1)
                    self.choice_set_file = open(pvalue, "w")
                    cols = self.eng.choice_set_columns()
                    tmp = ",".join(cols)
                    print(tmp, file=self.choice_set_file)
            elif pname == "capture_choice_set_obs":
                self.choice_set_obs = pvalue
            elif pname == "capture_choice_set_mkts":
                self.choice_set_mkts = pvalue
            elif pname == "revenue_alpha":
                # Not used in the core, it's set as a class variable in PathClass (later in the code)
                pass

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
                "segment_k_factor",
                "simple_cv100",
                "timeframe_demand_allocation",
                "tot_z_factor",
                "allow_unused_restrictions",
                "additional_settings",
                "connection_builder",
                "manual_paths",
                "speed_limits",
                "use_standard_todd_curves",
            ]:
                pass
            else:
                try:
                    self.eng.set_parm(pname, float(pvalue))
                except TypeError as err:
                    raise TypeError(f"Error setting parameter {pname} to {pvalue}") from err
        for pname, pvalue in config.simulation_controls.model_extra.items():
            print(f"extra simulation setting: {pname} = ", float(pvalue))
            self.eng.set_parm(pname, float(pvalue))
        if config.simulation_controls.additional_settings:
            self.eng.additional_settings(**config.simulation_controls.additional_settings)

        # There is a default array of DCPs, we'll override it with the data from the
        # input file (if available)
        if len(config.dcps) > 0:
            self.dcp_list = []
            for dcp_index, days_prior in enumerate(config.dcps):
                self.eng.add_dcp(dcp_index, days_prior)
                self.dcp_list.append(days_prior)
            # We need to add the last DCP, which is always 0, if not already in the list
            if self.dcp_list[-1] != 0:
                self.eng.add_dcp(len(self.dcp_list), 0)
                self.dcp_list.append(0)

    def _init_circuity(self, config):
        """
        Initialize circuity rules for the simulation.

        Parameters
        ----------
        config : Config
            Configuration object containing circuity rules.

        Notes
        -----
        Circuity rules define how passengers can connect through hubs
        and intermediate airports in their journey.
        """
        logger.info("Initializing circuity rules")
        for rule in config.circuity_rules:
            # Flatten the object into a dictionary,
            # SimulationEngine will iterate over it
            self.eng.add_circuity_rule(dict(rule))

    def _init_todd_curves(self, config):
        """
        Initialize TODD (Time-Of-Departure Demand) curves.

        Parameters
        ----------
        config : Config
            Configuration object containing TODD curve definitions.

        Notes
        -----
        TODD curves model how demand varies as the departure time approaches,
        which is crucial for revenue management optimization.
        """
        logger.info("Initializing TODD curves")
        for todd_name, todd in config.todd_curves.items():
            dwm = DecisionWindow(
                todd_name,
                k_factor=todd.k_factor,
                dwm_tod=[todd.probabilities.get(j) for j in range(24)],
                random_generator=self.random_generator,
            )
            self.todd_curves[todd_name] = dwm

    def _get_fare_restriction_num(self, restriction_name: str, *, ignore_when_missing: bool = False):
        """
        Get the numeric identifier for a fare restriction name.

        Parameters
        ----------
        restriction_name : str
            The name of the fare restriction.
        ignore_when_missing : bool, default False
            If True, return None when the restriction is not found instead
            of creating a new mapping.

        Returns
        -------
        int or None
            The numeric identifier for the restriction, or None if
            ignore_when_missing is True and the restriction is not found.
        """
        r = str(restriction_name)
        if ignore_when_missing:
            return self._fare_restriction_mapping.get_number_if_exists(r)
        return self._fare_restriction_mapping.get_number(r)

    def parse_restriction_flags(self, restriction_flags: int) -> list[str]:
        """
        Convert restriction flags to a list of restriction names.

        Parameters
        ----------
        restriction_flags : int
            Integer bit flags representing which restrictions are active.

        Returns
        -------
        list[str]
            List of restriction names corresponding to the set flags.
        """
        result = []
        rest_num = 1
        rest_names = self._fare_restriction_mapping.list_all()
        while restriction_flags:
            if restriction_flags & 1:
                result.append(rest_names[rest_num - 1])
            rest_num += 1
            restriction_flags >>= 1
        return result

    def get_restriction_name(self, restriction_num: int) -> str:
        """
        Convert restriction number to a restriction name.

        Parameters
        ----------
        restriction_num : int
            The numeric identifier for the restriction (must be >= 1).

        Returns
        -------
        str
            The name of the restriction.

        Raises
        ------
        IndexError
            If restriction_num is less than 1 or exceeds the number
            of defined restrictions.
        """
        if restriction_num < 1:
            raise IndexError(restriction_num)
        return self._fare_restriction_mapping.list_all()[restriction_num - 1]

    def _init_choice_models(self, config):
        """
        Initialize customer choice models.

        Parameters
        ----------
        config : Config
            Configuration object containing choice model definitions.

        Notes
        -----
        Choice models determine how passengers select among available
        flight options based on factors like price, schedule, and
        service attributes.
        """
        logger.info("Initializing choice models")
        for cm_name, cm in config.choice_models.items():
            self.choice_models[cm_name] = make_core_choice_model(
                cm, self.random_generator, self._fare_restriction_mapping, self.todd_curves
            )

    def _init_frat5_curves(self, config):
        """
        Initialize FRAT5 curves for revenue management.

        Parameters
        ----------
        config : Config
            Configuration object containing FRAT5 curve definitions.

        Notes
        -----
        FRAT5 curves define the fare ratio at which half (0.5) of the
        customers will buy up to the higher fare. These curves define how
        fare ratios change over time as departure approaches, used for revenue
        optimization decisions.
        """
        logger.info("Initializing Frat5 curves")
        for f5_name, f5_data in config.frat5_curves.items():
            f5 = Frat5(f5_name, f5_data.curve, max_cap=f5_data.max_cap)
            self.eng.add_frat5(f5)
            self.frat5curves[f5_name] = f5

    def _init_blf_curves(self, config):
        """These are currently grabbed by the RmStep"""
        pass

    def _init_load_factor_curves(self, config):
        logger.info("Initializing load factor curves")
        for lf_name, lf_curve in config.load_factor_curves.items():
            self.load_factor_curves[lf_name] = lf_curve

    def _init_carriers(self, config: Config):
        """
        Initialize carriers and their revenue management systems.

        Parameters
        ----------
        config : Config
            Configuration object containing carrier definitions including
            their associated revenue management systems.

        Notes
        -----
        This method sets up each carrier with its revenue management system,
        creating the necessary objects for managing inventory, pricing,
        and booking decisions.
        """
        logger.info("Initializing carriers")
        self.carriers_dict = {}
        self.rm_callbacks = {}
        for carrier_name, carrier_config in config.carriers.items():
            try:
                system_class = get_registered_rm_system(carrier_config.rm_system)
            except KeyError:
                raise ValueError(f"Unknown RM system: {carrier_config.rm_system}") from None
            # if `carrier_config.rm_system_options` is defined, use the options
            if carrier_config.rm_system_options is not None:
                # Define a callback-style RM system for this carrier
                system_def = carrier_config.rm_system_options.copy()
                system_def.pop("name", None)  # remove name if present
                rm_sys = system_class(carrier=carrier_name, cfg=config, **system_def)
                self.rm_callbacks[carrier_name] = rm_sys
            # otherwise, if `rm_system_options` is not explicitly False, and
            # the RM system name is NOT defined in the config nor in the old-style standard RM systems list,
            # but it IS a registered callback-style RM system, then also treat it as a callback-style RM system
            else:
                # Define a callback-style RM system with all default config for this carrier
                rm_sys = system_class(carrier=carrier_name, cfg=config)
                self.rm_callbacks[carrier_name] = rm_sys

            carrier = passengersim.core.Carrier(
                carrier_name,
                rm_sys.availability_control,
                store_q_history=carrier_config.store_q_history,
                truncation_rule=carrier_config.truncation_rule,
                proration_rule=carrier_config.proration_rule,
                history_length=carrier_config.history_length,
                cp_algorithm=carrier_config.cp_algorithm,
                cp_record_highest_closed_as_open=carrier_config.cp_record_highest_closed_as_open,
                cp_quantize=carrier_config.cp_quantize,
                cp_scale=carrier_config.cp_scale,
                cp_record=carrier_config.cp_record,
                cp_elasticity=carrier_config.cp_elasticity,
                cp_markets=carrier_config.cp_markets,
                cabin_ordering=carrier_config.cabin_ordering,
            )
            carrier.metadata = {
                "rm_system": {"name": carrier_config.rm_system, "options": carrier_config.rm_system_options},
            }
            self.carriers_dict[carrier_name] = carrier
            carrier.rm_sys = rm_sys
            self.daily_callback(rm_sys)
            frat5_name = carrier_config.frat5
            if not frat5_name and carrier_config.rm_system in config.rm_systems:
                frat5_name = config.rm_systems[carrier_config.rm_system].frat5
            if frat5_name is not None and frat5_name != "":
                # We want a deep copy of the Frat5 curve,
                # in case two carriers are using the same curve,
                # and we want to adjust one of them using ML
                f5_data = config.get_frat5_curve(frat5_name)
                f5 = Frat5(
                    f5_data.name,
                    f5_data.curve,
                    fare_adjustment_scale=carrier_config.fare_adjustment_scale,
                    max_cap=f5_data.max_cap,
                )
                carrier.frat5 = f5
            if carrier_config.load_factor_curve is not None and carrier_config.load_factor_curve != "":
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

            # Customer models, for CP
            for cm in carrier_config.customer_models:
                cust = CustomerModel(cm.name, "MNL")
                cust.add_parm("price", cm.price)
                cust.add_parm("nonstop", cm.nonstop)
                carrier.add_customer_model(cust)

            # Contextual CP Optimmizer
            if carrier_config.contextual_optimizer is not None:
                co = ContextualOptimizer()
                carrier.contextual_optimizer = co

            if carrier_config.cp_algorithm == "OPT":
                self.config.simulation_controls.capture_competitor_data = True

            self.eng.add_carrier(carrier)

        self.classes = config.classes

    def _init_airports(self, config: Config):
        """
        Initialize airports and their geographic information.

        Parameters
        ----------
        config : Config
            Configuration object containing airport/place definitions
            with coordinates and minimum connection times.

        Notes
        -----
        This method creates Airport objects with geographic coordinates
        used for distance calculations and minimum connection time (MCT)
        data for hub operations.
        """
        logger.info("Initializing airports")
        # Load the places into Airport objects.  We use lat/lon to get
        # great circle distance, and this also has the MCT data
        for code, p in config.places.items():
            assert isinstance(p, passengersim.config.Place)
            a = Airport(code, p.label)
            if p.lat is not None:
                a.latitude = p.lat
            if p.lon is not None:
                a.longitude = p.lon
            if p.country is not None:
                a.country = p.country
            if p.state is not None:
                a.state = p.state
            if p.mct is not None:
                assert isinstance(p.mct, passengersim.config.LimitConnectTime)
                a.mct_dd = p.mct.domestic_domestic
                a.mct_di = p.mct.domestic_international
                a.mct_id = p.mct.international_domestic
                a.mct_ii = p.mct.international_international
            if p.max_connect_time is not None:
                a.max_connect_time_dd = p.max_connect_time.domestic_domestic
                a.max_connect_time_di = p.max_connect_time.domestic_international
                a.max_connect_time_id = p.max_connect_time.international_domestic
                a.max_connect_time_ii = p.max_connect_time.international_international
            self.airports[code] = a
            self.eng.add_airport(a)

    def _init_booking_curves(self, config):
        logger.info("Initializing booking curves")
        self.curves = {}
        for curve_name, curve_config in config.booking_curves.items():
            bc = passengersim.core.BookingCurve(curve_name, random_generator=self.random_generator)
            # ensure that the curve is sorted in descending order by days prior
            sorted_days_prior = reversed(sorted(curve_config.curve.keys()))
            for days_prior in sorted_days_prior:
                pct = curve_config.curve[days_prior]
                bc.add_dcp(days_prior, pct)
            self.curves[curve_name] = bc

    def _init_demands(self, config):
        logger.info("Initializing demands")
        markets = {}
        market_multipliers = {}
        for mkt_config in config.markets:
            market_multipliers[f"{mkt_config.orig}~{mkt_config.dest}"] = mkt_config.demand_multiplier
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
            dmd = passengersim.core.Demand(
                segment=dmd_config.segment,
                market=mkt,
                deterministic=dmd_config.deterministic,
                base_demand=float(
                    dmd_config.base_demand * self.demand_multiplier * market_multipliers.get(mkt_ident, 1.0)
                ),
                price=dmd_config.reference_price,
                reference_price=dmd_config.reference_price,
            )
            if dmd_config.distance > 0.01:
                dmd.distance = dmd_config.distance
            elif dmd.orig in self.airports and dmd.dest in self.airports:
                dmd.distance = get_mileage(self.airports, dmd.orig, dmd.dest)

            # Get the choice model name to use for this demand.
            model_name = dmd_config.choice_model or dmd_config.segment
            cm = self.choice_models.get(model_name, None)
            if cm is not None:
                dmd.choice_model = cm
            else:
                raise ValueError(f"Choice model {model_name} not found for demand {dmd}")
            if dmd_config.emult is not None:
                dmd.emult = dmd_config.emult
            else:
                dmd.emult = cm.get_parameters()["emult"]
            if dmd_config.curve:
                curve_name = str(dmd_config.curve).strip()
                curve = self.curves[curve_name]
                dmd.add_curve(curve)
            else:
                # If there is no booking curve name attached, we will check if any booking
                # curves are defined. It is valid sometimes to have no booking curves if you
                # are not planning to actually run a simulation.  We will assume that if
                # curves are defined then they should be used.
                if self.curves:
                    raise ValueError(f"Booking curve not defined for demand {dmd}")
            if dmd_config.todd_curve in self.todd_curves:
                dmd.dwm = self.todd_curves[dmd_config.todd_curve]
            if dmd_config.group_sizes is not None:
                dmd.add_group_sizes(dmd_config.group_sizes)
            if dmd_config.prob_saturday_night is not None:
                dmd.prob_saturday_night = dmd_config.prob_saturday_night
            dmd.prob_num_days = dmd_config.prob_num_days
            dmd.prob_favored_carrier = calp

            for o in dmd_config.overrides:
                dmd.add_override(o.carrier, o.discount_pct, o.pref_adj)

            if dmd_config.dwm_tolerance > 0.0:
                dmd.dwm_tolerance = dmd_config.dwm_tolerance
            elif len(self.config.dwm_tolerance) > 0:
                for tolerance in self.config.dwm_tolerance:
                    if tolerance["min_dist"] <= dmd.distance <= tolerance["max_dist"]:
                        if dmd.segment in tolerance:
                            dmd.dwm_tolerance = tolerance[dmd.segment]
                        else:
                            raise Exception(f"DWM tolerance data is missing segment '{dmd.segment}'")

            self.eng.add_demand(dmd)
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
            fare.brand = fare_config.brand
            fare.cabin = fare_config.cabin
            fare.min_stay = fare_config.min_stay
            fare.saturday_night_required = fare_config.saturday_night_required
            if not disable_ap:
                fare.adv_purch = fare_config.advance_purchase
            for rest_code in fare_config.restrictions:
                rest_num = self._get_fare_restriction_num(rest_code, ignore_when_missing=True)
                if rest_num:
                    fare.add_restriction(rest_num)
                    discovered_restrictions.add(str(rest_code).casefold())
                else:
                    if config.simulation_controls.allow_unused_restrictions:
                        warnings.warn(
                            f"Restriction {rest_code!r} found in fares but not used in any choice model",
                            skip_file_prefixes=_warn_skips,
                            stacklevel=1,
                        )
                    else:
                        raise ValueError(f"Restriction {rest_code!r} found in fares but not used in any choice model")
            self.eng.add_fare(fare)
            if self.debug:
                print(f"Added fare: {fare}")
            # self.fares.append(fare)

        # check that all restrictions used in choice models are present in fares
        for r in self._fare_restriction_mapping.list_all():
            if r not in discovered_restrictions:
                if config.simulation_controls.allow_unused_restrictions:
                    warnings.warn(
                        f"Restriction {r!r} used in choice models but not found in fares",
                        skip_file_prefixes=_warn_skips,
                        stacklevel=1,
                    )
                else:
                    raise ValueError(f"Restriction {r!r} used in choice models but not found in fares")

        carriers = {cxr.name: cxr for cxr in self.eng.carriers}
        for path_config in config.paths:
            p = passengersim.core.Path(path_config.orig, path_config.dest, 0.0)
            p.path_quality_index = path_config.path_quality_index
            leg_index1 = path_config.legs[0]
            tmp_leg = self.legs[leg_index1]
            assert tmp_leg.orig == path_config.orig, "Path statement is corrupted, orig doesn't match"
            assert tmp_leg.leg_id == leg_index1
            p.add_leg(tmp_leg)
            i = 1
            while len(path_config.legs) > i:
                next_leg_id = path_config.legs[i]
                if next_leg_id > 0:
                    tmp_leg = self.legs[next_leg_id]
                    p.add_leg(self.legs[next_leg_id])
                i += 1
            if tmp_leg.dest != path_config.dest:
                raise ValueError(
                    f"Path is corrupted, final leg dest {tmp_leg.dest} doesn't match path dest {path_config.dest}"
                )
            path_carrier_name = tmp_leg.carrier_name
            if path_carrier_name not in carriers:
                raise ValueError(f"Carrier {path_carrier_name} not found")
            p.add_carrier(carriers[path_carrier_name])
            self.eng.add_path(p)

        # Go through and make sure things are linked correctly
        fares_dict = defaultdict(list)
        lowest_fare_dict = defaultdict(lambda: 9e9)
        highest_fare_dict = defaultdict(float)
        for f in self.eng.fares:
            od_key = (f.orig, f.dest)
            fares_dict[od_key].append(f)
            lowest_fare_dict[od_key] = min(lowest_fare_dict[od_key], f.price)
            highest_fare_dict[od_key] = max(highest_fare_dict[od_key], f.price)
        for dmd in self.eng.demands:
            tmp_fares = fares_dict[(dmd.orig, dmd.dest)]
            tmp_fares = sorted(tmp_fares, reverse=True, key=lambda p: p.price)
            for fare in tmp_fares:
                dmd.add_fare(fare)

            # Now set upper and lower bounds, these are used in continuous pricing
            # CP can never go lower than the lowest published fare
            lowest_published = lowest_fare_dict[(dmd.orig, dmd.dest)]
            highest_published = highest_fare_dict[(dmd.orig, dmd.dest)]
            for cxr in self.eng.carriers:
                cp_bounds = self.config.carriers[cxr.name].cp_bounds
                prev_fare = None
                for fare in tmp_fares:
                    if fare.carrier_name != cxr.name:
                        continue
                    if prev_fare is not None:
                        diff = prev_fare.price - fare.price
                        prev_fare.price_lower_bound = max(prev_fare.price - diff * cp_bounds, lowest_published)
                        fare.price_upper_bound = min(fare.price + diff * cp_bounds, highest_published)
                        # This provides a price floor, but will be overwritten
                        # each time through the loop EXCEPT for the lowest fare
                        fare.price_lower_bound = max(fare.price - diff * cp_bounds, lowest_published)
                    else:
                        ub = highest_published * (1.0 + self.config.carriers[cxr.name].cp_upper_bound)
                        fare.price_upper_bound = min(fare.price, ub)
                    prev_fare = fare

        logger.info("Initializing bucket decision fares")
        for leg in self.eng.legs:
            try:
                leg_market = self.eng.markets[f"{leg.orig}~{leg.dest}"]
            except KeyError:
                # no market for this leg, so no fares, that's ok
                continue
            assert len(leg_market.fares) > 0, f"No fares found for market {leg_market}"
            for fare in leg_market.fares:
                if fare.carrier_name == leg.carrier_name:
                    leg.set_bucket_blank_value(fare.booking_class, fare.price)

        self.eng.base_time = config.simulation_controls.reference_epoch()

    def _initialize_leg_cabin_bucket(self, config: Config):
        logger.info("Initializing legs, cabins, and buckets")
        self.legs = {}
        carriers = {}
        for carrier in self.eng.carriers:
            carriers[carrier.name] = carrier
        next_leg_id = 1
        for leg_config in config.legs:
            leg = make_core_leg(
                leg_config,
                carriers=carriers,
                next_leg_id=next_leg_id,
                places=self.airports,
                leg_id_exists=self.eng.leg_id_exists,
                booking_classes=self.config.carriers[leg_config.carrier].classes,
            )
            # update the proposed next leg id as needed
            while self.eng.leg_id_exists(next_leg_id):
                next_leg_id += 1
            self.eng.add_leg(leg)
            self.legs[leg.leg_id] = leg

    def set_classes(self, leg: passengersim.core.Leg, _cabin, debug=False):
        leg_classes = self.config.carriers[leg.carrier.name].classes
        cabin_code_list = [c.name for c in leg.cabins]
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
            else:
                cabin_code = bkg_class[0]
            if cabin_code not in cabin_code_list:
                continue
            b = passengersim.core.Bucket(bkg_class, alloc=auth, history=history_def)
            b.cabin = cabin_code
            leg.add_bucket(b)
            if debug:
                print("    Added Bucket", leg, bkg_class, auth)

    def setup_scenario(self) -> None:
        """
        Set up the scenario for the simulation.

        This will delete any existing data in the database under the same simulation
        name, build the connections if needed, and then call the vn_initial_mapping
        method to set up the initial mapping for the carriers using virtual nesting.
        """
        self.cnx.delete_experiment(self.eng.name)
        logger.debug("building connections")
        num_paths = self.eng.build_connections(**dict(self.config.simulation_controls.connection_builder))
        self.eng.compute_hhi()
        if num_paths and self.cnx.is_open:
            database.tables.create_table_path_defs(self.cnx._connection, self.eng.paths)
        logger.debug(f"Connections done, num_paths = {num_paths}")
        self.eng.initialize_bucket_ap_rules()

        # start with default number of timeframes
        num_timeframes_default = len(self.config.dcps)
        if len(self.config.dcps) and self.config.dcps[-1] == 0:
            num_timeframes_default -= 1

        # initialize pathclasses for each carrier, using settings from the carrier
        # to size the history buffers
        # Also, Q-demand can be forecasted by pathclass even in the absence of bookings
        for carrier in self.eng.carriers:
            self.eng.initialize_pathclasses(carrier.get_history_def(), carrier.name)
            try:
                self.vn_initial_mapping(carrier.name)
            except Exception as e:
                print(e)
        for _p in self.eng.paths:
            for _pc in _p.pathclasses:
                _pc.revenue_alpha = self.config.simulation_controls.revenue_alpha
                break  # We just set a class attribute, so no need to keep iterating
            break

        # TODO: only initialize nonstop linkage when needed?
        self.eng.initialize_nonstop_path_linkage()

        # Compute a sampling probability to get approximately the number of
        # choice sets requested
        if self.choice_set_file is not None and self.choice_set_obs > 0:
            tot_dmd = 0
            for d in self.config.demands:
                if len(self.choice_set_mkts) == 0 or (d.orig, d.dest) in self.choice_set_mkts:
                    tot_dmd += d.base_demand
            usable_samples = self.eng.num_trials * (self.eng.num_samples - self.eng.burn_samples)
            total_choice_sets = tot_dmd * usable_samples
            prob = self.choice_set_obs / total_choice_sets if total_choice_sets > 0 else 0
            self.eng.choice_set_sampling_probability = prob
            self.eng.choice_set_mkts = self.choice_set_mkts

        # must close all Python SQLite database connections, so that the
        # C++ simulation engine can open the database without locking issues
        self.cnx.close()

    def vn_initial_mapping(self, carrier_code):
        """
        Set up initial virtual nesting mapping for a carrier.

        Parameters
        ----------
        carrier_code : str
            The carrier code to set up virtual nesting mapping for.

        Notes
        -----
        This method assigns index values to path classes for carriers
        using virtual nesting, which allows revenue management systems
        to map between physical and virtual booking classes.
        """
        for path in self.eng.paths:
            if path.get_leg_carrier(0) == carrier_code:
                for i, pc in enumerate(path.pathclasses):
                    pc.set_index(0, i)

    def begin_sample(self, sample: int | None = None):
        """
        Begin processing a new sample in the simulation.

        Parameters
        ----------
        sample : int or None, optional
            The sample number to set. If None, the current sample number
            will be incremented by 1.

        Notes
        -----
        This method handles sample initialization including setting the
        random seed (if configured) and preparing the simulation state
        for the new sample.
        """
        if sample is None:
            # when sample is None, we simply increment the current sample number
            self.eng.sample += 1
        else:
            # otherwise, we set the sample number to the given value
            self.eng.sample = sample
        if self.eng.config.simulation_controls.random_seed is not None:
            self.reseed(
                [
                    self.eng.config.simulation_controls.random_seed,
                    self.eng.trial,
                    self.eng.sample,
                ]
            )
        self.eng.reset_counters()
        self.generate_demands()

    def end_sample(self):
        """
        End processing of the current sample.

        Notes
        -----
        This method records departure statistics to carrier-level counters,
        handles choice set and competitor data capture if configured,
        and performs other end-of-sample cleanup and data collection tasks.
        """

        # Record the departure statistics to carrier-level counters in the simulation
        self.eng.record_departure_statistics()

        # Roll histories to next sample
        self.eng.next_departure()

        # Commit data to the database
        if self.cnx:
            try:
                self.cnx.commit()
            except AttributeError:
                pass
        self.db_writer.commit()

        # Are we capturing choice-set data?
        if self.choice_set_file is not None:
            if self.eng.sample > self.eng.burn_samples:
                cs = self.eng.get_choice_set()
                for line in cs:
                    tmp = [str(z) for z in line]
                    tmp2 = ",".join(tmp)
                    print(tmp2, file=self.choice_set_file)
            self.eng.clear_choice_set()

        # Market share computation (MIDT-lite), might move to C++ in a future version
        alpha = 0.15
        for m in self.eng.markets.values():
            sold = float(m.sold)
            for a in self.eng.carriers:
                carrier_sold = m.get_carrier_sold(a.name)
                share = carrier_sold / sold if sold > 0 else 0
                if self.eng.sample > 1:
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
        self.eng.trial = trial
        logger.info("beginning trial %d", trial)
        self.eng.reset_trial_counters()

        for carrier in self.eng.carriers:
            # Initialize the histories all the various things that need them.
            # This is by-carrier, as the carriers may eventually have different
            # data requirements (sizes) for their history arrays.
            self.eng.initialize_histories(
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
        self.db_writer.final_write_to_sqlite(self.cnx._connection)
        self.save_simulation_state()

    def extract_and_reset_bid_price_traces(self):
        self.bid_price_traces[self.eng.trial] = {
            carrier.name: carrier.raw_bid_price_trace() for carrier in self.eng.carriers
        }
        self.displacement_traces[self.eng.trial] = {
            carrier.name: carrier.raw_displacement_cost_trace() for carrier in self.eng.carriers
        }
        for carrier in self.eng.carriers:
            carrier.reset_bid_price_trace()
            carrier.reset_displacement_cost_trace()

    def extract_segmentation_by_timeframe(self):
        # this should be run, if desired, at the end of each trial
        num_samples = self.eng.num_samples - self.eng.burn_samples
        top_level = {}
        for k in ("bookings", "revenue"):
            data = {}
            for carrier in self.eng.carriers:
                carrier_data = {}
                for segment, values in getattr(carrier, f"raw_{k}_by_segment_fare_dcp")().items():
                    carrier_data[segment] = (
                        pd.DataFrame.from_dict(values, "columns")
                        .rename_axis(index="days_prior", columns="booking_class")
                        .stack()
                    )
                if carrier_data:
                    data[carrier.name] = pd.concat(carrier_data, axis=1, names=["segment"]).fillna(0) / num_samples
            # add non-bookings to the data dict
            if k == "bookings":
                non_bookings = pd.DataFrame.from_dict(self.eng.nonbookings_by_segment_dcp(), "columns").rename_axis(
                    index="days_prior", columns="segment"
                )
                non_bookings["booking_class"] = "XX"
                data["NONE"] = non_bookings.reset_index().set_index(["days_prior", "booking_class"]) / num_samples
            if len(data) == 0:
                return None
            top_level[k] = pd.concat(data, axis=0, names=["carrier"])
        df = pd.concat(top_level, axis=1, names=["metric"])
        self.segmentation_data_by_timeframe[self.eng.trial] = df
        return df

    def save_simulation_state(self, force: pathlib.Path | None = None) -> None:
        if force:
            # just use the given filename if forced
            prepared_filename = force
        elif self.config.outputs.sim_state.save and (
            not self.config.outputs.sim_state.include_trials
            or self.eng.trial in self.config.outputs.sim_state.include_trials
        ):
            # If we are instructed to save the simulation state to a file,
            # figure out what filename to use.
            fmt = {
                "trial": self.eng.trial,
                "basename": self.config.outputs.filename_stem,
            }
            prepared_filename = self.config.outputs.get_output_filename("sim_state", timestamp=self._timestamp, **fmt)
        else:
            prepared_filename = None
        if prepared_filename:
            from ._saver import serialize_dynamic_state

            serialize_dynamic_state(self.eng, filename=prepared_filename)

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
        if self.eng.trial < 0:
            warnings.warn(
                "Trial must be started before running a sample, implicitly starting Trial 0",
                skip_file_prefixes=_warn_skips,
                stacklevel=1,
            )
            self.begin_trial(0)
        self.begin_sample()
        while True:
            event = self.eng.go()
            self._event_handler(event)
            if event is None or str(event) == "Done" or (event[0] == "Done"):
                assert self.eng.num_events() == 0, f"Event queue still has {self.eng.num_events()} events"
                break
        yield self.eng.sample
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
            n_samples_total = self.eng.num_trials * self.eng.num_samples

        self.begin_trial(trial)
        logger.info("running %d samples in trial %d", self.eng.num_samples, trial)
        for sample in range(self.eng.num_samples):
            sample_start_time = time.time()
            if self.eng.config.simulation_controls.double_capacity_until:
                # Just trying this, PODS has something similar during burn phase
                if sample == 0:
                    for leg in self.eng.legs:
                        leg.capacity = leg.capacity * 2
                elif sample == self.eng.config.simulation_controls.double_capacity_until:
                    for leg in self.eng.legs:
                        leg.capacity = int(leg.capacity / 2)

            self.begin_sample(sample)
            if update_freq is not None and self.eng.sample % update_freq == 0:
                total_rev, n = 0.0, 0
                carrier_info = ""
                for cxr in self.eng.carriers:
                    total_rev += cxr.revenue
                    n += 1
                    carrier_info += f"{', ' if n > 0 else ''}{cxr.name}=${cxr.revenue:8.0f}"
                dmd_b, dmd_l = 0, 0
                for dmd in self.eng.demands:
                    if dmd.business:
                        dmd_b += dmd.scenario_demand
                    else:
                        dmd_l += dmd.scenario_demand
                d_info = f", {int(dmd_b)}, {int(dmd_l)}"
                logger.info(f"Trial={self.eng.trial}, Sample={self.eng.sample}{carrier_info}{d_info}")

            # Loop on passengers
            while True:
                event = self.eng.go()
                memory_log(f"pre-run_carrier_models {event}")
                self._event_handler(event)
                memory_log(f"post-run_carrier_models {event}")
                if event is None or str(event) == "Done" or (event[0] == "Done"):
                    assert self.eng.num_events() == 0, f"Event queue still has {self.eng.num_events()} events"
                    break

            n_samples_done += 1
            self.sample_done_callback(n_samples_done, n_samples_total)
            self.end_sample()
            if progress is not None:
                progress.tick(refresh=(sample == 0))
            t = time.time() - sample_start_time
            logger.info("completed sample %i in %.2f secs", sample, t)

        self.eng.num_trials_completed += 1
        self.end_trial()

    def _run_sim(self, rich_progress: ProgressBar | None = None):
        update_freq = self.update_frequency
        logger.debug(f"run_sim, num_trials = {self.eng.num_trials}, num_samples = {self.eng.num_samples}")
        self.db_writer.update_db_write_flags()
        n_samples_total = self.eng.num_trials * self.eng.num_samples
        n_samples_done = 0
        self.sample_done_callback(n_samples_done, n_samples_total)
        if rich_progress is None:
            if self.eng.config.simulation_controls.show_progress_bar:
                progress = ProgressBar(total=n_samples_total)
            else:
                progress = DummyProgressBar()
        elif isinstance(rich_progress, Progress):
            if self.eng.config.simulation_controls.show_progress_bar:
                # if an external Progress object is provided, generate a
                # ProgressBar object from it
                progress = ProgressBar(total=n_samples_total, external_progress=rich_progress)
            else:
                progress = DummyProgressBar()
        else:
            raise TypeError("rich_progress must be a Progress object")
        with progress:
            for trial in range(self.eng.num_trials):
                self._run_single_trial(
                    trial,
                    n_samples_done,
                    n_samples_total,
                    progress,
                    update_freq,
                )

    def _run_sim_single_trial(self, trial: int, *, rich_progress: Progress | None = None):
        update_freq = self.update_frequency
        self.db_writer.update_db_write_flags()
        self.cnx.close()
        n_samples_total = self.eng.num_samples
        n_samples_done = 0
        self.sample_done_callback(n_samples_done, n_samples_total)
        if rich_progress is None:
            progress = DummyProgressBar()
        elif isinstance(rich_progress, Progress):
            progress = ProgressBar(total=n_samples_total, external_progress=rich_progress)
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

    def _event_handler(self, info: tuple = None):
        """
        Run carrier revenue management models in response to events.

        Parameters
        ----------
        info : tuple, optional
            Event information including event type and associated data. The
            first element is the event type (e.g., "dcp", "daily", "done"),
            followed by event type specific parameters such as recording day,
            DCP index, or callback function and its arguments.

        Notes
        -----
        This method processes various event types including callbacks,
        DCP events, passenger arrivals, and departures. It coordinates
        the execution of revenue management processes for all carriers.
        """
        what_had_happened_was = []
        try:
            event_type = info[0]

            if event_type.startswith("callback_"):
                # This is a callback function, not a string event type.
                # In this situation, the second element of `info` is the function
                # to call, and the remaining elements are arguments to pass to it,
                # after the simulation object itself.
                callback_t = event_type[9:]
                callback_f = info[1]
                result = callback_f(self, *info[2:])
                if isinstance(result, dict):
                    self.callback_data.update_data(callback_t, self.eng.trial, self.eng.sample, *info[2:], **result)
                return

            # For all other event types, the second element is the recording day,
            # and the third element is the DCP index.

            recording_day = info[1]  # could in theory be non-integer for fractional days
            dcp_index = info[2]
            if dcp_index == -1:
                dcp_index = len(self.dcp_list) - 1

            if event_type.lower() in {"dcp", "done"}:
                # For these event types, we update the tracking variables in
                # the simulation engine to reflect the latest DCP processed.
                self.eng.last_dcp = recording_day
                self.eng.last_dcp_index = dcp_index

            # The RM systems for the carriers used to be called here. This is no longer
            # necessary, as the carrier's RM systems are now all callbacks, which are
            # triggered using the regular callback process.

            # Internal simulation data capture that is normally done by RM systems
            if event_type.lower() in {"dcp", "done"}:
                self.eng.last_dcp = recording_day
                self.eng.last_dcp_index = dcp_index
                self.capture_dcp_data(dcp_index)
                what_had_happened_was.append("capture_dcp_close_data")

            # Web shopping
            if event_type.lower() in ["daily", "dcp"]:
                self.capture_competitor_data()  # Simulates 3Victors / Infare / etc.

            # Database capture
            if event_type.lower() == "daily":
                if self.cnx.use_sqlite() and self.eng.save_timeframe_details and recording_day > 0:
                    what_had_happened_was.append("write_to_sqlite daily")
                    _internal_log = self.db_writer.write_to_sqlite(
                        recording_day,
                        store_bid_prices=self.eng.config.db.store_leg_bid_prices,
                        intermediate_day=True,
                        store_displacements=self.eng.config.db.store_displacements,
                    )
            elif event_type.lower() in {"dcp", "done"}:
                if self.cnx.use_sqlite() and self.eng.save_timeframe_details:
                    what_had_happened_was.append("write_to_sqlite dcp")
                    _internal_log = self.db_writer.write_to_sqlite(
                        recording_day,
                        store_bid_prices=self.eng.config.db.store_leg_bid_prices,
                        intermediate_day=False,
                        store_displacements=self.eng.config.db.store_displacements,
                    )
                if event_type.lower() == "done" and "forecast_accuracy" in self.config.outputs.reports:
                    self.eng.capture_forecast_accuracy()
                if self.cnx.is_open:
                    self.cnx.save_details(self.db_writer, self.eng, recording_day)
                if self.file_writer is not None:
                    self.file_writer.save_details(self.eng, recording_day)

            # simulation statistics record
            if event_type.lower() in {"dcp", "done"}:
                self.eng.record_dcp_statistics(recording_day)
            self.eng.record_daily_statistics(recording_day)

        except Exception:
            # print(e)
            # print("Error in run_carrier_models")
            # print(f"{info=}")
            # print("what_had_happened_was=", what_had_happened_was)
            raise

    def capture_competitor_data(self):
        """
        Capture competitor pricing data for all markets.

        Notes
        -----
        This method shops for the lowest prices in each market and
        stores competitor pricing information that can be used by
        revenue management systems for competitive analysis.
        """
        if not self.config.simulation_controls.capture_competitor_data:
            # when this setting is False, this method becomes a no-op
            return
        for mkt in self.eng.markets.values():
            lowest = self.eng.shop(mkt.orig, mkt.dest)
            for cxr, price in lowest:
                mkt.set_competitor_price(cxr, price)

    def capture_dcp_data(self, dcp_index, closures_only=False):
        """
        Capture data control point (DCP) data for revenue management.

        Parameters
        ----------
        dcp_index : int
            The index of the data control point.
        closures_only : bool, default False
            Whether to capture only closure data or all DCP data.

        Notes
        -----
        This method captures seat availability, booking data, and other
        metrics at specific time points (DCPs) before departure, which
        is essential for revenue management decision-making.
        """
        for leg in self.eng.legs:
            leg.capture_dcp(dcp_index)
        for path in self.eng.paths:
            path.capture_dcp(dcp_index, closures_only=closures_only)
        for carrier in self.eng.carriers:
            if dcp_index > 0:
                carrier.current_tf_index += 1

    def _accum_by_tf(self, dcp_index):
        # This is now replaced by C++ native counters ...
        if dcp_index > 0:
            prev_dcp = self.dcp_list[dcp_index - 1]
            for f in self.eng.fares:
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

        In older versions of PassengerSim, the DCP events activated a variety
        of processes, including carrier RM system updates, internal simulation
        data capture, and database writes.  In the current version, carrier RM
        systems are triggered via callback events, but data capture and database
        writes are still tied to DCP events.
        """
        dcp_hour = self.eng.config.simulation_controls.dcp_hour
        if debug:
            tmp = datetime.fromtimestamp(self.eng.base_time, tz=UTC)
            print(f"Base Time is {tmp.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        for dcp_index, dcp in enumerate(self.dcp_list):
            if dcp == 0:
                continue
            event_time = int(self.eng.base_time - dcp * 86400 + 3600 * dcp_hour)
            if debug:
                tmp = datetime.fromtimestamp(event_time, tz=UTC)
                print(f"Added DCP {dcp} at {tmp.strftime('%Y-%m-%d %H:%M:%S %Z')}")
            info = ("DCP", dcp, dcp_index)
            rm_event = Event(info, event_time)
            self.eng.add_event(rm_event)

        # Now add the events for daily reoptimization
        max_days_prior = max(self.dcp_list)
        dcp_idx = 0
        for days_prior in reversed(range(max_days_prior)):
            if days_prior not in self.dcp_list:
                info = ("daily", days_prior, dcp_idx)
                event_time = int(self.eng.base_time - days_prior * 86400 + 3600 * dcp_hour)
                rm_event = Event(info, event_time)
                self.eng.add_event(rm_event)
            else:
                dcp_idx += 1

        # add events for begin and end sample callbacks
        self.add_callback_events()

    def generate_demands(self):
        """
        Generate demands following the procedure used in PODS.
        """
        self.generate_dcp_rm_events()
        generate_sample_demands(self.eng, self.eng.config.simulation_controls, allocate=False)
        allocate_sample_demands(self.eng, self.eng.config.simulation_controls)

    def generate_demands_gamma(self, system_rn=None, debug=False):
        """Using this as a quick test"""
        self.generate_dcp_rm_events()
        end_time = self.base_time
        cv100 = 0.3
        for dmd in self.eng.demands:
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
            self.eng.allocate_demand_to_tf_pods(dmd, num_pax, self.eng.tf_k_factor, int(end_time))
        total_events = 0
        return total_events

    def reseed(self, seed: int | list[int] | None = 42):
        """
        Reseed the simulation's random number generator.

        Parameters
        ----------
        seed : int, list[int], or None, default 42
            Seed value(s) for the random number generator. Can be a single
            integer, a list of integers, or None.

        Notes
        -----
        This method updates the random seed for the simulation's internal
        random number generator, affecting all subsequent random operations.
        """
        logger.debug("reseeding random_generator: %s", seed)
        try:
            self.eng.reseed(seed)
        except Exception as e:
            logger.error("Failed to reseed random_generator: %s", e)
            raise RuntimeError(f"Failed to reseed random_generator with seed {seed}") from e

    def _user_certificate(self, certificate_filename=None):
        if certificate_filename:
            from cryptography.x509 import load_pem_x509_certificate

            certificate_filename = pathlib.Path(certificate_filename)
            with certificate_filename.open("rb") as f:
                user_cert = load_pem_x509_certificate(f.read())
        else:
            user_cert = self.eng.config.license_certificate
        return user_cert

    def validate_license(self, certificate_filename=None, future: int = 0):
        user_cert = self._user_certificate(certificate_filename)
        return self.eng.validate_license(user_cert, future=future)

    def license_info(self, certificate_filename=None):
        user_cert = self._user_certificate(certificate_filename)
        return self.eng.license_info(user_cert)

    @property
    def config(self) -> Config:
        """The configuration used for this Simulation."""
        return self.eng.config

    def run(
        self,
        log_reports: bool = False,
        *,
        single_trial: int | None = None,
        summarizer: type[SimulationTablesT] | SimulationTablesT | None = None,
        rich_progress: Progress | None = None,
        cache_dir: pathlib.Path | None = None,
    ) -> SimulationTablesT:
        """
        Run the simulation and compute reports.

        Parameters
        ----------
        log_reports : bool
        single_trial : int, optional
            Run only a single trial, with the given trial number (to get
            the correct fixed random seed, for example).
        summarizer : type[SimulationTables] | SimulationTables, optional
            Use this summarizer to compute the reports. A valid summarizer
            must be a subclass or instance of GenericSimulationTables. If
            not provided, the default summarizer will be used.
        rich_progress : Progress, optional
            A rich Progress object to use for displaying progress.  If not
            provided, a new Progress object will be created unless the
            simulation configuration specifies not to show progress.

        Returns
        -------
        SimulationTables
        """
        summarizer = check_summarizer(summarizer)

        if cache_dir is not None:
            from ._cache_run import cache_run

            return cache_run(self, cache_dir=cache_dir, summarizer=summarizer, rich_progress=rich_progress)

        start_time = time.time()
        self.setup_scenario()
        if single_trial is not None:
            self._run_sim_single_trial(single_trial, rich_progress=rich_progress)
        else:
            self._run_sim(rich_progress=rich_progress)
        if self.choice_set_file is not None:
            self.choice_set_file.close()
        logger.info("Extracting summary results")
        if isinstance(summarizer, GenericSimulationTables):
            summary = summarizer._extract(self)
        elif issubclass(summarizer, GenericSimulationTables):
            summary = summarizer.extract(self)
        else:
            raise TypeError("summarizer must be an instance or subclass of GenericSimulationTables")

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
            if self.config.outputs.html:
                write_html_filename = self.config.outputs.get_output_filename(
                    "html",
                    timestamp=self._timestamp,
                )
                out_filename = summary.to_html(
                    write_html_filename,
                    add_timestamp=False,
                )
                summary._metadata["outputs.html_filename"] = out_filename
            if disk_output_file := self.config.outputs.get_output_filename(
                "disk",
                timestamp=self._timestamp,
            ):
                out_filename = summary.to_file(
                    disk_output_file,
                    add_timestamp_ext=False,
                )
                summary._metadata["outputs.disk_filename"] = out_filename
            if pickle_output_file := self.config.outputs.get_output_filename(
                "pickle",
                timestamp=self._timestamp,
            ):
                pkl_filename = summary.to_pickle(pickle_output_file)
                summary._metadata["outputs.pickle_filename"] = pkl_filename
            if excel_output_file := self.config.outputs.get_output_filename("excel", timestamp=self._timestamp):
                summary.to_xlsx(excel_output_file)

        logger.info(f"Th' th' that's all folks !!!    (Elapsed time = {round(time.time() - start_time, 2)})")
        return summary

    def run_trial(
        self,
        trial: int,
        summarizer: type[SimulationTablesT] | SimulationTablesT | None = None,
    ) -> SimulationTablesT:
        self.setup_scenario()
        self.eng.trial = trial

        summarizer = check_summarizer(summarizer)

        if not isinstance(summarizer, GenericSimulationTables) and not issubclass(summarizer, GenericSimulationTables):
            raise TypeError("summarizer must be an instance or subclass of GenericSimulationTables")

        update_freq = self.update_frequency
        logger.debug(f"run_sim, num_trials = {self.eng.num_trials}, num_samples = {self.eng.num_samples}")
        self.db_writer.update_db_write_flags()
        n_samples_total = self.eng.num_samples
        n_samples_done = 0
        self.sample_done_callback(n_samples_done, n_samples_total)
        if self.eng.config.simulation_controls.show_progress_bar:
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
        if isinstance(summarizer, GenericSimulationTables):
            summary = summarizer._extract(self)
        elif issubclass(summarizer, GenericSimulationTables):
            summary = summarizer.extract(self)
        else:
            raise TypeError("summarizer must be an instance or subclass of GenericSimulationTables")
        return summary

    def backup_db(self, dst: pathlib.Path | str | sqlite3.Connection):
        """Back up this database to another copy.

        Parameters
        ----------
        dst : Path-like or sqlite3.Connection
        """
        return self.cnx.backup(dst)

    def get_choice_parameters(self, choicemodel: str | ChoiceModel):
        """
        Get the parameters for a choice model.

        Parameters
        ----------
        choicemodel : str or ChoiceModel
            The choice model name (string) or ChoiceModel object to get
            parameters from.

        Returns
        -------
        dict
            Dictionary containing the choice model parameters, including
            restrictions and their associated sigma values.
        """
        if isinstance(choicemodel, str):
            choicemodel = self.choice_models[choicemodel]
        raw = choicemodel.get_parameters()
        r = raw.pop("restrictions", ())
        rsigma = raw.pop("restriction_sigmas", ())
        for rname, rval, rsig in zip(self._fare_restriction_mapping.list_all(), r, rsigma):
            raw[f"restrictions_{rname}"] = rval
            raw[f"restrictions_{rname}_sigma"] = rsig
        return raw

    def set_choice_parameters(self, choicemodel: str | ChoiceModel, values: dict[str, float]):
        """
        Set the parameters for a choice model.

        Parameters
        ----------
        choicemodel : str or ChoiceModel
            The choice model name (string) or ChoiceModel object to update.
        values : dict[str, float]
            Dictionary of parameter names and their new values. Can include
            restriction parameters using the format 'restrictions_{name}'.
        """
        if isinstance(choicemodel, str):
            choicemodel = self.choice_models[choicemodel]
        raw = choicemodel.get_parameters()
        for k, v in values.items():
            if k.startswith("restrictions_"):
                if k.endswith("_sigma"):
                    kr = k[13:-6]
                else:
                    kr = k[13:]
                position = self._fare_restriction_mapping.get_number(kr) - 1
                if k.endswith("_sigma"):
                    raw["restriction_sigmas"][position] = v
                else:
                    raw["restrictions"][position] = v
            else:
                raw[k] = v
        choicemodel.set_parameters(raw)

    def rm_data(self, carrier: str, kind: str, set_value: Any = None) -> Any:
        """Access RM data (forecasts, optimizers, etc.) from the simulation.

        If the requested data for the carrier and kind do not exist, an
        empty dictionary is returned.

        Parameters
        ----------
        carrier, kind : str
            The carrier name and data type.
        set_value : Any, optional
            If provided, this value will be set for the carrier and kind
            instead of retrieving the existing value.

        Returns
        -------
        Any
            The existing value for this carrier and data type, or an empty dict.
        """
        if set_value is not None:
            self._rm_data[(carrier, kind)] = set_value
            return None
        return self._rm_data.setdefault((carrier, kind), {})
