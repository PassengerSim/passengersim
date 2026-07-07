from pytest import fixture, raises

import passengersim as pax
from passengersim.experiments import Experiments
from passengersim.rm.emsr import ExpectedMarginalSeatRevenue
from passengersim.rm.standard_forecasting import StandardLegForecast
from passengersim.rm.systems import RmSys, RmSysOption, check_registered_rm_system, register_rm_system
from passengersim.rm.untruncation import LegDetruncation


@fixture
def tiny_cfg():
    cfg = pax.Config.from_yaml(pax.demo_network("3MKT"))
    cfg.db = None
    cfg.outputs.reports.clear()
    cfg.outputs._write_no_files()
    cfg.simulation_controls.burn_samples = 20
    cfg.simulation_controls.num_samples = 30
    cfg.simulation_controls.num_trials = 1
    return cfg


def test_rm_system_E(tiny_cfg, dataframe_regression):
    cfg = tiny_cfg
    # we will test that both the old-style and callback-style RM systems work
    # and give the same results
    experiments = Experiments(cfg, output_dir=False)

    @experiments
    def old_rm(cfg: pax.Config) -> pax.Config:
        cfg.carriers["AL1"].rm_system_options = None
        cfg.carriers["AL2"].rm_system_options = None
        return cfg

    @experiments
    def new_rm(cfg: pax.Config) -> pax.Config:
        cfg.carriers["AL1"].rm_system_options = {"name": "E"}
        cfg.carriers["AL2"].rm_system_options = {"name": "E"}
        return cfg

    ex_out = experiments.run(use_existing=False, write_report=False)
    for ex_summary in ex_out.values():
        assert ex_summary is not None
        dataframe_regression.check(ex_summary.carrier_history2, basename="rm_system_E_carrier_history")


def test_rm_system_E_bad_input(tiny_cfg):
    cfg = tiny_cfg
    assert check_registered_rm_system("E")
    assert cfg.carriers["AL1"].rm_system == "E"
    cfg.carriers["AL1"].rm_system_options = {"emsr_variant": "BAD_VARIANT"}
    with raises(ValueError, match="Unknown EMSR variant 'BAD_VARIANT'"):
        _ = pax.Simulation(cfg)


def test_rm_system_E_bad_arg(tiny_cfg):
    cfg = tiny_cfg
    assert check_registered_rm_system("E")
    assert cfg.carriers["AL1"].rm_system == "E"
    cfg.carriers["AL1"].rm_system_options = {"BAD_OPTION": 42}
    with raises(ValueError, match="Keyword argument 'BAD_OPTION' does not match any configuration parameter"):
        _ = pax.Simulation(cfg)


def test_rm_system_name_mismatch(tiny_cfg):
    cfg = tiny_cfg
    cfg.carriers["AL1"].rm_system = "P"
    cfg.carriers["AL1"].rm_system_options = {"name": "E"}
    with raises(ValueError, match=r"`rm_system` must match `rm_system_options\['name'\]`"):
        _ = pax.Simulation(cfg)


def test_rm_system_broken_default(tiny_cfg):
    @register_rm_system
    class Bad_E(RmSys):
        _name = "BAD_E_1"
        availability_control = "leg"
        actions = [
            LegDetruncation,
            StandardLegForecast,
            ExpectedMarginalSeatRevenue.configure(
                variant=RmSysOption("emsr_variant", default="BAD_VARIANT"),
            ),
        ]

    assert check_registered_rm_system("BAD_E_1")

    # check with both names
    cfg = tiny_cfg.model_copy(deep=True)
    cfg.carriers["AL1"].rm_system_options = {"name": "BAD_E_1"}
    cfg.carriers["AL1"].rm_system = "BAD_E_1"
    with raises(ValueError, match="Unknown EMSR variant 'BAD_VARIANT'"):
        _ = pax.Simulation(cfg)

    # check with only the config name and no rm_system_options
    cfg = tiny_cfg.model_copy(deep=True)
    cfg.carriers["AL1"].rm_system = "BAD_E_1"
    assert cfg.carriers["AL1"].rm_system_options is None
    with raises(ValueError, match="Unknown EMSR variant 'BAD_VARIANT'"):
        _ = pax.Simulation(cfg)

    # check with only the config name and empty rm_system_options
    cfg = tiny_cfg.model_copy(deep=True)
    cfg.carriers["AL1"].rm_system = "BAD_E_1"
    cfg.carriers["AL1"].rm_system_options = {}
    with raises(ValueError, match="Unknown EMSR variant 'BAD_VARIANT'"):
        _ = pax.Simulation(cfg)
