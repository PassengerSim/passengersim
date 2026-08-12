from types import SimpleNamespace

import pytest

from passengersim import Config, Simulation, SimulationTables, demo_network
from passengersim.rm.specialty_systems.fcfs import FirstComeFirstServed, NoProtection
from passengersim.rm.standard_forecasting import StandardLegForecast
from passengersim.rm.systems import get_registered_rm_system
from passengersim.rm.untruncation import LegDetruncation


def test_first_come_first_served_configuration():
    cfg = SimpleNamespace(dcps=[90, 60, 30, 0])

    rm_system = FirstComeFirstServed(carrier="AL1", cfg=cfg)

    assert get_registered_rm_system("FCFS") is FirstComeFirstServed
    assert rm_system.get_name() == "FCFS"
    assert rm_system.availability_control == "leg"
    assert [type(action) for action in rm_system.action_queue] == [
        LegDetruncation,
        StandardLegForecast,
        NoProtection,
    ]
    assert all(action.carrier == "AL1" for action in rm_system.action_queue)
    assert rm_system.action_queue[1].algorithm == "additive_pickup"
    assert rm_system.action_queue[2].run(sim=None, days_prior=90) is None


def test_first_come_first_served_3mkt_integration():
    cfg = Config.from_yaml(demo_network("3MKT/01-base"))
    cfg.simulation_controls.num_trials = 1
    cfg.simulation_controls.num_samples = 40
    cfg.simulation_controls.burn_samples = 30
    cfg.simulation_controls.show_progress_bar = False
    cfg.simulation_controls.allow_unused_restrictions = True
    cfg.outputs.reports.clear()
    cfg.outputs._write_no_files()
    cfg.db = None

    sim = Simulation(cfg)
    summary = sim.run()

    assert isinstance(summary, SimulationTables)
    assert sim.eng.num_trials_completed == 1
    assert sim.eng.num_samples == 40
    assert all(isinstance(carrier.rm_sys, FirstComeFirstServed) for carrier in sim.carriers_dict.values())
    assert summary.carriers["rm_system"].to_dict() == {"AL1": "FCFS", "AL2": "FCFS"}
    assert summary.carriers["avg_sold"].to_dict() == pytest.approx({"AL1": 295.5, "AL2": 297.1}, rel=0.02)
    assert summary.carriers["avg_rev"].to_dict() == pytest.approx({"AL1": 46_370.0, "AL2": 46_410.0}, rel=0.02)

    assert len(summary.legs) == 8
    assert (summary.legs["gt_sold"] > 0).all()
    assert (summary.legs["gt_sold"] <= summary.legs["gt_capacity"]).all()

    # With no protection levels, this model's accepted bookings remain in its lowest fare class.
    fare_class_sales = summary.fare_class_mix["sold"]
    other_classes = fare_class_sales.index.get_level_values("booking_class") != "Y5"
    assert fare_class_sales.loc[other_classes].eq(0).all()
    assert fare_class_sales.xs("Y5", level="booking_class").to_dict() == pytest.approx(
        {"AL1": 2_955, "AL2": 2_971}, rel=0.02
    )
