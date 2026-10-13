"""Native daily collection against independent observations and callback timing."""

from collections import defaultdict

import numpy as np
import pandas as pd
import pytest

import passengersim as pax
from passengersim.tracers import LegBidPriceTracer


def test_default_dcp_schedule_populates_native_history() -> None:
    """Omitting DCPs must retain daily collection over the driver's default horizon."""
    from passengersim.rm import register_rm_system
    from passengersim.rm.specialty_systems.fcfs import FirstComeFirstServed

    @register_rm_system
    class DefaultDcpTestRM(FirstComeFirstServed):
        """Use no protection or forecasting so no explicit DCPs are required."""

        _name = "DefaultDcpTestRM"

        def __init__(self, carrier: str, cfg: pax.Config | None = None) -> None:
            # This fixture has no demand or fare classes and needs no RM actions.
            # Forecasting actions independently require an explicit DCP schedule.
            self.action_queue = []

    config = pax.Config.model_validate(
        {
            "carriers": [{"name": "XX", "control": "none", "rm_system": DefaultDcpTestRM}],
            "legs": [
                {
                    "carrier": "XX",
                    "fltno": 101,
                    "orig": "A",
                    "dest": "B",
                    "dep_time": "08:00",
                    "arr_time": "10:00",
                    "capacity": 10,
                }
            ],
            "places": [
                {"name": "A", "label": "Airport A", "lat": 33.64, "lon": -84.43},
                {"name": "B", "label": "Airport B", "lat": 42.3656, "lon": -71.0098},
            ],
        }
    )
    assert config.dcps == []
    config.simulation_controls.num_trials = 1
    config.simulation_controls.num_samples = 3
    config.simulation_controls.burn_samples = 1
    config.simulation_controls.show_progress_bar = False
    config.outputs._write_no_files()
    config.outputs.reports.clear()
    config.db = None
    with pax.Simulation(config) as sim:
        assert sim.dcp_list[0] == 63
        assert sim.dcp_list[-1] == 0
        assert [sim.eng.get_days_prior(i) for i in range(sim.eng.num_dcps)] == sim.dcp_list
        # Exercise native recording without unrelated fare/market setup: this
        # programmatic fixture intentionally contains no demand or fare classes.
        sim.eng.set_parm("collect_leg_bid_prices", config.simulation_controls.collect_leg_bid_prices)
        for sample in (1, 2):
            sim.eng.sample = sample
            for day in range(64):
                sim.eng.record_daily_statistics(day)
        rows = sim.eng._raw_leg_bid_price_statistics()
        assert len(rows) == len(sim.legs) * 64
        assert {day for _, day, *_ in rows} == set(range(64))
        assert all(count == 2 and np.isfinite(mean) for _, _, count, mean, _ in rows)


@pytest.mark.parametrize("enabled", [False, True])
def test_native_collection_without_database(enabled: bool) -> None:
    config = pax.Config.from_yaml(pax.demo_network("3MKT/DEMO"))
    config.simulation_controls.num_trials = 2
    config.simulation_controls.num_samples = 4
    config.simulation_controls.burn_samples = 1
    config.simulation_controls.show_progress_bar = False
    config.simulation_controls.collect_leg_bid_prices = enabled
    config.outputs._write_no_files()
    config.outputs.reports.clear()
    config.db = None
    observations: dict[tuple[int, int], list[float]] = defaultdict(list)

    def observe(sim: pax.Simulation, days_prior: int) -> None:
        # This network has no intervening price changes between core statistics
        # and the default callback, so the observations provide an independent oracle.
        if sim.eng.sample >= sim.eng.burn_samples:
            for leg_id, leg in sim.legs.items():
                observations[leg_id, days_prior].append(leg.report_bid_price)

    with pax.Simulation(config) as sim:
        subset_id = next(iter(sim.legs))
        LegBidPriceTracer([subset_id]).attach(sim)
        sim.daily_callback(observe)
        summary = sim.run()
        assert summary.db is None
        assert list(summary.callback_data.leg_bid_prices.columns) == [subset_id]
        assert sim.eng._raw_leg_bid_price_statistics() == []
        if not enabled:
            assert sim._leg_bid_price_moments is None
            return
        rows = sim._leg_bid_price_moments.reset_index().itertuples(index=False, name=None)
        rows = list(rows)
        assert {(leg_id, day) for leg_id, day, *_ in rows} == {
            (leg_id, day) for leg_id in sim.legs for day in range(max(config.dcps) + 1)
        }
        for leg_id, day, count, mean, m2 in rows:
            values = np.asarray(observations[leg_id, day])
            assert count == len(values) == 6
            assert mean == pytest.approx(values.mean())
            assert m2 == pytest.approx(((values - values.mean()) ** 2).sum(), abs=1e-8)
            row = summary.leg_bid_price_history.loc[(leg_id, day)]
            assert row.bid_price_mean == pytest.approx(values.mean())
            assert row.bid_price_std_dev == pytest.approx(values.std(ddof=1), abs=1e-8)


def test_native_timing_before_later_callbacks() -> None:
    """Daily/DCP capture precedes late callbacks; departure follows all events."""
    config = pax.Config.from_yaml(pax.demo_network("3MKT/DEMO"))
    config.simulation_controls.num_trials = 1
    config.simulation_controls.num_samples = 3
    config.simulation_controls.burn_samples = 1
    config.simulation_controls.collect_leg_bid_prices = True
    config.simulation_controls.show_progress_bar = False
    config.outputs._write_no_files()
    config.outputs.reports.clear()
    config.db = None
    observed: dict[tuple[int, int], list[float]] = defaultdict(list)
    departure: dict[int, list[float]] = defaultdict(list)

    def change_prices(sim: pax.Simulation, days_prior: int) -> None:
        # A late callback sees the reporting price that was just recorded, then
        # makes a visible change before the independently scheduled tracer.
        for leg_id, leg in sim.legs.items():
            if sim.eng.sample >= sim.eng.burn_samples:
                observed[leg_id, days_prior].append(leg.report_bid_price)
            leg.bid_price = 12345.0 + days_prior
            if days_prior == 0 and sim.eng.sample >= sim.eng.burn_samples:
                departure[leg_id].append(leg.report_bid_price)

    change_prices.priority = 2
    with pax.Simulation(config) as sim:
        sim.daily_callback(change_prices)
        LegBidPriceTracer(list(sim.legs), priority=3).attach(sim)
        summary = sim.run()
    table = summary.leg_bid_price_history
    assert set(table.index.get_level_values("days_prior")) == set(range(max(config.dcps) + 1))
    assert any(day not in config.dcps for _, day in observed)
    for key, values in observed.items():
        assert table.loc[key, "n_observations"] == 2
        expected = departure[key[0]] if key[1] == 0 else values
        assert table.loc[key, "bid_price_mean"] == pytest.approx(np.mean(expected))
    traced = summary.callback_data.leg_bid_prices.xs("mean", level="statistic")
    # The first DCP has unsold legs, so the explicit price change is observable.
    first_day = max(config.dcps)
    assert (traced.loc[first_day] == 12345.0 + first_day).all()
    native_first = table.xs(first_day, level="days_prior").bid_price_mean
    assert (native_first != traced.loc[first_day].reindex(native_first.index)).all()


def test_collection_does_not_change_simulation_results() -> None:
    """Enabling collection must leave sales, revenues, and carrier traces intact."""
    config = pax.Config.from_yaml(pax.demo_network("3MKT/DEMO"))
    config.simulation_controls.num_trials = 1
    config.simulation_controls.num_samples = 4
    config.simulation_controls.burn_samples = 1
    config.simulation_controls.show_progress_bar = False
    config.outputs._write_no_files()
    config.outputs.reports.clear()
    config.db = None
    summaries = []
    for enabled in (False, True):
        config.simulation_controls.collect_leg_bid_prices = enabled
        with pax.Simulation(config) as sim:
            summaries.append(sim.run())
    for name in ("legs", "carriers", "bid_price_history"):
        pd.testing.assert_frame_equal(getattr(summaries[0], name), getattr(summaries[1], name))
