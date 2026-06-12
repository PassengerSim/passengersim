import sqlite3

import altair
import pandas as pd
import pytest

from passengersim import Simulation, demo_network, from_file
from passengersim.config import Config
from passengersim.contrast import Contrast
from passengersim.rm.specialty_systems.e_no_detruncation import E_NoDetruncation  # noqa: F401

DEFAULT_TOLERANCE = dict(rtol=2e-02, atol=1e-06)


@pytest.fixture(scope="module")
def retrospect_dir(tmp_path_factory):
    retrospect = tmp_path_factory.mktemp("retrospect")
    return retrospect


@pytest.fixture(scope="module")
def stored_results(retrospect_dir) -> Contrast:
    retrospect = retrospect_dir

    cfg = Config.from_yaml(demo_network("3MKT/08-untrunc-em"))
    cfg.simulation_controls.num_trials = 1
    cfg.simulation_controls.num_samples = 100
    cfg.simulation_controls.burn_samples = 50
    cfg.simulation_controls.show_progress_bar = False
    cfg.outputs.base_dir = retrospect_dir / "outputs1"
    summary1 = Simulation(cfg).run()
    summary1.to_file(retrospect.joinpath("detruncated.pxsim"))

    cfg = Config.from_yaml(demo_network("3MKT/05-emsrb"))
    cfg.carriers.AL1.rm_system = "E_NoDetruncation"
    cfg.carriers.AL2.rm_system = "E_NoDetruncation"
    cfg.simulation_controls.num_trials = 1
    cfg.simulation_controls.num_samples = 100
    cfg.simulation_controls.burn_samples = 50
    cfg.simulation_controls.show_progress_bar = False
    cfg.db.filename = retrospect.joinpath("simple.sqlite")
    cfg.outputs.base_dir = retrospect_dir / "outputs2"
    summary2 = Simulation(cfg).run()
    summary2.cnx.close()
    summary2.to_file(retrospect.joinpath("simple.pxsim"))

    simple = from_file(retrospect.joinpath("simple.pxsim"))
    untrunc = from_file(retrospect.joinpath("detruncated.pxsim"))

    comps = Contrast(Simple=simple, Untruncated=untrunc)
    return comps


def test_fig_carrier_revenues(stored_results, dataframe_regression):
    assert isinstance(stored_results, Contrast)
    fig = stored_results.fig_carrier_revenues()
    assert isinstance(fig, altair.TopLevelMixin)
    df = stored_results.fig_carrier_revenues(raw_df=True).reset_index(drop=True)
    dataframe_regression.check(df)


def test_fig_carrier_load_factors(stored_results, dataframe_regression):
    assert isinstance(stored_results, Contrast)
    fig = stored_results.fig_carrier_load_factors()
    assert isinstance(fig, altair.TopLevelMixin)
    df = stored_results.fig_carrier_load_factors(raw_df=True).reset_index(drop=True)
    dataframe_regression.check(df)


def test_fig_fare_class_mix(stored_results, dataframe_regression):
    assert isinstance(stored_results, Contrast)
    fig = stored_results.fig_fare_class_mix()
    assert isinstance(fig, altair.TopLevelMixin)
    df = stored_results.fig_fare_class_mix(raw_df=True).reset_index(drop=True)
    dataframe_regression.check(df)


def test_fig_bookings_by_timeframe(stored_results, dataframe_regression):
    assert isinstance(stored_results, Contrast)
    fig = stored_results.fig_bookings_by_timeframe(by_carrier="AL1", by_class=True, source_labels=True)
    assert isinstance(fig, altair.TopLevelMixin)
    df = stored_results.fig_bookings_by_timeframe(
        by_carrier="AL1", by_class=True, source_labels=True, raw_df=True
    ).reset_index(drop=True)
    dataframe_regression.check(df)


def test_arbitrary_sql(stored_results, dataframe_regression, retrospect_dir):
    cnx = sqlite3.connect(retrospect_dir.joinpath("simple.sqlite"))

    query = """
    SELECT
      sample, auth, sold, forecast_mean, forecast_stdev
    FROM
      leg_bucket_detail
    WHERE
      leg_id = 101
      AND days_prior = 21
      AND name = 'Y2'
      AND sample >= 90
    LIMIT 10
    """
    df = pd.read_sql_query(query, cnx)
    print(df)
    dataframe_regression.check(df)
