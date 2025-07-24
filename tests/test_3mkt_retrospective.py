import altair
import pytest

from passengersim import Simulation, demo_network
from passengersim.config import Config
from passengersim.contrast import Contrast
from passengersim.summary import SummaryTables

DEFAULT_TOLERANCE = dict(rtol=2e-02, atol=1e-06)


@pytest.fixture(scope="module")
def stored_results(tmp_path_factory) -> Contrast:
    retrospect = tmp_path_factory.mktemp("retrospect")

    cfg = Config.from_yaml(demo_network("3MKT/08-untrunc-em"))
    cfg.simulation_controls.num_trials = 1
    cfg.simulation_controls.num_samples = 400
    cfg.simulation_controls.show_progress_bar = False
    cfg.db.filename = retrospect.joinpath("untruncated.sqlite")
    summary1 = Simulation(cfg).run(summarizer=None)
    summary1.cnx.close()

    cfg = Config.from_yaml(demo_network("3MKT/05-emsrb"))
    cfg.simulation_controls.num_trials = 1
    cfg.simulation_controls.num_samples = 400
    cfg.simulation_controls.show_progress_bar = False
    cfg.db.filename = retrospect.joinpath("simple.sqlite")
    summary2 = Simulation(cfg).run(summarizer=None)
    summary2.cnx.close()

    simple = SummaryTables.from_sqlite(retrospect.joinpath("simple.sqlite"), additional="*")
    untrunc = SummaryTables.from_sqlite(retrospect.joinpath("untruncated.sqlite"), additional="*")

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


def test_arbitrary_sql(stored_results, dataframe_regression):
    df = stored_results["Simple"].cnx.dataframe(
        """
    SELECT
      sample, auth, sold, forecast_mean, forecast_stdev
    FROM
      leg_bucket_detail
    WHERE
      leg_id = 101
      AND days_prior = 21
      AND name = 'Y2'
      AND sample >= 100
    LIMIT 10
    """
    )
    dataframe_regression.check(df)
