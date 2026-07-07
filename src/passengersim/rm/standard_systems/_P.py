from passengersim.rm.probp import ProbabilisticBidPrice
from passengersim.rm.standard_forecasting import PathForecastDailyDecay, StandardPathForecast
from passengersim.rm.systems import RmSys, RmSysOption, register_rm_system
from passengersim.rm.untruncation import PathDetruncation
from passengersim.snapshot.filtering import NetworkSnapshotFilter


@register_rm_system
class P(RmSys):
    """RM system using standard path-level forecasting and ProBP.

    This RM system uses path-level bid price controls with probabilistic bid
    price (ProBP) optimization, along with EM untruncation and standard leg
    forecasting.

    Parameters
    ----------
    forecast_algorithm : {'additive_pickup', 'exp_smoothing', 'multiplicative_pickup'}, default 'additive_pickup'
        Specifies which leg-level forecasting algorithm to use for generating
        leg demand forecasts.  Options are 'additive_pickup', 'exp_smoothing', or
        'multiplicative_pickup'.  The default is 'additive_pickup'.
    bid_price_vector : bool, default True
        If True, enables bid price vector optimization in ProBP.  If False,
        uses scalar bid prices.
    sub_bp : bool, default False
        If True, use SubBP instead of ProBP.  Under ProBP the revenue from
        a connecting fare is prorated among the legs to determine the displacement
        cost, while for SubBP the displacement costs for each leg are found
        by using the bid prices from the other legs directly without proration.

    Notes
    -----
    This RM system consists of the following actions executed in order:

    1. **EM Untruncation of Path Demands**
        This step applies the EM algorithm to detruncate observed path
        sales into inferred true demand levels.  It runs only once at the
        beginning of each sample day, and detruncates demand for all timeframes.

    2. **Standard Path Forecasting**
        This step generates path-level demand forecasts using a standard
        class-based forecasting algorithm (additive pickup by default). It
        runs full computations to produce path forecasts for the entire
        booking horizon in one pass at the beginning of each sample day,
        and on later DCPs it simply moves a pointer forward through that
        array of forecasts to provide the correct forecast values at that
        time.

    3. **Path Forecast Daily Decay Adjustment**
        This step applies a daily decay adjustment to the path-level forecasts,
        to account for the changes in expected demand to come in between DCPs.
        It runs every day that isn't a DCP, to adjust the path forecasts
        accordingly.

    4. **ProBP Optimization**
        Optimizes path-level bid prices using the Probabilistic Bid Price
        (ProBP) algorithm. This step runs every day, to update the bid price
        controls based on the current path forecasts, current sales, and the
        ProBP optimization logic.
    """

    availability_control = "bp"
    """This RM system uses bid price availability controls."""

    actions = [
        PathDetruncation,
        StandardPathForecast.configure(
            algorithm=RmSysOption("forecast_algorithm", default="additive_pickup"),
        ),
        PathForecastDailyDecay,
        ProbabilisticBidPrice.configure(
            bid_price_vector=RmSysOption("bid_price_vector", expected_type=bool, default=True),
            capacity_sharing=RmSysOption("capacity_sharing", expected_type=bool, default=False),
            capacity_sharing_start_dcp_index=RmSysOption(
                "capacity_sharing_start_dcp_index", expected_type=int, default=0
            ),
            capacity_sharing_start_lf=RmSysOption("capacity_sharing_start_lf", expected_type=float, default=0.0),
            cabins=RmSysOption("cabins", default=None),
            use_sub_bp=RmSysOption("sub_bp", expected_type=bool, default=False),
            snapshot_filters=RmSysOption(
                "bp_snapshot_filters", expected_type=list[NetworkSnapshotFilter], default=None
            ),
        ),
    ]
