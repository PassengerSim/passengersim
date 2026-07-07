from passengersim.rm.emsr import ExpectedMarginalSeatRevenue
from passengersim.rm.standard_forecasting import StandardLegForecast
from passengersim.rm.systems import RmSys, RmSysOption, register_rm_system
from passengersim.rm.untruncation import LegDetruncation


@register_rm_system
class E(RmSys):
    """RM system using standard forecasting and EMSR-B.

    This RM system uses leg-level seat allocation controls with EMSR-B
    optimization, along with EM untruncation and standard leg forecasting.

    Parameters
    ----------
    emsr_variant : {'b', 'a'}, default='b'
        Specifies which variant of the EMSR algorithm to use, either 'b' for
        EMSR-B, or 'a' for EMSR-A.  At this time only 'b' is supported.
    forecast_algorithm : {'additive_pickup', 'exp_smoothing', 'multiplicative_pickup'}, default 'additive_pickup'
        Specifies which leg-level forecasting algorithm to use for generating
        leg demand forecasts.  Options are 'additive_pickup', 'exp_smoothing', or
        'multiplicative_pickup'.  The default is 'additive_pickup'.
    exp_smoothing_alpha : float, default 0.15
        Specifies the alpha parameter to use for exponential smoothing.  This
        parameter is only used if the `forecast_algorithm` is set to
        'exp_smoothing'.  The default value is 0.15.

    Notes
    -----
    This RM system consists of the following actions executed in order:

    1. **EM Untruncation of Leg Demands**
        This step applies the EM algorithm to detruncate observed leg
        sales into inferred true demand levels.  It runs only once at the
        beginning of each sample day, and detruncates demand for all timeframes.

    2. **Standard Leg Forecasting**
        This step generates leg-level demand forecasts using a standard
        class-based forecasting algorithm (additive pickup by default). It
        runs full computations to produce leg forecasts for the entire
        booking horizon in one pass at the beginning of each sample day,
        and on later DCPs it simply moves a pointer forward through that
        array of forecasts to provide the correct forecast values at that
        time.

    3. **EMSR-B Optimization**
        Optimizes leg-level seat availability using the Expected Marginal
        Seat Revenue Version B (EMSR-B) algorithm. This step runs on each
        DCP to update the seat availability controls based on the current
        leg forecasts, current sales, and the EMSR optimization logic.
    """

    availability_control = "leg"
    """This RM system uses leg-level class allocation availability controls."""

    actions = [
        LegDetruncation,
        StandardLegForecast.configure(
            algorithm=RmSysOption("forecast_algorithm", default="additive_pickup"),
            alpha=RmSysOption("exp_smoothing_alpha", expected_type=float, default=0.15),
        ),
        ExpectedMarginalSeatRevenue.configure(
            variant=RmSysOption("emsr_variant", default="b"),
            cabins=RmSysOption("cabins", default=None),
        ),
    ]
