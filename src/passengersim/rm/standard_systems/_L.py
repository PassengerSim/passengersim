from typing import Literal

from passengersim.rm.conditional_forecasting import ConditionalLegForecast
from passengersim.rm.emsr import ExpectedMarginalSeatRevenue
from passengersim.rm.systems import RmSys, RmSysOption, register_rm_system
from passengersim.rm.untruncation import LegDetruncation


@register_rm_system
class L(RmSys):
    """A standard RM system of type "L".

    This RM system uses leg-level seat allocation controls with EMSR-B
    optimization, along with EM untruncation of yieldable demand and hybrid
    conditional leg forecasting.

    Parameters
    ----------
    emsr_variant : {'b', 'a'}, default='b'
        Specifies which variant of the EMSR algorithm to use, either 'b' for
        EMSR-B, or 'a' for EMSR-A.  At this time only 'b' is supported.
    forecast_algorithm : {'additive_pickup', 'exp_smoothing', 'multiplicative_pickup'}, default 'additive_pickup'
        Specifies which leg-level forecasting algorithm to use for generating
        leg demand forecasts.  Options are 'additive_pickup', 'exp_smoothing', or
        'multiplicative_pickup'.  The default is 'additive_pickup'.
    fare_adjustment : {'mr', 'ki', None}, default 'mr'
        Specifies the type of fare adjustment to apply in the conditional path
        forecasting algorithm. The first option is 'mr' for marginal revenue
        adjustment, inspired by the work of Fiig, Isler, Hopperstad, and Belobaba (2010),
        which assumes a continuous pricing curve with a negative exponential
        customer sellup function. The second option is 'ki' for Karl Isler's method,
        which is a discrete pricing adjustment method that assumes no particular
        functional form of the customer sellup function. It is also possible to select
        `None` for no fare adjustment, although this is not usually recommended
        except for restricted product sets.
    fare_adjustment_scale : float, default 0.5
        Specifies the scale factor to use for fare adjustment in the conditional
        path forecasting algorithm. This factor determines the magnitude of the
        fare adjustment applied to the forecasts. A value of 1.0 applies the full
        adjustment, while values less than 1.0 scale down the adjustment. In
        practice, smaller values are used when the product sets are more restricted,
        and higher values are used when the product sets are less restricted.
        Theoretically, the full fare adjustment scale of 1.0 would be appropriate
        for completely unrestricted product sets, while no fare adjustment (0.0)
        should be used for heavily restricted product sets.  The default setting
        of 0.5 is a moderate adjustment that works reasonably well in cases with
        semi-restricted product sets.
    regression_weight : {None, 'sellup', 'sellup^2', 'fare', 'none'}, default None
        Specifies the type of regression weight to use in the conditional path
        forecasting algorithm. Options are 'sellup' to weight by sellup factor,
        'sellup^2' to weight by the square of the sellup factor, 'fare' to weight
        by fare amount, 'none' for no weighting.
    variance_rollup_algorithm : {'tf', 'dep'}, default 'tf'
        Specifies how to roll up variance when combining priceable and yieldable forecasts.
        Options are 'tf' to roll up variance by time frame, or 'dep' to roll up variance
        to departure. This setting has no effect if the `variance_is_ratio_of_mean`
        parameter is set to a value greater than zero.
    variance_is_ratio_of_mean : float, default 2.0
        Assume that the variance is this ratio of the mean. When this is set to a
        value greater than zero, the variance of the forecast is set to this fixed
        ratio of the mean.  Note that many algorithms for optimization use the
        forecast standard deviation, which is the square root of the variance, but it is
        the variance that is set to this ratio times the mean.
        When set to zero (the default), the variance is computed from mean squared error
        of the linear regression model used to compute the mean, and is not a fixed ratio
        of the mean. The default value of 2.0 is reflective of methods used in practice.
    max_cap : float, default 0.0
        Maximum weighting factor for the conditional forecast. If set to a value
        greater than zero, the weighting factor used in the conditional forecast is
        capped at this maximum value to prevent excessively high weights from extreme
        sellup factors.  The max cap will be more important when the regression weight
        is set to 'sellup^2', as this weighting can grow very large for extreme sellup
        factors. If this is applied, it is recommended to set the cap to 10.0, which is
        consistent with prior work on hybrid forecasting methods.
    q_allocation_algorithm : {'tf', 'dep'}, default 'tf'
        Specifies how to allocate variance from aggregate Q forecasts to
        class-level forecasts. Options are 'tf' to allocate variance by time frame,
        or 'dep' to allocate variance to departure. This setting has no effect if
        the `variance_is_ratio_of_mean` parameter is set to a value greater than zero.


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
        LegDetruncation.configure(
            fixed=dict(which_data="yieldable"),
        ),
        ConditionalLegForecast.configure(
            algorithm=RmSysOption("forecast_algorithm", default="additive_pickup"),
            fare_adjustment=RmSysOption("fare_adjustment", expected_type=Literal["mr", "ki", None], default="mr"),
            fare_adjustment_scale=RmSysOption("fare_adjustment_scale", expected_type=float, default=0.5),
            regression_weight=RmSysOption(
                "regression_weight", expected_type=Literal["sellup", "sellup^2", "fare", "none", None], default=None
            ),
            variance_rollup_algorithm=RmSysOption(
                "variance_rollup_algorithm", expected_type=Literal["tf", "dep"], default="tf"
            ),
            variance_is_ratio_of_mean=RmSysOption("variance_is_ratio_of_mean", expected_type=float, default=2.0),
            max_cap=RmSysOption("max_cap", expected_type=float, default=0.0),
            q_allocation_algorithm=RmSysOption(
                "q_allocation_algorithm", expected_type=Literal["tf", "dep"], default="tf"
            ),
        ),
        ExpectedMarginalSeatRevenue.configure(
            variant=RmSysOption("emsr_variant", default="b"),
        ),
    ]
