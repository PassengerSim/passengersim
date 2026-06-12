from typing import Literal

from passengersim.rm.conditional_forecasting import ConditionalPathForecast
from passengersim.rm.dynamic_prog import UnbucketedDynamicProgram
from passengersim.rm.standard_forecasting import PathForecastDailyDecay
from passengersim.rm.systems import RmSys, RmSysOption, register_rm_system
from passengersim.rm.untruncation import PathUntruncation


@register_rm_system
class V(RmSys):
    """A standard RM system of type "V".

    This RM system uses path-level bid price controls with UDP
    optimization, along with EM untruncation of yieldable demand and hybrid
    conditional path forecasting.

    Parameters
    ----------
    forecast_algorithm : {'additive_pickup', 'exp_smoothing', 'multiplicative_pickup'}, default 'additive_pickup'
        Specifies which leg-level forecasting algorithm to use for generating
        leg demand forecasts.  Options are 'additive_pickup', 'exp_smoothing', or
        'multiplicative_pickup'.  The default is 'additive_pickup'.
    arrivals_per_time_slice : float, default 0.5
        Specifies the expected number of customer arrivals per time slice
        used in the dynamic program optimization. This value affects the
        granularity of the discrete time approximation of the Poisson arrivals
        process; smaller values lead to finer granularity and a closer match
        to the theoretical continuous-time model, but increased computational
        time. Note that achieving a closer match to the continuous-time model
        may not always lead to better RM performance, as the actual customer
        arrival process may deviate from the Poisson assumption in practice
        (and in PassengerSim simulations). The default value is 0.5, which
        provides reasonably fast computational speed, while still capturing all
        forecast demand in the dynamic program.
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
        This step applies the EM algorithm to detruncate observed path
        sales into inferred true demand levels.  It runs only once at the
        beginning of each sample day, and detruncates demand for all timeframes.

    2. **Hybrid-Conditional Path Forecasting**
        This step generates path-level demand forecasts using a hybrid conditional
        class-based forecasting algorithm (additive pickup by default). It
        runs full computations to produce path forecasts for the entire
        booking horizon in one pass at the beginning of each sample day,
        and on later DCPs it simply moves a pointer forward through that
        array of forecasts to provide the correct forecast values at that
        time.

    3. **UDP Optimization**
        Optimizes path-level bid prices using the Unbucketed Dynamic Program
        (UDP) algorithm. This step re-solves the dynamic program with updated
        displacement values based on actual sales and current forecasts only on
        the DCPs; however, it still updates the bid prices every day, using daily
        average bid price vectors taken from the most recent DP solution..

    """

    availability_control = "bp"

    actions = [
        PathUntruncation.configure(
            fixed=dict(which_data="yieldable"),
        ),
        ConditionalPathForecast.configure(
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
        PathForecastDailyDecay,
        UnbucketedDynamicProgram.configure(
            arrivals_per_time_slice=RmSysOption("arrivals_per_time_slice", expected_type=float, default=0.5),
        ),
    ]
