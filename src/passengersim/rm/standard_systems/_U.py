from passengersim.rm.dynamic_prog import UnbucketedDynamicProgram
from passengersim.rm.standard_forecasting import StandardPathForecast
from passengersim.rm.systems import RmSys, RmSysOption, register_rm_system
from passengersim.rm.untruncation import PathUntruncation


@register_rm_system
class U(RmSys):
    """A standard RM system of type "U".

    This RM system uses path-level bid price controls with UDP
    optimization, along with EM untruncation and standard leg forecasting.

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
    em_initialization_method : {'default', 'pods'}, default 'default'
        Specifies the initialization method for the EM untruncation algorithm.
        The 'default' method uses all available data at each iteration, while the
        'pods' method uses only data from unclosed observations on the first EM
        iteration.

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

    3. **UDP Optimization**
        Optimizes path-level bid prices using the Unbucketed Dynamic Program
        (UDP) algorithm. This step re-solves the dynamic program with updated
        displacement values based on actual sales and current forecasts only on
        the DCPs; however, it still updates the bid prices every day, using daily
        average bid price vectors taken from the most recent DP solution..
    """

    availability_control = "bp"
    """This RM system uses bid price availability controls."""

    actions = [
        PathUntruncation.configure(
            initialization_method=RmSysOption("em_initialization_method", default="default"),
        ),
        StandardPathForecast.configure(
            algorithm=RmSysOption("forecast_algorithm", default="additive_pickup"),
        ),
        UnbucketedDynamicProgram.configure(
            arrivals_per_time_slice=RmSysOption("arrivals_per_time_slice", expected_type=float, default=0.5),
            normalization_method=RmSysOption("normalization_method", expected_type=int, default=0),
            cabins=RmSysOption("cabins", default=None),
            capacity_sharing=RmSysOption("capacity_sharing", expected_type=bool, default=False),
            capacity_sharing_start_dcp_index=RmSysOption(
                "capacity_sharing_start_dcp_index", expected_type=int, default=0
            ),
            capacity_sharing_start_lf=RmSysOption("capacity_sharing_start_lf", expected_type=float, default=0.0),
        ),
    ]
