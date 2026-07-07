from typing import Literal

from passengersim.rm.conditional_forecasting import ConditionalLegForecast
from passengersim.rm.dynamic_prog import LegDynamicProgram
from passengersim.rm.leg_value import LegValue
from passengersim.rm.q_forecasting import QLegForecast
from passengersim.rm.standard_forecasting import StandardLegForecast
from passengersim.rm.systems import RmSys, RmSysOption, register_rm_system
from passengersim.rm.untruncation import LegDetruncation


@register_rm_system
class LegDP(RmSys):
    """A Leg Dynamic Programming RM system."""

    availability_control = "bp"
    """This RM system uses bid price availability controls."""

    actions = [
        LegDetruncation.configure(
            initialization_method=RmSysOption("em_initialization_method", default="default"),
        ),
        StandardLegForecast.configure(
            algorithm=RmSysOption("forecast_algorithm", default="additive_pickup"),
        ),
        LegDynamicProgram.configure(
            arrivals_per_time_slice=RmSysOption("arrivals_per_time_slice", expected_type=float, default=0.5),
            normalization_method=RmSysOption("normalization_method", expected_type=int, default=0),
            fixed=dict(reoptimize="once"),
        ),
    ]


@register_rm_system
class CLegDP(RmSys):
    """A Leg Dynamic Programming RM system."""

    availability_control = "bp"
    """This RM system uses bid price availability controls."""

    actions = [
        LegDetruncation.configure(
            initialization_method=RmSysOption("em_initialization_method", default="default"),
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
        LegDynamicProgram.configure(
            arrivals_per_time_slice=RmSysOption("arrivals_per_time_slice", expected_type=float, default=0.5),
            normalization_method=RmSysOption("normalization_method", expected_type=int, default=0),
            snapshot_filters=RmSysOption("dp_snapshots", expected_type=list, default=None),
            fixed=dict(reoptimize="once"),
        ),
    ]


@register_rm_system
class QLegDP(RmSys):
    """A Leg Dynamic Programming RM system."""

    availability_control = "bp"
    """This RM system uses bid price availability controls."""

    actions = [
        LegValue.configure(
            minimum_pct_separation=RmSysOption(
                "fare_inversion_minimum_pct_separation", expected_type=float, default=0.05
            ),
        ),
        LegDetruncation.configure(
            algorithm=RmSysOption("yieldable_detruncation", expected_type=Literal["em", "none"], default="em"),
            fixed=dict(which_data="yieldable"),
        ),
        LegDetruncation.configure(
            algorithm=RmSysOption("priceable_detruncation", expected_type=Literal["em", "none"], default="em"),
            fixed=dict(which_data="priceable"),
        ),
        QLegForecast.configure(
            algorithm=RmSysOption("forecast_algorithm", default="additive_pickup"),
            fare_adjustment=RmSysOption("fare_adjustment", expected_type=Literal["mr", "ki", None], default="mr"),
            fare_adjustment_scale=RmSysOption("fare_adjustment_scale", expected_type=float, default=0.5),
            variance_rollup_algorithm=RmSysOption(
                "variance_rollup_algorithm", expected_type=Literal["tf", "dep"], default="tf"
            ),
            variance_is_ratio_of_mean=RmSysOption("variance_is_ratio_of_mean", expected_type=float, default=0.0),
            max_cap=RmSysOption("max_cap", expected_type=float, default=10.0),
            q_allocation_algorithm=RmSysOption(
                "q_allocation_algorithm", expected_type=Literal["tf", "dep"], default="tf"
            ),
        ),
        LegDynamicProgram.configure(
            arrivals_per_time_slice=RmSysOption("arrivals_per_time_slice", expected_type=float, default=0.5),
            normalization_method=RmSysOption("normalization_method", expected_type=int, default=0),
            snapshot_filters=RmSysOption("dp_snapshots", expected_type=list, default=None),
            fixed=dict(reoptimize="once"),
        ),
    ]
