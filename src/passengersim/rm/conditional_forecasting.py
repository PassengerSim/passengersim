from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from ._common import RmAction

if TYPE_CHECKING:
    from passengersim.config import Config
    from passengersim.driver import Simulation


class ConditionalPathForecast(RmAction):
    """
    Conditional path-level demand forecasting tool.
    """

    requires: set[str] = {"path_demand"}
    produces: set[str] = {"path_forecast"}
    frequency = "dcp"

    def __init__(
        self,
        *,
        algorithm: Literal["additive_pickup", "exp_smoothing", "multiplicative_pickup"] = "additive_pickup",
        alpha: float = 0.15,
        carrier: str = "",
        minimum_sample: int = 10,
        cfg: Config | None = None,
        fare_adjustment: Literal["mr", "ki", None] = "mr",
        fare_adjustment_scale: float = 0.25,
        regression_weight: Literal["sellup", "sellup^2", "fare", "none", None] = "sellup",
        variance_rollup_algorithm: Literal["tf", "dep"] = "tf",
        variance_is_ratio_of_mean: float = 0.0,
        max_cap: float = 0.0,
        q_allocation_algorithm: Literal["tf", "dep"] = "tf",
    ):
        super().__init__(
            carrier=carrier,
            minimum_sample=minimum_sample,
            cfg=cfg,
        )

        self.algorithm = algorithm
        """
        Forecasting algorithm.

        There are several available forecasting algorithms:

        `additive_pickup`
            is an additive pickup model, which generates a forecast by considering the
            "pickup", or the number of new sales in a booking class, in each time
            period (DCP).  This model is additive in that the forecast of demand yet
            to come at given time is computed as the sum of forecast pickups in all
            future time periods.  This forecasting model does not consider the level
            of demand already accumulated, only the demand expected in the future. The
            forecast is made considering the results from the prior 26 sample days.
            The additive pickup model ignores the value of the alpha parameter, and it
            can safely be omitted when using this algorithm.

        `exp_smoothing`
            is an exponential smoothing model.  This model uses the `alpha` parameter
            to control the amount of smoothing applied.  It does not (currently)
            incorporate trend effects or seasonality.

        `multiplicative_pickup`
            is a multiplicative pickup model.  This model is in development.
        """

        self.alpha = alpha
        """Exponential smoothing factor.

        This setting is ignored if the forecast algorithm is not "exp_smoothing".
        """

        self.fare_adjustment: Literal["mr", "ki", None] = fare_adjustment
        """Fare adjustment algorithm to use with hybrid or conditional forecasting.

        This setting is ignored for other forecast types.
        """

        self.fare_adjustment_scale = fare_adjustment_scale
        """Fare adjustment scale factor to use with hybrid forecasting.

        This setting is ignored for forecast types other than hybrid, or if
        the `fare_adjustment` setting is None."""

        self.weighted_by_ratio: bool = True
        """Weight fare adjustment by the ratio of priceable and yieldable demand.

        When set to True, the fare adjustment is weighted by the ratio of priceable
        and yieldable forecasted demand, so that the fare adjustment is applied more
        heavily when more of the combined demand is priceable.  When set to False, the
        fare adjustment is applied uniformly and in full without regard to the ratio of
        priceable and yieldable demand.
        """

        self.regression_weight = regression_weight

        self.q_allocation_algorithm: Literal["tf", "dep"] = q_allocation_algorithm
        """How to allocate variance from aggregate Q forecasts to class-level forecasts."""

        self.variance_rollup_algorithm = variance_rollup_algorithm
        """How to roll up variance when combining priceable and yieldable forecasts."""

        self.variance_is_ratio_of_mean: float = variance_is_ratio_of_mean
        """For conditional forecasting, assume that the variance is this ratio of the mean.

        When this is set to a value greater than zero, the variance of the forecast is set to
        this fixed ratio of the mean.  Note that many algorithms for optimization use the
        forecast standard deviation, which is the square root of the variance, but it is
        the variance that is set to this ratio times the mean.

        When set to zero (the default), the variance is computed from mean squared error
        of the linear regression model used to compute the mean.

        This setting is used only for conditional forecasting.
        """

        self.max_cap: float = max_cap
        """
        Maximum sellup weighting factor for the conditional forecast.

        If set to a value greater than zero, the weighting factor used in
        the regression model for conditional forecasting is capped at this value.
        If set to zero (the default), there is no cap applied.
        """

    def _apply_on_objects(self, sim: Simulation):
        return sim.eng.paths.set_filters(carrier=self.carrier)

    def run(self, sim: Simulation, days_prior: int):
        if not self.should_run(sim, days_prior):
            return

        dcp_index = self.get_dcp_index(days_prior)

        # Get the carrier object
        carrier_obj = None
        for a in sim.eng.carriers:
            if a.name == self.carrier:
                carrier_obj = a
                break
        if carrier_obj is None:
            raise ValueError(f"Carrier '{self.carrier}' not found for conditional forecasting")

        things = self._apply_on_objects(sim)
        for thing in things:
            # Get the Frat5 curve for this specific market
            f5 = carrier_obj.get_frat5_mkt(thing.orig, thing.dest)
            if f5 is None:
                raise ValueError(
                    f"Frat5 curve not found for conditional forecasting for market {thing.orig}-{thing.dest}"
                )

            snapshot_instruction = None
            # snapshot_instruction = get_snapshot_instruction(sim, path=thing, only_type="forecast", debug=debug)
            if dcp_index == 0:
                thing.forecast.compute_simple_fare_adjustments(
                    algorithm=self.fare_adjustment,
                    frat5=f5,
                    scale_factor=self.fare_adjustment_scale,
                    snapshot_instruction=snapshot_instruction,
                )
                thing.forecast.compute_conditional_q_forecast(
                    f5,
                    0,
                    regression_weight=self.regression_weight,
                    max_cap=self.max_cap,
                    snapshot_instruction=snapshot_instruction,
                    variance_is_ratio_of_mean=self.variance_is_ratio_of_mean,
                )

                # The forecast has now been created in the q_forecast of the thing.
                # Now we allocate the Q demand to the pathclasses/buckets.
                thing.forecast.allocate_q_demand(
                    f5, 0, snapshot_instruction, allocation_algorithm=self.q_allocation_algorithm
                )

                # compute the yieldable forecasts (if needed).  This must be done before computing the
                # fare adjustments if "weighted_by_ratio" is True, as in that case the fare adjustment is weighted by
                # the ratio of priceable and yieldable forecasted demand.  If "weighted_by_ratio" is False, then it is
                # not critical to compute the yieldable forecasts before the fare adjustments, but it does not
                # adversely affect the results to do so.
                # TODO: check if actually needed?  This may do a lot of compute to get to zero if not needed
                thing.forecast.compute_forecasts(
                    dcp_index,
                    self.algorithm,
                    snapshot_instruction=snapshot_instruction,
                    recompute=True,
                    event_time=sim.eng.last_event_time,
                    which_data="yieldable",
                )

                # compute the "actual" fare adjustment after the Q demand has been allocated
                # This does two things: zero out forecasted Q demand in classes that have a negative
                # adjusted price, and weight the fare adjustment by the ratio of priceable and yieldable demand
                if self.fare_adjustment is not None:
                    thing.forecast.compute_fare_adjustments(
                        self.fare_adjustment,
                        f5,
                        snapshot_instruction=snapshot_instruction,
                        weighted_by_ratio=self.weighted_by_ratio,
                        scale_factor=self.fare_adjustment_scale,
                    )

                thing.forecast.combine_forecasts(
                    dcp_index,
                    rollup_algorithm=self.variance_rollup_algorithm,
                    snapshot_instruction=snapshot_instruction,
                )
                thing.forecast.move_forecast_pointers(dcp_index, snapshot_instruction=snapshot_instruction)
            else:
                # just update cached forecast values
                thing.forecast.move_forecast_pointers(dcp_index, snapshot_instruction=snapshot_instruction)


class ConditionalLegForecast(ConditionalPathForecast):
    """
    Conditional leg-level demand forecasting tool.
    """

    requires: set[str] = set("leg_demand")

    produces: set[str] = set("leg_forecast")

    def _apply_on_objects(self, sim: Simulation):
        return sim.eng.legs.set_filters(carrier=self.carrier)
