from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from ._common import RmAction

if TYPE_CHECKING:
    from passengersim.config import Config
    from passengersim.driver import Simulation


class StandardLegForecast(RmAction):
    """
    Standard leg-level demand forecasting tool.
    """

    requires: set[str] = {"leg_demand"}
    produces: set[str] = {"leg_forecast"}
    frequency = "dcp"

    def __init__(
        self,
        *,
        algorithm: Literal["additive_pickup", "exp_smoothing", "multiplicative_pickup"] = "additive_pickup",
        alpha: float = 0.15,
        carrier: str = "",
        minimum_sample: int = 10,
        cfg: Config | None = None,
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

    def run(self, sim: Simulation, days_prior: int):
        if not self.should_run(sim, days_prior):
            return
        dcp_index = self.get_dcp_index(days_prior)
        # we will only recompute the full forecast on the first DCP. Subsequent DCPs
        # will reuse the forecast computed at DCP 0.
        recompute = dcp_index == 0

        for leg in sim.eng.legs.set_filters(carrier=self.carrier):
            leg.forecast.compute_forecasts(dcp_index, self.algorithm, None, recompute=recompute)


class StandardPathForecast(RmAction):
    """
    Standard path-level demand forecasting tool.
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

    def run(self, sim: Simulation, days_prior: int):
        if not self.should_run(sim, days_prior):
            return

        dcp_index = self.get_dcp_index(days_prior)

        # we will only recompute the full forecast on the first DCP. Subsequent DCPs
        # will reuse the forecast computed at DCP 0.
        recompute = dcp_index == 0

        for pth in sim.eng.paths.set_filters(carrier=self.carrier):
            pth.forecast.compute_forecasts(dcp_index, self.algorithm, None, recompute=recompute)


class PathForecastDailyDecay(RmAction):
    """
    Apply daily decay to path-level demand forecasts.
    """

    frequency = "non_dcp"

    def __init__(
        self,
        *,
        carrier: str = "",
        minimum_sample: int = 10,
        cfg: Config | None = None,
    ):
        super().__init__(
            carrier=carrier,
            minimum_sample=minimum_sample,
            cfg=cfg,
        )

    def run(self, sim: Simulation, days_prior: int):
        if not self.should_run(sim, days_prior):
            return

        engine = sim.eng
        # dcp_index = self.get_dcp_index(days_prior, allow_between=True)
        # # When does this timeframe end?
        # current_ts = engine.last_event_time
        # departure_ts = engine.base_time
        # end_tf_ts = departure_ts
        # tf_remaining_days = -1
        # if (dcp_index + 1) <= engine.num_dcps:
        #     tf_remaining_days = days_prior - engine.get_days_prior(dcp_index + 1)
        #     end_tf_ts = current_ts + tf_remaining_days * 86400
        # tf_remaining_days_at_begin = days_prior - engine.get_days_prior(dcp_index)
        # begin_tf_ts = current_ts + tf_remaining_days_at_begin * 86400
        # if tf_remaining_days < 0:
        #     raise RuntimeError("tf_remaining_days is negative")

        for p in engine.paths.set_filters(carrier=self.carrier):
            # snapshot_instruction = get_snapshot_instruction(engine, path=p, only_type="forecast_adj", debug=debug)
            # snapshot_instruction.mode = "a"
            snapshot_instruction = False  # TODO: implement snapshot instruction properly
            p.forecast.move_forecast_pointers(days_prior=days_prior, snapshot_instruction=snapshot_instruction)
