from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from ._common import RmAction

if TYPE_CHECKING:
    from passengersim.config import Config
    from passengersim.driver import Simulation


class LegUntruncation(RmAction):
    """
    Leg-level demand untruncation action.

    This action will untruncate demand on legs using the specified algorithm.
    It is called only once at the beginning of each sample.
    """

    produces: set[str] = {"leg_demand"}
    frequency = "begin_sample"

    def __init__(
        self,
        *,
        carrier: str = "",
        minimum_sample: int = 10,
        cfg: Config | None = None,
        algorithm: Literal["em", "em_py", "none", "naive1", "naive2", "naive3", "pd", "mfem"] = "em",
        which_data: Literal["total", "yieldable", "priceable"] = "total",
        maxiter: int = 20,
        tolerance: float = 0.01,
        initialization_method: Literal["default", "pods"] = "default",
        minimum_mu: float = 0.01,
        minimum_sigma: float = 0.1,
    ):
        super().__init__(
            carrier=carrier,
            minimum_sample=minimum_sample,
            cfg=cfg,
        )

        self.which_data = which_data
        """
        Which data to use for untruncation.
        """

        self.algorithm: Literal["em", "em_py", "none", "naive1", "naive2", "naive3", "pd", "mfem"] = algorithm
        """
        Untruncation algorithm.

        There are several available algorithms:

        `none`
            applies no untruncation, and assumes that demand was the same as sales.
            Applying this algorithm is still important even if no detruncation is
            desired, as PassengerSim tracks historical demand separately from sales
            and without "none" the historical demand used in forecasting would be
            zero.

        `em`
            is an expectation-maximization model.

        `em_py`
            is an expectation-maximization model implemented in Python.  It is slow
            but useful for educational purposes.

        `naive1`
            is not recommended for use.

        `naive2`
            is not recommended for use.

        `naive3`
            is not recommended for use.

        `pd`
            is a projection-detruncation model, based on the method developed by
            Hopperstad and described by Weatherford & Polt.

        `mfem`
            Multi-Flight Expectation Maximization.  Takes into account spill and recapture across multiple departures
        """

        self.maxiter = maxiter
        """
        Maximum number of iterations for the EM and PD algorithms.
        """

        self.tolerance = tolerance
        """
        Tolerance for the EM and PD algorithms.
        """

        self.initialization_method: Literal["default", "pods"] = initialization_method
        """
        Method for initializing the EM algorithm.

        The default method is to use all available data on the first EM iteration.  The pods method is to use
        only the data from unclosed observations on the first EM iteration.
        """

        self.minimum_mu = minimum_mu
        """
        Minimum value for the mean of the demand distribution.

        If the computed mean is less than this value, untruncation will result in
        zero demand. Setting this to a very small but non-zero value can help avoid
        thin-path problems, where untruncation results in some non-zero demand on
        every path-class, even though many path-classes have zero historical sales and
        probably will stay that way.
        """

        self.minimum_sigma = minimum_sigma
        """
        Minimum value for the standard deviation of the demand distribution.

        If the computed sigma is less than this value, this value is used instead.
        """

    def run(self, sim: Simulation, days_prior: int):
        if not self.should_run(sim, days_prior):
            return

        dcp_index = self.get_dcp_index(days_prior)

        # MFEM runs multiple legs
        if self.algorithm == "mfem":
            raise NotImplementedError("MFEM untruncation is not implemented yet.")
            # leg_mfem(sim, _carrier, _dcp_index)
            # return

        for leg in sim.eng.legs.set_filters(carrier=self.carrier):
            if self.algorithm in ["em", "none", "naive1", "naive3"]:
                # snapshot_instruction = get_snapshot_instruction(
                #     sim, leg=leg, only_type="leg_untruncation", debug=_debug
                # )
                snapshot_instruction = None
                leg.untruncate_demand(
                    dcp_index,
                    self.algorithm,
                    snapshot_instruction,
                    maxiter=self.maxiter,
                    tolerance=self.tolerance,
                    pods_initialization=(self.initialization_method == "pods"),
                    minimum_mu=self.minimum_mu,
                    minimum_sigma=self.minimum_sigma,
                    which_data=self.which_data,
                )

            else:
                raise NotImplementedError(f"Untruncation algorithm '{self.algorithm}' is not implemented.")
                # for bkt in leg.buckets:
                #     self.do_single_bucket(leg, bkt, _dcp_index, _debug)


class PathUntruncation(RmAction):
    """
    Path-level demand untruncation tool.
    """

    produces: set[str] = {"path_demand"}
    frequency = "begin_sample"

    def __init__(
        self,
        *,
        carrier: str = "",
        minimum_sample: int = 10,
        cfg: Config | None = None,
        algorithm: Literal["em", "em_py", "none", "naive1", "naive2", "naive3", "pd", "mfem"] = "em",
        which_data: Literal["total", "yieldable", "priceable"] = "total",
        maxiter: int = 20,
        tolerance: float = 0.01,
        initialization_method: Literal["default", "pods"] = "default",
        minimum_mu: float = 0.01,
        minimum_sigma: float = 0.1,
    ):
        super().__init__(
            carrier=carrier,
            minimum_sample=minimum_sample,
            cfg=cfg,
        )

        self.which_data = which_data
        """
        Which data to use for untruncation.
        """

        self.algorithm = algorithm
        """
        Untruncation algorithm.

        There are several available algorithms:

        `none`
            applies no untruncation, and assumes that demand was the same as sales.
            Applying this algorithm is still important even if no detruncation is
            desired, as PassengerSim tracks historical demand separately from sales
            and without "none" the historical demand used in forecasting would be
            zero.

        `em`
            is an expectation-maximization model.

        `em_py`
            is an expectation-maximization model implemented in Python.  It is slow
            but useful for educational purposes.

        `naive1`
            is not recommended for use.

        `naive2`
            is not recommended for use.

        `naive3`
            is not recommended for use.

        `pd`
            is a projection-detruncation model, based on the method developed by
            Hopperstad and described by Weatherford & Polt.

        `mfem`
            Multi-Flight Expectation Maximization.  Takes into account spill and recapture across multiple departures
        """

        self.maxiter = maxiter
        """
        Maximum number of iterations for the EM and PD algorithms.
        """

        self.tolerance = tolerance
        """
        Tolerance for the EM and PD algorithms.
        """

        self.initialization_method = initialization_method
        """
        Method for initializing the EM algorithm.

        The default method is to use all available data on the first EM iteration.  The pods method is to use
        only the data from unclosed observations on the first EM iteration.
        """

        self.minimum_mu = minimum_mu
        """
        Minimum value for the mean of the demand distribution.

        If the computed mean is less than this value, untruncation will result in
        zero demand. Setting this to a very small but non-zero value can help avoid
        thin-path problems, where untruncation results in some non-zero demand on
        every path-class, even though many path-classes have zero historical sales and
        probably will stay that way.
        """

        self.minimum_sigma = minimum_sigma
        """
        Minimum value for the standard deviation of the demand distribution.

        If the computed sigma is less than this value, this value is used instead.
        """

    def run(self, sim: Simulation, days_prior: int):
        if not self.should_run(sim, days_prior):
            return

        dcp_index = self.get_dcp_index(days_prior)

        # MFEM runs multiple legs
        if self.algorithm == "mfem":
            raise NotImplementedError("MFEM untruncation is not implemented yet.")
            # leg_mfem(sim, _carrier, _dcp_index)
            # return

        for pth in sim.eng.paths.set_filters(carrier=self.carrier):
            if self.algorithm in ["em", "none", "naive1", "naive3"]:
                # snapshot_instruction = get_snapshot_instruction(
                #     sim, leg=leg, only_type="leg_untruncation", debug=_debug
                # )
                snapshot_instruction = None
                pth.untruncate_demand(
                    dcp_index,
                    self.algorithm,
                    snapshot_instruction,
                    maxiter=self.maxiter,
                    tolerance=self.tolerance,
                    pods_initialization=(self.initialization_method == "pods"),
                    minimum_mu=self.minimum_mu,
                    minimum_sigma=self.minimum_sigma,
                    which_data=self.which_data,
                )

            else:
                raise NotImplementedError(f"Untruncation algorithm '{self.algorithm}' is not implemented.")
                # for bkt in leg.buckets:
                #     self.do_single_bucket(leg, bkt, _dcp_index, _debug)
