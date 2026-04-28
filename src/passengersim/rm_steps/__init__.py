#
# A step in the RM system
#  - Untruncation / Forecasting / Adjustments / Optimization / etc.
#  - This is the abstract base class, each component will inherit from this
#  - It defines the means to link the processing steps and check they fit together
#
# Alan W, January 2023
# (c) PassengerSim LLC
#

from abc import ABC, abstractmethod

from passengersim.config.rm_steps import RmStepBase

try:
    from passengersim_core import SimulationEngine
except ImportError:
    SimulationEngine = None


class RmStep(RmStepBase, ABC):
    name: str = ""

    @property
    def requires(self) -> list[str]:
        """
        A list of things that this RM step requires before it can be run.

        This allows the driver to check that the flow is correct. For example, if
        forecasting produces "leg_forecast" but optimization requires "path_forecast"
        then an exception will be thrown.

        This property is defined in the base class as an empty list, but
        derived classes that implement the RmStep interface should override this
        property if they do require anything in particular.
        """
        return []

    @property
    def produces(self) -> list[str]:
        """
        A list of things this RM step produces, which may be required by later steps.

        This allows the driver to check that the flow is correct. For example, if
        forecasting produces "leg_forecast" but optimization requires "path_forecast"
        then an exception will be thrown.

        This property is defined in the base class as an empty list, but
        derived classes that implement the RmStep interface should override this
        property if they do produce anything that may be required by a later step.
        """
        return []

    @property
    def require_availability_control(self) -> set[str] | None:
        """
        A set of availability controls that this RM step requires.

        If the RM step requires a specific availability control, or one of a set of
        availability controls, this property should be overridden to return the
        required availability control(s).
        """
        return None

    debug_filters: list[str] = []

    def __str__(self):
        return f"RM Step {self.__class__.__name__}: {super().__str__()}"

    @abstractmethod
    def run(
        self,
        sim: SimulationEngine,
        carrier: str,
        dcp_index: int,
        dcp: int,
        debug: bool = False,
    ):
        """This is where the magic happens.

        Every RM step implementation must override this method to do whatever it
        is going to do.
        """
