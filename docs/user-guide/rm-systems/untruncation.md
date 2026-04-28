# Untruncation

Untruncation is a part of most revenue management systems.  It is a mathematical
process whereby we estimate the number of customers there would have been for a
particular product, assuming we would have offered the product for sale to all
comers.  In the cases where we actually did offer the product to all, then there
is nothing for this algorithm to do beyond counting our actual sales.  However,
many times our RM systems will limit the number of customers we actually accept,
and our actual sales are "truncated".  Untruncation is needed to approximate how
many customers were lost.

In PassengerSim, untruncation is included as a step within an RM system, typically
within the DCP process before any forecasting or optimization steps.


::: passengersim.rm.untruncation.LegUntruncation
    options:
      show_root_heading: true
      show_root_full_path: false
      show_source: false
      members:
        - which_data
        - algorithm
        - initialization_method

::: passengersim.rm.untruncation.PathUntruncation
    options:
      show_root_heading: true
      show_root_full_path: false
      show_source: false
      members:
        - which_data
        - algorithm
        - initialization_method
