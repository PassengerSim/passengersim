# EMSR Optimization

Optimization is the most fundamental part of revenue management systems, is it
is the process used to tailor the set of products being offered to maximize revenue.
It typically occurs after untruncation and forecasting.

PassengerSim offers several different optimization algorithms. One widely used
algorithm is called EMSR (expected marginal seat revenue), which has a few variants,
generally labels as "A", "B", and "C".

::: passengersim.rm.emsr.ExpectedMarginalSeatRevenue
    options:
      show_root_heading: true
      show_root_full_path: false
      show_source: false
      members:
        - variant
