
# Demand Generation

Demand generation is one of the first steps in simulating a travel day (i.e., a sample in the simulation).
The demand generation process is used to create the pool of potential customers that will be used in the
simulation for a given sample.  The total pool of customers is generated at the beginning of the sample,
before any sales occur, and every customer is scheduled to arrive at fixed time before departure.

This process occurs in two steps. The first step is to generate the total demand for each passenger type
in each market on each travel day.  The second step is to distribute this total demand over the booking
time periods (i.e. to assign an arrival time to each customer in the pool).


## Generating Total Demands

For every market and passenger type, we have a base demand given as an exogenous input to the simulation.
This base demand is used as the center point for this demand generation process, which is the
mean of a normal distribution from which we will draw a random value to represent the total demand
for that market and passenger type on the given departure day.  Due to the use of a normal distribution,
we may have some negative demand values in some samples, which are truncated to zero, so the actual
mean demand across samples may be slightly higher than the provided base demand.

The random draw of total demand for a given market and passenger type on a given departure day is
controlled by a number of "k-factors" that are used to introduce correlation across various dimensions of
demand.  Each k factor also contributes to the overall variance of demand across samples.

- [`sys_k_factor`][passengersim.config.simulation_controls.SimulationSettings.sys_k_factor] induces correlation
    across the entire system.  For example, if we have a "high demand" day across the system, we would expect that
    demand is higher than average across most markets and passenger types in the system.
- [`mkt_k_factor`][passengersim.config.simulation_controls.SimulationSettings.mkt_k_factor] induces correlation
    across passenger segments within each market. For example, if we have a "high demand" day in the BOS-LAS market,
    we would expect that both business and leisure demand in that market would tend to be higher than average, but
    there would be no effect on demand in other markets.
- [`segment_k_factor`][passengersim.config.simulation_controls.SimulationSettings.segment_k_factor] induces
    correlation across markets within each passenger segment. For example we may have a "high demand" day for
    business passengers, which would see business demand tend to be larger in all markets, but leisure demand
    would not be affected. Note this is *not* the same as the perhaps mis-named "passenger type k-factor"
    historically used in PODS; there is no PODS setting that actually does what this k-factor does.
- [`simple_k_factor`][passengersim.config.simulation_controls.SimulationSettings.pax_type_k_factor] adds
    independent, uncorrelated variance to the total demand for each market and passenger segment. Note that in
    early in previous versions of PassengerSim, this was called `pax_type_k_factor`, following from the PODS
    implementation and terminology. However, that name was somewhat misleading relative to what this k factor
    does, so that name has been deprecated in favor of the clearer terminology.

These k-factors are implemented by attaching them to standard normal random variables (i.e., random draws with
mean of zero and standard deviation of one) that are generated for each sample.  The random numbers are multiplied
by the corresponding k-factor and added to a unit base demand (i.e., 1) to create a new unit base demand for that market and passenger
type on that departure day.  For the system-level k-factor, we use the sample single random draw that is applied
to every market and passenger type in the system.  For the market-level k-factor, we draw a unique random number
for each market that is applied to all passenger types in that market.  For the segment-level k-factor, we draw a
unique random number for each passenger segment, and apply it to all markets for that segment.  For the simple
k-factor, we draw a unique random number for each demand that is applied only to that demand.

This unit base demand is then scaled by the provided base demand for that market and passenger type to get the
mean of a demand distribution for that market and passenger type on that departure day.
