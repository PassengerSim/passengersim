# How PassengerSim Works

PasengerSim is a tool that simulates several aspects of passenger travel, including
airline revenue management operations, market level passenger demand, and individual
customer choice processes.

## Counting Simulations

A simulation run consists of a number of independent trials, and each trial is
made up of a sequence of dependent samples -- earlier samples in a trial are used
to develop forecasts and train optimization algorithms used by carriers in later
samples of the same trial.

The number of trials is set by the
[`num_trials`][passengersim.config.simulation_controls.SimulationSettings.num_trials]
configuration, and the number of samples in each trial is set by
[`num_samples`][passengersim.config.simulation_controls.SimulationSettings.num_samples].
Both values can be found in the
[`simulation_controls`][passengersim.config.Config.simulation_controls]
configuration inputs.

We can think of a sample as a "typical" departure day.  When generating results,
the first X samples from each trial as these are during a "burn period" when the
simulation is getting started and sufficient history is being generated to use
for forecasts and other steps.  The nuber of samples in the burn period is set by
the [`burn_samples`][passengersim.config.simulation_controls.SimulationSettings.burn_samples]
configuration value.


## Simulation Randomness

### Demand Generation K-Factors {k-factors}

There are multiple sources of variability that is introduced in the simulation.
Variability in the level of demand by passenger type on any given day for any
origin-destination pair is controlled by a number of k-factors, which are used
to create some correlation across various dimensions of demand.

Three k-factors are used to introduce correlation in demand across all markets in
the system and between business/leisure demand within a market. The equation to
generate the mean demand for a given market and passenger type
is given as:

$\mu'_{OD-Biz} = \mu_{OD-Biz} + (SRN \times k_{sys}) + (MRN \times k_{mkt}) + (PRN \times k_{paxtype})$

where SRN, MRN, and PRN are random numbers associated with the system, market,
and passenger k-factors, respectively.

The intuition behind using three k-factors is that even across a "typical" departure
day (like Wednesdays) we may have high demand days across the system and low demand
days across the system.  Likewise, if business demand is above average, we expect
that leisure demand may also be above average. Intuitively, when generating demands,
for a given sample (or departure day in the simulation), we add the term
(SRN x `sys_k_factor`) to every single market and passenger in the system for that
departure day, we add a unique (MRN x `mkt_k_factor`) to each origin-destination
market in the sample, and we add a unique (PRN x `pax_type_k_factor`) for each
passenger type within the market. The addition of the (SRN x `sys_k_factor`) to
"everything" in the sample creates a system-level correlation (i.e., demand on
a given departure date or sample could be "high" across the system or "low"
compared to average).

The addition of the (MRN x `mkt_k_factor`) to a market creates a market-level
correlation (i.e., if business demand is running higher than average in the
market, we expect leisure demand will run higher than average in general as
well).

The use of the (PRN x `pax_type_k_factor`) ensures that there is some random
component that is independent between business and leisure passengers in a
given market.

In addition to the k-factors that are used to introduce correlation, we have a z-factor.
We assume that actual demand, given mean demand, will vary across samples (or
departure dates) according to a constant z-factor given as $\mu$ divided by $\sigma^2$.

Given different levels of aggregation, we expect that `sys_k_factor` < `mkt_k_factor` < `pax_type_k_factor`.

Once we generate the means and std dev for correlated demands by OD and pax type
using the methodology described above, we use these to generate the “actual” demands
for a sample using the equation

$\mu'_{OD-Biz} + NRV x \sigma'_{OD-Biz}$,

where NRV is a normal random variable.


### Demand Allocation to Time Periods

The demand generation above is used to find the total demand for each passenger
type in each market on each travel day.  This total value is subsequently distributed
over the booking time periods.

While the methodology for this is not (yet) explained here, note that the `tf_k_factor`
is used to generate variability in how demand is allocated to the different time
frames (also using the booking curves as a key input).
