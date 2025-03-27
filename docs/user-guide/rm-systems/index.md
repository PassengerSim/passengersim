# RM Systems

A revenue management (RM) system is defined by one or more steps. The steps tell
the simulation which demand untruncation, demand forecasting, and optimization
algorithms to use.  These steps also provide information on the kind of forecast
and optimization used (i.e., leg or path) and if path-level information should be
aggregated to leg-level information before a step is performed.  Finally, these
steps allow the user to specify algorithm-specific parameters, e.g., when using
exponential smoothing the user can specify the smoothing constant, alpha.


Each carrier should have an RM system that it uses. In PassengerSim, users have
the ability to create a single RM system and assign it to all carriers, or to
create multiple RM systems and assign different RM systems to different carriers.

Below is an example that defines 4 RM systems.  It will be helpful to use these
examples to understand the different step_types and options associated with each
step_type.

```{yaml}
rm_systems:
  fcfs:
    steps: []
  rm_no_detruncation:
    processes:
      DCP:
      - step_type: untruncation
        name: untruncation
        algorithm: none
        kind: leg
      - step_type: forecast
        name: forecast
        algorithm: additive_pickup
        alpha: 0.1
        kind: leg
      - step_type: emsr
        name: optimization
        algorithm: emsrb
        kind: leg
  rm_em:
    steps:
      - step_type: untruncation
        name: untruncation
        algorithm: none
        kind: leg
      - step_type: forecast
        name: forecast
        algorithm: additive_pickup
        alpha: 0.15
        kind: leg
      - step_type: fcfs
        name: optimization
  rm_probp:
    steps:
      - step_type: untruncation
        name: untruncation
        algorithm: em
        kind: path
      - step_type: forecast
        name: path_forecast
        algorithm: exp_smoothing
        alpha: 0.15
        kind: path
      - step_type: probp
        name: optimization
      - step_type: aggregation
        name: aggregate
      - step_type: emsr
        name: optimization
        algorithm: emsrb
        kind: leg
```

The first RM system is based on a first-come, first-serve approach (named `fcfs`).
No steps are defined for `fcfs` as there is no demand detruncation, demand
forecasting, or optimization done with FCFS.  If step_types are defined when
`fcfs` is explicitly specified as the optimization algorithm, they will be ignored.

The second RM system, named `rm_no_detruncation`, is leg-based and uses an `additive_pickup`
forecasting model with EMSRb and no demand detrunction. The alpha parameter that
is specified in the forecast `step_type` will be ignored as it is not used for
the additive pick-up model.

The third RM system, named `rm_em`, is using the expectation-maximization (EM)
method of detruncation with an exponential smoothing demand forecasting approach
that has a smoothing constant of alpha of 0.15 with and EMSRb optimizer. Both the
forecasts and optimization are done using leg-level inputs.

Finally, the fourth RM system, named `rm_probp`, is also using the EM method of
detruncation with an exponential smoothing dmeand forecasting approach that has
a smoothing constant or alpha of 0.15.  Unlike in `rm_em` however, the
untruncation and forecasting steps are done at the path level. The optimization
step is based on probp that first finds displacement costs at a path level, then
aggregates them to a leg-level in the aggregation step type, and finally calculates
protection levels using EMSRb with leg-level demand inputs.

Given an overview of how RM systems are constructed, let's now look at each
step in detail.

## Untruncation

```{yaml}
- step_type: untruncation
  name: untruncation
  algorithm: none, em, naive1, naive2
  kind: leg, path
```

There are three untruncation (also called detruncation) algorithms.  The first,
`em`, is based on the expectation maximization method.  The `naive1` and `naive2`
methods are based on Shebelov presentation.
The untruncation steps can be performed at the path or leg level.

<!-- LG note:
need to verify and add details here.
Am I able to add links (behind a password) to course notes/reference where they are explained?
--->

## Forecast

```{yaml}
- step_type: 'forecast'
  algorithm: { additive_pickup, exp_smoothing }
  alpha: float
  kind: { leg, path, hybrid }
```

There are two forecasting algorithms: exponential smoothing (that uses a smoothing
or alpha parameter) and additive pickup model.  The exponential smoothing model
does not (currently) incorporate trend or seasonality.

The additive pickup model generates a forecast by considering the "pickup", or the
number of new sales in a booking class, in each time period (DCP).  This model is additive
in that the forecast of demand yet to come at given time is computed as the sum
of forecast pickups in all future time periods.  This forecasting model does not
consider the level of demand already accumulated, only the demand expected in the
future.  The forecast is made considering the results from the prior 26 sample days.
The additive pickup model ignores the value of the alpha parameter, and it should be
omitted when using this algorithm.

Either forecast algorithm can be performed based on data collected at either the
leg or path level. However, if the forecast is at the leg level then detruncation
must also have been performed at the same level.

## EMSR Optimization

```{yaml}
- step_type: emsr
  name: optimization
  algorithm: emsra, emsrb
  kind: leg
```

The step_type emsr is used for algorithms based on the expected marginal seat
revenue approach and also can be used for fcfs. (although the first RM system
defined as fcfs in the example above is a cleaner way to specify the fcfs option).

## ProBP Optimization

```{yaml}
- step_type: probp
  name: optimization
```

The step_type probp does just as the name suggests - uses the probabilistic bid
price algorithm to determine path-based displacment costs.  After these are found,
two more steps are needed - the first is to aggregate path-level information to
leg-level information and do the probp proration? and the next is to use the
leg-level inputs to find protection levels using emsrb.
