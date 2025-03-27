# Forecasting

Forecasting is a key part of revenue management systems.  You need to know how
many customers of each type you should expect, so you can tailor the set of
products being offered to maximize revenue.

In PassengerSim, forecasting is included as a step within an RM system, typically
within the DCP process, after untruncation and before any optimization.

```yaml title="example.yaml" hl_lines="8-10"
rm_systems:
  basic_emsr_b:
    processes:
      DCP:
      - step_type: untruncation
        algorithm: em
        kind: leg
      - step_type: forecast #(1)!
        algorithm: additive_pickup #(2)!
        kind: leg #(3)!
      - step_type: emsr
        algorithm: b
        kind: leg
```
{ .annotate }

1.  The `step_type` for forecasting must be `forecast`, this is how PassengerSim
    identifies what to do in this step.
2.  Several different algorithms are available for forecasting, see
    [below][passengersim.rm.ForecastStep.algorithm] for details.
3.  Forecasts can be made at the leg or path level, see
    [below][passengersim.rm.ForecastStep.kind] for details.


::: passengersim.rm.ForecastStep
    options:
      show_root_heading: true
      show_root_full_path: false
      show_source: false
      members:
        - algorithm
        - kind
        - alpha
