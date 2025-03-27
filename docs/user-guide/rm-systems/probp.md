# ProBP Optimization

Optimization is the most fundamental part of revenue management systems, is it
is the process used to tailor the set of products being offered to maximize revenue.
It typically occurs after untruncation and forecasting.

PassengerSim offers several different optimization algorithms, including
probabilistic bid price (ProBP) optimization.

```yaml title="example.yaml" hl_lines="3 12"
rm_systems:
  basic_probp:
    availability_control: bp #(2)!
    processes:
      DCP:
      - step_type: untruncation
        algorithm: em
        kind: leg
      - step_type: forecast
        algorithm: additive_pickup
        kind: leg
      - step_type: probp #(1)!
```
{ .annotate }

1.  The `step_type` for probabilistic bid price optimization is `probp`, this is
    how PassengerSim identifies what to do in this step.
2.  To apply the ProBP optimization results, the `rm_system` should be set to use
    `bp` (bid price) availability controls.


::: passengersim.rm.ProBpStep
    options:
      show_root_heading: true
      show_root_full_path: false
      show_source: false
      members:
        - snapshot_filters
