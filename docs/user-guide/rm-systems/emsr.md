# EMSR Optimization

Optimization is the most fundamental part of revenue management systems, is it
is the process used to tailor the set of products being offered to maximize revenue.
It typically occurs after untruncation and forecasting.

PassengerSim offers several different optimization algorithms. One widely used
algorithm is called EMSR (expected marginal seat revenue), which has a few variants,
generally labels as "A", "B", and "C".

```yaml title="example.yaml" hl_lines="11-13"
rm_systems:
  basic_emsr_b:
    processes:
      DCP:
      - step_type: untruncation
        algorithm: em
        kind: leg
      - step_type: forecast
        algorithm: additive_pickup
        kind: leg
      - step_type: emsr #(1)!
        algorithm: b
        kind: leg
```
{ .annotate }

1.  The `step_type` for EMSR optimization is `emsr`, this is how PassengerSim
    identifies what to do in this step.


::: passengersim.rm.EmsrStep
    options:
      show_root_heading: true
      show_root_full_path: false
      show_source: false
      members:
        - algorithm
        - kind
        - snapshot_filters
