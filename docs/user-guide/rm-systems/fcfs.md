# FCFS Allocation

First come first served (FCFS) is a simple method for allocating capacity to
customers, and it operates pretty much as you would expect: customers whom arrive
first are offered products, no attempt is made to optimize for anything.

This process of capacity allocation will also occur if no RM optimization algorithm
is applied, but the explicit FCFS step type allow the user to be intentional
about selecting this algorithm, and to attach snapshot filters to the simulation
if desired.

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
      - step_type: fcfs #(1)!
```
{ .annotate }

1.  The `step_type` for first come, first served is `fcfs`.


::: passengersim.rm.FcfsStep
    options:
      show_root_heading: true
      show_root_full_path: false
      show_source: false
      members:
        - snapshot_filters
