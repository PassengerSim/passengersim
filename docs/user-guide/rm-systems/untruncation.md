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

```yaml title="example.yaml" hl_lines="5-7"
rm_systems:
  basic_emsr_b: #(4)!
    processes:
      DCP:
      - step_type: untruncation #(1)!
        algorithm: em #(2)!
        kind: leg #(3)!
      - step_type: forecast
        algorithm: additive_pickup
        kind: leg
      - step_type: emsr
        algorithm: b
        kind: leg
```
{ .annotate }

1.  The `step_type` must be `untruncation`, as this is how PassengerSim
    identifies what to do in this step.
2.  Several different algorithms are available for untruncation, see
    [below][passengersim.rm.UntruncationStep.algorithm] for details.
3.  Untruncation can be done at the leg or path level, see
    [below][passengersim.rm.UntruncationStep.kind] for details.
4.  This is showing that `basic_emsr_b` is the name of this RM system.
    Elsewhere in the configuration (not shown in this example snippet) you will
    define carriers, and each will be assigned an RM system using these names.


::: passengersim.rm.UntruncationStep
    options:
      show_root_heading: true
      show_root_full_path: false
      show_source: false
      members:
        - algorithm
        - kind
