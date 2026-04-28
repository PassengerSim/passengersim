# Forecasting

Forecasting is a key part of revenue management systems.  You need to know how
many customers of each type you should expect, so you can tailor the set of
products being offered to maximize revenue.

In PassengerSim, forecasting is included as a step within an RM system, typically
after untruncation and before any optimization.

```python title="E.py" hl_lines="10-16"
from passengersim.rm.emsr import ExpectedMarginalSeatRevenue
from passengersim.rm.standard_forecasting import StandardLegForecast
from passengersim.rm.systems import RmSys, RmSysOption, register_rm_system
from passengersim.rm.untruncation import LegUntruncation

@register_rm_system
class E(RmSys):
    actions = [
        LegUntruncation,
        StandardLegForecast.configure(  #(1)!
            algorithm=RmSysOption("forecast_algorithm", default="additive_pickup"),  #(2)!
            alpha=RmSysOption(
                "exp_smoothing_alpha", expected_type=float, default=0.15
            ),
        ),
        ExpectedMarginalSeatRevenue.configure(
            variant=RmSysOption("emsr_variant", default="b"),
        ),
    ]
```
{ .annotate }

1.  The forecaster (in this example, a standard leg forecast) is included as a
    step here.  Selected options are passed to it via the `configure` method,
    which allows options to be set from the RM system configuration.
2.  The configurable option name for the RM system is specified here as
    `"forecast_algorithm"`, to clarify its purpose. The attribute being controlled
    on the `StandardLegForecast` is simply `algorithm`, which is clear within the
    context of forecast step alone, but potentially ambiguous in the context of
    the RM system as a whole.


::: passengersim.rm.standard_forecasting.StandardLegForecast
    options:
      show_root_heading: true
      show_root_full_path: false
      show_source: false
      members:
        - algorithm
        - alpha

::: passengersim.rm.standard_forecasting.StandardPathForecast
    options:
      show_root_heading: true
      show_root_full_path: false
      show_source: false
      members:
        - algorithm
        - alpha
