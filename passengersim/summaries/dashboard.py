import ipywidgets as widgets
from IPython import display

from . import SimulationTables


def default_dashboard(summary: SimulationTables):
    out1 = widgets.Output()
    out2 = widgets.Output()
    out1.clear_output()
    out2.clear_output()
    with out1:
        display.display(summary.fig_carrier_revenues().properties(width=300))
        display.display(summary.fig_carrier_load_factors().properties(width=300))
        display.display(summary.fig_carrier_rasm().properties(width=300))
    with out2:
        display.display(summary.fig_fare_class_mix().properties(width=300))
    return widgets.HBox([out1, out2])
