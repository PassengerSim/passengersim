from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import pandas as pd

from .generic import SimulationTable_add_item

if TYPE_CHECKING:
    from passengersim import Simulation

    from . import SimulationTables


def extract_carriers(sim: Simulation) -> pd.DataFrame:
    """Extract carrier-level summary data from a Simulation."""
    eng = sim.sim
    num_samples = eng.num_trials_completed * (eng.num_samples - eng.burn_samples)

    carrier_asm = defaultdict(float)
    carrier_rpm = defaultdict(float)
    carrier_leg_lf = defaultdict(float)
    carrier_leg_count = defaultdict(float)
    for leg in eng.legs:
        carrier_name = leg.carrier.name
        carrier_asm[carrier_name] += leg.distance * leg.capacity * num_samples
        carrier_rpm[carrier_name] += leg.distance * leg.gt_sold
        carrier_leg_lf[carrier_name] += leg.gt_sold / (leg.capacity * num_samples)
        carrier_leg_count[carrier_name] += 1

    carrier_data = []
    for carrier in sim.sim.carriers:
        avg_rev = carrier.gt_revenue / num_samples
        rpm = carrier_rpm[carrier.name] / num_samples
        avg_leg_lf = (
            100 * carrier_leg_lf[carrier.name] / max(carrier_leg_count[carrier.name], 1)
        )
        # Add up total ancillaries
        tot_anc_rev = 0.0
        for anc in carrier.ancillaries:
            tot_anc_rev += anc.price * anc.sold
        carrier_data.append(
            {
                "name": carrier.name,
                "control": carrier.control,
                "avg_rev": avg_rev,
                "avg_sold": carrier.gt_sold / num_samples,
                "truncation_rule": carrier.truncation_rule,
                "avg_leg_lf": avg_leg_lf,
                "asm": carrier_asm[carrier.name] / num_samples,
                "rpm": rpm,
                "ancillary_rev": tot_anc_rev,
            }
        )
    return pd.DataFrame(carrier_data).set_index("name")


def aggregate_carriers(summaries: list[SimulationTables]) -> pd.DataFrame | None:
    """Aggregate leg-level summaries."""
    table_avg = []
    for s in summaries:
        frame = s._raw_carriers
        if frame is not None:
            table_avg.append(
                frame.set_index(["control", "truncation_rule"], append=True)
            )
    n = len(table_avg)
    while len(table_avg) > 1:
        table_avg[0] = table_avg[0].add(table_avg.pop(1), fill_value=0)
    if table_avg:
        table_avg[0] /= n
        return table_avg[0].reset_index(["control", "truncation_rule"])
    return None


SimulationTable_add_item(
    "carriers",
    aggregation_func=aggregate_carriers,
    extraction_func=extract_carriers,
    computed_fields={
        "avg_price": "avg_rev / avg_sold",
        "yield": "avg_rev / rpm",
        "sys_lf": "100.0 * rpm / asm",
    },
    doc="Carrier-level summary data.",
)
