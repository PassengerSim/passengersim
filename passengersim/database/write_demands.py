from __future__ import annotations

from passengersim_core import SimulationEngine

from passengersim.database import Database


def save_demand_to_database(cnx: Database, sim: SimulationEngine, dcp: int):
    """Write simulation demands to the database."""
    data_list = []
    for dmd in sim.demands:
        data_list.append(
            (
                sim.name,
                sim.iteration,
                sim.trial,
                sim.sample,
                dcp,
                dmd.orig,
                dmd.dest,
                dmd.segment,
                dmd.scenario_demand,
                dmd.sold,
                dmd.revenue,
            )
        )
    cursor = cnx.cursor()
    sql = """
    INSERT INTO demand_detail (
        scenario,       -- 1
        iteration,      -- 2
        trial,          -- 3
        sample,         -- 4
        days_prior,     -- 5
        orig,           -- 6
        dest,           -- 7
        segment,        -- 8
        sample_demand,  -- 9
        sold,           -- 10
        revenue         -- 11
    ) VALUES """ + cnx.sql_placeholders(11)
    cursor.executemany(sql, data_list)
