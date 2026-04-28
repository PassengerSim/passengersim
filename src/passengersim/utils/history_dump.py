import os

from passengersim_core import SimulationEngine

from passengersim.driver import Simulation
from passengersim.utils.compression import deserialize_from_file, serialize_to_file


def dump_history(sim: Simulation | SimulationEngine, filename: str | None = None) -> dict:
    """Dump the history of all pathclasses and buckets to a dict."""
    if isinstance(sim, SimulationEngine):
        eng = sim
    elif hasattr(sim, "eng") and isinstance(sim.eng, SimulationEngine):
        eng = sim.eng
    else:
        raise TypeError("sim must be a SimulationEngine or a Simulation object.")
    if not isinstance(eng, SimulationEngine):
        raise TypeError("eng must be a SimulationEngine or a Simulation object.")
    dump = {}
    dump["sample"] = eng.sample
    dump["paths"] = {}
    for path in eng.paths:
        path_data = dump["paths"][path.path_id] = {}
        for pc in path.pathclasses:
            path_data[pc.booking_class] = pc.history.as_arrays()
    dump["legs"] = {}
    for leg in eng.legs:
        leg_data = dump["legs"][leg.leg_id] = {}
        for bkt in leg.buckets:
            leg_data[bkt.booking_class] = bkt.history.as_arrays()
    if filename:
        serialize_to_file(filename, dump)
    return dump


def load_history(sim: Simulation | SimulationEngine, dump: dict | str | os.PathLike):
    """Load the history of all pathclasses and buckets from a dict."""
    if isinstance(sim, SimulationEngine):
        eng = sim
    elif hasattr(sim, "eng") and isinstance(sim.eng, SimulationEngine):
        eng = sim.eng
    else:
        raise TypeError("sim must be a SimulationEngine or a Simulation object.")
    if not isinstance(eng, SimulationEngine):
        raise TypeError("eng must be a SimulationEngine or a Simulation object.")
    if isinstance(dump, str | os.PathLike):
        dump = deserialize_from_file(dump)
    for path_id, path_data in dump["paths"].items():
        path = eng.paths.select(path_id=path_id)
        for booking_class, data in path_data.items():
            pc = path.pathclasses.select(booking_class=booking_class)
            pc.history.from_arrays(data)
    for leg_id, leg_data in dump["legs"].items():
        leg = eng.legs.select(leg_id=leg_id)
        for booking_class, data in leg_data.items():
            bkt = leg.buckets.select(booking_class=booking_class)
            bkt.history.from_arrays(data)
    eng.sample = dump["sample"]
