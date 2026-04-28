import pathlib
from typing import Any

import dill as pickle

from passengersim.core import Bucket, Leg, Path, PathClass, SimulationEngine
from passengersim.utils.compression import deserialize_from_bytes, serialize_to_bytes


def _collect_counters(obj):
    return obj._get_counters()


def _restore_counters(obj, counters: dict[str, int | float]):
    obj._set_counters(counters)


def _save_byclass(bkt: Bucket | PathClass) -> dict[str, Any]:
    data = {}
    data["counters"] = _collect_counters(bkt)
    data["history"] = pickle.dumps(bkt.history)
    return data


def _restore_byclass(bkt: Bucket | PathClass, data: dict) -> None:
    _restore_counters(bkt, data["counters"])
    bkt._adopt_history = pickle.loads(data["history"])


def _save_withclass(obj: Leg | Path):
    data = {
        "counters": _collect_counters(obj),
        "byclass": {bkt.booking_class: _save_byclass(bkt) for bkt in obj._byclass},
    }
    # Q History if stored
    try:
        q_history = obj.q_forecast.history
    except AttributeError:
        pass
    else:
        data["q_history"] = pickle.dumps(q_history)
    return data


def _restore_withclass(obj: Leg | Path, data: dict):
    _restore_counters(obj, data["counters"])
    for b, v in data["byclass"].items():
        _restore_byclass(obj._byclass.select(booking_class=b), v)


def serialize_dynamic_state(
    eng: SimulationEngine, *, b64: bool = False, filename: str | pathlib.Path | None = None
) -> bytes:
    data = {}
    data["legs"] = {leg.leg_id: _save_withclass(leg) for leg in eng.legs}
    data["paths"] = {pth.path_id: _save_withclass(pth) for pth in eng.paths}
    data["sim"] = {"counters": _collect_counters(eng)}
    out = serialize_to_bytes(data, b64=b64)
    if filename is not None:
        filename = pathlib.Path(filename)
        filename.parent.mkdir(parents=True, exist_ok=True)
        filename.write_bytes(out)
    return out


def restore_dynamic_state(
    eng: SimulationEngine, state: bytes | None = None, filename: str | pathlib.Path | None = None
):
    if state is None:
        if filename is None:
            raise ValueError("must provide state or filename")
        filename = pathlib.Path(filename)
        state = filename.read_bytes()
    data = deserialize_from_bytes(state)
    _restore_counters(eng, data.get("sim", {}).get("counters", {}))
    for leg in eng.legs:
        leg_data = data["legs"].get(leg.leg_id)
        if leg_data is not None:
            _restore_withclass(leg, leg_data)
    for pth in eng.paths:
        pth_data = data["paths"].get(pth.path_id)
        if pth_data is not None:
            _restore_withclass(pth, pth_data)
