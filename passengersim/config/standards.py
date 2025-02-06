import sys
import textwrap

import addicty
import yaml

from passengersim import demo_network


def standard_rm_systems_raw() -> addicty.Dict:
    with open(demo_network("standard-rm-systems.yaml")) as f:
        d = addicty.Dict.load(f, freeze=False, Loader=yaml.CSafeLoader)
    if "rm_systems" not in d:
        d.rm_systems = addicty.Dict()
    return d.rm_systems


def describe_standard_rm_systems(stream=sys.stdout) -> str | None:
    msg = ""
    d = standard_rm_systems_raw()
    for k in d:
        msg += f"{k}:\n"
        msg += textwrap.fill(
            d[k].get("description", "no description available"),
            70,
            initial_indent="  ",
            subsequent_indent="  ",
        )
        msg += "\n"
    if stream is None:
        return msg
    print(msg, file=stream)
