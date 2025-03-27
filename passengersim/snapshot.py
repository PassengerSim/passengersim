from __future__ import annotations

from passengersim.config import SnapshotFilter, SnapshotInstruction  # noqa


def get_snapshot_instruction(
    sim,
    filters: list[SnapshotFilter] | None = None,
    only_type: str | None = None,
    break_on_first: bool = True,
    leg=None,
    path=None,
    carrier: str | None = None,
    debug: bool = False,
) -> SnapshotInstruction:
    """
    Check to see if this is a time we should emit snapshot outputs.

    Parameters
    ----------
    sim : SimulationEngine
    filters : list[SnapshotFilter], optional
        The list of filters to check against.  If not provided, the list of
        filters in `sim.snapshot_filters` is used.
    only_type : str, optional
        If set, only snapshot filters matching this type will be processed.
    break_on_first : bool, default True
        If set, only the first triggering snapshot filter will be applied.
        Usually there won't be more than one snapshot filter in a list that would
        trigger at the same point anyhow.
    leg : core.Leg, optional
        If provided, this Leg is used to check against the carrier, origin,
        destination, and flt_no values in each filter.
    path : core.Path, optional
        If provided, this Leg is used to check against the origin, destination,
        and flt_no values (for the first leg in the path) in each filter.
    carrier : str, optional
        If provided, this carrier used to check each filter.
    debug : bool, default False
        When this flag is set, snapshot output is activated and the resulting
        SnapshotInstruction will have no file set, so that the output is printed
        directly to the console (stdout), not a snapshot file.

    Returns
    -------
    SnapshotInstruction
    """
    if debug:
        return SnapshotInstruction(True)

    # resolve down from Simulation to SimulationEngine
    if hasattr(sim, "sim"):
        sim = sim.sim

    # the default return value of this
    snapshot_instruction = SnapshotInstruction(False)
    if filters is not None:
        snapshot_filters = filters
    else:
        snapshot_filters = sim.snapshot_filters
    if snapshot_filters:
        for sf in snapshot_filters:
            if only_type is not None:
                if sf.type == only_type:
                    # if we're only looking for a specific type, and we found it
                    snapshot_instruction = sf.run(
                        sim, leg=leg, path=path, carrier=carrier
                    )
                    if snapshot_instruction and break_on_first:
                        break
            else:
                snapshot_instruction = sf.run(sim, leg=leg, path=path, carrier=carrier)
                if snapshot_instruction and break_on_first:
                    break
    return snapshot_instruction
