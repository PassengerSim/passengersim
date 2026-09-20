# Schedule Editor

PassengerSim includes an interactive flight-leg schedule editor for Jupyter notebooks. It uses the
same hub-centered layout as `fig_hub_schedule`, while allowing each leg to be dragged left or right.
A move shifts departure and arrival by the same amount, so flight duration remains unchanged.

The editor always makes a deep copy of the supplied configuration. There is no in-place editing
mode, and the source configuration is never modified.

## Create an editor

Install PassengerSim with its `lab` extra so the Jupyter widget dependencies are available, then
create and display an editor:

```python
import passengersim as pax
from passengersim.config.editor import edit_hub_schedule

cfg = pax.Config.from_yaml(pax.demo_network("3MKT"))
editor = edit_hub_schedule(cfg, "ORD", snap_minutes=5)
editor
```

Drag a flight line horizontally to revise its schedule. The toolbar can change the snapping interval
and undo, redo, or reset edits. Clicking a flight without moving it highlights connections from the
path set that existed when the editor was created. These connections remain static during editing.

Use `changes` to inspect the current revisions:

```python
editor.changes
```

The table contains the original and edited local times and the signed change in minutes for each
modified leg.

## Color legs by tags

The `coloring` argument follows the hub-schedule figure's category selectors and adds support for
leg tags:

```python
# Color by carrier (the default).
editor = edit_hub_schedule(cfg, "ORD", coloring="carrier")

# Color by spoke airport or by a tag on each spoke Place.
editor = edit_hub_schedule(cfg, "ORD", coloring="spoke")
editor = edit_hub_schedule(cfg, "ORD", coloring="spoke.region")

# Color by a tag stored on each Leg.
editor = edit_hub_schedule(cfg, "ORD", coloring="leg.fleet")
```

Categories have stable colors and appear in a compact, horizontally scrollable legend. Legs without
the requested tag use a separate `(missing)` category, so a partially tagged schedule remains
editable and makes its missing metadata visible.

## Control row density

Rows use 6 pixels of vertical spacing by default. This allows at least 80 legs to appear
simultaneously in the default 600-pixel schedule viewport. Dense rows omit individual text labels to
prevent overlap; hover over a line for its leg, route, times, color category, duration, and
capacity. The editor-owned tooltip appears immediately on hover, without the delay of a
browser-native tooltip.

Set `row_height` from 5 through 24 pixels when a different balance is useful. At 12 pixels and
above, the editor also displays a text label beside each leg:

```python
editor = edit_hub_schedule(cfg, "ORD", row_height=14)
editor
```

## Finish and rebuild connections

The primary **Finish & rebuild connections** action replaces paths using the edited schedule.
Rebuilding occurs only when editing is finished, never during a drag. After the action completes,
retrieve the copied configuration with:

```python
edited_cfg = editor.edited_config
```

To finish without rebuilding, select **Skip connection rebuild** before clicking the Finish action.
This choice is intentionally explicit because the retained paths may no longer satisfy
connection-time rules.

The equivalent programmatic operations are:

```python
# Default: finish and replace connections.
edited_cfg = editor.finish()

# Explicitly retain the existing static paths instead.
edited_cfg = editor.finish(rebuild_connections=False)
```

Calling `finish` disables further schedule changes. If connection rebuilding fails, the error is
shown in the editor and editing remains available so the schedule can be adjusted or the operation
retried.

## Filter by carrier

Use `carrier` to display and edit only one carrier's legs at the hub:

```python
editor = edit_hub_schedule(cfg, "ORD", carrier="AL1")
editor
```

Legs excluded by this filter remain unchanged in the copied configuration.
