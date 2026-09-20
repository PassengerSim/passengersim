# Hub Capacity and Bank Editor

PassengerSim includes an interactive Jupyter widget for assigning carrier legs to hub banks and
editing their capacities. Its schedule uses the same dense, hub-centered layout and `coloring`
selectors as the schedule editor.

The editor always makes a deep copy of the supplied configuration. It has no in-place mode, and
finishing returns the edited copy without rebuilding connections.

## Create an editor

`carrier` is required. Only that carrier's legs arriving at or departing from the hub are visible
and editable:

```python
import passengersim as pax
from passengersim.config.editor import edit_hub_capacity

cfg = pax.Config.from_yaml(pax.demo_network("3MKT"))
editor = edit_hub_capacity(cfg, "ORD", carrier="AL1")
editor
```

Hidden legs are outside the editor's scope. Their bank tags are not highlighted, included in totals,
or changed—even when a hidden leg has the same bank name as a visible leg.

## Create and edit banks

A bank name is stored in a leg tag. The tag key defaults to `bank` and can be changed with
`bank_tag`:

```python
editor = edit_hub_capacity(
    cfg,
    "ORD",
    carrier="AL1",
    bank_tag="schedule_bank",
)
```

To create a bank, enter its name, select **New**, click each leg that should be included, and select
**Save members**. To revise a bank, activate it and select **Edit members** before toggling legs.
Membership changes are staged until saved, so **Cancel** leaves all tags unchanged.

Each visible leg can have only one value for the bank tag, making banks mutually exclusive. A leg
does not need to belong to any bank. Assigning a leg to a new bank replaces its previous bank tag
value.

Clicking a leg outside membership-editing mode selects it and activates its current bank. All
visible members of that bank are highlighted, and the editor shows separate total capacities for
arrivals at the hub and departures from the hub. Bank creation, renaming, deletion, and membership
changes can all be undone and redone.

The equivalent Python operations are available on the widget:

```python
editor.new_bank("morning")
editor.toggle_bank_leg(101)
editor.toggle_bank_leg(111)
editor.save_bank()

editor.activate_bank("morning")
editor.active_bank_totals
```

## Edit capacity

Select a leg and choose **Edit capacity**. For a leg whose capacity is a single integer, the pop-up
includes the carrier's capacity levels discovered when the editor was created, ordered from smallest
to largest. You can choose one of those presets or enter any nonnegative integer. The preset list
remains fixed during the editing session.

For a leg with cabin-specific capacity, such as `{"Y": 150, "J": 12}`, the pop-up instead displays a
separate integer field for every existing cabin. It does not display total-capacity presets. Cabin
keys cannot be added or removed through the editor.

Select **Delete leg** in the capacity pop-up to mark the selected leg for deletion. This sets a
scalar capacity to zero or sets every cabin capacity to zero. You can obtain the same result by
entering zero directly. The marked leg remains visible with a dashed line and can be restored with
**Undo** or by entering a positive capacity until editing is finished.

The programmatic interface applies the same validation:

```python
# Scalar capacity
editor.set_capacity(101, 110)

# Existing per-cabin capacity
editor.set_capacity(102, {"Y": 150, "J": 12})

# Mark a leg for deletion at finalization
editor.set_capacity(103, 0)
```

When a changed leg belongs to the active bank, arrival or departure capacity totals update
immediately. Use `changes` to inspect all current visible-leg bank and capacity differences from the
supplied configuration:

```python
editor.changes
```

## Color and size the display

The `coloring` argument accepts the same selectors as the schedule editor:

```python
edit_hub_capacity(cfg, "ORD", carrier="AL1", coloring="carrier")
edit_hub_capacity(cfg, "ORD", carrier="AL1", coloring="spoke")
edit_hub_capacity(cfg, "ORD", carrier="AL1", coloring="spoke.region")
edit_hub_capacity(cfg, "ORD", carrier="AL1", coloring="leg.fleet")
```

Rows use 6 pixels of vertical spacing by default, allowing at least 80 legs in the default viewport.
Use `row_height` from 5 through 24 pixels to change the density. Leg details appear immediately when
the pointer enters a flight line.

## Finish editing

Select **Finish editing**, or call `finish`, after saving or canceling any staged bank membership
changes:

```python
edited_cfg = editor.finish()
```

The returned object is the editor's independent configuration copy. The source `cfg` remains
unchanged, and further editor changes are disabled. Every visible carrier-filtered hub leg whose
total capacity is zero is removed from the returned copy. For a cabin-specific leg, this means all
cabin capacities must be zero. Hidden legs remain untouched even when they have zero capacity.
