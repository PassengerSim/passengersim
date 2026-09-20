const SVG_NS = "http://www.w3.org/2000/svg";

function svgElement(name, attributes = {}) {
  const element = document.createElementNS(SVG_NS, name);
  for (const [key, value] of Object.entries(attributes)) {
    element.setAttribute(key, String(value));
  }
  return element;
}

function render({ model, el }) {
  el.classList.add("passengersim-schedule-editor", "passengersim-capacity-editor");
  el.innerHTML = `
    <section class="pse-shell" aria-label="Hub capacity and bank editor">
      <header class="pse-header">
        <div>
          <h3 class="pse-title"></h3>
          <div class="pse-subtitle">Select a leg to activate its bank, edit its capacity, or delete it.</div>
        </div>
        <div class="pse-toolbar" role="toolbar" aria-label="Capacity editing controls">
          <button type="button" class="pse-undo">Undo</button>
          <button type="button" class="pse-redo">Redo</button>
          <button type="button" class="pse-reset">Reset</button>
        </div>
      </header>
      <div class="pse-status" role="status" aria-live="polite">
        <span class="pse-status-message"></span>
        <span class="pse-modified"></span>
      </div>
      <div class="pse-legend" aria-label="Schedule color legend"></div>
      <section class="pce-bank-panel" aria-label="Bank controls">
        <div class="pce-bank-controls">
          <label>Active bank
            <select class="pce-bank-select" aria-label="Active bank"></select>
          </label>
          <label>Bank name
            <input class="pce-bank-name" type="text" autocomplete="off" aria-label="Bank name">
          </label>
          <div class="pce-bank-buttons">
            <button type="button" class="pce-new-bank">New</button>
            <button type="button" class="pce-edit-bank">Edit members</button>
            <button type="button" class="pce-rename-bank">Rename</button>
            <button type="button" class="pce-delete-bank">Delete</button>
            <button type="button" class="pce-save-bank">Save members</button>
            <button type="button" class="pce-cancel-bank">Cancel</button>
          </div>
        </div>
        <div class="pce-bank-totals" aria-live="polite">
          <span class="pce-total-placeholder">No bank selected</span>
          <span class="pce-arrival-total"></span>
          <span class="pce-departure-total"></span>
        </div>
      </section>
      <div class="pse-chart-viewport">
        <div class="pse-chart"></div>
      </div>
      <div class="pse-tooltip" role="tooltip" hidden></div>
      <section class="pce-leg-inspector" aria-label="Selected leg capacity">
        <div>
          <div class="pce-inspector-label">Selected leg</div>
          <div class="pce-selected-summary">Click a leg to select it.</div>
        </div>
        <div class="pce-capacity-summary" hidden>
          <span class="pce-capacity-value"></span>
          <button type="button" class="pce-edit-capacity">Edit capacity</button>
        </div>
      </section>
      <div class="pce-capacity-popover" role="dialog" aria-modal="false" aria-label="Edit leg capacity" hidden></div>
      <footer class="pse-footer">
        <button type="button" class="pse-finish">Finish editing</button>
      </footer>
    </section>
  `;

  const root = el.querySelector(".pse-shell");
  const title = root.querySelector(".pse-title");
  const chart = root.querySelector(".pse-chart");
  const viewport = root.querySelector(".pse-chart-viewport");
  const undo = root.querySelector(".pse-undo");
  const redo = root.querySelector(".pse-redo");
  const reset = root.querySelector(".pse-reset");
  const statusMessage = root.querySelector(".pse-status-message");
  const modified = root.querySelector(".pse-modified");
  const legend = root.querySelector(".pse-legend");
  const bankSelect = root.querySelector(".pce-bank-select");
  const bankName = root.querySelector(".pce-bank-name");
  const newBank = root.querySelector(".pce-new-bank");
  const editBank = root.querySelector(".pce-edit-bank");
  const renameBank = root.querySelector(".pce-rename-bank");
  const deleteBank = root.querySelector(".pce-delete-bank");
  const saveBank = root.querySelector(".pce-save-bank");
  const cancelBank = root.querySelector(".pce-cancel-bank");
  const totalPlaceholder = root.querySelector(".pce-total-placeholder");
  const arrivalTotal = root.querySelector(".pce-arrival-total");
  const departureTotal = root.querySelector(".pce-departure-total");
  const selectedSummary = root.querySelector(".pce-selected-summary");
  const capacitySummary = root.querySelector(".pce-capacity-summary");
  const capacityValue = root.querySelector(".pce-capacity-value");
  const editCapacity = root.querySelector(".pce-edit-capacity");
  const capacityPopover = root.querySelector(".pce-capacity-popover");
  const legTooltip = root.querySelector(".pse-tooltip");
  const finish = root.querySelector(".pse-finish");
  let eventSequence = 0;

  function sendEvent(event) {
    eventSequence += 1;
    model.set("_event", {
      ...event,
      nonce: `${Date.now()}-${eventSequence}`,
    });
    model.save_changes();
  }

  function formatTime(timestamp, timezone, includeDate = true) {
    const options = {
      timeZone: timezone,
      hour: "2-digit",
      minute: "2-digit",
      hour12: false,
    };
    if (includeDate) {
      options.weekday = "short";
      options.month = "short";
      options.day = "numeric";
    }
    return new Intl.DateTimeFormat("en-US", options).format(new Date(timestamp * 1000));
  }

  function displayCapacity(capacity) {
    if (capacity !== null && typeof capacity === "object") {
      return Object.entries(capacity)
        .map(([cabin, seats]) => `${cabin}: ${seats}`)
        .join(" · ");
    }
    return `${capacity}`;
  }

  function positionTooltip(clientX, clientY) {
    const margin = 8;
    const offset = 12;
    const rootBounds = root.getBoundingClientRect();
    const localX = clientX - rootBounds.left;
    const localY = clientY - rootBounds.top;
    let left = localX + offset;
    let top = localY + offset;
    if (
      rootBounds.left + left + legTooltip.offsetWidth > window.innerWidth - margin ||
      left + legTooltip.offsetWidth > rootBounds.width - margin
    ) {
      left = localX - legTooltip.offsetWidth - offset;
    }
    if (rootBounds.top + top + legTooltip.offsetHeight > window.innerHeight - margin) {
      top = localY - legTooltip.offsetHeight - offset;
    }
    const maxLeft = Math.max(margin, rootBounds.width - legTooltip.offsetWidth - margin);
    legTooltip.style.left = `${Math.min(maxLeft, Math.max(margin, left))}px`;
    legTooltip.style.top = `${Math.max(margin, top)}px`;
  }

  function showLegTooltip(text, legId, clientX, clientY) {
    legTooltip.textContent = text;
    legTooltip.dataset.legId = String(legId);
    legTooltip.hidden = false;
    positionTooltip(clientX, clientY);
  }

  function hideLegTooltip() {
    legTooltip.hidden = true;
    delete legTooltip.dataset.legId;
  }

  function hideCapacityPopover() {
    capacityPopover.hidden = true;
    capacityPopover.replaceChildren();
  }

  function positionCapacityPopover() {
    const anchor = editCapacity.getBoundingClientRect();
    const rootBounds = root.getBoundingClientRect();
    const margin = 8;
    const maxLeft = Math.max(margin, rootBounds.width - capacityPopover.offsetWidth - margin);
    const left = Math.min(maxLeft, Math.max(margin, anchor.left - rootBounds.left));
    let top = anchor.bottom - rootBounds.top + 6;
    if (rootBounds.top + top + capacityPopover.offsetHeight > window.innerHeight - margin) {
      top = anchor.top - rootBounds.top - capacityPopover.offsetHeight - 6;
    }
    capacityPopover.style.left = `${left}px`;
    capacityPopover.style.top = `${Math.max(margin, top)}px`;
  }

  function showCapacityPopover() {
    const legs = model.get("_legs_data") || [];
    const state = model.get("_state") || {};
    const settings = model.get("_settings") || {};
    const leg = legs.find((item) => item.leg_id === state.selected_leg_id);
    if (!leg) return;

    capacityPopover.replaceChildren();
    const heading = document.createElement("div");
    heading.className = "pce-popover-heading";
    heading.textContent = `${leg.carrier}${leg.fltno} · ${leg.orig} → ${leg.dest}`;
    capacityPopover.append(heading);

    const form = document.createElement("form");
    form.className = "pce-capacity-form";
    let collectCapacity;

    if (leg.capacity_kind === "cabins") {
      const cabinFields = document.createElement("div");
      cabinFields.className = "pce-cabin-fields";
      for (const [cabin, seats] of Object.entries(leg.capacity)) {
        const label = document.createElement("label");
        label.textContent = cabin;
        const input = document.createElement("input");
        input.type = "number";
        input.min = "0";
        input.step = "1";
        input.required = true;
        input.value = String(seats);
        input.dataset.cabin = cabin;
        label.append(input);
        cabinFields.append(label);
      }
      form.append(cabinFields);
      collectCapacity = () =>
        Object.fromEntries(
          [...cabinFields.querySelectorAll("input")].map((input) => [
            input.dataset.cabin,
            Number(input.value),
          ]),
        );
    } else {
      const presets = document.createElement("div");
      presets.className = "pce-capacity-presets";
      const presetLabel = document.createElement("span");
      presetLabel.textContent = "Common carrier capacities";
      presets.append(presetLabel);

      const inputLabel = document.createElement("label");
      inputLabel.className = "pce-scalar-capacity";
      inputLabel.textContent = "Capacity";
      const input = document.createElement("input");
      input.type = "number";
      input.min = "0";
      input.step = "1";
      input.required = true;
      input.value = String(leg.capacity);
      inputLabel.append(input);

      for (const preset of settings.common_capacities || []) {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "pce-preset";
        button.textContent = String(preset);
        button.addEventListener("click", () => {
          input.value = String(preset);
          input.focus();
        });
        presets.append(button);
      }
      form.append(presets, inputLabel);
      collectCapacity = () => Number(input.value);
    }

    const actions = document.createElement("div");
    actions.className = "pce-popover-actions";
    const cancel = document.createElement("button");
    cancel.type = "button";
    cancel.textContent = "Cancel";
    const deleteLeg = document.createElement("button");
    deleteLeg.type = "button";
    deleteLeg.className = "pce-delete-leg";
    deleteLeg.textContent = "Delete leg";
    deleteLeg.title = "Set capacity to zero; the leg will be removed when editing finishes";
    const apply = document.createElement("button");
    apply.type = "submit";
    apply.className = "pce-apply-capacity";
    apply.textContent = "Apply";
    actions.append(deleteLeg, cancel, apply);
    form.append(actions);
    capacityPopover.append(form);
    capacityPopover.hidden = false;
    positionCapacityPopover();
    form.querySelector("input")?.focus();

    cancel.addEventListener("click", hideCapacityPopover);
    deleteLeg.addEventListener("click", () => {
      const zeroCapacity =
        leg.capacity_kind === "cabins"
          ? Object.fromEntries(Object.keys(leg.capacity).map((cabin) => [cabin, 0]))
          : 0;
      sendEvent({ type: "set_capacity", leg_id: leg.leg_id, capacity: zeroCapacity });
      hideCapacityPopover();
    });
    form.addEventListener("submit", (event) => {
      event.preventDefault();
      if (!form.reportValidity()) return;
      sendEvent({ type: "set_capacity", leg_id: leg.leg_id, capacity: collectCapacity() });
      hideCapacityPopover();
    });
  }

  function renderLegend() {
    const settings = model.get("_settings") || {};
    const entries = settings.legend || [];
    legend.replaceChildren();
    legend.hidden = entries.length === 0;
    if (!entries.length) return;

    const legendTitle = document.createElement("span");
    legendTitle.className = "pse-legend-title";
    legendTitle.textContent = `${settings.coloring_title || "Color"}:`;
    legend.append(legendTitle);
    for (const entry of entries) {
      const item = document.createElement("span");
      item.className = "pse-legend-item";
      const swatch = document.createElement("span");
      swatch.className = "pse-legend-swatch";
      swatch.style.backgroundColor = entry.color;
      const label = document.createElement("span");
      label.textContent = entry.label;
      item.append(swatch, label);
      legend.append(item);
    }
  }

  function renderControls() {
    const legs = model.get("_legs_data") || [];
    const settings = model.get("_settings") || {};
    const state = model.get("_state") || {};
    const status = model.get("_status") || {};
    const finished = Boolean(status.finished);
    const membershipEditing = Boolean(state.membership_editing);
    const activeBank = state.active_bank ?? "";

    title.textContent = `${settings.carrier} Hub Capacity at ${settings.hub}`;
    statusMessage.textContent = status.message || "";
    root.dataset.phase = status.phase || "editing";
    root.dataset.membershipEditing = membershipEditing ? "true" : "false";
    modified.textContent = `${status.modified_count || 0} modified`;

    bankSelect.replaceChildren();
    const noBank = document.createElement("option");
    noBank.value = "";
    noBank.textContent = "(No active bank)";
    bankSelect.append(noBank);
    const bankChoices = [...(state.banks || [])];
    if (activeBank && !bankChoices.includes(activeBank)) bankChoices.push(activeBank);
    for (const bank of bankChoices.sort()) {
      const option = document.createElement("option");
      option.value = bank;
      option.textContent = bank;
      bankSelect.append(option);
    }
    bankSelect.value = activeBank;
    if (document.activeElement !== bankName) bankName.value = activeBank;

    undo.disabled = !status.can_undo || finished;
    redo.disabled = !status.can_redo || finished;
    reset.disabled = finished || !(status.modified_count > 0);
    bankSelect.disabled = finished || membershipEditing;
    bankName.disabled = finished || membershipEditing;
    newBank.disabled = finished || membershipEditing;
    editBank.disabled = finished || membershipEditing || !activeBank;
    renameBank.disabled = finished || membershipEditing || !activeBank;
    deleteBank.disabled = finished || membershipEditing || !activeBank;
    saveBank.hidden = !membershipEditing;
    cancelBank.hidden = !membershipEditing;
    saveBank.disabled = finished || !membershipEditing;
    cancelBank.disabled = finished || !membershipEditing;
    finish.disabled = finished || membershipEditing;

    const totals = state.totals || {};
    const hasActiveBank = Boolean(activeBank);
    totalPlaceholder.hidden = hasActiveBank;
    arrivalTotal.hidden = !hasActiveBank;
    departureTotal.hidden = !hasActiveBank;
    if (hasActiveBank) {
      arrivalTotal.textContent = `Arrivals: ${totals.arrival_capacity || 0} seats (${totals.arrival_legs || 0} legs)`;
      departureTotal.textContent = `Departures: ${totals.departure_capacity || 0} seats (${totals.departure_legs || 0} legs)`;
    }

    const selected = legs.find((leg) => leg.leg_id === state.selected_leg_id);
    if (selected) {
      selectedSummary.textContent = `${selected.carrier}${selected.fltno} · ${selected.orig} → ${selected.dest}`;
      const deletionLabel = selected.pending_deletion ? " · Will be removed when finished" : "";
      capacityValue.textContent =
        `Capacity: ${displayCapacity(selected.capacity)} (total ${selected.total_capacity})${deletionLabel}`;
      capacitySummary.hidden = false;
      editCapacity.disabled = finished;
    } else {
      selectedSummary.textContent = "Click a leg to select it.";
      capacitySummary.hidden = true;
      hideCapacityPopover();
    }

    renderLegend();
  }

  function renderChart() {
    hideLegTooltip();
    const legs = (model.get("_legs_data") || []).map((leg) => ({ ...leg }));
    const settings = model.get("_settings") || {};
    const state = model.get("_state") || {};
    const status = model.get("_status") || {};
    const interactionDisabled = Boolean(status.finished);
    const preferredWidth = Number(settings.width || 900);
    const availableWidth = Math.floor(chart.getBoundingClientRect().width || preferredWidth);
    const width = Math.max(560, Math.min(preferredWidth, availableWidth || preferredWidth));
    const rowHeight = Number(settings.row_height || 6);
    const showLabels = rowHeight >= 12;
    const axisY = 32;
    const firstRowY = 48;
    const height = Math.max(
      Number(settings.min_height || 180),
      firstRowY + Math.max(0, legs.length - 1) * rowHeight + 8,
    );
    const left = showLabels ? 132 : 16;
    const right = 16;
    const plotWidth = Math.max(240, width - left - right);
    const domainStart = Number(settings.domain_start);
    const domainEnd = Number(settings.domain_end);
    const timezone = settings.hub_timezone || "UTC";
    const hasActiveBank = state.active_bank !== null && state.active_bank !== undefined;

    viewport.style.minHeight = `${settings.min_height || 180}px`;
    viewport.style.maxHeight = `${settings.max_height || 600}px`;
    chart.innerHTML = "";

    if (!legs.length) {
      const empty = document.createElement("div");
      empty.className = "pse-empty";
      empty.textContent = `No ${settings.carrier} legs arrive at or depart from ${settings.hub}.`;
      chart.append(empty);
      return;
    }

    const svg = svgElement("svg", {
      class: "pse-svg",
      width,
      height,
      viewBox: `0 0 ${width} ${height}`,
      role: "img",
      "aria-label": `${settings.carrier} hub capacity and bank editor at ${settings.hub}`,
    });
    chart.append(svg);

    const scaleX = (timestamp) =>
      left + ((timestamp - domainStart) / (domainEnd - domainStart)) * plotWidth;

    const axis = svgElement("g", { class: "pse-axis" });
    axis.append(
      svgElement("line", {
        x1: left,
        x2: width - right,
        y1: axisY,
        y2: axisY,
      }),
    );
    const tickCount = Math.max(4, Math.floor(plotWidth / 125));
    for (let index = 0; index <= tickCount; index += 1) {
      const timestamp = domainStart + ((domainEnd - domainStart) * index) / tickCount;
      const x = scaleX(timestamp);
      axis.append(svgElement("line", { x1: x, x2: x, y1: axisY - 4, y2: axisY + 4 }));
      const label = svgElement("text", {
        x,
        y: axisY - 9,
        "text-anchor": index === 0 ? "start" : index === tickCount ? "end" : "middle",
      });
      label.textContent = formatTime(timestamp, timezone, true);
      axis.append(label);
    }
    const axisTitle = svgElement("text", {
      class: "pse-axis-title",
      x: left + plotWidth / 2,
      y: 13,
      "text-anchor": "middle",
    });
    axisTitle.textContent = `Time at ${settings.hub} (${timezone})`;
    axis.append(axisTitle);
    svg.append(axis);

    for (const [index, leg] of legs.entries()) {
      const y = firstRowY + index * rowHeight;
      const classes = ["pse-leg-row"];
      if (leg.changed) classes.push("pse-changed");
      if (leg.pending_deletion) classes.push("pce-pending-deletion");
      if (leg.selected) classes.push("pse-selected");
      if (hasActiveBank && !leg.in_active_bank) classes.push("pse-dimmed");
      if (leg.in_active_bank) classes.push("pce-active");
      if (leg.staged) classes.push("pce-staged");
      const row = svgElement("g", {
        class: classes.join(" "),
        "data-leg-id": leg.leg_id,
      });

      if (rowHeight >= 10) {
        row.append(
          svgElement("line", {
            class: "pse-row-guide",
            x1: left,
            x2: width - right,
            y1: y,
            y2: y,
          }),
        );
      }

      if (showLabels) {
        const label = svgElement("text", {
          class: "pse-leg-label",
          x: left - 10,
          y: y + 4,
          "text-anchor": "end",
        });
        const direction = leg.direction === "outbound" ? "→" : "←";
        label.textContent = `${leg.carrier}${leg.fltno} ${direction} ${leg.spoke}`;
        row.append(label);
      }

      const bankLabel = leg.bank || "(none)";
      const deletionLabel = leg.pending_deletion ? "\nPending: leg will be removed when editing finishes" : "";
      const tooltipText =
        `Leg ${leg.leg_id} · ${leg.carrier}${leg.fltno}\n` +
        `${leg.orig} → ${leg.dest}\n` +
        `${settings.coloring_title || "Color"}: ${leg.color_group}\n` +
        `Depart: ${leg.dep_local}\nArrive: ${leg.arr_local}\n` +
        `Duration: ${leg.duration_minutes} min\n` +
        `Bank: ${bankLabel}\nCapacity: ${displayCapacity(leg.capacity)} (total ${leg.total_capacity})${deletionLabel}`;
      const selectable = svgElement("g", {
        class: "pce-selectable",
        tabindex: interactionDisabled ? -1 : 0,
        role: "button",
        "aria-label": `${tooltipText.replaceAll("\n", ". ")}. ${
          state.membership_editing ? "Toggle bank membership" : "Select leg"
        }.`,
      });
      const x1 = scaleX(leg.hub_time);
      const x2 = scaleX(leg.far_time);
      const visualLine = svgElement("line", {
        class: "pse-flight-line",
        x1,
        x2,
        y1: y,
        y2: y,
        stroke: leg.color,
      });
      const hitLine = svgElement("line", {
        class: "pse-flight-hit pce-flight-hit",
        x1,
        x2,
        y1: y,
        y2: y,
        "stroke-width": Math.max(5, rowHeight),
      });
      const markerSize = rowHeight < 10 ? 4 : 8;
      const diamond = svgElement("rect", {
        class: "pse-hub-marker",
        x: x1 - markerSize / 2,
        y: y - markerSize / 2,
        width: markerSize,
        height: markerSize,
        fill: leg.color,
        transform: `rotate(45 ${x1} ${y})`,
      });
      selectable.append(visualLine, hitLine, diamond);
      row.append(selectable);
      svg.append(row);

      const selectOrToggle = () => {
        if (interactionDisabled) return;
        hideCapacityPopover();
        sendEvent({
          type: state.membership_editing ? "toggle_bank_leg" : "select_leg",
          leg_id: leg.leg_id,
        });
      };
      hitLine.addEventListener("pointerenter", (event) => {
        showLegTooltip(tooltipText, leg.leg_id, event.clientX, event.clientY);
      });
      hitLine.addEventListener("pointermove", (event) => {
        positionTooltip(event.clientX, event.clientY);
      });
      hitLine.addEventListener("pointerleave", hideLegTooltip);
      hitLine.addEventListener("click", selectOrToggle);
      selectable.addEventListener("focus", () => {
        const bounds = hitLine.getBoundingClientRect();
        showLegTooltip(tooltipText, leg.leg_id, bounds.left + bounds.width / 2, bounds.top);
      });
      selectable.addEventListener("blur", hideLegTooltip);
      selectable.addEventListener("keydown", (event) => {
        if (event.key !== "Enter" && event.key !== " ") return;
        event.preventDefault();
        selectOrToggle();
      });
    }
  }

  const requireBankName = (eventType) => {
    sendEvent({ type: eventType, bank: bankName.value });
  };
  const onUndo = () => sendEvent({ type: "undo" });
  const onRedo = () => sendEvent({ type: "redo" });
  const onReset = () => sendEvent({ type: "reset" });
  const onBankChange = () => sendEvent({ type: "activate_bank", bank: bankSelect.value || null });
  const onNewBank = () => requireBankName("new_bank");
  const onEditBank = () => sendEvent({ type: "begin_bank_edit" });
  const onRenameBank = () => requireBankName("rename_bank");
  const onDeleteBank = () => sendEvent({ type: "delete_bank" });
  const onSaveBank = () => sendEvent({ type: "save_bank" });
  const onCancelBank = () => sendEvent({ type: "cancel_bank_edit" });
  const onEditCapacity = () => showCapacityPopover();
  const onFinish = () => sendEvent({ type: "finish" });
  const onDocumentKeydown = (event) => {
    if (event.key === "Escape" && !capacityPopover.hidden) hideCapacityPopover();
  };

  undo.addEventListener("click", onUndo);
  redo.addEventListener("click", onRedo);
  reset.addEventListener("click", onReset);
  bankSelect.addEventListener("change", onBankChange);
  newBank.addEventListener("click", onNewBank);
  editBank.addEventListener("click", onEditBank);
  renameBank.addEventListener("click", onRenameBank);
  deleteBank.addEventListener("click", onDeleteBank);
  saveBank.addEventListener("click", onSaveBank);
  cancelBank.addEventListener("click", onCancelBank);
  editCapacity.addEventListener("click", onEditCapacity);
  finish.addEventListener("click", onFinish);
  document.addEventListener("keydown", onDocumentKeydown);

  const onLegs = () => {
    hideCapacityPopover();
    renderControls();
    renderChart();
  };
  const onSettings = () => {
    renderControls();
    renderChart();
  };
  const onState = () => {
    renderControls();
    renderChart();
  };
  const onStatus = () => {
    renderControls();
    renderChart();
  };
  model.on("change:_legs_data", onLegs);
  model.on("change:_settings", onSettings);
  model.on("change:_state", onState);
  model.on("change:_status", onStatus);

  let resizeFrame = null;
  const resizeObserver = new ResizeObserver(() => {
    if (resizeFrame !== null) cancelAnimationFrame(resizeFrame);
    resizeFrame = requestAnimationFrame(renderChart);
  });
  resizeObserver.observe(chart);

  renderControls();
  renderChart();

  return () => {
    resizeObserver.disconnect();
    if (resizeFrame !== null) cancelAnimationFrame(resizeFrame);
    hideLegTooltip();
    hideCapacityPopover();
    undo.removeEventListener("click", onUndo);
    redo.removeEventListener("click", onRedo);
    reset.removeEventListener("click", onReset);
    bankSelect.removeEventListener("change", onBankChange);
    newBank.removeEventListener("click", onNewBank);
    editBank.removeEventListener("click", onEditBank);
    renameBank.removeEventListener("click", onRenameBank);
    deleteBank.removeEventListener("click", onDeleteBank);
    saveBank.removeEventListener("click", onSaveBank);
    cancelBank.removeEventListener("click", onCancelBank);
    editCapacity.removeEventListener("click", onEditCapacity);
    finish.removeEventListener("click", onFinish);
    document.removeEventListener("keydown", onDocumentKeydown);
    model.off("change:_legs_data", onLegs);
    model.off("change:_settings", onSettings);
    model.off("change:_state", onState);
    model.off("change:_status", onStatus);
  };
}

export default { render };
