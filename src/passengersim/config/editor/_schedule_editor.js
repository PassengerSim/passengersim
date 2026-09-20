const SVG_NS = "http://www.w3.org/2000/svg";

function svgElement(name, attributes = {}) {
  const element = document.createElementNS(SVG_NS, name);
  for (const [key, value] of Object.entries(attributes)) {
    element.setAttribute(key, String(value));
  }
  return element;
}

function render({ model, el }) {
  el.classList.add("passengersim-schedule-editor");
  el.innerHTML = `
    <section class="pse-shell" aria-label="Flight schedule editor">
      <header class="pse-header">
        <div>
          <h3 class="pse-title"></h3>
          <div class="pse-subtitle">Drag a flight horizontally; its duration remains fixed.</div>
        </div>
        <div class="pse-toolbar" role="toolbar" aria-label="Schedule editing controls">
          <label class="pse-snap-label">Snap
            <select class="pse-snap" aria-label="Drag snapping interval">
              <option value="1">1 min</option>
              <option value="5">5 min</option>
              <option value="10">10 min</option>
              <option value="15">15 min</option>
              <option value="30">30 min</option>
            </select>
          </label>
          <button type="button" class="pse-undo">Undo</button>
          <button type="button" class="pse-redo">Redo</button>
          <button type="button" class="pse-reset">Reset</button>
        </div>
      </header>
      <div class="pse-status" role="status" aria-live="polite">
        <span class="pse-status-message"></span>
        <span class="pse-drag-readout"></span>
        <span class="pse-modified"></span>
      </div>
      <div class="pse-legend" aria-label="Schedule color legend"></div>
      <div class="pse-chart-viewport">
        <div class="pse-chart"></div>
      </div>
      <div class="pse-tooltip" role="tooltip" hidden></div>
      <footer class="pse-footer">
        <label class="pse-skip-label">
          <input type="checkbox" class="pse-skip-rebuild">
          Skip connection rebuild
        </label>
        <button type="button" class="pse-finish">Finish &amp; rebuild connections</button>
      </footer>
    </section>
  `;

  const root = el.querySelector(".pse-shell");
  const title = root.querySelector(".pse-title");
  const chart = root.querySelector(".pse-chart");
  const viewport = root.querySelector(".pse-chart-viewport");
  const snap = root.querySelector(".pse-snap");
  const undo = root.querySelector(".pse-undo");
  const redo = root.querySelector(".pse-redo");
  const reset = root.querySelector(".pse-reset");
  const statusMessage = root.querySelector(".pse-status-message");
  const dragReadout = root.querySelector(".pse-drag-readout");
  const modified = root.querySelector(".pse-modified");
  const legend = root.querySelector(".pse-legend");
  const legTooltip = root.querySelector(".pse-tooltip");
  const skipRebuild = root.querySelector(".pse-skip-rebuild");
  const finish = root.querySelector(".pse-finish");
  let selectedLegId = null;
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

  function renderControls() {
    const settings = model.get("_settings") || {};
    const status = model.get("_status") || {};
    const busy = status.phase === "rebuilding";
    const finished = Boolean(status.finished);

    title.textContent = settings.carrier
      ? `${settings.carrier} Flight Schedule at ${settings.hub}`
      : `Flight Schedule at ${settings.hub}`;
    if (![...snap.options].some((option) => Number(option.value) === settings.snap_minutes)) {
      const option = document.createElement("option");
      option.value = String(settings.snap_minutes);
      option.textContent = `${settings.snap_minutes} min`;
      snap.append(option);
    }
    if (document.activeElement !== snap) {
      snap.value = String(settings.snap_minutes || 5);
    }

    statusMessage.textContent = status.message || "";
    root.dataset.phase = status.phase || "editing";
    modified.textContent = `${status.modified_count || 0} modified`;
    undo.disabled = !status.can_undo || busy || finished;
    redo.disabled = !status.can_redo || busy || finished;
    reset.disabled = busy || finished || !(status.modified_count > 0);
    snap.disabled = busy || finished;
    skipRebuild.disabled = busy || finished;
    finish.disabled = busy || finished;
    finish.textContent = skipRebuild.checked
      ? "Finish without rebuilding"
      : "Finish & rebuild connections";

    const entries = settings.legend || [];
    legend.replaceChildren();
    legend.hidden = entries.length === 0;
    if (entries.length) {
      const legendTitle = document.createElement("span");
      legendTitle.className = "pse-legend-title";
      legendTitle.textContent = `${settings.coloring_title || "Color"}:`;
      legend.append(legendTitle);
      entries.forEach((entry) => {
        const item = document.createElement("span");
        item.className = "pse-legend-item";
        const swatch = document.createElement("span");
        swatch.className = "pse-legend-swatch";
        swatch.style.backgroundColor = entry.color;
        const label = document.createElement("span");
        label.textContent = entry.label;
        item.append(swatch, label);
        legend.append(item);
      });
    }
  }

  function renderChart() {
    hideLegTooltip();
    const legs = (model.get("_legs_data") || []).map((leg) => ({ ...leg }));
    const settings = model.get("_settings") || {};
    const status = model.get("_status") || {};
    const interactionDisabled = Boolean(status.finished) || status.phase === "rebuilding";
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

    viewport.style.minHeight = `${settings.min_height || 180}px`;
    viewport.style.maxHeight = `${settings.max_height || 600}px`;
    chart.innerHTML = "";

    if (!legs.length) {
      const empty = document.createElement("div");
      empty.className = "pse-empty";
      empty.textContent = "No legs match this hub and carrier filter.";
      chart.append(empty);
      return;
    }

    const svg = svgElement("svg", {
      class: "pse-svg",
      width,
      height,
      viewBox: `0 0 ${width} ${height}`,
      role: "img",
      "aria-label": `Editable flight schedule at ${settings.hub}`,
    });
    chart.append(svg);

    const scaleX = (timestamp) =>
      left + ((timestamp - domainStart) / (domainEnd - domainStart)) * plotWidth;
    const secondsPerPixel = (domainEnd - domainStart) / plotWidth;

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

    function applySelection() {
      const rows = svg.querySelectorAll(".pse-leg-row");
      if (selectedLegId === null || selectedLegId === undefined) {
        rows.forEach((row) => {
          row.classList.remove("pse-selected", "pse-dimmed");
        });
        return;
      }

      const selected = legs.find((leg) => leg.leg_id === selectedLegId);
      if (selected === undefined) {
        selectedLegId = null;
        rows.forEach((row) => {
          row.classList.remove("pse-selected", "pse-dimmed");
        });
        return;
      }

      const connected = new Set(selected.connected_leg_ids);
      rows.forEach((row) => {
        const rowId = Number(row.dataset.legId);
        row.classList.toggle("pse-selected", rowId === selectedLegId);
        if (!connected.has(rowId)) {
          row.classList.add("pse-dimmed");
        } else {
          row.classList.remove("pse-dimmed");
        }
      });
    }

    legs.forEach((leg, index) => {
      const y = firstRowY + index * rowHeight;
      const row = svgElement("g", {
        class: `pse-leg-row ${leg.changed ? "pse-changed" : ""}`,
        "data-leg-id": leg.leg_id,
      });
      row.dataset.legId = String(leg.leg_id);

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

      const changed = leg.changed ? `\nChange: ${leg.delta_minutes >= 0 ? "+" : ""}${leg.delta_minutes} min` : "";
      const tooltipText =
        `Leg ${leg.leg_id} · ${leg.carrier}${leg.fltno}\n` +
        `${leg.orig} → ${leg.dest}\n` +
        `${settings.coloring_title || "Color"}: ${leg.color_group}\n` +
        `Depart: ${leg.dep_local}\nArrive: ${leg.arr_local}\n` +
        `Duration: ${leg.duration_minutes} min · Capacity: ${leg.capacity}${changed}`;
      const draggable = svgElement("g", {
        class: "pse-draggable",
        tabindex: interactionDisabled ? -1 : 0,
        role: "button",
        "aria-label": `${tooltipText.replaceAll("\n", ". ")}. Use left and right arrows to move this leg.`,
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
        class: "pse-flight-hit",
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
      draggable.append(visualLine, hitLine, diamond);
      row.append(draggable);
      svg.append(row);

      let dragState = null;
      hitLine.addEventListener("pointerenter", (event) => {
        if (dragState === null) {
          showLegTooltip(tooltipText, leg.leg_id, event.clientX, event.clientY);
        }
      });
      hitLine.addEventListener("pointerdown", (event) => {
        if (interactionDisabled || event.button !== 0) return;
        event.preventDefault();
        hideLegTooltip();
        hitLine.setPointerCapture(event.pointerId);
        dragState = {
          pointerId: event.pointerId,
          startX: event.clientX,
          deltaSeconds: 0,
        };
        draggable.classList.add("pse-dragging");
      });
      hitLine.addEventListener("pointermove", (event) => {
        if (!dragState) {
          positionTooltip(event.clientX, event.clientY);
          return;
        }
        if (event.pointerId !== dragState.pointerId) return;
        const snapSeconds = Number(snap.value || settings.snap_minutes || 5) * 60;
        const rawSeconds = (event.clientX - dragState.startX) * secondsPerPixel;
        const deltaSeconds = Math.round(rawSeconds / snapSeconds) * snapSeconds;
        dragState.deltaSeconds = deltaSeconds;
        const deltaPixels = deltaSeconds / secondsPerPixel;
        draggable.setAttribute("transform", `translate(${deltaPixels} 0)`);
        const minutes = deltaSeconds / 60;
        dragReadout.textContent = `${leg.carrier}${leg.fltno}: ${minutes >= 0 ? "+" : ""}${minutes} min`;
      });
      hitLine.addEventListener("pointerleave", () => {
        if (dragState === null) hideLegTooltip();
      });
      const endDrag = (event) => {
        if (!dragState || event.pointerId !== dragState.pointerId) return;
        const deltaSeconds = dragState.deltaSeconds;
        dragState = null;
        draggable.classList.remove("pse-dragging");
        dragReadout.textContent = "";
        if (deltaSeconds !== 0) {
          sendEvent({ type: "move", leg_id: leg.leg_id, delta_seconds: deltaSeconds });
        } else {
          draggable.removeAttribute("transform");
          selectedLegId = selectedLegId === leg.leg_id ? null : leg.leg_id;
          applySelection();
        }
      };
      hitLine.addEventListener("pointerup", endDrag);
      hitLine.addEventListener("pointercancel", (event) => {
        if (!dragState || event.pointerId !== dragState.pointerId) return;
        dragState = null;
        hideLegTooltip();
        draggable.classList.remove("pse-dragging");
        draggable.removeAttribute("transform");
        dragReadout.textContent = "";
      });
      draggable.addEventListener("focus", () => {
        const bounds = hitLine.getBoundingClientRect();
        showLegTooltip(tooltipText, leg.leg_id, bounds.left + bounds.width / 2, bounds.top);
      });
      draggable.addEventListener("blur", hideLegTooltip);
      draggable.addEventListener("keydown", (event) => {
        if (interactionDisabled || (event.key !== "ArrowLeft" && event.key !== "ArrowRight")) return;
        event.preventDefault();
        const directionMultiplier = event.key === "ArrowLeft" ? -1 : 1;
        const deltaSeconds = directionMultiplier * Number(snap.value || settings.snap_minutes || 5) * 60;
        sendEvent({ type: "move", leg_id: leg.leg_id, delta_seconds: deltaSeconds });
      });
    });

    if (!legs.some((leg) => leg.leg_id === selectedLegId)) {
      selectedLegId = null;
    }
    applySelection();
  }

  const onUndo = () => sendEvent({ type: "undo" });
  const onRedo = () => sendEvent({ type: "redo" });
  const onReset = () => sendEvent({ type: "reset" });
  const onSnapChange = () => sendEvent({ type: "set_snap", snap_minutes: Number(snap.value) });
  const onSkipChange = () => renderControls();
  const onFinish = () => {
    finish.disabled = true;
    sendEvent({ type: "finish", skip_rebuild: skipRebuild.checked });
  };
  undo.addEventListener("click", onUndo);
  redo.addEventListener("click", onRedo);
  reset.addEventListener("click", onReset);
  snap.addEventListener("change", onSnapChange);
  skipRebuild.addEventListener("change", onSkipChange);
  finish.addEventListener("click", onFinish);

  const onLegs = () => renderChart();
  const onSettings = () => {
    renderControls();
    renderChart();
  };
  const onStatus = () => {
    renderControls();
    renderChart();
  };
  model.on("change:_legs_data", onLegs);
  model.on("change:_settings", onSettings);
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
    undo.removeEventListener("click", onUndo);
    redo.removeEventListener("click", onRedo);
    reset.removeEventListener("click", onReset);
    snap.removeEventListener("change", onSnapChange);
    skipRebuild.removeEventListener("change", onSkipChange);
    finish.removeEventListener("click", onFinish);
    model.off("change:_legs_data", onLegs);
    model.off("change:_settings", onSettings);
    model.off("change:_status", onStatus);
  };
}

export default { render };
