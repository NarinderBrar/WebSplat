import type { EditorTool, HsvAdjust, ScreenPoint, ScreenRect, SelectionMode } from "./editor/types.ts";
import GaussianSplatViewer from "./viewer/gaussian-splat-viewer.ts";
import type { RenderQualityMode } from "./world/types.ts";

interface SelectionToolOptions {
  colorThreshold: number;
  screenRadius: number;
  selectionMode: SelectionMode;
  selectBehind: boolean;
  depthRangeFactor: number;
}

interface PaintBrushOptions {
  color: [number, number, number];
  mixFactor: number;
  screenRadius: number;
}

interface EditorState {
  getTool(): EditorTool;
  getSelectionOptions(): SelectionToolOptions;
  getPaintOptions(): PaintBrushOptions;
  getHsvAdjust(): HsvAdjust;
  setMoveActive(active: boolean): void;
}

const canvas = document.querySelector<HTMLCanvasElement>("#renderCanvas");
if (!canvas) throw new Error("Canvas element #renderCanvas was not found.");

const renderCanvas = canvas;
const params = new URLSearchParams(window.location.search);
const splatSource = params.get("splat") ?? undefined;
const qualityMode = parseQualityMode(params.get("quality"));
const optimized = parseBooleanFlag(params.get("optimized"), false);

async function main(): Promise<void> {
  try {
    const viewer = await GaussianSplatViewer.create({
      canvas: renderCanvas,
      source: splatSource,
      qualityMode,
      optimized,
    });
    viewer.start();
    viewer.setVisualizationMode(1);
    const editorState = createEditorToolbar(renderCanvas, viewer);
    bindEditorTools(renderCanvas, viewer, editorState);
  } catch (error) {
    console.error(error);

    const warning = document.createElement("div");
    warning.className = "webgpu-warning";
    warning.textContent = "WebGPU not supported";
    document.body.append(warning);
  }
}

main();

function bindEditorTools(
  canvas: HTMLCanvasElement,
  viewer: GaussianSplatViewer,
  editorState: EditorState,
): void {
  let pointerDownX = 0;
  let pointerDownY = 0;
  let pointerDownButton = -1;
  let isPaintDragging = false;
  let isBrushDragging = false;
  let dragOverlay: HTMLDivElement | null = null;
  let lassoOverlay: SVGSVGElement | null = null;
  let lassoPath: SVGPolylineElement | null = null;
  let lassoPoints: ScreenPoint[] = [];

  canvas.addEventListener("pointerdown", (event) => {
    pointerDownX = event.clientX;
    pointerDownY = event.clientY;
    pointerDownButton = event.button;

    if (event.button !== 0) {
      return;
    }

    const tool = editorState.getTool();

    if (tool === "marqueeSelect") {
      dragOverlay = createDragOverlay("marquee-selection-box");
      updateRectOverlay(dragOverlay, pointerDownX, pointerDownY, pointerDownX, pointerDownY);
    } else if (tool === "lassoSelect") {
      lassoPoints = [{ x: event.clientX, y: event.clientY }];
      const overlay = createLassoOverlay();
      lassoOverlay = overlay.svg;
      lassoPath = overlay.path;
      updateLassoPath(lassoPath, lassoPoints);
    } else if (tool === "brushSelect") {
      isBrushDragging = false;
    } else if (tool === "paintBrush") {
      isPaintDragging = true;
      paintAt(event);
    }
  });

  canvas.addEventListener("pointermove", (event) => {
    if (pointerDownButton !== 0) {
      return;
    }

    const tool = editorState.getTool();

    if (tool === "marqueeSelect" && dragOverlay) {
      updateRectOverlay(dragOverlay, pointerDownX, pointerDownY, event.clientX, event.clientY);
    } else if (tool === "lassoSelect" && lassoPath) {
      const last = lassoPoints[lassoPoints.length - 1];
      const dx = event.clientX - last.x;
      const dy = event.clientY - last.y;

      if (dx * dx + dy * dy >= 4) {
        lassoPoints.push({ x: event.clientX, y: event.clientY });
        updateLassoPath(lassoPath, lassoPoints);
      }
    } else if (tool === "brushSelect") {
      isBrushDragging = true;
      const options = editorState.getSelectionOptions();
      void viewer.selectSimilarColorDragAt(
        event.clientX,
        event.clientY,
        options.colorThreshold,
        options.selectionMode,
        options.selectBehind,
        options.depthRangeFactor,
      );
    } else if (tool === "paintBrush" && isPaintDragging) {
      paintAt(event);
    }
  });

  canvas.addEventListener("pointerup", (event) => {
    if (pointerDownButton !== 0) {
      cleanupDragUi();
      return;
    }

    pointerDownButton = -1;
    isPaintDragging = false;
    const wasBrushDragging = isBrushDragging;
    isBrushDragging = false;
    const tool = editorState.getTool();
    const dx = event.clientX - pointerDownX;
    const dy = event.clientY - pointerDownY;
    const movedSq = dx * dx + dy * dy;

    if (tool === "brushSelect" && movedSq <= 16 && !wasBrushDragging) {
      const options = editorState.getSelectionOptions();
      void viewer.selectSimilarColorAt(
        event.clientX,
        event.clientY,
        options.colorThreshold,
        options.selectionMode,
        options.selectBehind,
        options.depthRangeFactor,
      );
    } else if (tool === "circleSelect" && movedSq <= 16) {
      const options = editorState.getSelectionOptions();
      void viewer.selectCircleAt(
        event.clientX,
        event.clientY,
        options.screenRadius,
        options.selectionMode,
      );
    } else if (tool === "marqueeSelect" && dragOverlay && movedSq > 16) {
      const rect = toCanvasRect(canvas, pointerDownX, pointerDownY, event.clientX, event.clientY);
      const partial = event.clientX < pointerDownX;
      void viewer.selectMarquee(rect, partial, editorState.getSelectionOptions().selectionMode);
    } else if (tool === "lassoSelect" && lassoPoints.length >= 3) {
      const points = simplifyScreenPoints(lassoPoints).map((point) => clientToCanvasPoint(canvas, point.x, point.y));
      void viewer.selectLasso(points, editorState.getSelectionOptions().selectionMode);
    }

    cleanupDragUi();
  });

  function paintAt(event: PointerEvent): void {
    const options = editorState.getPaintOptions();
    viewer.paintBrushAt(
      event.clientX,
      event.clientY,
      options.screenRadius,
      options.color,
      options.mixFactor,
    );
  }

  function cleanupDragUi(): void {
    dragOverlay?.remove();
    lassoOverlay?.remove();
    dragOverlay = null;
    lassoOverlay = null;
    lassoPath = null;
    lassoPoints = [];
  }
}

function createEditorToolbar(canvas: HTMLCanvasElement, viewer: GaussianSplatViewer): EditorState {
  let tool: EditorTool = "orbit";
  let hsvEditActive = false;
  let colorizeEditActive = false;
  let transformActive = false;
  const selectionOptions: SelectionToolOptions = {
    colorThreshold: 0.14,
    screenRadius: 12,
    selectionMode: "normal",
    selectBehind: true,
    depthRangeFactor: 2.5,
  };
  const paintOptions: PaintBrushOptions = {
    color: [0.85, 0.15, 0.15],
    mixFactor: 0.35,
    screenRadius: 18,
  };
  const hsvAdjust: HsvAdjust = {
    hue: 0,
    saturation: 1,
    value: 1,
  };
  const toolbar = document.createElement("div");
  const optionsBar = document.createElement("div");
  const toolButtons = new Map<EditorTool, HTMLButtonElement>();
  const selectionControls = document.createElement("div");
  const paintControls = document.createElement("div");
  const hsvControls = document.createElement("div");
  const colorizeControls = document.createElement("div");
  const hideControls = document.createElement("div");
  const transformControls = document.createElement("div");
  const thresholdInput = createNumberInput("Threshold", selectionOptions.colorThreshold, 0.01, 1, 0.01);
  const radiusInput = createNumberInput("Radius", selectionOptions.screenRadius, 1, 96, 1);
  const modeControl = createSelectionModeControl(selectionOptions);
  const depthRangeInput = createNumberInput("Depth range", selectionOptions.depthRangeFactor, 1.0, 10, 0.1);
  const selectBehindToggle = createSelectBehindToggle(selectionOptions, depthRangeInput.label);
  const paintColorInput = createColorInput("Paint", "#d92626");
  const paintMixInput = createRangeInput("Mix", 35, 1, 100, 1);
  const paintRadiusInput = createNumberInput("Radius", paintOptions.screenRadius, 1, 128, 1);
  const hueInput = createRangeInput("Hue", 0, -180, 180, 1);
  const saturationInput = createRangeInput("Sat", 100, 0, 200, 1);
  const valueInput = createRangeInput("Val", 100, 0, 200, 1);
  const colorizeInput = createColorInput("Color", "#d82626");
  const hsvApplyButton = createActionButton("Apply");
  const hsvCancelButton = createActionButton("Cancel");
  const colorizeApplyButton = createActionButton("Apply");
  const colorizeCancelButton = createActionButton("Cancel");
  const hideSelectedButton = createActionButton("Hide Selected");
  const unhideAllButton = createActionButton("Unhide All");
  const transformHint = document.createElement("span");
  const visualizer = document.createElement("div");

  toolbar.className = "tool-rail";
  optionsBar.className = "tool-options-bar";
  selectionControls.className = "tool-settings selection-controls";
  paintControls.className = "tool-settings paint-controls";
  hsvControls.className = "tool-settings hsv-controls";
  colorizeControls.className = "tool-settings colorize-controls";
  hideControls.className = "tool-settings hide-controls";
  transformControls.className = "tool-settings move-controls";
  transformHint.className = "move-tool-status";
  transformHint.textContent = "Gizmo";
  visualizer.className = "selection-radius-visualizer";

  const tools: Array<[EditorTool, string, string]> = [
    ["orbit", "Orbit", orbitIconSvg()],
    ["brushSelect", "Brush select", brushIconSvg()],
    ["paintBrush", "Paint brush", paintBrushIconSvg()],
    ["circleSelect", "Circle select", circleIconSvg()],
    ["marqueeSelect", "Marquee select", marqueeIconSvg()],
    ["lassoSelect", "Lasso select", lassoIconSvg()],
    ["hsv", "HSV", hsvIconSvg()],
    ["colorize", "Colorize", colorizeIconSvg()],
    ["hide", "Hide", hideIconSvg()],
    ["move", "Move", moveIconSvg()],
    ["rotate", "Rotate", rotateIconSvg()],
    ["scale", "Scale", scaleIconSvg()],
    ["copy", "Copy selected", copyIconSvg()],
  ];

  for (const [toolId, label, icon] of tools) {
    const button = createToolButton(label, icon);
    const handler = toolId === "copy"
      ? () => { const count = viewer.duplicateSelectedSplats(); if (count > 0) setTool("orbit"); }
      : () => setTool(toolId);
    button.addEventListener("click", handler);
    toolButtons.set(toolId, button);
    toolbar.append(button);
  }

  const densityToggle = document.createElement("button");
  densityToggle.className = "tool-mode-button";
  densityToggle.title = "Density culling: show 30% splats in dense areas";
  densityToggle.setAttribute("aria-label", "Density culling");
  densityToggle.innerHTML = `<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="9" fill="none" stroke="currentColor"/><circle cx="12" cy="12" r="3" fill="currentColor" stroke="none"/></svg>`;
  let densityCullingActive = false;
  densityToggle.addEventListener("click", () => {
    densityCullingActive = !densityCullingActive;
    densityToggle.classList.toggle("is-active", densityCullingActive);
    viewer.setDensityCulling(densityCullingActive);
  });
  toolbar.append(densityToggle);

  selectionControls.append(modeControl, thresholdInput.label, radiusInput.label, selectBehindToggle, depthRangeInput.label);

  depthRangeInput.input.addEventListener("input", () => {
    selectionOptions.depthRangeFactor = depthRangeInput.input.valueAsNumber || 2.5;
  });
  paintControls.append(paintColorInput.label, paintMixInput.label, paintRadiusInput.label);
  hsvControls.append(hueInput.label, saturationInput.label, valueInput.label, hsvApplyButton, hsvCancelButton);
  colorizeControls.append(colorizeInput.label, colorizeApplyButton, colorizeCancelButton);
  hideControls.append(hideSelectedButton, unhideAllButton);
  transformControls.append(transformHint);
  optionsBar.append(selectionControls, paintControls, hsvControls, colorizeControls, hideControls, transformControls);
  const vizBar = document.createElement("div");
  vizBar.className = "viz-bar";
  const vizModes: Array<{ mode: number; label: string; svg: string }> = [
    { mode: 0, label: "Normal", svg: `<svg viewBox="0 0 16 16"><circle cx="8" cy="8" r="5" fill="currentColor" stroke="none"/></svg>` },
    { mode: 1, label: "Particle cloud", svg: `<svg viewBox="0 0 16 16"><circle cx="5" cy="8" r="2.2" fill="currentColor" stroke="none"/><circle cx="11" cy="6" r="1.6" fill="currentColor" stroke="none"/><circle cx="10" cy="11" r="1.8" fill="currentColor" stroke="none"/></svg>` },
    { mode: 2, label: "Random colors", svg: `<svg viewBox="0 0 16 16"><circle cx="8" cy="8" r="6" fill="none" stroke="currentColor"/><path d="M5 6h6M5 8h6M5 10h6" stroke="currentColor" stroke-width="1.2"/></svg>` },
  ];
  let activeVizMode = 1;
  for (const viz of vizModes) {
    const btn = document.createElement("button");
    btn.className = "viz-button" + (viz.mode === activeVizMode ? " is-active" : "");
    btn.title = viz.label;
    btn.setAttribute("aria-label", viz.label);
    btn.innerHTML = viz.svg;
    btn.addEventListener("click", () => {
      activeVizMode = viz.mode;
      viewer.setVisualizationMode(viz.mode);
      for (const child of vizBar.children) {
        (child as HTMLElement).classList.toggle("is-active", child === btn);
      }
    });
    vizBar.append(btn);
  }
  document.body.append(toolbar, optionsBar, visualizer, vizBar);

  thresholdInput.input.addEventListener("input", () => {
    selectionOptions.colorThreshold = thresholdInput.input.valueAsNumber || 0.14;
  });
  radiusInput.input.addEventListener("input", () => {
    selectionOptions.screenRadius = radiusInput.input.valueAsNumber || 12;
    updateVisualizerSize();
  });
  paintColorInput.input.addEventListener("input", () => {
    paintOptions.color = hexToRgb(paintColorInput.input.value);
  });
  paintMixInput.input.addEventListener("input", () => {
    paintOptions.mixFactor = Math.max(0, Math.min(1, (paintMixInput.input.valueAsNumber || 35) / 100));
  });
  paintRadiusInput.input.addEventListener("input", () => {
    paintOptions.screenRadius = paintRadiusInput.input.valueAsNumber || 18;
    updateVisualizerSize();
  });

  const updateHsv = (): void => {
    hsvAdjust.hue = (hueInput.input.valueAsNumber || 0) / 360;
    hsvAdjust.saturation = (saturationInput.input.valueAsNumber || 100) / 100;
    hsvAdjust.value = (valueInput.input.valueAsNumber || 100) / 100;

    if (!hsvEditActive) {
      hsvEditActive = viewer.beginHsvEdit() > 0;
    }

    if (hsvEditActive) {
      viewer.previewHsvEdit(hsvAdjust);
    }
  };

  for (const input of [hueInput.input, saturationInput.input, valueInput.input]) {
    input.addEventListener("input", updateHsv);
  }

  hsvApplyButton.addEventListener("click", () => {
    if (hsvEditActive) {
      viewer.commitHsvEdit();
      hsvEditActive = false;
    }
  });
  hsvCancelButton.addEventListener("click", () => {
    if (hsvEditActive) {
      viewer.cancelHsvEdit();
      hsvEditActive = false;
    }
    resetHsvInputs();
  });

  const updateColorize = (): void => {
    if (!colorizeEditActive) {
      colorizeEditActive = viewer.beginColorizeEdit() > 0;
    }

    if (colorizeEditActive) {
      viewer.previewColorizeEdit(hexToRgb(colorizeInput.input.value));
    }
  };

  colorizeInput.input.addEventListener("input", updateColorize);
  colorizeApplyButton.addEventListener("click", () => {
    if (colorizeEditActive) {
      viewer.commitColorizeEdit();
      colorizeEditActive = false;
    }
  });
  colorizeCancelButton.addEventListener("click", () => {
    if (colorizeEditActive) {
      viewer.cancelColorizeEdit();
      colorizeEditActive = false;
    }
  });
  hideSelectedButton.addEventListener("click", () => {
    viewer.hideSelectedSplats();
  });
  unhideAllButton.addEventListener("click", () => {
    viewer.unhideAllSplats();
  });

  canvas.addEventListener("pointermove", (event) => {
    if (tool !== "circleSelect" && tool !== "paintBrush") {
      return;
    }

    const radius = tool === "paintBrush" ? paintOptions.screenRadius : selectionOptions.screenRadius;
    visualizer.style.transform = `translate(${event.clientX - radius}px, ${event.clientY - radius}px)`;
  });
  canvas.addEventListener("pointerleave", () => visualizer.classList.add("is-hidden"));
  canvas.addEventListener("pointerenter", () => visualizer.classList.remove("is-hidden"));

  function setTool(nextTool: EditorTool): void {
    if (hsvEditActive) {
      viewer.cancelHsvEdit();
      hsvEditActive = false;
      resetHsvInputs();
    }

    if (colorizeEditActive) {
      viewer.cancelColorizeEdit();
      colorizeEditActive = false;
    }

    if (transformActive && !isTransformTool(nextTool)) {
      viewer.commitMoveSelected();
      transformActive = false;
    }

    tool = nextTool;
    canvas.dataset.toolMode = tool;
    viewer.setOrbitControlsEnabled(tool === "orbit");
    viewer.setSelectionHighlightVisible(tool !== "hsv" && tool !== "colorize" && tool !== "paintBrush");
    selectionControls.hidden = !isSelectionTool(tool);
    thresholdInput.label.hidden = tool !== "brushSelect";
    selectBehindToggle.hidden = tool !== "brushSelect";
    depthRangeInput.label.hidden = tool !== "brushSelect" || selectionOptions.selectBehind;
    radiusInput.label.hidden = tool !== "circleSelect";
    paintControls.hidden = tool !== "paintBrush";
    hsvControls.hidden = tool !== "hsv";
    colorizeControls.hidden = tool !== "colorize";
    hideControls.hidden = tool !== "hide";
    transformControls.hidden = !isTransformTool(tool);
    visualizer.hidden = tool !== "circleSelect" && tool !== "paintBrush";

    for (const [toolId, button] of toolButtons) {
      const active = toolId === tool;
      button.classList.toggle("is-active", active);
      button.setAttribute("aria-pressed", String(active));
    }

    if (tool === "hsv") {
      hsvEditActive = viewer.beginHsvEdit() > 0;
    } else if (tool === "colorize") {
      colorizeEditActive = viewer.beginColorizeEdit() > 0;
      updateColorize();
    } else if (isTransformTool(tool) && !transformActive) {
      transformActive = viewer.beginMoveSelected(tool);
    } else if (isTransformTool(tool)) {
      viewer.setTransformToolMode(tool);
    }
  }

  function resetHsvInputs(): void {
    hueInput.input.value = "0";
    saturationInput.input.value = "100";
    valueInput.input.value = "100";
    hsvAdjust.hue = 0;
    hsvAdjust.saturation = 1;
    hsvAdjust.value = 1;
  }

  function updateVisualizerSize(): void {
    const radius = tool === "paintBrush" ? paintOptions.screenRadius : selectionOptions.screenRadius;
    const diameter = radius * 2;
    visualizer.style.width = `${diameter}px`;
    visualizer.style.height = `${diameter}px`;
  }

  updateVisualizerSize();
  setTool(tool);

  return {
    getTool: () => tool,
    getSelectionOptions: () => ({ ...selectionOptions }),
    getPaintOptions: () => ({ ...paintOptions }),
    getHsvAdjust: () => ({ ...hsvAdjust }),
    setMoveActive: (active) => {
      transformActive = active;
    },
  };
}

function isSelectionTool(tool: EditorTool): boolean {
  return tool === "brushSelect" || tool === "circleSelect" || tool === "marqueeSelect" || tool === "lassoSelect";
}

function isTransformTool(tool: EditorTool): tool is "move" | "rotate" | "scale" {
  return tool === "move" || tool === "rotate" || tool === "scale";
}

function createSelectBehindToggle(
  options: SelectionToolOptions,
  depthRangeLabel?: HTMLLabelElement,
): HTMLLabelElement {
  const label = document.createElement("label");
  label.className = "selection-control";

  const checkbox = document.createElement("input");
  checkbox.type = "checkbox";
  checkbox.checked = options.selectBehind;
  checkbox.addEventListener("change", () => {
    options.selectBehind = checkbox.checked;
    if (depthRangeLabel) {
      depthRangeLabel.hidden = checkbox.checked;
    }
  });
  label.append(checkbox, " Behind");
  return label;
}

function createSelectionModeControl(options: SelectionToolOptions): HTMLDivElement {
  const control = document.createElement("div");
  const modes: Array<[SelectionMode, string]> = [
    ["normal", "Normal"],
    ["additive", "Add"],
    ["subtractive", "Sub"],
  ];

  control.className = "selection-mode-control";

  for (const [mode, label] of modes) {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = label;
    button.addEventListener("click", () => {
      options.selectionMode = mode;
      for (const child of control.querySelectorAll("button")) {
        child.classList.toggle("is-active", child === button);
      }
    });
    button.classList.toggle("is-active", mode === options.selectionMode);
    control.append(button);
  }

  return control;
}

function createToolButton(label: string, svg: string): HTMLButtonElement {
  const button = document.createElement("button");
  button.className = "tool-mode-button";
  button.type = "button";
  button.title = label;
  button.setAttribute("aria-label", label);
  button.innerHTML = svg;
  return button;
}

function createActionButton(label: string): HTMLButtonElement {
  const button = document.createElement("button");
  button.className = "tool-action-button";
  button.type = "button";
  button.textContent = label;
  return button;
}

function createNumberInput(
  text: string,
  value: number,
  min: number,
  max: number,
  step: number,
): { label: HTMLLabelElement; input: HTMLInputElement } {
  const label = document.createElement("label");
  const input = document.createElement("input");

  label.className = "selection-control";
  label.textContent = text;
  input.type = "range";
  input.className = "tool-range";
  input.min = String(min);
  input.max = String(max);
  input.step = String(step);
  input.value = String(value);
  label.append(input);

  return { label, input };
}

function createRangeInput(
  text: string,
  value: number,
  min: number,
  max: number,
  step: number,
): { label: HTMLLabelElement; input: HTMLInputElement } {
  const field = createNumberInput(text, value, min, max, step);
  field.input.type = "range";
  field.input.className = "tool-range";
  return field;
}

function createColorInput(
  text: string,
  value: string,
): { label: HTMLLabelElement; input: HTMLInputElement } {
  const label = document.createElement("label");
  const input = document.createElement("input");

  label.className = "selection-control color-control";
  label.textContent = text;
  input.type = "color";
  input.value = value;
  label.append(input);

  return { label, input };
}

function createDragOverlay(className: string): HTMLDivElement {
  const overlay = document.createElement("div");
  overlay.className = className;
  document.body.append(overlay);
  return overlay;
}

function updateRectOverlay(overlay: HTMLElement, x0: number, y0: number, x1: number, y1: number): void {
  const minX = Math.min(x0, x1);
  const minY = Math.min(y0, y1);
  overlay.style.transform = `translate(${minX}px, ${minY}px)`;
  overlay.style.width = `${Math.abs(x1 - x0)}px`;
  overlay.style.height = `${Math.abs(y1 - y0)}px`;
}

function createLassoOverlay(): { svg: SVGSVGElement; path: SVGPolylineElement } {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  const path = document.createElementNS("http://www.w3.org/2000/svg", "polyline");
  svg.classList.add("lasso-selection-overlay");
  path.setAttribute("fill", "none");
  svg.append(path);
  document.body.append(svg);
  return { svg, path };
}

function updateLassoPath(path: SVGPolylineElement, points: readonly ScreenPoint[]): void {
  path.setAttribute("points", points.map((point) => `${point.x},${point.y}`).join(" "));
}

function simplifyScreenPoints(points: readonly ScreenPoint[]): ScreenPoint[] {
  const simplified: ScreenPoint[] = [];

  for (const point of points) {
    const last = simplified[simplified.length - 1];

    if (!last) {
      simplified.push(point);
      continue;
    }

    const dx = point.x - last.x;
    const dy = point.y - last.y;

    if (dx * dx + dy * dy >= 9) {
      simplified.push(point);
    }
  }

  return simplified;
}

function clientToCanvasPoint(canvas: HTMLCanvasElement, clientX: number, clientY: number): ScreenPoint {
  const rect = canvas.getBoundingClientRect();
  return {
    x: ((clientX - rect.left) / rect.width) * canvas.width,
    y: ((clientY - rect.top) / rect.height) * canvas.height,
  };
}

function toCanvasRect(canvas: HTMLCanvasElement, x0: number, y0: number, x1: number, y1: number): ScreenRect {
  const a = clientToCanvasPoint(canvas, x0, y0);
  const b = clientToCanvasPoint(canvas, x1, y1);
  return {
    minX: Math.min(a.x, b.x),
    minY: Math.min(a.y, b.y),
    maxX: Math.max(a.x, b.x),
    maxY: Math.max(a.y, b.y),
  };
}

function orbitIconSvg(): string {
  return `
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M4 12c0-3.7 3.6-6.7 8-6.7s8 3 8 6.7-3.6 6.7-8 6.7" />
      <path d="M8 15.4 4.6 18.8 8 22.2" />
      <path d="M12 9.2a2.8 2.8 0 1 1 0 5.6 2.8 2.8 0 0 1 0-5.6Z" />
    </svg>
  `;
}

function brushIconSvg(): string {
  return `
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="m5 3 6.8 16.4 2.1-6.1 6.1-2.1L5 3Z" />
      <path d="m14 14 4.8 4.8" />
    </svg>
  `;
}

function paintBrushIconSvg(): string {
  return `
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M15 4 5 14" />
      <path d="m14 5 5 5" />
      <path d="M4 15c3 0 5 2 5 5-2.7 0-5-2.2-5-5Z" />
      <path d="M12 7 17 2l5 5-5 5" />
    </svg>
  `;
}

function circleIconSvg(): string {
  return `
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <circle cx="12" cy="12" r="8" />
      <path d="M12 5v14M5 12h14" />
    </svg>
  `;
}

function marqueeIconSvg(): string {
  return `
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M5 5h14v14H5z" />
      <path d="M8 5v14M16 5v14M5 8h14M5 16h14" />
    </svg>
  `;
}

function lassoIconSvg(): string {
  return `
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M5 13c-1.2-4.4 3-8.4 8-7.3 5.1 1.1 8 5.2 5.7 8.5-2.2 3.1-8.4 2.8-10.3 1.2" />
      <path d="M8.4 15.4 5.5 21" />
    </svg>
  `;
}

function hsvIconSvg(): string {
  return `
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M12 4a8 8 0 1 0 8 8" />
      <path d="M12 4v8l7 4" />
    </svg>
  `;
}

function colorizeIconSvg(): string {
  return `
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M12 3a7 7 0 0 0-7 7c0 5 7 11 7 11s7-6 7-11a7 7 0 0 0-7-7Z" />
      <path d="M9 10a3 3 0 0 0 6 0" />
    </svg>
  `;
}

function hideIconSvg(): string {
  return `
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M3 12s3.4-6 9-6 9 6 9 6-3.4 6-9 6-9-6-9-6Z" />
      <path d="m4 4 16 16" />
      <path d="M9.9 9.9a3 3 0 0 0 4.2 4.2" />
    </svg>
  `;
}

function hexToRgb(value: string): [number, number, number] {
  const hex = value.startsWith("#") ? value.slice(1) : value;
  const packed = Number.parseInt(hex.padEnd(6, "0").slice(0, 6), 16);
  return [
    ((packed >> 16) & 255) / 255,
    ((packed >> 8) & 255) / 255,
    (packed & 255) / 255,
  ];
}

function moveIconSvg(): string {
  return `
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M12 3v18M3 12h18" />
      <path d="m12 3-3 3M12 3l3 3M12 21l-3-3M12 21l3-3M3 12l3-3M3 12l3 3M21 12l-3-3M21 12l-3 3" />
    </svg>
  `;
}

function rotateIconSvg(): string {
  return `
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M4 12a8 8 0 0 1 13.7-5.6" />
      <path d="M18 3v5h-5" />
      <path d="M20 12a8 8 0 0 1-13.7 5.6" />
      <path d="M6 21v-5h5" />
    </svg>
  `;
}

function scaleIconSvg(): string {
  return `
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M4 20 20 4" />
      <path d="M14 4h6v6" />
      <path d="M10 20H4v-6" />
      <path d="M7 7h10v10H7z" />
    </svg>
  `;
}

function copyIconSvg(): string {
  return `
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <rect x="5" y="5" width="13" height="13" rx="1" />
      <path d="M17 7h2a1 1 0 0 1 1 1v11a2 2 0 0 1-2 2H7a1 1 0 0 1-1-1v-2" />
    </svg>
  `;
}

function parseQualityMode(value: string | null): RenderQualityMode {
  if (value === "quality" || value === "balanced" || value === "gpu-balanced" || value === "performance") {
    return value;
  }

  return "performance";
}

function parseBooleanFlag(value: string | null, fallback: boolean): boolean {
  if (value === null) {
    return fallback;
  }

  return value === "1" || value === "true" || value === "yes" || value === "on";
}
