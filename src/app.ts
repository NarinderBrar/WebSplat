import type { EditorTool, HsvAdjust, ScreenPoint, ScreenRect, SelectionMode } from "./editor/types.ts";
import GaussianSplatViewer from "./viewer/gaussian-splat-viewer.ts";
import type { RenderQualityMode } from "./world/types.ts";

interface SelectionToolOptions {
  colorThreshold: number;
  screenRadius: number;
  selectionMode: SelectionMode;
}

interface EditorState {
  getTool(): EditorTool;
  getSelectionOptions(): SelectionToolOptions;
  getHsvAdjust(): HsvAdjust;
  setMoveActive(active: boolean): void;
}

const canvas = document.querySelector<HTMLCanvasElement>("#renderCanvas");
if (!canvas) throw new Error("Canvas element #renderCanvas was not found.");

const renderCanvas = canvas;
const params = new URLSearchParams(window.location.search);
const splatSource = params.get("splat") ?? undefined;
const qualityMode = parseQualityMode(params.get("quality"));

async function main(): Promise<void> {
  try {
    const viewer = await GaussianSplatViewer.create({
      canvas: renderCanvas,
      source: splatSource,
      qualityMode,
    });
    viewer.start();
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
    }
  });

  canvas.addEventListener("pointerup", (event) => {
    if (pointerDownButton !== 0) {
      cleanupDragUi();
      return;
    }

    pointerDownButton = -1;
    const tool = editorState.getTool();
    const dx = event.clientX - pointerDownX;
    const dy = event.clientY - pointerDownY;
    const movedSq = dx * dx + dy * dy;

    if (tool === "brushSelect" && movedSq <= 16) {
      const options = editorState.getSelectionOptions();
      void viewer.selectBrushAt(
        event.clientX,
        event.clientY,
        options.screenRadius,
        options.colorThreshold,
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
  let moveActive = false;
  const selectionOptions: SelectionToolOptions = {
    colorThreshold: 0.14,
    screenRadius: 12,
    selectionMode: "normal",
  };
  const hsvAdjust: HsvAdjust = {
    hue: 0,
    saturation: 1,
    value: 1,
  };
  const toolbar = document.createElement("div");
  const toolButtons = new Map<EditorTool, HTMLButtonElement>();
  const selectionControls = document.createElement("div");
  const hsvControls = document.createElement("div");
  const colorizeControls = document.createElement("div");
  const moveControls = document.createElement("div");
  const thresholdInput = createNumberInput("Threshold", selectionOptions.colorThreshold, 0.01, 1, 0.01);
  const radiusInput = createNumberInput("Radius", selectionOptions.screenRadius, 1, 96, 1);
  const modeControl = createSelectionModeControl(selectionOptions);
  const hueInput = createRangeInput("Hue", 0, -180, 180, 1);
  const saturationInput = createRangeInput("Sat", 100, 0, 200, 1);
  const valueInput = createRangeInput("Val", 100, 0, 200, 1);
  const colorizeInput = createColorInput("Color", "#d82626");
  const moveHint = document.createElement("span");
  const visualizer = document.createElement("div");

  toolbar.className = "tool-mode-toolbar editor-toolbar";
  selectionControls.className = "tool-settings selection-controls";
  hsvControls.className = "tool-settings hsv-controls";
  colorizeControls.className = "tool-settings colorize-controls";
  moveControls.className = "tool-settings move-controls";
  moveHint.className = "move-tool-status";
  moveHint.textContent = "Gizmo";
  visualizer.className = "selection-radius-visualizer";

  const tools: Array<[EditorTool, string, string]> = [
    ["orbit", "Orbit", orbitIconSvg()],
    ["brushSelect", "Brush select", brushIconSvg()],
    ["marqueeSelect", "Marquee select", marqueeIconSvg()],
    ["lassoSelect", "Lasso select", lassoIconSvg()],
    ["hsv", "HSV", hsvIconSvg()],
    ["colorize", "Colorize", colorizeIconSvg()],
    ["move", "Move", moveIconSvg()],
  ];

  for (const [toolId, label, icon] of tools) {
    const button = createToolButton(label, icon);
    button.addEventListener("click", () => setTool(toolId));
    toolButtons.set(toolId, button);
    toolbar.append(button);
  }

  selectionControls.append(modeControl, thresholdInput.label, radiusInput.label);
  hsvControls.append(hueInput.label, saturationInput.label, valueInput.label);
  colorizeControls.append(colorizeInput.label);
  moveControls.append(moveHint);
  toolbar.append(selectionControls, hsvControls, colorizeControls, moveControls);
  document.body.append(toolbar, visualizer);

  thresholdInput.input.addEventListener("input", () => {
    selectionOptions.colorThreshold = thresholdInput.input.valueAsNumber || 0.14;
  });
  radiusInput.input.addEventListener("input", () => {
    selectionOptions.screenRadius = radiusInput.input.valueAsNumber || 12;
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
    input.addEventListener("change", () => {
      if (hsvEditActive) {
        viewer.commitHsvEdit();
        hsvEditActive = false;
      }
    });
  }

  const updateColorize = (): void => {
    if (!colorizeEditActive) {
      colorizeEditActive = viewer.beginColorizeEdit() > 0;
    }

    if (colorizeEditActive) {
      viewer.previewColorizeEdit(hexToRgb(colorizeInput.input.value));
    }
  };

  colorizeInput.input.addEventListener("input", updateColorize);
  colorizeInput.input.addEventListener("change", () => {
    if (colorizeEditActive) {
      viewer.commitColorizeEdit();
      colorizeEditActive = false;
    }
  });

  canvas.addEventListener("pointermove", (event) => {
    if (tool !== "brushSelect") {
      return;
    }

    visualizer.style.transform = `translate(${event.clientX - selectionOptions.screenRadius}px, ${event.clientY - selectionOptions.screenRadius}px)`;
  });
  canvas.addEventListener("pointerleave", () => visualizer.classList.add("is-hidden"));
  canvas.addEventListener("pointerenter", () => visualizer.classList.remove("is-hidden"));

  function setTool(nextTool: EditorTool): void {
    if (hsvEditActive) {
      viewer.commitHsvEdit();
      hsvEditActive = false;
    }

    if (colorizeEditActive) {
      viewer.commitColorizeEdit();
      colorizeEditActive = false;
    }

    if (moveActive && nextTool !== "move") {
      viewer.commitMoveSelected();
      moveActive = false;
    }

    tool = nextTool;
    canvas.dataset.toolMode = tool;
    viewer.setOrbitControlsEnabled(tool === "orbit");
    viewer.setSelectionHighlightVisible(tool !== "hsv" && tool !== "colorize");
    selectionControls.hidden = !isSelectionTool(tool);
    hsvControls.hidden = tool !== "hsv";
    colorizeControls.hidden = tool !== "colorize";
    moveControls.hidden = tool !== "move";
    visualizer.hidden = tool !== "brushSelect";

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
    } else if (tool === "move" && !moveActive) {
      moveActive = viewer.beginMoveSelected();
    }
  }

  function updateVisualizerSize(): void {
    const diameter = selectionOptions.screenRadius * 2;
    visualizer.style.width = `${diameter}px`;
    visualizer.style.height = `${diameter}px`;
  }

  updateVisualizerSize();
  setTool(tool);

  return {
    getTool: () => tool,
    getSelectionOptions: () => ({ ...selectionOptions }),
    getHsvAdjust: () => ({ ...hsvAdjust }),
    setMoveActive: (active) => {
      moveActive = active;
    },
  };
}

function isSelectionTool(tool: EditorTool): boolean {
  return tool === "brushSelect" || tool === "marqueeSelect" || tool === "lassoSelect";
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
  input.type = "number";
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

function parseQualityMode(value: string | null): RenderQualityMode {
  if (value === "quality" || value === "balanced" || value === "gpu-balanced" || value === "performance") {
    return value;
  }

  return "performance";
}
