import GaussianSplatViewer from "./viewer/gaussian-splat-viewer.ts";
import type { RenderQualityMode } from "./world/types.ts";

type InteractionMode = "orbit" | "select";
interface SelectionToolOptions {
  colorThreshold: number;
  screenRadius: number;
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
    const modeState = createToolModeToolbar(renderCanvas);
    bindSelectionClick(renderCanvas, viewer, modeState);
  } catch (error) {
    console.error(error);

    const warning = document.createElement("div");
    warning.className = "webgpu-warning";
    warning.textContent = "WebGPU not supported";
    document.body.append(warning);
  }
}

main();

function bindSelectionClick(
  canvas: HTMLCanvasElement,
  viewer: GaussianSplatViewer,
  modeState: { getMode(): InteractionMode; getSelectionOptions(): SelectionToolOptions },
): void {
  let pointerDownX = 0;
  let pointerDownY = 0;
  let pointerDownButton = -1;

  canvas.addEventListener("pointerdown", (event) => {
    pointerDownX = event.clientX;
    pointerDownY = event.clientY;
    pointerDownButton = event.button;
  });

  canvas.addEventListener("pointerup", (event) => {
    if (pointerDownButton !== 0) {
      return;
    }

    if (modeState.getMode() !== "select") {
      return;
    }

    const dx = event.clientX - pointerDownX;
    const dy = event.clientY - pointerDownY;

    if (dx * dx + dy * dy > 16) {
      return;
    }

    const options = modeState.getSelectionOptions();
    viewer.selectSimilarColorInRadiusAt(
      event.clientX,
      event.clientY,
      options.screenRadius,
      options.colorThreshold,
    );
  });
}

function createToolModeToolbar(
  canvas: HTMLCanvasElement,
): { getMode(): InteractionMode; getSelectionOptions(): SelectionToolOptions } {
  let mode: InteractionMode = "orbit";
  const selectionOptions: SelectionToolOptions = {
    colorThreshold: 0.14,
    screenRadius: 12,
  };
  const toolbar = document.createElement("div");
  const orbitButton = createToolButton("Orbit", orbitIconSvg());
  const selectButton = createToolButton("Color select", selectIconSvg());
  const controls = document.createElement("div");
  const thresholdInput = createNumberInput("Threshold", selectionOptions.colorThreshold, 0.01, 1, 0.01);
  const radiusInput = createNumberInput("Radius", selectionOptions.screenRadius, 1, 96, 1);
  const visualizer = document.createElement("div");

  toolbar.className = "tool-mode-toolbar";
  controls.className = "selection-controls";
  controls.append(thresholdInput.label, radiusInput.label);
  visualizer.className = "selection-radius-visualizer";
  toolbar.append(orbitButton, selectButton, controls);
  document.body.append(toolbar);
  document.body.append(visualizer);

  const setMode = (nextMode: InteractionMode): void => {
    mode = nextMode;
    canvas.dataset.toolMode = mode;
    controls.hidden = mode !== "select";
    visualizer.hidden = mode !== "select";
    orbitButton.classList.toggle("is-active", mode === "orbit");
    selectButton.classList.toggle("is-active", mode === "select");
    orbitButton.setAttribute("aria-pressed", String(mode === "orbit"));
    selectButton.setAttribute("aria-pressed", String(mode === "select"));
  };

  const updateVisualizerSize = (): void => {
    const diameter = selectionOptions.screenRadius * 2;
    visualizer.style.width = `${diameter}px`;
    visualizer.style.height = `${diameter}px`;
  };

  thresholdInput.input.addEventListener("input", () => {
    selectionOptions.colorThreshold = thresholdInput.input.valueAsNumber || 0.14;
  });
  radiusInput.input.addEventListener("input", () => {
    selectionOptions.screenRadius = radiusInput.input.valueAsNumber || 12;
    updateVisualizerSize();
  });
  canvas.addEventListener("pointermove", (event) => {
    if (mode !== "select") {
      return;
    }

    visualizer.style.transform = `translate(${event.clientX - selectionOptions.screenRadius}px, ${event.clientY - selectionOptions.screenRadius}px)`;
  });
  canvas.addEventListener("pointerleave", () => {
    visualizer.classList.add("is-hidden");
  });
  canvas.addEventListener("pointerenter", () => {
    visualizer.classList.remove("is-hidden");
  });
  orbitButton.addEventListener("click", () => setMode("orbit"));
  selectButton.addEventListener("click", () => setMode("select"));
  updateVisualizerSize();
  setMode(mode);

  return {
    getMode: () => mode,
    getSelectionOptions: () => ({ ...selectionOptions }),
  };
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

function orbitIconSvg(): string {
  return `
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M4 12c0-3.7 3.6-6.7 8-6.7s8 3 8 6.7-3.6 6.7-8 6.7" />
      <path d="M8 15.4 4.6 18.8 8 22.2" />
      <path d="M12 9.2a2.8 2.8 0 1 1 0 5.6 2.8 2.8 0 0 1 0-5.6Z" />
    </svg>
  `;
}

function selectIconSvg(): string {
  return `
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="m5 3 6.8 16.4 2.1-6.1 6.1-2.1L5 3Z" />
      <path d="m14 14 4.8 4.8" />
    </svg>
  `;
}

function parseQualityMode(value: string | null): RenderQualityMode {
  if (value === "quality" || value === "balanced" || value === "gpu-balanced" || value === "performance") {
    return value;
  }

  return "performance";
}
