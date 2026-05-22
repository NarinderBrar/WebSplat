import GaussianSplatViewer from "./viewer/gaussian-splat-viewer.ts";
import type { RenderQualityMode } from "./world/types.ts";

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
  } catch (error) {
    console.error(error);

    const warning = document.createElement("div");
    warning.className = "webgpu-warning";
    warning.textContent = "WebGPU not supported";
    document.body.append(warning);
  }
}

main();

function parseQualityMode(value: string | null): RenderQualityMode {
  if (value === "quality" || value === "performance") {
    return value;
  }

  return "balanced";
}
