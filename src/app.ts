import GaussianSplatViewer from "./viewer/gaussian-splat-viewer.ts";

const canvas = document.querySelector<HTMLCanvasElement>("#renderCanvas");
if (!canvas) throw new Error("Canvas element #renderCanvas was not found.");

const renderCanvas = canvas;
async function main(): Promise<void> {
  try {
    const viewer = await GaussianSplatViewer.create({ canvas: renderCanvas });
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
