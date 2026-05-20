import { WebGPUEngine } from "@babylonjs/core/Engines/webgpuEngine";

export default class BabylonEngine {
  private constructor() {}

  static async create(canvas: HTMLCanvasElement): Promise<WebGPUEngine> {
    const isSupported = await WebGPUEngine.IsSupportedAsync;

    if (!isSupported) {
      throw new Error("WebGPU is not available in this browser.");
    }

    const engine = new WebGPUEngine(canvas, {
      antialias: true,
      adaptToDeviceRatio: true,
    });

    await engine.initAsync();

    return engine;
  }
}
