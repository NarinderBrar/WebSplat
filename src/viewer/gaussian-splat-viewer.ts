import type { Scene } from "@babylonjs/core/scene";
import type { WebGPUEngine } from "@babylonjs/core/Engines/webgpuEngine";
import BabylonEngine from "../engine/babylon-engine.ts";
import ViewerScene from "../engine/viewer-scene.ts";

export interface GaussianSplatViewerOptions {
  canvas: HTMLCanvasElement;
}

export default class GaussianSplatViewer {
  private readonly engine: WebGPUEngine;
  private readonly scene: Scene;
  private isRunning = false;

  private constructor(engine: WebGPUEngine, scene: Scene) {
    this.engine = engine;
    this.scene = scene;
  }

  static async create(
    options: GaussianSplatViewerOptions,
  ): Promise<GaussianSplatViewer> {
    const engine = await BabylonEngine.create(options.canvas);
    const viewerScene = new ViewerScene(engine, options.canvas);

    return new GaussianSplatViewer(engine, viewerScene.scene);
  }

  start(): void {
    if (this.isRunning) {
      return;
    }

    this.isRunning = true;
    this.engine.runRenderLoop(() => {
      this.scene.render();
    });

    window.addEventListener("resize", this.resize);
  }

  stop(): void {
    if (!this.isRunning) {
      return;
    }

    this.engine.stopRenderLoop();
    window.removeEventListener("resize", this.resize);
    this.isRunning = false;
  }

  dispose(): void {
    this.stop();
    this.scene.dispose();
    this.engine.dispose();
  }

  private readonly resize = (): void => {
    this.engine.resize();
  };
}
