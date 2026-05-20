import { OrbitCamera } from "../camera/orbit-camera";
import { GpuContext } from "../renderer/gpu-context";
import { GaussianRenderer } from "../renderer/gaussian-renderer";

export interface GaussianSplatViewerOptions {
  canvas: HTMLCanvasElement;
}

export default class GaussianSplatViewer {
  private readonly gpu: GpuContext;
  private readonly renderer: GaussianRenderer;
  private readonly camera: OrbitCamera;
  private rafId: number | null = null;
  private isRunning = false;

  private constructor(
    gpu: GpuContext,
    renderer: GaussianRenderer,
    camera: OrbitCamera,
  ) {
    this.gpu = gpu;
    this.renderer = renderer;
    this.camera = camera;
  }

  static async create(
    options: GaussianSplatViewerOptions,
  ): Promise<GaussianSplatViewer> {
    const gpu = await GpuContext.create(options.canvas);
    const renderer = new GaussianRenderer(gpu);
    const camera = new OrbitCamera(gpu.device, gpu.canvas);

    return new GaussianSplatViewer(gpu, renderer, camera);
  }

  start(): void {
    if (this.isRunning) {
      return;
    }

    this.isRunning = true;
    this.loop();
    window.addEventListener("resize", this.resize);
  }

  stop(): void {
    if (!this.isRunning) {
      return;
    }

    if (this.rafId !== null) {
      cancelAnimationFrame(this.rafId);
      this.rafId = null;
    }

    window.removeEventListener("resize", this.resize);
    this.isRunning = false;
  }

  dispose(): void {
    this.stop();
    this.camera.dispose();
    this.renderer.dispose();
    this.gpu.dispose();
  }

  private readonly resize = (): void => {
    this.gpu.resize();
  };

  private readonly loop = (): void => {
    this.camera.update();
    this.renderer.render();
    this.rafId = requestAnimationFrame(this.loop);
  };
}
