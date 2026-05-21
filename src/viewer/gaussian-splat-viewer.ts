import { OrbitCamera } from "../camera/orbit-camera";
import { GpuContext } from "../renderer/gpu-context";
import { GaussianRenderer } from "../renderer/gaussian-renderer";
import { createDemoSplatSource } from "../splats/demo-splat-source";
import { SplatBuffer } from "../splats/splatBuffer";
import { loadSplatSource } from "../splats/splatLoader";

export interface GaussianSplatViewerOptions {
  canvas: HTMLCanvasElement;
  source?: string | ArrayBuffer;
}

export default class GaussianSplatViewer {
  private readonly gpu: GpuContext;
  private readonly renderer: GaussianRenderer;
  private readonly camera: OrbitCamera;
  private readonly splatBuffer: SplatBuffer;
  private rafId: number | null = null;
  private isRunning = false;

  private constructor(
    gpu: GpuContext,
    renderer: GaussianRenderer,
    camera: OrbitCamera,
    splatBuffer: SplatBuffer,
  ) {
    this.gpu = gpu;
    this.renderer = renderer;
    this.camera = camera;
    this.splatBuffer = splatBuffer;
  }

  static async create(
    options: GaussianSplatViewerOptions,
  ): Promise<GaussianSplatViewer> {
    const gpu = await GpuContext.create(options.canvas);
    const renderer = new GaussianRenderer(gpu);
    const camera = new OrbitCamera(
      gpu.device,
      gpu.canvas,
      renderer.getCameraBindGroupLayout(),
    );
    const splatData = await loadSplatSource(options.source ?? createDemoSplatSource());
    const splatBuffer = new SplatBuffer();

    splatBuffer.setData(splatData);
    splatBuffer.createBuffers(gpu.device);
    renderer.setSplatBuffer(splatBuffer);

    return new GaussianSplatViewer(gpu, renderer, camera, splatBuffer);
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
    this.splatBuffer.dispose();
    this.camera.dispose();
    this.renderer.dispose();
    this.gpu.dispose();
  }

  getBabylonCamera() {
    return this.camera.getBabylonCamera();
  }

  getBabylonScene() {
    return this.camera.getBabylonScene();
  }

  private readonly resize = (): void => {
    this.gpu.resize();
  };

  private readonly loop = (): void => {
    this.camera.update();
    this.splatBuffer.sortByView(this.camera.getViewMatrix(), this.gpu.device);
    this.renderer.render(this.camera.uniforms);
    this.rafId = requestAnimationFrame(this.loop);
  };
}
