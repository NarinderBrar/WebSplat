import { OrbitCamera } from "../camera/orbit-camera";
import { GpuContext } from "../renderer/gpu-context";
import { GaussianRenderer } from "../renderer/gaussian-renderer";
import { createDemoSplatSource } from "../splats/demo-splat-source";
import { SplatBuffer } from "../splats/splatBuffer";
import { loadSplatSource } from "../splats/splatLoader";
import { SplatWorld } from "../world/splat-world";

export interface GaussianSplatViewerOptions {
  canvas: HTMLCanvasElement;
  source?: string | ArrayBuffer;
}

export default class GaussianSplatViewer {
  private readonly gpu: GpuContext;
  private readonly renderer: GaussianRenderer;
  private readonly camera: OrbitCamera;
  private readonly splatBuffer: SplatBuffer;
  private readonly world: SplatWorld;
  private rafId: number | null = null;
  private isRunning = false;

  private constructor(
    gpu: GpuContext,
    renderer: GaussianRenderer,
    camera: OrbitCamera,
    splatBuffer: SplatBuffer,
    world: SplatWorld,
  ) {
    this.gpu = gpu;
    this.renderer = renderer;
    this.camera = camera;
    this.splatBuffer = splatBuffer;
    this.world = world;
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
    const world = SplatWorld.fromSplatData(splatData);
    const splatBuffer = new SplatBuffer();

    splatBuffer.setData(world.getSplatData());
    splatBuffer.createBuffers(gpu.device);
    splatBuffer.createStableIdBuffers(gpu.device, world.getSplatData());
    splatBuffer.createChunkMetadataBuffer(gpu.device, world.packGpuMetadata());
    renderer.setSplatBuffer(splatBuffer);

    return new GaussianSplatViewer(gpu, renderer, camera, splatBuffer, world);
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

  getWorld(): SplatWorld {
    return this.world;
  }

  private readonly resize = (): void => {
    this.gpu.resize();
  };

  private readonly loop = (): void => {
    this.camera.update();
    const visibility = this.world.updateVisibility(this.camera.getViewProjectionMatrix());
    this.splatBuffer.sortByView(
      this.camera.getViewMatrix(),
      this.gpu.device,
      visibility.chunks,
    );
    this.renderer.render(this.camera.uniforms);
    this.rafId = requestAnimationFrame(this.loop);
  };
}
