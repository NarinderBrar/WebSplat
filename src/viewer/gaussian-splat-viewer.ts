import { OrbitCamera } from "../camera/orbit-camera";
import { GpuChunkCullPass } from "../passes/gpuChunkCullPass";
import { IdPickingPass } from "../passes/idPickingPass";
import { GpuContext } from "../renderer/gpu-context";
import { GaussianRenderer } from "../renderer/gaussian-renderer";
import { createDemoSplatSource } from "../splats/demo-splat-source";
import { SplatBuffer } from "../splats/splatBuffer";
import { loadSplatSource } from "../splats/splatLoader";
import { DebugStatsOverlay } from "./debug-stats-overlay";
import { SplatWorld } from "../world/splat-world";
import type { RenderQualityMode } from "../world/types";

export interface GaussianSplatViewerOptions {
  canvas: HTMLCanvasElement;
  source?: string | ArrayBuffer;
  qualityMode?: RenderQualityMode;
}

export default class GaussianSplatViewer {
  private readonly gpu: GpuContext;
  private readonly renderer: GaussianRenderer;
  private readonly camera: OrbitCamera;
  private readonly splatBuffer: SplatBuffer;
  private readonly world: SplatWorld;
  private readonly gpuChunkCullPass: GpuChunkCullPass;
  private readonly idPickingPass: IdPickingPass;
  private readonly debugStatsOverlay: DebugStatsOverlay;
  private readonly qualityMode: RenderQualityMode;
  private rafId: number | null = null;
  private isRunning = false;
  private lastFrameMs = 16.6;

  private constructor(
    gpu: GpuContext,
    renderer: GaussianRenderer,
    camera: OrbitCamera,
    splatBuffer: SplatBuffer,
    world: SplatWorld,
    gpuChunkCullPass: GpuChunkCullPass,
    idPickingPass: IdPickingPass,
    debugStatsOverlay: DebugStatsOverlay,
    qualityMode: RenderQualityMode,
  ) {
    this.gpu = gpu;
    this.renderer = renderer;
    this.camera = camera;
    this.splatBuffer = splatBuffer;
    this.world = world;
    this.gpuChunkCullPass = gpuChunkCullPass;
    this.idPickingPass = idPickingPass;
    this.debugStatsOverlay = debugStatsOverlay;
    this.qualityMode = qualityMode;
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
    splatBuffer.createSelectionMaskBuffer(gpu.device);
    splatBuffer.createIndirectArgsBuffer(gpu.device);
    splatBuffer.createChunkOrderCache(world.getChunks());

    const chunkMetadataBuffer = splatBuffer.getChunkMetadataBuffer();

    if (!chunkMetadataBuffer) {
      throw new Error("Chunk metadata buffer must exist before GPU culling is created.");
    }

    const gpuChunkCullPass = new GpuChunkCullPass(gpu.device, {
      chunkMetadataBuffer,
      cameraBindGroupLayout: renderer.getCameraBindGroupLayout(),
      chunkCount: world.getChunks().length,
      splatCount: world.getSplatData().count,
    });
    renderer.setSplatBuffer(splatBuffer);
    const idPickingPass = new IdPickingPass(gpu.device, renderer.getCameraBindGroupLayout());
    idPickingPass.setSplatBuffer(splatBuffer);

    return new GaussianSplatViewer(
      gpu,
      renderer,
      camera,
      splatBuffer,
      world,
      gpuChunkCullPass,
      idPickingPass,
      new DebugStatsOverlay(),
      options.qualityMode ?? "balanced",
    );
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
    this.debugStatsOverlay.dispose();
    this.idPickingPass.dispose();
    this.gpuChunkCullPass.dispose();
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

  async pickSplatAt(clientX: number, clientY: number): Promise<number | null> {
    const rect = this.gpu.canvas.getBoundingClientRect();
    const x = ((clientX - rect.left) / rect.width) * this.gpu.canvas.width;
    const y = ((clientY - rect.top) / rect.height) * this.gpu.canvas.height;
    const indirectArgsBuffer = this.splatBuffer.getIndirectArgsBuffer();

    if (!indirectArgsBuffer) {
      return null;
    }

    this.idPickingPass.resize(this.gpu.canvas.width, this.gpu.canvas.height);

    const encoder = this.gpu.beginFrame();
    this.idPickingPass.encode(encoder, this.camera.uniforms, indirectArgsBuffer);
    this.idPickingPass.copyPixelToReadback(encoder, x, y);
    this.gpu.submit(encoder);

    return this.idPickingPass.readCopiedSplatId();
  }

  private readonly resize = (): void => {
    this.gpu.resize();
  };

  private readonly loop = (): void => {
    const frameStart = performance.now();
    this.camera.update();
    const cullStart = performance.now();
    this.world.updateVisibility(this.camera.getViewProjectionMatrix());
    const cpuCullMs = performance.now() - cullStart;
    const plans = this.world.createRenderPlans(
      this.camera.getViewMatrix(),
      this.camera.getProjectionMatrix(),
      this.gpu.canvas.height,
      this.qualityMode,
      this.lastFrameMs,
    );
    const visibleTelemetry = this.splatBuffer.buildVisibleSplatIndicesFromChunkPlans(
      plans,
      this.camera.getViewMatrix(),
      this.gpu.device,
    );
    this.renderer.render(this.camera.uniforms);
    this.lastFrameMs = performance.now() - frameStart;
    this.debugStatsOverlay.update(this.world.getDebugStats(
      this.splatBuffer.getRenderCount(),
      {
        frameMs: this.lastFrameMs,
        estimatedFps: 1000 / Math.max(0.001, this.lastFrameMs),
        cpuCullMs,
        localOrderRefreshMs: visibleTelemetry.localOrderRefreshMs,
        visibleIndexBuildMs: visibleTelemetry.visibleIndexBuildMs,
      },
    ));
    this.rafId = requestAnimationFrame(this.loop);
  };
}
