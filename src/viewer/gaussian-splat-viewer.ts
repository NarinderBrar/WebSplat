import { OrbitCamera } from "../camera/orbit-camera";
import { GpuChunkCullPass } from "../passes/gpuChunkCullPass";
import { GpuDepthBinPass } from "../passes/gpuDepthBinPass";
import { GpuTilePressurePass, type GpuTilePressureTelemetry } from "../passes/gpuTilePressurePass";
import { IdPickingPass } from "../passes/idPickingPass";
import { GpuContext } from "../renderer/gpu-context";
import { GaussianRenderer } from "../renderer/gaussian-renderer";
import { createDemoSplatSource } from "../splats/demo-splat-source";
import { SplatBuffer } from "../splats/splatBuffer";
import { loadSplatSource } from "../splats/splatLoader";
import { DebugStatsOverlay } from "./debug-stats-overlay";
import { SplatWorld } from "../world/splat-world";
import type { GpuRenderBackend, RenderQualityMode } from "../world/types";
import type { TileBudgetOptions } from "../world/types";

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
  private readonly gpuDepthBinPass: GpuDepthBinPass | null;
  private readonly gpuTilePressurePass: GpuTilePressurePass | null;
  private readonly idPickingPass: IdPickingPass;
  private readonly debugStatsOverlay: DebugStatsOverlay;
  private readonly qualityMode: RenderQualityMode;
  private readonly renderBackend: GpuRenderBackend;
  private rafId: number | null = null;
  private isRunning = false;
  private lastFrameMs = 16.6;
  private lastRafTime = performance.now();
  private lastGpuTilePressure: GpuTilePressureTelemetry = {
    testedSplats: 0,
    maxTileSplats: 0,
    overloadedTiles: 0,
  };

  private constructor(
    gpu: GpuContext,
    renderer: GaussianRenderer,
    camera: OrbitCamera,
    splatBuffer: SplatBuffer,
    world: SplatWorld,
    gpuChunkCullPass: GpuChunkCullPass,
    gpuDepthBinPass: GpuDepthBinPass | null,
    gpuTilePressurePass: GpuTilePressurePass | null,
    idPickingPass: IdPickingPass,
    debugStatsOverlay: DebugStatsOverlay,
    qualityMode: RenderQualityMode,
    renderBackend: GpuRenderBackend,
  ) {
    this.gpu = gpu;
    this.renderer = renderer;
    this.camera = camera;
    this.splatBuffer = splatBuffer;
    this.world = world;
    this.gpuChunkCullPass = gpuChunkCullPass;
    this.gpuDepthBinPass = gpuDepthBinPass;
    this.gpuTilePressurePass = gpuTilePressurePass;
    this.idPickingPass = idPickingPass;
    this.debugStatsOverlay = debugStatsOverlay;
    this.qualityMode = qualityMode;
    this.renderBackend = renderBackend;
  }

  static async create(
    options: GaussianSplatViewerOptions,
  ): Promise<GaussianSplatViewer> {
    const gpu = await GpuContext.create(options.canvas);
    gpu.setRenderScale(getRenderScale(options.qualityMode ?? "gpu-balanced"));
    const renderer = new GaussianRenderer(gpu);
    renderer.setQualityLevel(0);
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
    const renderBackend = getRenderBackend(options.qualityMode ?? "performance");
    const gpuDepthBinPass = createGpuDepthBinPass(
      gpu.device,
      renderer.getCameraBindGroupLayout(),
      splatBuffer,
      world.getSplatData().count,
      options.qualityMode ?? "performance",
      gpu.canvas.height,
      renderBackend,
    );

    if (gpuDepthBinPass) {
      const buffers = gpuDepthBinPass.getBuffers();
      splatBuffer.adoptGpuVisibleBuffers(
        buffers.visibleSplatIndicesBuffer,
        buffers.indirectArgsBuffer,
      );
      renderer.setGpuDepthBinPass(gpuDepthBinPass);
    }

    const gpuTilePressurePass = createGpuTilePressurePass(
      gpu.device,
      renderer.getCameraBindGroupLayout(),
      splatBuffer,
      world.getSplatData().count,
      options.qualityMode ?? "performance",
    );

    if (gpuTilePressurePass) {
      renderer.setGpuTilePressurePass(gpuTilePressurePass);
    }

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
      gpuDepthBinPass,
      gpuTilePressurePass,
      idPickingPass,
      new DebugStatsOverlay(),
      options.qualityMode ?? "performance",
      renderBackend,
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
    this.gpuTilePressurePass?.dispose();
    this.gpuDepthBinPass?.dispose();
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

  async selectSimilarColorAt(clientX: number, clientY: number): Promise<number> {
    const splatId = await this.pickSplatAt(clientX, clientY);

    if (splatId === null) {
      this.splatBuffer.clearSelection(this.gpu.device);
      return 0;
    }

    const picked = this.world.lookupSplat(splatId);

    if (!picked) {
      this.splatBuffer.clearSelection(this.gpu.device);
      return 0;
    }

    return this.splatBuffer.selectConnectedSimilarColor(
      this.gpu.device,
      picked.globalIndex,
      picked.chunk,
      this.world,
    );
  }

  async selectSimilarColorInRadiusAt(
    clientX: number,
    clientY: number,
    screenRadius: number,
    colorThreshold: number,
  ): Promise<number> {
    const rect = this.gpu.canvas.getBoundingClientRect();
    const x = ((clientX - rect.left) / rect.width) * this.gpu.canvas.width;
    const y = ((clientY - rect.top) / rect.height) * this.gpu.canvas.height;

    return this.splatBuffer.selectConnectedSimilarColorInScreenRadiusProgressive(
      this.gpu.device,
      this.world,
      {
        screenX: x,
        screenY: y,
        screenRadius: Math.max(1, screenRadius * (this.gpu.canvas.width / Math.max(1, rect.width))),
        viewportWidth: this.gpu.canvas.width,
        viewportHeight: this.gpu.canvas.height,
        viewMatrix: this.camera.getViewMatrix(),
        viewProjectionMatrix: this.camera.getViewProjectionMatrix(),
        colorThreshold,
      },
    );
  }

  private readonly resize = (): void => {
    this.gpu.resize();
  };

  private readonly loop = (): void => {
    const frameStart = performance.now();
    this.lastFrameMs = frameStart - this.lastRafTime;
    this.lastRafTime = frameStart;
    this.camera.update();
    let cpuCullMs = 0;
    let localOrderRefreshMs = 0;
    let visibleIndexBuildMs = 0;
    let tileCulledSplats = 0;
    let tileTestedSplats = 0;
    let tileProtectedSplats = 0;
    const gpuTilePressure = this.gpuTilePressurePass?.pollTelemetry() ?? this.lastGpuTilePressure;
    this.lastGpuTilePressure = gpuTilePressure;

    if (this.renderBackend === "cpuChunkBinned") {
      const cullStart = performance.now();
      this.world.updateVisibility(this.camera.getViewProjectionMatrix());
      cpuCullMs = performance.now() - cullStart;
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
        createTileBudgetOptions(
          this.qualityMode,
          this.gpu.canvas.width,
          this.gpu.canvas.height,
          this.camera.getViewProjectionMatrix(),
          gpuTilePressure,
        ),
      );
      localOrderRefreshMs = visibleTelemetry.localOrderRefreshMs;
      visibleIndexBuildMs = visibleTelemetry.visibleIndexBuildMs;
      tileCulledSplats = visibleTelemetry.tileCulledSplats;
      tileTestedSplats = visibleTelemetry.tileTestedSplats;
      tileProtectedSplats = visibleTelemetry.tileProtectedSplats;
    }

    this.renderer.render(this.camera.uniforms);
    this.debugStatsOverlay.update(this.world.getDebugStats(
      this.splatBuffer.getRenderCount(),
      {
        backend: this.renderBackend,
        frameMs: this.lastFrameMs,
        estimatedFps: 1000 / Math.max(0.001, this.lastFrameMs),
        cpuCullMs,
        localOrderRefreshMs,
        visibleIndexBuildMs,
        tileCulledSplats,
        tileTestedSplats,
        tileProtectedSplats,
        gpuTileTestedSplats: gpuTilePressure.testedSplats,
        gpuMaxTileSplats: gpuTilePressure.maxTileSplats,
        gpuOverloadedTiles: gpuTilePressure.overloadedTiles,
      },
    ));
    this.rafId = requestAnimationFrame(this.loop);
  };
}

function getRenderScale(qualityMode: RenderQualityMode): number {
  if (qualityMode === "performance") {
    return 0.6;
  }

  if (qualityMode === "gpu-balanced") {
    return 0.8;
  }

  return 1;
}

function getRenderBackend(qualityMode: RenderQualityMode): GpuRenderBackend {
  void qualityMode;
  return "cpuChunkBinned";
}

function createTileBudgetOptions(
  qualityMode: RenderQualityMode,
  viewportWidth: number,
  viewportHeight: number,
  viewProjectionMatrix: Float32Array,
  gpuTilePressure?: GpuTilePressureTelemetry,
): TileBudgetOptions {
  if (qualityMode === "quality") {
    return {
      enabled: false,
      tileSize: 64,
      maxSplatsPerTile: Number.POSITIVE_INFINITY,
      maxProtectedScreenRadius: 0,
      protectedNearDepth: Number.POSITIVE_INFINITY,
      viewportWidth,
      viewportHeight,
      viewProjectionMatrix,
    };
  }

  const maxSplatsPerTile = chooseCpuTileBudget(qualityMode, gpuTilePressure);

  return {
    enabled: qualityMode === "performance" || qualityMode === "gpu-balanced",
    tileSize: 64,
    maxSplatsPerTile,
    maxProtectedScreenRadius: qualityMode === "performance" ? 32 : 24,
    protectedNearDepth: qualityMode === "performance" ? 25 : 40,
    viewportWidth,
    viewportHeight,
    viewProjectionMatrix,
  };
}

function chooseCpuTileBudget(
  qualityMode: RenderQualityMode,
  gpuTilePressure?: GpuTilePressureTelemetry,
): number {
  const baseBudget = qualityMode === "performance" ? 6_000 : 12_000;

  if (qualityMode !== "performance" || !gpuTilePressure) {
    return baseBudget;
  }

  if (gpuTilePressure.maxTileSplats > 24_000) {
    return 3_500;
  }

  if (gpuTilePressure.maxTileSplats > 12_000) {
    return 4_500;
  }

  return baseBudget;
}

function createGpuTilePressurePass(
  device: GPUDevice,
  cameraBindGroupLayout: GPUBindGroupLayout,
  splatBuffer: SplatBuffer,
  splatCount: number,
  qualityMode: RenderQualityMode,
): GpuTilePressurePass | null {
  if (qualityMode === "quality") {
    return null;
  }

  const positionBuffer = splatBuffer.getPositionBuffer();
  const visibleSplatIndicesBuffer = splatBuffer.getVisibleSplatIndicesBuffer();

  if (!positionBuffer || !visibleSplatIndicesBuffer) {
    return null;
  }

  return new GpuTilePressurePass(device, {
    cameraBindGroupLayout,
    positionBuffer,
    visibleSplatIndicesBuffer,
    maxVisibleSplatCount: splatCount,
    tileSize: 64,
    overloadThreshold: qualityMode === "performance" ? 6_000 : 12_000,
  });
}

function createGpuDepthBinPass(
  device: GPUDevice,
  cameraBindGroupLayout: GPUBindGroupLayout,
  splatBuffer: SplatBuffer,
  splatCount: number,
  qualityMode: RenderQualityMode,
  viewportHeight: number,
  renderBackend: GpuRenderBackend,
): GpuDepthBinPass | null {
  if (renderBackend !== "gpuDepthBinned") {
    return null;
  }

  const positionBuffer = splatBuffer.getPositionBuffer();
  const covarianceBuffer = splatBuffer.getCovarianceBuffer();
  const opacityBuffer = splatBuffer.getOpacityBuffer();

  if (!positionBuffer || !covarianceBuffer || !opacityBuffer) {
    throw new Error("Position, covariance and opacity buffers must exist before GPU depth binning is created.");
  }

  return new GpuDepthBinPass(device, {
    cameraBindGroupLayout,
    positionBuffer,
    covarianceBuffer,
    opacityBuffer,
    splatCount,
    qualityMode,
    viewportHeight,
  });
}
