import { GizmoManager } from "@babylonjs/core/Gizmos/gizmoManager";
import { Quaternion, Vector3 } from "@babylonjs/core/Maths/math.vector";
import { TransformNode } from "@babylonjs/core/Meshes/transformNode";
import { OrbitCamera } from "../camera/orbit-camera";
import type { HsvAdjust, ScreenPoint, ScreenRect, SelectionMode } from "../editor/types";
import { GpuChunkCullPass } from "../passes/gpuChunkCullPass";
import { ComputeSortPass } from "../passes/computeSortPass";
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
  optimized?: boolean;
}

export default class GaussianSplatViewer {
  private readonly gpu: GpuContext;
  private readonly renderer: GaussianRenderer;
  private readonly camera: OrbitCamera;
  private readonly splatBuffer: SplatBuffer;
  private world: SplatWorld;
  private readonly gpuChunkCullPass: GpuChunkCullPass;
  private readonly gpuDepthBinPass: GpuDepthBinPass | null;
  private readonly gpuTilePressurePass: GpuTilePressurePass | null;
  private readonly computeSortPass: ComputeSortPass | null;
  private readonly idPickingPass: IdPickingPass;
  private readonly debugStatsOverlay: DebugStatsOverlay;
  private readonly qualityMode: RenderQualityMode;
  private readonly renderBackend: GpuRenderBackend;
  private readonly optimized: boolean;
  private rafId: number | null = null;
  private isRunning = false;
  private lastFrameMs = 16.6;
  private lastRafTime = performance.now();
  private lastGpuTilePressure: GpuTilePressureTelemetry = {
    testedSplats: 0,
    maxTileSplats: 0,
    overloadedTiles: 0,
  };
  private moveGizmoManager: GizmoManager | null = null;
  private moveTransformNode: TransformNode | null = null;
  private moveStartPosition: Vector3 | null = null;
  private transformStartRotation: Quaternion | null = null;
  private transformStartScaling: Vector3 | null = null;
  private moveActive = false;
  private densityCullingEnabled = false;

  private constructor(
    gpu: GpuContext,
    renderer: GaussianRenderer,
    camera: OrbitCamera,
    splatBuffer: SplatBuffer,
    world: SplatWorld,
    gpuChunkCullPass: GpuChunkCullPass,
    gpuDepthBinPass: GpuDepthBinPass | null,
    gpuTilePressurePass: GpuTilePressurePass | null,
    computeSortPass: ComputeSortPass | null,
    idPickingPass: IdPickingPass,
    debugStatsOverlay: DebugStatsOverlay,
    qualityMode: RenderQualityMode,
    renderBackend: GpuRenderBackend,
    optimized: boolean,
  ) {
    this.gpu = gpu;
    this.renderer = renderer;
    this.camera = camera;
    this.splatBuffer = splatBuffer;
    this.world = world;
    this.gpuChunkCullPass = gpuChunkCullPass;
    this.gpuDepthBinPass = gpuDepthBinPass;
    this.gpuTilePressurePass = gpuTilePressurePass;
    this.computeSortPass = computeSortPass;
    this.idPickingPass = idPickingPass;
    this.debugStatsOverlay = debugStatsOverlay;
    this.qualityMode = qualityMode;
    this.renderBackend = renderBackend;
    this.optimized = optimized;
  }

  static async create(
    options: GaussianSplatViewerOptions,
  ): Promise<GaussianSplatViewer> {
    const optimized = options.optimized ?? false;
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
    splatBuffer.createHiddenMaskBuffer(gpu.device);
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
    const renderBackend = optimized
      ? getRenderBackend(options.qualityMode ?? "performance")
      : "cpuChunkBinned";
    const gpuDepthBinPass = createGpuDepthBinPass(
      gpu.device,
      renderer.getCameraBindGroupLayout(),
      splatBuffer,
      world.getSplatData().count,
      options.qualityMode ?? "performance",
      gpu.canvas.height,
      renderBackend,
      optimized,
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
      optimized,
    );

    if (gpuTilePressurePass) {
      renderer.setGpuTilePressurePass(gpuTilePressurePass);
    }

    renderer.setSplatBuffer(splatBuffer);

    const computeSortPass = createComputeSortPass(
      gpu.device,
      renderer.getCameraBindGroupLayout(),
      world.getSplatData().count,
      renderBackend,
      optimized,
    );

    if (computeSortPass) {
      renderer.setComputeSortPass(computeSortPass);
    }

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
      computeSortPass,
      idPickingPass,
      new DebugStatsOverlay(),
      options.qualityMode ?? "performance",
      renderBackend,
      optimized,
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
    this.moveGizmoManager?.dispose();
    this.moveTransformNode?.dispose();
    this.debugStatsOverlay.dispose();
    this.idPickingPass.dispose();
    this.computeSortPass?.dispose();
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

  setVisualizationMode(mode: number): void {
    this.renderer.setVisualizationMode(mode);
  }

  setDensityCulling(enabled: boolean): void {
    this.densityCullingEnabled = enabled;
  }

  getWorld(): SplatWorld {
    return this.world;
  }

  setOrbitControlsEnabled(enabled: boolean): void {
    this.camera.setControlsEnabled(enabled);
    this.camera.setGizmoPointerEnabled(enabled);
  }

  setSelectionHighlightVisible(visible: boolean): void {
    this.splatBuffer.setSelectionHighlightVisible(this.gpu.device, visible);
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

  async selectSimilarColorAt(
    clientX: number,
    clientY: number,
    colorThreshold: number,
    selectionMode: SelectionMode = "normal",
    selectBehind: boolean = true,
    depthRangeFactor: number = 2.5,
  ): Promise<number> {
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

    return this.splatBuffer.selectConnectedSimilarColorProgressive(
      this.gpu.device,
      picked.globalIndex,
      picked.chunk,
      this.world,
      {
        colorThreshold,
        selectBehind,
        depthRangeFactor: selectBehind ? undefined : depthRangeFactor,
        viewProjectionMatrix: selectBehind ? undefined : this.camera.getViewProjectionMatrix(),
        viewportWidth: selectBehind ? undefined : this.gpu.canvas.width,
        viewportHeight: selectBehind ? undefined : this.gpu.canvas.height,
        visibleChunks: selectBehind ? undefined : this.world.getVisibility().chunks,
      },
      selectionMode,
    );
  }

  async selectSimilarColorDragAt(
    clientX: number,
    clientY: number,
    colorThreshold: number,
    selectionMode: SelectionMode = "normal",
    selectBehind: boolean = true,
    depthRangeFactor: number = 2.5,
  ): Promise<number> {
    const rect = this.gpu.canvas.getBoundingClientRect();
    const x = ((clientX - rect.left) / rect.width) * this.gpu.canvas.width;
    const y = ((clientY - rect.top) / rect.height) * this.gpu.canvas.height;
    const chunks = this.world.getVisibility().chunks;

    const splatIndex = this.splatBuffer.findNearestSplatAtScreenPos(
      x,
      y,
      this.gpu.canvas.width,
      this.gpu.canvas.height,
      this.camera.getViewProjectionMatrix(),
      chunks,
    );

    if (splatIndex === null) {
      return 0;
    }

    const chunk = this.splatBuffer.getChunkForSplatIndex(splatIndex);

    if (!chunk) {
      return 0;
    }

    return this.splatBuffer.selectConnectedSimilarColorProgressive(
      this.gpu.device,
      splatIndex,
      chunk,
      this.world,
      {
        colorThreshold,
        selectBehind,
        depthRangeFactor: selectBehind ? undefined : depthRangeFactor,
        viewProjectionMatrix: selectBehind ? undefined : this.camera.getViewProjectionMatrix(),
        viewportWidth: selectBehind ? undefined : this.gpu.canvas.width,
        viewportHeight: selectBehind ? undefined : this.gpu.canvas.height,
        visibleChunks: selectBehind ? undefined : this.world.getVisibility().chunks,
      },
      selectionMode,
    );
  }

  async selectCircleAt(
    clientX: number,
    clientY: number,
    screenRadius: number,
    selectionMode: SelectionMode = "normal",
  ): Promise<number> {
    const rect = this.gpu.canvas.getBoundingClientRect();
    const x = ((clientX - rect.left) / rect.width) * this.gpu.canvas.width;
    const y = ((clientY - rect.top) / rect.height) * this.gpu.canvas.height;

    return this.splatBuffer.selectCircleProgressive(
      this.gpu.device,
      {
        screenX: x,
        screenY: y,
        screenRadius: Math.max(1, screenRadius * (this.gpu.canvas.width / Math.max(1, rect.width))),
        viewportWidth: this.gpu.canvas.width,
        viewportHeight: this.gpu.canvas.height,
        viewMatrix: this.camera.getViewMatrix(),
        viewProjectionMatrix: this.camera.getViewProjectionMatrix(),
        colorThreshold: 0,
        selectionMode,
      },
    );
  }

  async selectMarquee(rect: ScreenRect, partial: boolean, selectionMode: SelectionMode): Promise<number> {
    return this.splatBuffer.selectMarqueeProgressive(
      this.gpu.device,
      {
        rect,
        partial,
        viewportWidth: this.gpu.canvas.width,
        viewportHeight: this.gpu.canvas.height,
        viewMatrix: this.camera.getViewMatrix(),
        viewProjectionMatrix: this.camera.getViewProjectionMatrix(),
        selectionMode,
        chunks: this.world.getVisibility().chunks,
      },
    );
  }

  async selectLasso(points: readonly ScreenPoint[], selectionMode: SelectionMode): Promise<number> {
    return this.splatBuffer.selectLassoProgressive(
      this.gpu.device,
      {
        points,
        viewportWidth: this.gpu.canvas.width,
        viewportHeight: this.gpu.canvas.height,
        viewMatrix: this.camera.getViewMatrix(),
        viewProjectionMatrix: this.camera.getViewProjectionMatrix(),
        selectionMode,
        chunks: this.world.getVisibility().chunks,
      },
    );
  }

  paintBrushAt(
    clientX: number,
    clientY: number,
    screenRadius: number,
    color: [number, number, number],
    mixFactor: number,
  ): number {
    const rect = this.gpu.canvas.getBoundingClientRect();
    const x = ((clientX - rect.left) / rect.width) * this.gpu.canvas.width;
    const y = ((clientY - rect.top) / rect.height) * this.gpu.canvas.height;

    return this.splatBuffer.paintScreenRadius(
      this.gpu.device,
      {
        screenX: x,
        screenY: y,
        screenRadius: Math.max(1, screenRadius * (this.gpu.canvas.width / Math.max(1, rect.width))),
        viewportWidth: this.gpu.canvas.width,
        viewportHeight: this.gpu.canvas.height,
        viewMatrix: this.camera.getViewMatrix(),
        viewProjectionMatrix: this.camera.getViewProjectionMatrix(),
        color,
        mixFactor,
        chunks: this.world.getVisibility().chunks,
      },
    );
  }

  beginHsvEdit(): number {
    return this.splatBuffer.beginHsvEdit();
  }

  previewHsvEdit(adjust: HsvAdjust): void {
    this.splatBuffer.previewHsvEdit(this.gpu.device, adjust);
  }

  commitHsvEdit(): void {
    this.splatBuffer.commitHsvEdit();
  }

  cancelHsvEdit(): void {
    this.splatBuffer.cancelHsvEdit(this.gpu.device);
  }

  beginColorizeEdit(): number {
    return this.splatBuffer.beginColorizeEdit();
  }

  previewColorizeEdit(target: [number, number, number]): void {
    this.splatBuffer.previewColorizeEdit(this.gpu.device, target);
  }

  commitColorizeEdit(): void {
    this.splatBuffer.commitColorizeEdit();
  }

  cancelColorizeEdit(): void {
    this.splatBuffer.cancelColorizeEdit(this.gpu.device);
  }

  hideSelectedSplats(): number {
    return this.splatBuffer.hideSelectedSplats(this.gpu.device);
  }

  unhideAllSplats(): void {
    this.splatBuffer.unhideAllSplats(this.gpu.device);
  }

  duplicateSelectedSplats(): number {
    const result = this.splatBuffer.prepareDuplicateData();

    if (!result) {
      return 0;
    }

    const newWorld = SplatWorld.fromSplatData(result.combined);
    const worldData = newWorld.getSplatData();
    this.splatBuffer.rebuildFromWorld(this.gpu.device, worldData, result.oldCount, result.oldHiddenMask);
    this.splatBuffer.createChunkOrderCache(newWorld.getChunks());
    this.splatBuffer.createChunkMetadataBuffer(this.gpu.device, newWorld.packGpuMetadata());
    this.renderer.setSplatBuffer(this.splatBuffer);
    this.idPickingPass.setSplatBuffer(this.splatBuffer);
    this.world = newWorld;

    if (this.moveActive) {
      this.commitMoveSelected();
    }

    return worldData.count - result.oldCount;
  }

  beginMoveSelected(mode: "move" | "rotate" | "scale" = "move"): boolean {
    const centroid = this.splatBuffer.beginMoveEdit();

    if (!centroid) {
      return false;
    }

    this.ensureMoveGizmo();
    if (!this.moveTransformNode || !this.moveGizmoManager) {
      return false;
    }

    this.moveTransformNode.position.set(centroid[0], centroid[1], centroid[2]);
    this.moveTransformNode.rotation.set(0, 0, 0);
    this.moveTransformNode.rotationQuaternion = Quaternion.Identity();
    this.moveTransformNode.scaling.set(1, 1, 1);
    this.moveStartPosition = this.moveTransformNode.position.clone();
    this.transformStartRotation = this.moveTransformNode.rotationQuaternion.clone();
    this.transformStartScaling = this.moveTransformNode.scaling.clone();
    this.configureTransformGizmo(mode);
    this.moveGizmoManager.attachToNode(this.moveTransformNode);
    this.camera.setGizmoPointerEnabled(true);
    this.camera.setControlsEnabled(false);
    this.moveActive = true;
    return true;
  }

  previewMoveSelected(delta: [number, number, number]): void {
    this.splatBuffer.previewMoveEdit(this.gpu.device, delta);
  }

  setTransformToolMode(mode: "move" | "rotate" | "scale"): void {
    this.configureTransformGizmo(mode);

    if (this.moveTransformNode && this.moveGizmoManager) {
      this.moveGizmoManager.attachToNode(this.moveTransformNode);
      this.camera.setGizmoPointerEnabled(true);
      this.camera.setControlsEnabled(false);
    }
  }

  commitMoveSelected(): void {
    this.splatBuffer.commitMoveEdit();
    this.splatBuffer.createChunkMetadataBuffer(this.gpu.device, this.world.packGpuMetadata());
    this.moveActive = false;
    this.moveStartPosition = null;
    this.transformStartRotation = null;
    this.transformStartScaling = null;
    this.moveGizmoManager?.attachToNode(null);
    this.camera.setGizmoPointerEnabled(false);
  }

  private readonly resize = (): void => {
    this.gpu.resize();
  };

  private readonly loop = (): void => {
    const frameStart = performance.now();
    this.lastFrameMs = frameStart - this.lastRafTime;
    this.lastRafTime = frameStart;
    this.camera.update();
    this.updateMoveGizmoEdit();
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

      if (this.densityCullingEnabled) {
        const DENSITY_THRESHOLD = 500;
        for (const plan of plans) {
          const chunk = this.world.getChunkById(plan.chunkId);
          if (!chunk) continue;
          const dx = chunk.boundsMax[0] - chunk.boundsMin[0];
          const dy = chunk.boundsMax[1] - chunk.boundsMin[1];
          const dz = chunk.boundsMax[2] - chunk.boundsMin[2];
          const volume = Math.max(1e-8, dx * dy * dz);
          const density = chunk.localSortedIndicesCount / volume;
          if (density > DENSITY_THRESHOLD) {
            plan.lodStep = Math.max(plan.lodStep, 4);
          }
        }
      }

      const visibleTelemetry = this.splatBuffer.buildVisibleSplatIndicesFromChunkPlans(
        plans,
        this.camera.getViewMatrix(),
        this.gpu.device,
        createTileBudgetOptions(
          this.qualityMode,
          this.optimized,
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
    } else if (this.renderBackend === "gpuRadixSorted") {
      const cullStart = performance.now();
      this.world.updateVisibility(this.camera.getViewProjectionMatrix());
      cpuCullMs = performance.now() - cullStart;
      visibleIndexBuildMs = 0;
      this.splatBuffer.setGpuRenderCountEstimate(this.world.getVisibility().splatCount);
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

  private ensureMoveGizmo(): void {
    if (!this.moveTransformNode) {
      this.moveTransformNode = new TransformNode("selectedSplatsTransform", this.camera.getBabylonScene());
    }

    if (!this.moveGizmoManager) {
      this.moveGizmoManager = new GizmoManager(this.camera.getBabylonScene());
      this.moveGizmoManager.boundingBoxGizmoEnabled = false;
      this.moveGizmoManager.usePointerToAttachGizmos = false;
    }
  }

  private configureTransformGizmo(mode: "move" | "rotate" | "scale"): void {
    if (!this.moveGizmoManager) {
      return;
    }

    this.moveGizmoManager.positionGizmoEnabled = mode === "move";
    this.moveGizmoManager.rotationGizmoEnabled = mode === "rotate";
    this.moveGizmoManager.scaleGizmoEnabled = mode === "scale";
  }

  private updateMoveGizmoEdit(): void {
    if (
      !this.moveActive ||
      !this.moveTransformNode ||
      !this.moveStartPosition ||
      !this.transformStartRotation ||
      !this.transformStartScaling
    ) {
      return;
    }

    const delta = this.moveTransformNode.position.subtract(this.moveStartPosition);
    const rotation = this.moveTransformNode.rotationQuaternion ?? Quaternion.FromEulerVector(this.moveTransformNode.rotation);
    const relativeRotation = rotation.multiply(this.transformStartRotation.conjugate());
    const scaling = this.moveTransformNode.scaling;
    this.splatBuffer.previewTransformEdit(
      this.gpu.device,
      [delta.x, delta.y, delta.z],
      [relativeRotation.x, relativeRotation.y, relativeRotation.z, relativeRotation.w],
      [
        scaling.x / Math.max(1e-6, this.transformStartScaling.x),
        scaling.y / Math.max(1e-6, this.transformStartScaling.y),
        scaling.z / Math.max(1e-6, this.transformStartScaling.z),
      ],
    );
  }
}

function getRenderScale(qualityMode: RenderQualityMode): number {
  void qualityMode;
  return 1;
}

function getRenderBackend(qualityMode: RenderQualityMode): GpuRenderBackend {
  if (qualityMode === "performance" || qualityMode === "gpu-balanced") {
    return "gpuRadixSorted";
  }
  return "cpuChunkBinned";
}

function createTileBudgetOptions(
  qualityMode: RenderQualityMode,
  optimized: boolean,
  viewportWidth: number,
  viewportHeight: number,
  viewProjectionMatrix: Float32Array,
  gpuTilePressure?: GpuTilePressureTelemetry,
): TileBudgetOptions {
  if (!optimized || qualityMode === "quality") {
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
  optimized: boolean,
): GpuTilePressurePass | null {
  if (!optimized || qualityMode === "quality") {
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

function createComputeSortPass(
  device: GPUDevice,
  cameraBindGroupLayout: GPUBindGroupLayout,
  splatCount: number,
  renderBackend: GpuRenderBackend,
  optimized: boolean,
): ComputeSortPass | null {
  if (!optimized || renderBackend !== "gpuRadixSorted") {
    return null;
  }

  const pass = new ComputeSortPass(device, cameraBindGroupLayout);
  pass.ensureBuffers(splatCount);
  return pass;
}

function createGpuDepthBinPass(
  device: GPUDevice,
  cameraBindGroupLayout: GPUBindGroupLayout,
  splatBuffer: SplatBuffer,
  splatCount: number,
  qualityMode: RenderQualityMode,
  viewportHeight: number,
  renderBackend: GpuRenderBackend,
  optimized: boolean,
): GpuDepthBinPass | null {
  if (!optimized || renderBackend !== "gpuDepthBinned") {
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
