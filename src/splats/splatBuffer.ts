import type { ChunkRenderPlan, SplatChunk, TileBudgetOptions, Vector3Tuple, WorldSplatData } from "../world/types";
import type { HsvAdjust, ScreenPoint, ScreenRect, SelectionMode } from "../editor/types";

export interface ChunkNeighborLookup {
  getNeighborChunks(chunk: SplatChunk, radius?: number): readonly SplatChunk[];
}

export interface SplatData {
  positions: Float32Array;
  colors: Float32Array;
  opacities: Float32Array;
  covariances: Float32Array;
  shCoefficients: Float32Array;
  count: number;
}

export interface StableSplatData extends SplatData {
  splatIds: Uint32Array;
  chunkIds: Uint32Array;
  localIndices: Uint32Array;
}

export interface SplatRange {
  splatStart: number;
  splatCount: number;
}

export interface GpuCullingBuffers {
  visibleChunkFlagsBuffer: GPUBuffer;
  visibleChunkIndicesBuffer: GPUBuffer;
  visibleChunkCounterBuffer: GPUBuffer;
  visibleSplatIndicesBuffer: GPUBuffer;
  visibleSplatCounterBuffer: GPUBuffer;
  indirectArgsBuffer: GPUBuffer;
}

export interface VisibleIndexBuildTelemetry {
  localOrderRefreshMs: number;
  visibleIndexBuildMs: number;
  refreshedChunkCount: number;
  tileCulledSplats: number;
  tileTestedSplats: number;
  tileProtectedSplats: number;
}

interface ColorClusterSelectionOptions {
  colorThreshold: number;
  neighborRadius: number;
  maxCandidateChunks: number;
  maxSelectedSplats: number;
  selectBehind?: boolean;
  depthRangeFactor?: number;
  viewProjectionMatrix?: Float32Array;
  viewportWidth?: number;
  viewportHeight?: number;
  visibleChunks?: readonly SplatChunk[];
}

interface ChunkSelectionIndex {
  chunk: SplatChunk;
  spatialCellSize: number;
  spatialCells: Map<string, Uint32Array>;
  colorBins: Map<number, Uint32Array>;
}

export interface ScreenColorSelectionOptions {
  screenX: number;
  screenY: number;
  screenRadius: number;
  viewportWidth: number;
  viewportHeight: number;
  viewMatrix: Float32Array;
  viewProjectionMatrix: Float32Array;
  colorThreshold: number;
  depthTolerance?: number;
  selectionMode?: SelectionMode;
  chunks?: readonly SplatChunk[];
}

export interface ScreenMarqueeSelectionOptions {
  rect: ScreenRect;
  partial: boolean;
  viewportWidth: number;
  viewportHeight: number;
  viewMatrix: Float32Array;
  viewProjectionMatrix: Float32Array;
  selectionMode: SelectionMode;
  chunks?: readonly SplatChunk[];
}

export interface ScreenLassoSelectionOptions {
  points: readonly ScreenPoint[];
  viewportWidth: number;
  viewportHeight: number;
  viewMatrix: Float32Array;
  viewProjectionMatrix: Float32Array;
  selectionMode: SelectionMode;
  chunks?: readonly SplatChunk[];
}

export interface PaintBrushOptions {
  screenX: number;
  screenY: number;
  screenRadius: number;
  viewportWidth: number;
  viewportHeight: number;
  viewMatrix: Float32Array;
  viewProjectionMatrix: Float32Array;
  color: [number, number, number];
  mixFactor: number;
  chunks?: readonly SplatChunk[];
}

const LOCAL_SORT_DIRECTION_EPSILON = 0.015;
const COLOR_BIN_COUNT = 32;

function createStorageBuffer(device: GPUDevice, data: Float32Array, label: string): GPUBuffer {
  const buffer = device.createBuffer({
    label,
    size: data.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });

  device.queue.writeBuffer(
    buffer,
    0,
    data.buffer as ArrayBuffer,
    data.byteOffset,
    data.byteLength,
  );
  return buffer;
}

function createIndexBuffer(device: GPUDevice, data: Uint32Array, label: string): GPUBuffer {
  const buffer = device.createBuffer({
    label,
    size: data.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });

  device.queue.writeBuffer(
    buffer,
    0,
    data.buffer as ArrayBuffer,
    data.byteOffset,
    data.byteLength,
  );
  return buffer;
}

export class SplatBuffer {
  private positions: Float32Array | null = null;
  private colors: Float32Array | null = null;
  private opacities: Float32Array | null = null;
  private covariances: Float32Array | null = null;
  private shCoefficients: Float32Array | null = null;
  private order: Uint32Array | null = null;
  private visibleSplatIndices: Uint32Array | null = null;
  private selectionMask: Uint32Array | null = null;
  private hiddenMask: Uint32Array | null = null;
  private chunkLocalSortedIndices: Uint32Array | null = null;
  private splatChunkIds: Uint32Array | null = null;
  private chunksById = new Map<number, SplatChunk>();
  private selectionIndicesByChunkId = new Map<number, ChunkSelectionIndex>();
  private depths: Float32Array | null = null;
  private count: number = 0;
  private renderCount: number = 0;

  private positionBuffer: GPUBuffer | null = null;
  private colorBuffer: GPUBuffer | null = null;
  private opacityBuffer: GPUBuffer | null = null;
  private covarianceBuffer: GPUBuffer | null = null;
  private orderBuffer: GPUBuffer | null = null;
  private visibleSplatIndicesBuffer: GPUBuffer | null = null;
  private shBuffer: GPUBuffer | null = null;
  private splatIdBuffer: GPUBuffer | null = null;
  private chunkIdBuffer: GPUBuffer | null = null;
  private localIndexBuffer: GPUBuffer | null = null;
  private chunkMetadataBuffer: GPUBuffer | null = null;
  private chunkMetadataBufferSize = 0;
  private selectionMaskBuffer: GPUBuffer | null = null;
  private hiddenMaskBuffer: GPUBuffer | null = null;
  private indirectArgsBuffer: GPUBuffer | null = null;
  private ownsVisibleSplatIndicesBuffer = true;
  private ownsIndirectArgsBuffer = true;
  private selectionGeneration = 0;
  private selectedSplatCount = 0;
  private selectionHighlightVisible = true;
  private hiddenSelectionMask: Uint32Array | null = null;
  private hsvEditOriginalColors: Float32Array | null = null;
  private hsvEditIndices: Uint32Array | null = null;
  private colorizeEditOriginalColors: Float32Array | null = null;
  private colorizeEditIndices: Uint32Array | null = null;
  private moveEditOriginalPositions: Float32Array | null = null;
  private moveEditIndices: Uint32Array | null = null;
  private moveEditCentroid: Vector3Tuple | null = null;

  public setData(data: SplatData): void {
    this.positions = data.positions;
    this.colors = data.colors;
    this.opacities = data.opacities;
    this.covariances = data.covariances;
    this.shCoefficients = data.shCoefficients;
    this.count = data.count;
    this.renderCount = data.count;
    this.order = new Uint32Array(this.count);
    this.visibleSplatIndices = new Uint32Array(this.count);
    this.selectionMask = new Uint32Array(this.count);
    this.hiddenMask = new Uint32Array(this.count);
    this.selectedSplatCount = 0;
    this.depths = new Float32Array(this.count);

    for (let i = 0; i < this.count; i++) {
      this.order[i] = i;
      this.visibleSplatIndices[i] = i;
    }
  }

  public createBuffers(device: GPUDevice): void {
    if (this.positions) {
      this.positionBuffer = createStorageBuffer(device, this.positions, "SplatPositions");
    }

    if (this.colors) {
      this.colorBuffer = createStorageBuffer(device, this.colors, "SplatColors");
    }

    if (this.opacities) {
      this.opacityBuffer = createStorageBuffer(device, this.opacities, "SplatOpacities");
    }

    if (this.covariances) {
      this.covarianceBuffer = createStorageBuffer(device, this.covariances, "SplatCovariances");
    }

    if (this.order) {
      this.orderBuffer = createIndexBuffer(device, this.order, "SplatRenderOrder");
      this.visibleSplatIndicesBuffer = createIndexBuffer(device, this.order, "VisibleSplatIndices");
      this.ownsVisibleSplatIndicesBuffer = true;
    }

    if (this.shCoefficients && this.shCoefficients.byteLength > 0) {
      this.shBuffer = createStorageBuffer(device, this.shCoefficients, "SplatSHCoefficients");
    }
  }

  public createStableIdBuffers(device: GPUDevice, data: StableSplatData): void {
    this.splatIdBuffer = createIndexBuffer(device, data.splatIds, "StableSplatIds");
    this.chunkIdBuffer = createIndexBuffer(device, data.chunkIds, "StableSplatChunkIds");
    this.localIndexBuffer = createIndexBuffer(device, data.localIndices, "StableSplatLocalIndices");
  }

  public createChunkMetadataBuffer(device: GPUDevice, metadata: ArrayBuffer): void {
    if (!this.chunkMetadataBuffer || this.chunkMetadataBufferSize !== metadata.byteLength) {
      this.chunkMetadataBuffer?.destroy();
      this.chunkMetadataBuffer = device.createBuffer({
        label: "SplatChunkMetadata",
        size: metadata.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      this.chunkMetadataBufferSize = metadata.byteLength;
    }

    device.queue.writeBuffer(this.chunkMetadataBuffer, 0, metadata);
  }

  public createSelectionMaskBuffer(device: GPUDevice): void {
    this.selectionMaskBuffer?.destroy();
    this.selectionMask = new Uint32Array(this.count);
    this.selectedSplatCount = 0;
    this.selectionMaskBuffer = device.createBuffer({
      label: "SelectionMask",
      size: Math.max(4, this.count * Uint32Array.BYTES_PER_ELEMENT),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.uploadSelectionMask(device);
  }

  public createHiddenMaskBuffer(device: GPUDevice): void {
    this.hiddenMaskBuffer?.destroy();
    this.hiddenMask = new Uint32Array(this.count);
    this.hiddenMaskBuffer = device.createBuffer({
      label: "HiddenMask",
      size: Math.max(4, this.count * Uint32Array.BYTES_PER_ELEMENT),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.uploadHiddenMask(device);
  }

  public createIndirectArgsBuffer(device: GPUDevice): void {
    this.indirectArgsBuffer?.destroy();
    this.indirectArgsBuffer = device.createBuffer({
      label: "SplatIndirectDrawArgs",
      size: 4 * Uint32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    this.ownsIndirectArgsBuffer = true;
    this.updateIndirectArgs(device);
  }

  public createChunkOrderCache(chunks: readonly SplatChunk[]): void {
    this.chunksById.clear();
    this.selectionIndicesByChunkId.clear();
    this.chunkLocalSortedIndices = new Uint32Array(this.count);
    this.splatChunkIds = new Uint32Array(this.count);

    for (const chunk of chunks) {
      this.chunksById.set(chunk.id, chunk);
      chunk.localSortedIndicesOffset = chunk.splatStart;
      chunk.localSortedIndicesCount = chunk.splatCount;
      chunk.localOrderCacheVersion = 0;
      chunk.lastSortDirection = null;

      for (let i = 0; i < chunk.splatCount; i++) {
        const splatIndex = chunk.splatStart + i;
        this.chunkLocalSortedIndices[chunk.localSortedIndicesOffset + i] = splatIndex;
        this.splatChunkIds[splatIndex] = chunk.id;
      }

      const selectionIndex = this.createChunkSelectionIndex(chunk);

      if (selectionIndex) {
        this.selectionIndicesByChunkId.set(chunk.id, selectionIndex);
      }
    }
  }

  public getCount(): number {
    return this.count;
  }

  public getRenderCount(): number {
    return this.renderCount;
  }

  public getPositionBuffer(): GPUBuffer | null {
    return this.positionBuffer;
  }

  public getColorBuffer(): GPUBuffer | null {
    return this.colorBuffer;
  }

  public getOpacityBuffer(): GPUBuffer | null {
    return this.opacityBuffer;
  }

  public getCovarianceBuffer(): GPUBuffer | null {
    return this.covarianceBuffer;
  }

  public getOrderBuffer(): GPUBuffer | null {
    return this.orderBuffer;
  }

  public getVisibleSplatIndicesBuffer(): GPUBuffer | null {
    return this.visibleSplatIndicesBuffer;
  }

  public getShBuffer(): GPUBuffer | null {
    return this.shBuffer;
  }

  public getSplatIdBuffer(): GPUBuffer | null {
    return this.splatIdBuffer;
  }

  public getChunkIdBuffer(): GPUBuffer | null {
    return this.chunkIdBuffer;
  }

  public getLocalIndexBuffer(): GPUBuffer | null {
    return this.localIndexBuffer;
  }

  public getChunkMetadataBuffer(): GPUBuffer | null {
    return this.chunkMetadataBuffer;
  }

  public getSelectionMaskBuffer(): GPUBuffer | null {
    return this.selectionMaskBuffer;
  }

  public getHiddenMaskBuffer(): GPUBuffer | null {
    return this.hiddenMaskBuffer;
  }

  public getIndirectArgsBuffer(): GPUBuffer | null {
    return this.indirectArgsBuffer;
  }

  public adoptGpuVisibleBuffers(
    visibleSplatIndicesBuffer: GPUBuffer,
    indirectArgsBuffer: GPUBuffer,
  ): void {
    if (this.ownsVisibleSplatIndicesBuffer) {
      this.visibleSplatIndicesBuffer?.destroy();
    }

    if (this.ownsIndirectArgsBuffer) {
      this.indirectArgsBuffer?.destroy();
    }

    this.visibleSplatIndicesBuffer = visibleSplatIndicesBuffer;
    this.indirectArgsBuffer = indirectArgsBuffer;
    this.ownsVisibleSplatIndicesBuffer = false;
    this.ownsIndirectArgsBuffer = false;
    this.renderCount = this.count;
  }

  public sortByView(
    viewMatrix: Float32Array,
    device: GPUDevice,
    visibleRanges?: readonly SplatRange[],
  ): void {
    if (!this.positions || !this.order || !this.visibleSplatIndices || !this.depths || !this.orderBuffer) {
      return;
    }

    let writeIndex = 0;
    const ranges = visibleRanges ?? [{ splatStart: 0, splatCount: this.count }];

    for (const range of ranges) {
      const end = Math.min(this.count, range.splatStart + range.splatCount);

      for (let i = range.splatStart; i < end; i++) {
        this.order[writeIndex] = i;
        writeIndex++;
      }
    }

    this.renderCount = writeIndex;

    for (let i = 0; i < this.renderCount; i++) {
      const splatIndex = this.order[i];
      const base = splatIndex * 3;
      this.depths[splatIndex] =
        viewMatrix[2] * this.positions[base] +
        viewMatrix[6] * this.positions[base + 1] +
        viewMatrix[10] * this.positions[base + 2] +
        viewMatrix[14];
    }

    this.order.subarray(0, this.renderCount).sort(
      (a, b) => this.getDepth(b) - this.getDepth(a),
    );
    this.visibleSplatIndices.set(this.order.subarray(0, this.renderCount), 0);

    device.queue.writeBuffer(
      this.orderBuffer,
      0,
      this.order.buffer as ArrayBuffer,
      this.order.byteOffset,
      this.renderCount * Uint32Array.BYTES_PER_ELEMENT,
    );
    this.uploadVisibleSplatIndices(device);
    this.updateIndirectArgs(device);
  }

  public sortAllByView(viewMatrix: Float32Array, device: GPUDevice): void {
    if (!this.positions || !this.order || !this.visibleSplatIndices || !this.depths || !this.orderBuffer) {
      return;
    }

    this.renderCount = this.count;

    for (let i = 0; i < this.renderCount; i++) {
      const base = i * 3;
      this.depths[i] =
        viewMatrix[2] * this.positions[base] +
        viewMatrix[6] * this.positions[base + 1] +
        viewMatrix[10] * this.positions[base + 2] +
        viewMatrix[14];
      this.order[i] = i;
    }

    this.order.sort((a, b) => this.depths![b] - this.depths![a]);
    this.visibleSplatIndices.set(this.order.subarray(0, this.renderCount), 0);
    device.queue.writeBuffer(
      this.orderBuffer,
      0,
      this.order.buffer as ArrayBuffer,
      this.order.byteOffset,
      this.renderCount * Uint32Array.BYTES_PER_ELEMENT,
    );
    this.uploadVisibleSplatIndices(device);
    this.updateIndirectArgs(device);
  }

  public buildVisibleSplatIndicesFromChunkPlans(
    plans: readonly ChunkRenderPlan[],
    viewMatrix: Float32Array,
    device: GPUDevice,
    tileBudget?: TileBudgetOptions,
  ): VisibleIndexBuildTelemetry {
    const start = performance.now();
    let refreshMs = 0;
    let refreshedChunkCount = 0;
    let tileCulledSplats = 0;
    let tileTestedSplats = 0;
    let tileProtectedSplats = 0;
    const tileState = tileBudget?.enabled
      ? createTileBudgetState(tileBudget)
      : null;

    if (!this.positions || !this.depths || !this.visibleSplatIndices || !this.chunkLocalSortedIndices) {
      return {
        localOrderRefreshMs: 0,
        visibleIndexBuildMs: 0,
        refreshedChunkCount: 0,
        tileCulledSplats: 0,
        tileTestedSplats: 0,
        tileProtectedSplats: 0,
      };
    }

    const viewDirection: Vector3Tuple = [
      viewMatrix[2],
      viewMatrix[6],
      viewMatrix[10],
    ];
    let writeIndex = 0;

    for (const plan of plans) {
      const chunk = this.chunksById.get(plan.chunkId);

      if (!chunk) {
        continue;
      }

      if (this.shouldRefreshChunkOrder(chunk, viewDirection)) {
        const refreshStart = performance.now();
        this.refreshChunkOrder(chunk, viewMatrix, viewDirection);
        refreshMs += performance.now() - refreshStart;
        refreshedChunkCount++;
      }

      const sourceStart = chunk.localSortedIndicesOffset;
      const sourceEnd = sourceStart + chunk.localSortedIndicesCount;

      for (let i = sourceStart; i < sourceEnd; i += plan.lodStep) {
        const splatIndex = this.chunkLocalSortedIndices[i];
        const useTileBudget = tileState && shouldApplyTileBudget(plan, tileState.options);

        if (tileState && !useTileBudget) {
          tileProtectedSplats++;
        }

        if (useTileBudget) {
          tileTestedSplats++;

          if (!this.acceptTileBudget(splatIndex, tileState)) {
            tileCulledSplats++;
            continue;
          }
        }

        this.visibleSplatIndices[writeIndex] = splatIndex;
        writeIndex++;
      }
    }

    this.renderCount = writeIndex;
    this.uploadVisibleSplatIndices(device);
    this.updateIndirectArgs(device);

    return {
      localOrderRefreshMs: refreshMs,
      visibleIndexBuildMs: performance.now() - start,
      refreshedChunkCount,
      tileCulledSplats,
      tileTestedSplats,
      tileProtectedSplats,
    };
  }

  public adoptGpuCullingBuffers(buffers: GpuCullingBuffers): void {
    if (this.ownsVisibleSplatIndicesBuffer) {
      this.visibleSplatIndicesBuffer?.destroy();
    }

    if (this.ownsIndirectArgsBuffer) {
      this.indirectArgsBuffer?.destroy();
    }

    this.visibleSplatIndicesBuffer = buffers.visibleSplatIndicesBuffer;
    this.indirectArgsBuffer = buffers.indirectArgsBuffer;
    this.ownsVisibleSplatIndicesBuffer = false;
    this.ownsIndirectArgsBuffer = false;
  }

  public setGpuRenderCountEstimate(renderCount: number): void {
    this.renderCount = Math.max(0, Math.min(this.count, renderCount));
  }

  public isSplatHidden(splatIndex: number): boolean {
    return (this.hiddenMask?.[splatIndex] ?? 0) !== 0;
  }

  public selectConnectedSimilarColor(
    device: GPUDevice,
    seedGlobalIndex: number,
    seedChunk: SplatChunk,
    world: ChunkNeighborLookup,
    options: Partial<ColorClusterSelectionOptions> = {},
    selectionMode: SelectionMode = "normal",
  ): number {
    if (!this.positions || !this.colors || !this.selectionMask) {
      return 0;
    }

    const selectionOptions: ColorClusterSelectionOptions = {
      colorThreshold: options.colorThreshold ?? 0.14,
      neighborRadius: options.neighborRadius ?? Math.max(0.015, seedChunk.radius * 0.035),
      maxCandidateChunks: options.maxCandidateChunks ?? 27,
      maxSelectedSplats: options.maxSelectedSplats ?? 250_000,
    };

    this.prepareSelectionMode(selectionMode);

    const selectedCount = this.addConnectedSimilarColorCluster(
      seedGlobalIndex,
      seedChunk,
      world,
      selectionOptions,
      selectionMode,
    );

    this.uploadSelectionMask(device);
    return selectedCount;
  }

  public async selectConnectedSimilarColorProgressive(
    device: GPUDevice,
    seedGlobalIndex: number,
    seedChunk: SplatChunk,
    world: ChunkNeighborLookup,
    options: Partial<ColorClusterSelectionOptions> = {},
    selectionMode: SelectionMode = "normal",
  ): Promise<number> {
    if (!this.positions || !this.colors || !this.selectionMask) {
      return 0;
    }

    const selectionOptions: ColorClusterSelectionOptions = {
      colorThreshold: options.colorThreshold ?? 0.14,
      neighborRadius: options.neighborRadius ?? Math.max(0.015, seedChunk.radius * 0.035),
      maxCandidateChunks: options.maxCandidateChunks ?? 27,
      maxSelectedSplats: options.maxSelectedSplats ?? 250_000,
    };

    const generation = ++this.selectionGeneration;
    this.prepareSelectionMode(selectionMode);

    if (generation !== this.selectionGeneration) {
      return 0;
    }

    if (selectionMode === "normal") {
      this.uploadSelectionMask(device);
      await nextAnimationFrame();
    }

    if (generation !== this.selectionGeneration) {
      return 0;
    }

    const selectedCount = await this.addConnectedSimilarColorClusterProgressive(
      device,
      generation,
      seedGlobalIndex,
      seedChunk,
      world,
      selectionOptions,
      selectionMode,
    );

    this.uploadSelectionMask(device);
    return selectedCount;
  }

  public selectConnectedSimilarColorInScreenRadius(
    device: GPUDevice,
    world: ChunkNeighborLookup,
    options: ScreenColorSelectionOptions,
  ): number {
    if (!this.positions || !this.colors || !this.visibleSplatIndices || !this.selectionMask) {
      return 0;
    }

    this.prepareSelectionMode(options.selectionMode ?? "normal");

    const seedIndices = this.collectScreenRadiusSeeds(options);
    let selectedCount = 0;

    for (const seedIndex of seedIndices) {
      if ((options.selectionMode ?? "normal") !== "subtractive" && this.selectionMask[seedIndex] !== 0) {
        continue;
      }

      const seedChunk = this.getChunkForSplatIndex(seedIndex);

      if (!seedChunk) {
        continue;
      }

      selectedCount += this.addConnectedSimilarColorCluster(
        seedIndex,
        seedChunk,
        world,
        {
          colorThreshold: options.colorThreshold,
          neighborRadius: Math.max(0.015, seedChunk.radius * 0.035),
          maxCandidateChunks: 27,
          maxSelectedSplats: 250_000,
        },
        options.selectionMode ?? "normal",
      );
    }

    this.uploadSelectionMask(device);
    return selectedCount;
  }

  public async selectConnectedSimilarColorInScreenRadiusProgressive(
    device: GPUDevice,
    world: ChunkNeighborLookup,
    options: ScreenColorSelectionOptions,
  ): Promise<number> {
    if (!this.positions || !this.colors || !this.visibleSplatIndices || !this.selectionMask) {
      return 0;
    }

    const generation = ++this.selectionGeneration;
    const selectionMode = options.selectionMode ?? "normal";
    this.prepareSelectionMode(selectionMode);

    if (selectionMode === "normal") {
      this.uploadSelectionMask(device);
      await nextAnimationFrame();
    }

    if (generation !== this.selectionGeneration) {
      return 0;
    }

    const seedIndices = this.collectScreenRadiusSeeds(options);
    let selectedCount = 0;

    for (const seedIndex of seedIndices) {
      if (generation !== this.selectionGeneration) {
        return selectedCount;
      }

      if (selectionMode !== "subtractive" && this.selectionMask[seedIndex] !== 0) {
        continue;
      }

      const seedChunk = this.getChunkForSplatIndex(seedIndex);

      if (!seedChunk) {
        continue;
      }

      selectedCount += await this.addConnectedSimilarColorClusterProgressive(
        device,
        generation,
        seedIndex,
        seedChunk,
        world,
        {
          colorThreshold: options.colorThreshold,
          neighborRadius: Math.max(0.015, seedChunk.radius * 0.035),
          maxCandidateChunks: 27,
          maxSelectedSplats: 250_000,
        },
        selectionMode,
      );
    }

    this.uploadSelectionMask(device);
    return selectedCount;
  }

  public async selectMarqueeProgressive(
    device: GPUDevice,
    options: ScreenMarqueeSelectionOptions,
  ): Promise<number> {
    return this.selectProjectedGeometryProgressive(device, {
      ...options,
      contains: (projected) => {
        if (options.partial) {
          const radius = Math.max(1, projected.radius);
          return projected.x + radius >= options.rect.minX &&
            projected.x - radius <= options.rect.maxX &&
            projected.y + radius >= options.rect.minY &&
            projected.y - radius <= options.rect.maxY;
        }

        return projected.x >= options.rect.minX &&
          projected.x <= options.rect.maxX &&
          projected.y >= options.rect.minY &&
          projected.y <= options.rect.maxY;
      },
    });
  }

  public async selectLassoProgressive(
    device: GPUDevice,
    options: ScreenLassoSelectionOptions,
  ): Promise<number> {
    if (options.points.length < 3) {
      return 0;
    }

    return this.selectProjectedGeometryProgressive(device, {
      ...options,
      contains: (projected) => pointInPolygon(projected.x, projected.y, options.points),
    });
  }

  public async selectCircleProgressive(
    device: GPUDevice,
    options: ScreenColorSelectionOptions,
  ): Promise<number> {
    const { screenX, screenY, screenRadius } = options;
    const radiusSq = screenRadius * screenRadius;

    return this.selectProjectedGeometryProgressive(device, {
      viewportWidth: options.viewportWidth,
      viewportHeight: options.viewportHeight,
      viewMatrix: options.viewMatrix,
      viewProjectionMatrix: options.viewProjectionMatrix,
      selectionMode: options.selectionMode ?? "normal",
      chunks: options.chunks,
      contains: (projected) => {
        const dx = projected.x - screenX;
        const dy = projected.y - screenY;
        return dx * dx + dy * dy <= radiusSq;
      },
    });
  }

  public paintScreenRadius(device: GPUDevice, options: PaintBrushOptions): number {
    if (!this.positions || !this.colors) {
      return 0;
    }

    const chunks = options.chunks ?? [...this.chunksById.values()];
    const radiusSq = options.screenRadius * options.screenRadius;
    const mix = Math.max(0, Math.min(1, options.mixFactor));
    const selectionOnly = this.selectedSplatCount > 0;
    const [paintH, paintS] = rgbToHsv(options.color[0], options.color[1], options.color[2]);
    let paintedCount = 0;

    if (mix <= 0) {
      return 0;
    }

    for (const chunk of chunks) {
      const end = chunk.splatStart + chunk.splatCount;

      for (let splatIndex = chunk.splatStart; splatIndex < end; splatIndex++) {
        if (this.isSplatHidden(splatIndex)) {
          continue;
        }

        if (selectionOnly && this.selectionMask?.[splatIndex] === 0) {
          continue;
        }

        const projected = this.projectSplatToScreen(splatIndex, options);

        if (!projected) {
          continue;
        }

        const dx = projected.x - options.screenX;
        const dy = projected.y - options.screenY;
        const distanceSq = dx * dx + dy * dy;

        if (distanceSq > radiusSq) {
          continue;
        }

        const falloff = Math.max(0, 1 - Math.sqrt(distanceSq) / Math.max(1, options.screenRadius));
        const strength = mix * falloff;
        const base = splatIndex * 3;
        const [sourceH, sourceS, sourceV] = rgbToHsv(
          this.colors[base],
          this.colors[base + 1],
          this.colors[base + 2],
        );
        const hueDelta = shortestHueDelta(sourceH, paintH);
        const nextH = wrap01(sourceH + hueDelta * strength);
        const nextS = clamp01(sourceS + (paintS - sourceS) * strength);
        const [r, g, b] = hsvToRgb(nextH, nextS, sourceV);
        this.colors[base] = r;
        this.colors[base + 1] = g;
        this.colors[base + 2] = b;
        paintedCount++;
      }
    }

    if (paintedCount > 0) {
      this.uploadColors(device);
    }

    return paintedCount;
  }

  public getSelectedCount(): number {
    return this.selectedSplatCount;
  }

  public getSelectedCentroid(): Vector3Tuple | null {
    if (!this.positions || !this.selectionMask) {
      return null;
    }

    let x = 0;
    let y = 0;
    let z = 0;
    let count = 0;

    for (let i = 0; i < this.count; i++) {
      if (this.selectionMask[i] === 0) {
        continue;
      }

      const base = i * 3;
      x += this.positions[base];
      y += this.positions[base + 1];
      z += this.positions[base + 2];
      count++;
    }

    return count === 0 ? null : [x / count, y / count, z / count];
  }

  public beginHsvEdit(): number {
    if (!this.colors || !this.selectionMask) {
      this.hsvEditOriginalColors = null;
      this.hsvEditIndices = null;
      return 0;
    }

    const indices = this.collectSelectedIndices();
    this.hsvEditIndices = indices;
    this.hsvEditOriginalColors = new Float32Array(indices.length * 3);

    for (let i = 0; i < indices.length; i++) {
      const sourceBase = indices[i] * 3;
      const editBase = i * 3;
      this.hsvEditOriginalColors[editBase] = this.colors[sourceBase];
      this.hsvEditOriginalColors[editBase + 1] = this.colors[sourceBase + 1];
      this.hsvEditOriginalColors[editBase + 2] = this.colors[sourceBase + 2];
    }

    return indices.length;
  }

  public previewHsvEdit(device: GPUDevice, adjust: HsvAdjust): void {
    if (!this.colors || !this.hsvEditIndices || !this.hsvEditOriginalColors) {
      return;
    }

    for (let i = 0; i < this.hsvEditIndices.length; i++) {
      const editBase = i * 3;
      const [h, s, v] = rgbToHsv(
        this.hsvEditOriginalColors[editBase],
        this.hsvEditOriginalColors[editBase + 1],
        this.hsvEditOriginalColors[editBase + 2],
      );
      const [r, g, b] = hsvToRgb(
        wrap01(h + adjust.hue),
        clamp01(s * adjust.saturation),
        clamp01(v * adjust.value),
      );
      const targetBase = this.hsvEditIndices[i] * 3;
      this.colors[targetBase] = r;
      this.colors[targetBase + 1] = g;
      this.colors[targetBase + 2] = b;
    }

    this.uploadColors(device);
  }

  public commitHsvEdit(): void {
    this.hsvEditIndices = null;
    this.hsvEditOriginalColors = null;
    this.rebuildSelectionIndices();
  }

  public cancelHsvEdit(device: GPUDevice): void {
    if (!this.colors || !this.hsvEditIndices || !this.hsvEditOriginalColors) {
      this.hsvEditIndices = null;
      this.hsvEditOriginalColors = null;
      return;
    }

    for (let i = 0; i < this.hsvEditIndices.length; i++) {
      const sourceBase = i * 3;
      const targetBase = this.hsvEditIndices[i] * 3;
      this.colors[targetBase] = this.hsvEditOriginalColors[sourceBase];
      this.colors[targetBase + 1] = this.hsvEditOriginalColors[sourceBase + 1];
      this.colors[targetBase + 2] = this.hsvEditOriginalColors[sourceBase + 2];
    }

    this.hsvEditIndices = null;
    this.hsvEditOriginalColors = null;
    this.uploadColors(device);
  }

  public beginColorizeEdit(): number {
    if (!this.colors || !this.selectionMask) {
      this.colorizeEditOriginalColors = null;
      this.colorizeEditIndices = null;
      return 0;
    }

    const indices = this.collectSelectedIndices();
    this.colorizeEditIndices = indices;
    this.colorizeEditOriginalColors = new Float32Array(indices.length * 3);

    for (let i = 0; i < indices.length; i++) {
      const sourceBase = indices[i] * 3;
      const editBase = i * 3;
      this.colorizeEditOriginalColors[editBase] = this.colors[sourceBase];
      this.colorizeEditOriginalColors[editBase + 1] = this.colors[sourceBase + 1];
      this.colorizeEditOriginalColors[editBase + 2] = this.colors[sourceBase + 2];
    }

    return indices.length;
  }

  public previewColorizeEdit(device: GPUDevice, target: [number, number, number]): void {
    if (!this.colors || !this.colorizeEditIndices || !this.colorizeEditOriginalColors) {
      return;
    }

    const [targetH, targetS] = rgbToHsv(target[0], target[1], target[2]);

    for (let i = 0; i < this.colorizeEditIndices.length; i++) {
      const editBase = i * 3;
      const [, , originalValue] = rgbToHsv(
        this.colorizeEditOriginalColors[editBase],
        this.colorizeEditOriginalColors[editBase + 1],
        this.colorizeEditOriginalColors[editBase + 2],
      );
      const [r, g, b] = hsvToRgb(targetH, targetS, originalValue);
      const targetBase = this.colorizeEditIndices[i] * 3;
      this.colors[targetBase] = r;
      this.colors[targetBase + 1] = g;
      this.colors[targetBase + 2] = b;
    }

    this.uploadColors(device);
  }

  public commitColorizeEdit(): void {
    this.colorizeEditIndices = null;
    this.colorizeEditOriginalColors = null;
    this.rebuildSelectionIndices();
  }

  public cancelColorizeEdit(device: GPUDevice): void {
    if (!this.colors || !this.colorizeEditIndices || !this.colorizeEditOriginalColors) {
      this.colorizeEditIndices = null;
      this.colorizeEditOriginalColors = null;
      return;
    }

    for (let i = 0; i < this.colorizeEditIndices.length; i++) {
      const sourceBase = i * 3;
      const targetBase = this.colorizeEditIndices[i] * 3;
      this.colors[targetBase] = this.colorizeEditOriginalColors[sourceBase];
      this.colors[targetBase + 1] = this.colorizeEditOriginalColors[sourceBase + 1];
      this.colors[targetBase + 2] = this.colorizeEditOriginalColors[sourceBase + 2];
    }

    this.colorizeEditIndices = null;
    this.colorizeEditOriginalColors = null;
    this.uploadColors(device);
  }

  public beginMoveEdit(): Vector3Tuple | null {
    if (!this.positions || !this.selectionMask) {
      this.moveEditOriginalPositions = null;
      this.moveEditIndices = null;
      return null;
    }

    const centroid = this.getSelectedCentroid();

    if (!centroid) {
      this.moveEditOriginalPositions = null;
      this.moveEditIndices = null;
      this.moveEditCentroid = null;
      return null;
    }

    const indices = this.collectSelectedIndices();
    this.moveEditIndices = indices;
    this.moveEditOriginalPositions = new Float32Array(indices.length * 3);
    this.moveEditCentroid = centroid;

    for (let i = 0; i < indices.length; i++) {
      const sourceBase = indices[i] * 3;
      const editBase = i * 3;
      this.moveEditOriginalPositions[editBase] = this.positions[sourceBase];
      this.moveEditOriginalPositions[editBase + 1] = this.positions[sourceBase + 1];
      this.moveEditOriginalPositions[editBase + 2] = this.positions[sourceBase + 2];
    }

    return centroid;
  }

  public previewMoveEdit(device: GPUDevice, delta: Vector3Tuple): void {
    if (!this.positions || !this.moveEditIndices || !this.moveEditOriginalPositions) {
      return;
    }

    for (let i = 0; i < this.moveEditIndices.length; i++) {
      const editBase = i * 3;
      const targetBase = this.moveEditIndices[i] * 3;
      this.positions[targetBase] = this.moveEditOriginalPositions[editBase] + delta[0];
      this.positions[targetBase + 1] = this.moveEditOriginalPositions[editBase + 1] + delta[1];
      this.positions[targetBase + 2] = this.moveEditOriginalPositions[editBase + 2] + delta[2];
    }

    this.uploadPositions(device);
  }

  public previewTransformEdit(
    device: GPUDevice,
    translation: Vector3Tuple,
    rotation: [number, number, number, number],
    scale: Vector3Tuple,
  ): void {
    if (!this.positions || !this.moveEditIndices || !this.moveEditOriginalPositions || !this.moveEditCentroid) {
      return;
    }

    const [qx, qy, qz, qw] = normalizeQuaternion(rotation);

    for (let i = 0; i < this.moveEditIndices.length; i++) {
      const editBase = i * 3;
      const targetBase = this.moveEditIndices[i] * 3;
      const x = (this.moveEditOriginalPositions[editBase] - this.moveEditCentroid[0]) * scale[0];
      const y = (this.moveEditOriginalPositions[editBase + 1] - this.moveEditCentroid[1]) * scale[1];
      const z = (this.moveEditOriginalPositions[editBase + 2] - this.moveEditCentroid[2]) * scale[2];
      const rotated = rotateVectorByQuaternion(x, y, z, qx, qy, qz, qw);
      this.positions[targetBase] = this.moveEditCentroid[0] + translation[0] + rotated[0];
      this.positions[targetBase + 1] = this.moveEditCentroid[1] + translation[1] + rotated[1];
      this.positions[targetBase + 2] = this.moveEditCentroid[2] + translation[2] + rotated[2];
    }

    this.uploadPositions(device);
  }

  public commitMoveEdit(): void {
    this.moveEditIndices = null;
    this.moveEditOriginalPositions = null;
    this.moveEditCentroid = null;
    this.rebuildSelectionIndices();
    for (const chunk of this.chunksById.values()) {
      this.refreshChunkBounds(chunk);
      chunk.isDirty = true;
      chunk.editVersion++;
      chunk.lastSortDirection = null;
    }
  }

  public clearSelection(device: GPUDevice): void {
    if (!this.selectionMask) {
      return;
    }

    this.selectionGeneration++;
    this.selectionMask.fill(0);
    this.selectedSplatCount = 0;
    this.uploadSelectionMask(device);
  }

  public setSelectionHighlightVisible(device: GPUDevice, visible: boolean): void {
    if (this.selectionHighlightVisible === visible) {
      return;
    }

    this.selectionHighlightVisible = visible;
    this.uploadSelectionMask(device);
  }

  private compactSortedIndices(): void {
    if (!this.chunkLocalSortedIndices || !this.hiddenMask) {
      return;
    }

    for (const [, chunk] of this.chunksById) {
      const oldStart = chunk.localSortedIndicesOffset;
      const oldCount = chunk.localSortedIndicesCount;
      const oldEnd = oldStart + oldCount;
      let write = oldStart;

      for (let i = oldStart; i < oldEnd; i++) {
        const splatIndex = this.chunkLocalSortedIndices[i];

        if (this.hiddenMask[splatIndex] !== 0) {
          continue;
        }

        this.chunkLocalSortedIndices[write] = splatIndex;
        write++;
      }

      const newCount = write - oldStart;

      if (newCount !== oldCount) {
        chunk.localSortedIndicesCount = newCount;
        chunk.lastSortDirection = null;
        chunk.localOrderCacheVersion++;
      }
    }
  }

  public hideSelectedSplats(device: GPUDevice): number {
    if (!this.selectionMask || !this.hiddenMask) {
      return 0;
    }

    let hiddenCount = 0;

    for (let i = 0; i < this.count; i++) {
      if (this.selectionMask[i] === 0 || this.hiddenMask[i] !== 0) {
        continue;
      }

      this.hiddenMask[i] = 1;
      hiddenCount++;
    }

    this.uploadHiddenMask(device);
    this.compactSortedIndices();
    return hiddenCount;
  }

  public unhideAllSplats(device: GPUDevice): void {
    if (!this.hiddenMask) {
      return;
    }

    this.hiddenMask.fill(0);
    this.uploadHiddenMask(device);
    this.compactSortedIndices();
  }

  public dispose(): void {
    this.positionBuffer?.destroy();
    this.colorBuffer?.destroy();
    this.opacityBuffer?.destroy();
    this.covarianceBuffer?.destroy();
    this.orderBuffer?.destroy();
    if (this.ownsVisibleSplatIndicesBuffer) {
      this.visibleSplatIndicesBuffer?.destroy();
    }
    this.shBuffer?.destroy();
    this.splatIdBuffer?.destroy();
    this.chunkIdBuffer?.destroy();
    this.localIndexBuffer?.destroy();
    this.chunkMetadataBuffer?.destroy();
    this.selectionMaskBuffer?.destroy();
    this.hiddenMaskBuffer?.destroy();
    if (this.ownsIndirectArgsBuffer) {
      this.indirectArgsBuffer?.destroy();
    }
    this.positionBuffer = null;
    this.colorBuffer = null;
    this.opacityBuffer = null;
    this.covarianceBuffer = null;
    this.orderBuffer = null;
    this.visibleSplatIndicesBuffer = null;
    this.shBuffer = null;
    this.splatIdBuffer = null;
    this.chunkIdBuffer = null;
    this.localIndexBuffer = null;
    this.chunkMetadataBuffer = null;
    this.chunkMetadataBufferSize = 0;
    this.selectionMaskBuffer = null;
    this.hiddenMaskBuffer = null;
    this.indirectArgsBuffer = null;
    this.ownsVisibleSplatIndicesBuffer = true;
    this.ownsIndirectArgsBuffer = true;
    this.positions = null;
    this.colors = null;
    this.opacities = null;
    this.covariances = null;
    this.shCoefficients = null;
    this.order = null;
    this.visibleSplatIndices = null;
    this.selectionMask = null;
    this.hiddenSelectionMask = null;
    this.hiddenMask = null;
    this.selectedSplatCount = 0;
    this.chunkLocalSortedIndices = null;
    this.splatChunkIds = null;
    this.chunksById.clear();
    this.selectionIndicesByChunkId.clear();
    this.depths = null;
    this.count = 0;
    this.renderCount = 0;
  }

  public prepareDuplicateData(): { combined: SplatData; oldCount: number; oldHiddenMask: Uint32Array | null } | null {
    if (!this.positions || !this.colors || !this.opacities || !this.covariances || !this.shCoefficients || this.selectedSplatCount === 0) {
      return null;
    }

    const selectedIndices = this.collectSelectedIndices();
    const copyCount = selectedIndices.length;
    const oldCount = this.count;
    const newCount = oldCount + copyCount;
    const shStride = oldCount > 0 ? this.shCoefficients.length / oldCount : 0;
    const newPositions = new Float32Array(newCount * 3);
    const newColors = new Float32Array(newCount * 3);
    const newOpacities = new Float32Array(newCount);
    const newCovariances = new Float32Array(newCount * 6);
    const newShCoefficients = new Float32Array(newCount * shStride);

    newPositions.set(this.positions);
    newColors.set(this.colors);
    newOpacities.set(this.opacities);
    newCovariances.set(this.covariances);
    newShCoefficients.set(this.shCoefficients);

    for (let i = 0; i < copyCount; i++) {
      const srcIdx = selectedIndices[i];
      const dstIdx = oldCount + i;
      const srcBase = srcIdx * 3;
      const dstBase = dstIdx * 3;
      newPositions[dstBase] = this.positions[srcBase];
      newPositions[dstBase + 1] = this.positions[srcBase + 1];
      newPositions[dstBase + 2] = this.positions[srcBase + 2];
      newColors[dstBase] = this.colors[srcBase];
      newColors[dstBase + 1] = this.colors[srcBase + 1];
      newColors[dstBase + 2] = this.colors[srcBase + 2];
      newOpacities[dstIdx] = this.opacities[srcIdx];
      const srcCovBase = srcIdx * 6;
      const dstCovBase = dstIdx * 6;
      for (let j = 0; j < 6; j++) {
        newCovariances[dstCovBase + j] = this.covariances[srcCovBase + j];
      }
      for (let sh = 0; sh < shStride; sh++) {
        newShCoefficients[dstIdx * shStride + sh] = this.shCoefficients[srcIdx * shStride + sh];
      }
    }

    const oldHiddenMask = this.hiddenMask ? new Uint32Array(this.hiddenMask) : null;

    return {
      combined: {
        positions: newPositions,
        colors: newColors,
        opacities: newOpacities,
        covariances: newCovariances,
        shCoefficients: newShCoefficients,
        count: newCount,
      },
      oldCount,
      oldHiddenMask,
    };
  }

  public rebuildFromWorld(
    device: GPUDevice,
    worldData: WorldSplatData,
    oldCount: number,
    oldHiddenMask: Uint32Array | null,
  ): void {
    this.destroyGpuBuffers();
    this.setData(worldData);
    this.createBuffers(device);
    this.createStableIdBuffers(device, worldData);
    this.createIndirectArgsBuffer(device);

    this.createSelectionMaskBuffer(device);
    for (let i = 0; i < this.count; i++) {
      if (worldData.splatIds[i] >= oldCount) {
        this.selectionMask![i] = 1;
        this.selectedSplatCount++;
      }
    }
    this.uploadSelectionMask(device);

    this.createHiddenMaskBuffer(device);
    if (oldHiddenMask) {
      for (let i = 0; i < this.count; i++) {
        const splatId = worldData.splatIds[i];
        if (splatId < oldCount) {
          this.hiddenMask![i] = oldHiddenMask[splatId];
        }
      }
      this.uploadHiddenMask(device);
    }

    this.selectionGeneration++;
  }

  private destroyGpuBuffers(): void {
    this.positionBuffer?.destroy();
    this.colorBuffer?.destroy();
    this.opacityBuffer?.destroy();
    this.covarianceBuffer?.destroy();
    this.orderBuffer?.destroy();
    if (this.ownsVisibleSplatIndicesBuffer) {
      this.visibleSplatIndicesBuffer?.destroy();
      this.ownsVisibleSplatIndicesBuffer = true;
    }
    this.shBuffer?.destroy();
    this.splatIdBuffer?.destroy();
    this.chunkIdBuffer?.destroy();
    this.localIndexBuffer?.destroy();
    this.chunkMetadataBuffer?.destroy();
    this.selectionMaskBuffer?.destroy();
    this.hiddenMaskBuffer?.destroy();
    if (this.ownsIndirectArgsBuffer) {
      this.indirectArgsBuffer?.destroy();
      this.ownsIndirectArgsBuffer = true;
    }
    this.positionBuffer = null;
    this.colorBuffer = null;
    this.opacityBuffer = null;
    this.covarianceBuffer = null;
    this.orderBuffer = null;
    this.visibleSplatIndicesBuffer = null;
    this.shBuffer = null;
    this.splatIdBuffer = null;
    this.chunkIdBuffer = null;
    this.localIndexBuffer = null;
    this.chunkMetadataBuffer = null;
    this.chunkMetadataBufferSize = 0;
    this.selectionMaskBuffer = null;
    this.hiddenMaskBuffer = null;
    this.indirectArgsBuffer = null;
  }

  private getDepth(splatIndex: number): number {
    return this.depths?.[splatIndex] ?? 0;
  }

  private shouldRefreshChunkOrder(chunk: SplatChunk, viewDirection: Vector3Tuple): boolean {
    if (!chunk.lastSortDirection) {
      return true;
    }

    const dot =
      viewDirection[0] * chunk.lastSortDirection[0] +
      viewDirection[1] * chunk.lastSortDirection[1] +
      viewDirection[2] * chunk.lastSortDirection[2];

    return 1 - Math.abs(dot) > LOCAL_SORT_DIRECTION_EPSILON;
  }

  private refreshChunkOrder(
    chunk: SplatChunk,
    viewMatrix: Float32Array,
    viewDirection: Vector3Tuple,
  ): void {
    if (!this.positions || !this.depths || !this.chunkLocalSortedIndices) {
      return;
    }

    const start = chunk.localSortedIndicesOffset;
    const end = start + chunk.localSortedIndicesCount;

    for (let i = start; i < end; i++) {
      const splatIndex = this.chunkLocalSortedIndices[i];
      const base = splatIndex * 3;
      this.depths[splatIndex] =
        viewMatrix[2] * this.positions[base] +
        viewMatrix[6] * this.positions[base + 1] +
        viewMatrix[10] * this.positions[base + 2] +
        viewMatrix[14];
    }

    this.chunkLocalSortedIndices.subarray(start, end).sort(
      (a, b) => this.getDepth(b) - this.getDepth(a),
    );
    chunk.lastSortDirection = [...viewDirection];
    chunk.localOrderCacheVersion++;
  }

  private uploadVisibleSplatIndices(device: GPUDevice): void {
    if (!this.visibleSplatIndicesBuffer || !this.visibleSplatIndices) {
      return;
    }

    device.queue.writeBuffer(
      this.visibleSplatIndicesBuffer,
      0,
      this.visibleSplatIndices.buffer as ArrayBuffer,
      this.visibleSplatIndices.byteOffset,
      this.renderCount * Uint32Array.BYTES_PER_ELEMENT,
    );
  }

  private updateIndirectArgs(device: GPUDevice): void {
    if (!this.indirectArgsBuffer) {
      return;
    }

    device.queue.writeBuffer(
      this.indirectArgsBuffer,
      0,
      new Uint32Array([6, this.renderCount, 0, 0]),
    );
  }

  private uploadSelectionMask(device: GPUDevice): void {
    if (!this.selectionMask || !this.selectionMaskBuffer) {
      return;
    }

    if (!this.selectionHighlightVisible) {
      if (!this.hiddenSelectionMask || this.hiddenSelectionMask.length !== this.selectionMask.length) {
        this.hiddenSelectionMask = new Uint32Array(this.selectionMask.length);
      }

      device.queue.writeBuffer(
        this.selectionMaskBuffer,
        0,
        this.hiddenSelectionMask.buffer as ArrayBuffer,
        this.hiddenSelectionMask.byteOffset,
        this.hiddenSelectionMask.byteLength,
      );
      return;
    }

    device.queue.writeBuffer(
      this.selectionMaskBuffer,
      0,
      this.selectionMask.buffer as ArrayBuffer,
      this.selectionMask.byteOffset,
      this.selectionMask.byteLength,
    );
  }

  private uploadHiddenMask(device: GPUDevice): void {
    if (!this.hiddenMask || !this.hiddenMaskBuffer) {
      return;
    }

    device.queue.writeBuffer(
      this.hiddenMaskBuffer,
      0,
      this.hiddenMask.buffer as ArrayBuffer,
      this.hiddenMask.byteOffset,
      this.hiddenMask.byteLength,
    );
  }

  private uploadColors(device: GPUDevice): void {
    if (!this.colors || !this.colorBuffer) {
      return;
    }

    device.queue.writeBuffer(
      this.colorBuffer,
      0,
      this.colors.buffer as ArrayBuffer,
      this.colors.byteOffset,
      this.colors.byteLength,
    );
  }

  private uploadPositions(device: GPUDevice): void {
    if (!this.positions || !this.positionBuffer) {
      return;
    }

    device.queue.writeBuffer(
      this.positionBuffer,
      0,
      this.positions.buffer as ArrayBuffer,
      this.positions.byteOffset,
      this.positions.byteLength,
    );
  }

  private prepareSelectionMode(selectionMode: SelectionMode): void {
    if (selectionMode === "normal") {
      this.selectionMask?.fill(0);
      this.selectedSplatCount = 0;
    }
  }

  private applySelectionCandidate(splatIndex: number, selectionMode: SelectionMode): boolean {
    if (!this.selectionMask) {
      return false;
    }

    if (selectionMode === "subtractive") {
      const changed = this.selectionMask[splatIndex] !== 0;
      this.selectionMask[splatIndex] = 0;
      if (changed) {
        this.selectedSplatCount = Math.max(0, this.selectedSplatCount - 1);
      }
      return changed;
    }

    const changed = this.selectionMask[splatIndex] === 0;
    this.selectionMask[splatIndex] = 1;
    if (changed) {
      this.selectedSplatCount++;
    }
    return changed;
  }

  private collectSelectedIndices(): Uint32Array {
    if (!this.selectionMask) {
      return new Uint32Array();
    }

    const indices: number[] = [];

    for (let i = 0; i < this.selectionMask.length; i++) {
      if (this.selectionMask[i] !== 0) {
        indices.push(i);
      }
    }

    return Uint32Array.from(indices);
  }

  private rebuildSelectionIndices(): void {
    this.selectionIndicesByChunkId.clear();

    for (const chunk of this.chunksById.values()) {
      const selectionIndex = this.createChunkSelectionIndex(chunk);

      if (selectionIndex) {
        this.selectionIndicesByChunkId.set(chunk.id, selectionIndex);
      }
    }
  }

  private refreshChunkBounds(chunk: SplatChunk): void {
    if (!this.positions || chunk.splatCount <= 0) {
      return;
    }

    const min: Vector3Tuple = [Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY];
    const max: Vector3Tuple = [Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY];
    const end = chunk.splatStart + chunk.splatCount;

    for (let splatIndex = chunk.splatStart; splatIndex < end; splatIndex++) {
      const base = splatIndex * 3;
      const x = this.positions[base];
      const y = this.positions[base + 1];
      const z = this.positions[base + 2];
      min[0] = Math.min(min[0], x);
      min[1] = Math.min(min[1], y);
      min[2] = Math.min(min[2], z);
      max[0] = Math.max(max[0], x);
      max[1] = Math.max(max[1], y);
      max[2] = Math.max(max[2], z);
    }

    chunk.boundsMin = min;
    chunk.boundsMax = max;
    chunk.center = [
      (min[0] + max[0]) * 0.5,
      (min[1] + max[1]) * 0.5,
      (min[2] + max[2]) * 0.5,
    ];
    const dx = max[0] - chunk.center[0];
    const dy = max[1] - chunk.center[1];
    const dz = max[2] - chunk.center[2];
    chunk.radius = Math.sqrt(dx * dx + dy * dy + dz * dz);
  }

  private async selectProjectedGeometryProgressive(
    device: GPUDevice,
    options: {
      viewportWidth: number;
      viewportHeight: number;
      viewMatrix: Float32Array;
      viewProjectionMatrix: Float32Array;
      selectionMode: SelectionMode;
      chunks?: readonly SplatChunk[];
      contains(projected: { x: number; y: number; depth: number; radius: number }): boolean;
    },
  ): Promise<number> {
    if (!this.positions || !this.selectionMask) {
      return 0;
    }

    const generation = ++this.selectionGeneration;
    this.prepareSelectionMode(options.selectionMode);

    if (options.selectionMode === "normal") {
      this.uploadSelectionMask(device);
      await nextAnimationFrame();
    }

    const gridCols = Math.max(4, Math.ceil(options.viewportWidth / 16));
    const gridRows = Math.max(4, Math.ceil(options.viewportHeight / 16));
    const depthGrid = new Float32Array(gridCols * gridRows);
    depthGrid.fill(Number.POSITIVE_INFINITY);

    const chunks = options.chunks ?? [...this.chunksById.values()];
    let changedCount = 0;
    let processedSinceUpload = 0;
    let sliceStart = performance.now();

    for (const chunk of chunks) {
      if (generation !== this.selectionGeneration) {
        return changedCount;
      }

      const end = chunk.splatStart + chunk.splatCount;

      for (let splatIndex = chunk.splatStart; splatIndex < end; splatIndex++) {
        if (this.isSplatHidden(splatIndex)) {
          continue;
        }

        const projected = this.projectSplatToScreen(splatIndex, options);

        if (!projected || !options.contains(projected)) {
          continue;
        }

        const gx0 = Math.max(0, Math.floor((projected.x - projected.radius) / options.viewportWidth * gridCols));
        const gy0 = Math.max(0, Math.floor((projected.y - projected.radius) / options.viewportHeight * gridRows));
        const gx1 = Math.min(gridCols - 1, Math.floor((projected.x + projected.radius) / options.viewportWidth * gridCols));
        const gy1 = Math.min(gridRows - 1, Math.floor((projected.y + projected.radius) / options.viewportHeight * gridRows));
        let isFront = false;

        for (let gy = gy0; gy <= gy1; gy++) {
          for (let gx = gx0; gx <= gx1; gx++) {
            const cellDepth = depthGrid[gy * gridCols + gx];

            if (projected.depth <= cellDepth + Math.max(0.01, cellDepth * 0.008)) {
              isFront = true;
            }
          }
        }

        if (!isFront) {
          continue;
        }

        if (this.applySelectionCandidate(splatIndex, options.selectionMode)) {
          changedCount++;

          for (let gy = gy0; gy <= gy1; gy++) {
            for (let gx = gx0; gx <= gx1; gx++) {
              const idx = gy * gridCols + gx;
              if (projected.depth < depthGrid[idx]) {
                depthGrid[idx] = projected.depth;
              }
            }
          }
        }

        processedSinceUpload++;

        if (processedSinceUpload >= 8192 || performance.now() - sliceStart > 6) {
          this.uploadSelectionMask(device);
          await nextAnimationFrame();
          processedSinceUpload = 0;
          sliceStart = performance.now();

          if (generation !== this.selectionGeneration) {
            return changedCount;
          }
        }
      }
    }

    this.uploadSelectionMask(device);
    return changedCount;
  }

  private projectSplatToScreen(
    splatIndex: number,
    options: {
      viewportWidth: number;
      viewportHeight: number;
      viewMatrix: Float32Array;
      viewProjectionMatrix: Float32Array;
    },
  ): { x: number; y: number; depth: number; radius: number } | null {
    if (!this.positions) {
      return null;
    }

    const base = splatIndex * 3;
    const x = this.positions[base];
    const y = this.positions[base + 1];
    const z = this.positions[base + 2];
    const m = options.viewProjectionMatrix;
    const clipX = m[0] * x + m[4] * y + m[8] * z + m[12];
    const clipY = m[1] * x + m[5] * y + m[9] * z + m[13];
    const clipW = m[3] * x + m[7] * y + m[11] * z + m[15];

    if (clipW <= 0.001) {
      return null;
    }

    const ndcX = clipX / clipW;
    const ndcY = clipY / clipW;

    if (ndcX < -1.2 || ndcX > 1.2 || ndcY < -1.2 || ndcY > 1.2) {
      return null;
    }

    const view = options.viewMatrix;
    const depth = Math.abs(view[2] * x + view[6] * y + view[10] * z + view[14]);
    return {
      x: (ndcX * 0.5 + 0.5) * options.viewportWidth,
      y: (0.5 - ndcY * 0.5) * options.viewportHeight,
      depth,
      radius: estimateScreenRadius(this.covariances, splatIndex, clipW, options.viewportHeight),
    };
  }

  private collectScreenRadiusSeeds(options: ScreenColorSelectionOptions): number[] {
    if (!this.positions || !this.visibleSplatIndices) {
      return [];
    }

    const radiusSq = options.screenRadius * options.screenRadius;
    const candidates: Array<{ index: number; distanceSq: number; depth: number }> = [];
    const m = options.viewProjectionMatrix;
    const view = options.viewMatrix;
    let nearestDepth = Number.POSITIVE_INFINITY;

    const chunkCandidates = this.collectScreenRadiusChunks(options);

    for (const chunk of chunkCandidates) {
      const end = chunk.splatStart + chunk.splatCount;

      for (let splatIndex = chunk.splatStart; splatIndex < end; splatIndex++) {
        if (this.isSplatHidden(splatIndex)) {
          continue;
        }

      const base = splatIndex * 3;
      const x = this.positions[base];
      const y = this.positions[base + 1];
      const z = this.positions[base + 2];
      const clipX = m[0] * x + m[4] * y + m[8] * z + m[12];
      const clipY = m[1] * x + m[5] * y + m[9] * z + m[13];
      const clipW = m[3] * x + m[7] * y + m[11] * z + m[15];

      if (clipW <= 0.001) {
        continue;
      }

      const ndcX = clipX / clipW;
      const ndcY = clipY / clipW;

      if (ndcX < -1 || ndcX > 1 || ndcY < -1 || ndcY > 1) {
        continue;
      }

      const pixelX = (ndcX * 0.5 + 0.5) * options.viewportWidth;
      const pixelY = (0.5 - ndcY * 0.5) * options.viewportHeight;
      const dx = pixelX - options.screenX;
      const dy = pixelY - options.screenY;
      const distanceSq = dx * dx + dy * dy;

      if (distanceSq <= radiusSq) {
        const depth = Math.abs(
          view[2] * x +
          view[6] * y +
          view[10] * z +
          view[14],
        );

        candidates.push({ index: splatIndex, distanceSq, depth });
        nearestDepth = Math.min(nearestDepth, depth);
      }
      }
    }

    if (!Number.isFinite(nearestDepth)) {
      return [];
    }

    const depthTolerance = options.depthTolerance ?? Math.max(0.025, nearestDepth * 0.015);
    const frontLayerDepth = nearestDepth + depthTolerance;

    const frontCandidates = candidates
      .filter((candidate) => candidate.depth <= frontLayerDepth)
      .sort((a, b) => a.distanceSq - b.distanceSq);

    return this.pickColorDiverseSeeds(frontCandidates, options.colorThreshold);
  }

  private collectScreenRadiusChunks(options: ScreenColorSelectionOptions): SplatChunk[] {
    const chunks: SplatChunk[] = [];

    for (const index of this.selectionIndicesByChunkId.values()) {
      const bounds = this.projectChunkScreenBounds(index.chunk, options);

      if (!bounds) {
        continue;
      }

      const closestX = Math.max(bounds.minX, Math.min(options.screenX, bounds.maxX));
      const closestY = Math.max(bounds.minY, Math.min(options.screenY, bounds.maxY));
      const dx = closestX - options.screenX;
      const dy = closestY - options.screenY;

      if (dx * dx + dy * dy <= options.screenRadius * options.screenRadius) {
        chunks.push(index.chunk);
      }
    }

    return chunks;
  }

  private projectChunkScreenBounds(
    chunk: SplatChunk,
    options: ScreenColorSelectionOptions,
  ): { minX: number; minY: number; maxX: number; maxY: number } | null {
    const corners = [
      [chunk.boundsMin[0], chunk.boundsMin[1], chunk.boundsMin[2]],
      [chunk.boundsMax[0], chunk.boundsMin[1], chunk.boundsMin[2]],
      [chunk.boundsMin[0], chunk.boundsMax[1], chunk.boundsMin[2]],
      [chunk.boundsMax[0], chunk.boundsMax[1], chunk.boundsMin[2]],
      [chunk.boundsMin[0], chunk.boundsMin[1], chunk.boundsMax[2]],
      [chunk.boundsMax[0], chunk.boundsMin[1], chunk.boundsMax[2]],
      [chunk.boundsMin[0], chunk.boundsMax[1], chunk.boundsMax[2]],
      [chunk.boundsMax[0], chunk.boundsMax[1], chunk.boundsMax[2]],
    ];
    const m = options.viewProjectionMatrix;
    let minX = Number.POSITIVE_INFINITY;
    let minY = Number.POSITIVE_INFINITY;
    let maxX = Number.NEGATIVE_INFINITY;
    let maxY = Number.NEGATIVE_INFINITY;
    let projected = false;

    for (const corner of corners) {
      const x = corner[0];
      const y = corner[1];
      const z = corner[2];
      const clipX = m[0] * x + m[4] * y + m[8] * z + m[12];
      const clipY = m[1] * x + m[5] * y + m[9] * z + m[13];
      const clipW = m[3] * x + m[7] * y + m[11] * z + m[15];

      if (clipW <= 0.001) {
        continue;
      }

      const ndcX = clipX / clipW;
      const ndcY = clipY / clipW;
      const pixelX = (ndcX * 0.5 + 0.5) * options.viewportWidth;
      const pixelY = (0.5 - ndcY * 0.5) * options.viewportHeight;
      minX = Math.min(minX, pixelX);
      minY = Math.min(minY, pixelY);
      maxX = Math.max(maxX, pixelX);
      maxY = Math.max(maxY, pixelY);
      projected = true;
    }

    if (!projected || maxX < 0 || maxY < 0 || minX > options.viewportWidth || minY > options.viewportHeight) {
      return null;
    }

    return { minX, minY, maxX, maxY };
  }

  private pickColorDiverseSeeds(
    candidates: ReadonlyArray<{ index: number; distanceSq: number }>,
    colorThreshold: number,
  ): number[] {
    if (!this.colors) {
      return [];
    }

    const maxSeeds = 2048;
    const colorBinSize = Math.max(0.025, colorThreshold * 0.75);
    const seedSet = new Set<number>();
    const nearestByColor = new Map<string, { index: number; distanceSq: number }>();

    for (const candidate of candidates) {
      if (seedSet.size < 512) {
        seedSet.add(candidate.index);
      }

      const key = this.selectionColorKey(candidate.index, colorBinSize);
      const previous = nearestByColor.get(key);

      if (!previous || candidate.distanceSq < previous.distanceSq) {
        nearestByColor.set(key, candidate);
      }
    }

    for (const candidate of nearestByColor.values()) {
      seedSet.add(candidate.index);

      if (seedSet.size >= maxSeeds) {
        break;
      }
    }

    return [...seedSet];
  }

  private selectionColorKey(splatIndex: number, binSize: number): string {
    if (!this.colors) {
      return "0,0,0";
    }

    const base = splatIndex * 3;
    return `${Math.floor(this.colors[base] / binSize)},` +
      `${Math.floor(this.colors[base + 1] / binSize)},` +
      `${Math.floor(this.colors[base + 2] / binSize)}`;
  }

  private addConnectedSimilarColorCluster(
    seedGlobalIndex: number,
    seedChunk: SplatChunk,
    world: ChunkNeighborLookup,
    options: ColorClusterSelectionOptions,
    selectionMode: SelectionMode,
  ): number {
    if (!this.positions || !this.colors || !this.selectionMask) {
      return 0;
    }

    const seedBase = seedGlobalIndex * 3;
    if (this.isSplatHidden(seedGlobalIndex)) {
      return 0;
    }

    const seedR = this.colors[seedBase];
    const seedG = this.colors[seedBase + 1];
    const seedB = this.colors[seedBase + 2];
    const thresholdSq = options.colorThreshold * options.colorThreshold;
    const neighborRadiusSq = options.neighborRadius * options.neighborRadius;
    const candidates = this.collectColorCandidates(seedChunk, world, seedR, seedG, seedB, thresholdSq, options);

    if (!candidates.has(seedGlobalIndex)) {
      candidates.set(seedGlobalIndex, true);
    }

    if (
      options.selectBehind === false &&
      options.viewProjectionMatrix &&
      options.viewportWidth != null &&
      options.viewportHeight != null
    ) {
      this.filterColorCandidatesByDepth(
        candidates,
        options.viewProjectionMatrix,
        options.viewportWidth,
        options.viewportHeight,
        seedGlobalIndex,
        options.depthRangeFactor ?? 2.5,
        options.visibleChunks,
      );
    }

    const spatialHash = this.buildSelectionSpatialHash(candidates, options.neighborRadius);
    return this.floodSimilarColorCluster(
      seedGlobalIndex,
      candidates,
      spatialHash,
      neighborRadiusSq,
      options.maxSelectedSplats,
      selectionMode,
    );
  }

  private async addConnectedSimilarColorClusterProgressive(
    device: GPUDevice,
    generation: number,
    seedGlobalIndex: number,
    seedChunk: SplatChunk,
    world: ChunkNeighborLookup,
    options: ColorClusterSelectionOptions,
    selectionMode: SelectionMode,
  ): Promise<number> {
    if (!this.positions || !this.colors || !this.selectionMask || generation !== this.selectionGeneration) {
      return 0;
    }

    const seedBase = seedGlobalIndex * 3;
    if (this.isSplatHidden(seedGlobalIndex)) {
      return 0;
    }

    const seedR = this.colors[seedBase];
    const seedG = this.colors[seedBase + 1];
    const seedB = this.colors[seedBase + 2];
    const thresholdSq = options.colorThreshold * options.colorThreshold;
    const neighborRadiusSq = options.neighborRadius * options.neighborRadius;
    const candidates = this.collectColorCandidates(seedChunk, world, seedR, seedG, seedB, thresholdSq, options);

    if (!candidates.has(seedGlobalIndex)) {
      candidates.set(seedGlobalIndex, true);
    }

    if (
      options.selectBehind === false &&
      options.viewProjectionMatrix &&
      options.viewportWidth != null &&
      options.viewportHeight != null
    ) {
      this.filterColorCandidatesByDepth(
        candidates,
        options.viewProjectionMatrix,
        options.viewportWidth,
        options.viewportHeight,
        seedGlobalIndex,
        options.depthRangeFactor ?? 2.5,
        options.visibleChunks,
      );
    }

    const spatialHash = this.buildSelectionSpatialHash(candidates, options.neighborRadius);
    return this.floodSimilarColorClusterProgressive(
      device,
      generation,
      seedGlobalIndex,
      candidates,
      spatialHash,
      neighborRadiusSq,
      options.maxSelectedSplats,
      selectionMode,
    );
  }

  public getChunkForSplatIndex(splatIndex: number): SplatChunk | undefined {
    const chunkId = this.splatChunkIds?.[splatIndex];
    return chunkId === undefined ? undefined : this.chunksById.get(chunkId);
  }

  private createChunkSelectionIndex(chunk: SplatChunk): ChunkSelectionIndex | null {
    if (!this.positions || !this.colors) {
      return null;
    }

    const spatialCellSize = Math.max(0.015, chunk.radius * 0.035);
    const spatialBuckets = new Map<string, number[]>();
    const colorBuckets = new Map<number, number[]>();
    const end = chunk.splatStart + chunk.splatCount;

    for (let splatIndex = chunk.splatStart; splatIndex < end; splatIndex++) {
      if (this.isSplatHidden(splatIndex)) {
        continue;
      }

      const spatialKey = this.selectionCellKey(splatIndex, spatialCellSize);
      const spatialBucket = spatialBuckets.get(spatialKey);

      if (spatialBucket) {
        spatialBucket.push(splatIndex);
      } else {
        spatialBuckets.set(spatialKey, [splatIndex]);
      }

      const colorKey = this.quantizedColorKey(splatIndex);
      const colorBucket = colorBuckets.get(colorKey);

      if (colorBucket) {
        colorBucket.push(splatIndex);
      } else {
        colorBuckets.set(colorKey, [splatIndex]);
      }
    }

    return {
      chunk,
      spatialCellSize,
      spatialCells: freezeIndexBuckets(spatialBuckets),
      colorBins: freezeIndexBuckets(colorBuckets),
    };
  }

  private quantizedColorKey(splatIndex: number): number {
    if (!this.colors) {
      return 0;
    }

    const base = splatIndex * 3;
    return packedColorBin(
      colorBin(this.colors[base]),
      colorBin(this.colors[base + 1]),
      colorBin(this.colors[base + 2]),
    );
  }

  private collectColorCandidates(
    seedChunk: SplatChunk,
    world: ChunkNeighborLookup,
    seedR: number,
    seedG: number,
    seedB: number,
    thresholdSq: number,
    options: ColorClusterSelectionOptions,
  ): Map<number, true> {
    const candidates = new Map<number, true>();
    const chunks = [
      seedChunk,
      ...world.getNeighborChunks(seedChunk, 1),
    ].slice(0, options.maxCandidateChunks);

    if (!this.colors) {
      return candidates;
    }

    for (const chunk of chunks) {
      const sourceIndices = this.getColorCandidateIndicesForChunk(
        chunk,
        seedR,
        seedG,
        seedB,
        options.colorThreshold,
      );

      for (const i of sourceIndices) {
        if (this.isSplatHidden(i)) {
          continue;
        }

        const base = i * 3;
        const dr = this.colors[base] - seedR;
        const dg = this.colors[base + 1] - seedG;
        const db = this.colors[base + 2] - seedB;

        if (dr * dr + dg * dg + db * db <= thresholdSq) {
          candidates.set(i, true);
        }
      }
    }

    return candidates;
  }

  private getColorCandidateIndicesForChunk(
    chunk: SplatChunk,
    r: number,
    g: number,
    b: number,
    threshold: number,
  ): number[] {
    const index = this.selectionIndicesByChunkId.get(chunk.id);

    if (!index) {
      const fallback: number[] = [];
      const end = chunk.splatStart + chunk.splatCount;

      for (let i = chunk.splatStart; i < end; i++) {
        fallback.push(i);
      }

      return fallback;
    }

    const radius = Math.max(1, Math.ceil(threshold * COLOR_BIN_COUNT));
    const cr = colorBin(r);
    const cg = colorBin(g);
    const cb = colorBin(b);
    const result: number[] = [];

    for (let dz = -radius; dz <= radius; dz++) {
      const bz = cb + dz;

      if (bz < 0 || bz >= COLOR_BIN_COUNT) {
        continue;
      }

      for (let dy = -radius; dy <= radius; dy++) {
        const by = cg + dy;

        if (by < 0 || by >= COLOR_BIN_COUNT) {
          continue;
        }

        for (let dx = -radius; dx <= radius; dx++) {
          const bx = cr + dx;

          if (bx < 0 || bx >= COLOR_BIN_COUNT) {
            continue;
          }

          const bucket = index.colorBins.get(packedColorBin(bx, by, bz));

          if (bucket) {
            for (const splatIndex of bucket) {
              result.push(splatIndex);
            }
          }
        }
      }
    }

    return result;
  }

  private buildSelectionSpatialHash(candidates: ReadonlyMap<number, true>, cellSize: number): Map<string, number[]> {
    const hash = new Map<string, number[]>();

    if (!this.positions) {
      return hash;
    }

    for (const splatIndex of candidates.keys()) {
      const key = this.selectionCellKey(splatIndex, cellSize);
      const bucket = hash.get(key);

      if (bucket) {
        bucket.push(splatIndex);
      } else {
        hash.set(key, [splatIndex]);
      }
    }

    return hash;
  }

  private floodSimilarColorCluster(
    seedGlobalIndex: number,
    candidates: ReadonlyMap<number, true>,
    spatialHash: ReadonlyMap<string, readonly number[]>,
    neighborRadiusSq: number,
    maxSelectedSplats: number,
    selectionMode: SelectionMode,
  ): number {
    if (!this.positions || !this.selectionMask || !candidates.has(seedGlobalIndex)) {
      return 0;
    }

    const queue = [seedGlobalIndex];
    const visited = new Set<number>();

    while (queue.length > 0 && visited.size < maxSelectedSplats) {
      const splatIndex = queue.shift();

      if (splatIndex === undefined || visited.has(splatIndex)) {
        continue;
      }

      visited.add(splatIndex);
      const changed = this.applySelectionCandidate(splatIndex, selectionMode);
      void changed;

      for (const neighborIndex of this.findSelectionNeighbors(splatIndex, spatialHash, neighborRadiusSq)) {
        if (!visited.has(neighborIndex)) {
          queue.push(neighborIndex);
        }
      }
    }

    return visited.size;
  }

  private async floodSimilarColorClusterProgressive(
    device: GPUDevice,
    generation: number,
    seedGlobalIndex: number,
    candidates: ReadonlyMap<number, true>,
    spatialHash: ReadonlyMap<string, readonly number[]>,
    neighborRadiusSq: number,
    maxSelectedSplats: number,
    selectionMode: SelectionMode,
  ): Promise<number> {
    if (!this.positions || !this.selectionMask || !candidates.has(seedGlobalIndex)) {
      return 0;
    }

    const queue = [seedGlobalIndex];
    const visited = new Set<number>();
    let readIndex = 0;
    let selectedCount = 0;
    let sliceStart = performance.now();
    let processedSinceUpload = 0;

    while (readIndex < queue.length && visited.size < maxSelectedSplats) {
      if (generation !== this.selectionGeneration) {
        return selectedCount;
      }

      const splatIndex = queue[readIndex++];

      if (visited.has(splatIndex)) {
        continue;
      }

      visited.add(splatIndex);

      if (this.applySelectionCandidate(splatIndex, selectionMode)) {
        selectedCount++;
      }

      for (const neighborIndex of this.findSelectionNeighbors(splatIndex, spatialHash, neighborRadiusSq)) {
        if (!visited.has(neighborIndex)) {
          queue.push(neighborIndex);
        }
      }

      processedSinceUpload++;

      if (processedSinceUpload >= 4096 || performance.now() - sliceStart > 6) {
        this.uploadSelectionMask(device);
        await nextAnimationFrame();
        sliceStart = performance.now();
        processedSinceUpload = 0;
      }
    }

    this.uploadSelectionMask(device);
    return selectedCount;
  }

  private findSelectionNeighbors(
    splatIndex: number,
    spatialHash: ReadonlyMap<string, readonly number[]>,
    neighborRadiusSq: number,
  ): number[] {
    if (!this.positions) {
      return [];
    }

    const base = splatIndex * 3;
    const x = this.positions[base];
    const y = this.positions[base + 1];
    const z = this.positions[base + 2];
    const cellSize = Math.sqrt(neighborRadiusSq);
    const cx = Math.floor(x / cellSize);
    const cy = Math.floor(y / cellSize);
    const cz = Math.floor(z / cellSize);
    const neighbors: number[] = [];

    for (let dz = -1; dz <= 1; dz++) {
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          const bucket = spatialHash.get(`${cx + dx},${cy + dy},${cz + dz}`);

          if (!bucket) {
            continue;
          }

          for (const candidateIndex of bucket) {
            if (candidateIndex === splatIndex) {
              continue;
            }

            const candidateBase = candidateIndex * 3;
            const px = this.positions[candidateBase] - x;
            const py = this.positions[candidateBase + 1] - y;
            const pz = this.positions[candidateBase + 2] - z;

            if (px * px + py * py + pz * pz <= neighborRadiusSq) {
              neighbors.push(candidateIndex);
            }
          }
        }
      }
    }

    return neighbors;
  }

  private selectionCellKey(splatIndex: number, cellSize: number): string {
    if (!this.positions) {
      return "0,0,0";
    }

    const base = splatIndex * 3;
    return `${Math.floor(this.positions[base] / cellSize)},` +
      `${Math.floor(this.positions[base + 1] / cellSize)},` +
      `${Math.floor(this.positions[base + 2] / cellSize)}`;
  }

  private filterColorCandidatesByDepth(
    candidates: Map<number, true>,
    viewProjectionMatrix: Float32Array,
    viewportWidth: number,
    viewportHeight: number,
    seedIndex: number | undefined = undefined,
    depthRangeFactor: number = 2.5,
    visibleChunks?: readonly SplatChunk[],
  ): void {
    if (!this.positions) {
      return;
    }

    const gridSize = 16;
    const cols = Math.max(1, Math.ceil(viewportWidth / gridSize));
    const rows = Math.max(1, Math.ceil(viewportHeight / gridSize));
    const depthGrid = new Float32Array(cols * rows).fill(Number.POSITIVE_INFINITY);
    const m = viewProjectionMatrix;

    let seedDepth = 0;

    const projectSplat = (splatIndex: number): { pixelCol: number; pixelRow: number; depth: number; valid: boolean } => {
      const base = splatIndex * 3;
      const x = this.positions![base];
      const y = this.positions![base + 1];
      const z = this.positions![base + 2];
      const clipX = m[0] * x + m[4] * y + m[8] * z + m[12];
      const clipY = m[1] * x + m[5] * y + m[9] * z + m[13];
      const clipW = m[3] * x + m[7] * y + m[11] * z + m[15];

      if (clipW <= 0.001) {
        return { pixelCol: 0, pixelRow: 0, depth: 0, valid: false };
      }

      const ndcX = clipX / clipW;
      const ndcY = clipY / clipW;
      const pixelX = (ndcX * 0.5 + 0.5) * viewportWidth;
      const pixelY = (0.5 - ndcY * 0.5) * viewportHeight;
      const col = Math.max(0, Math.min(cols - 1, Math.floor(pixelX / gridSize)));
      const row = Math.max(0, Math.min(rows - 1, Math.floor(pixelY / gridSize)));

      return { pixelCol: col, pixelRow: row, depth: Math.abs(clipW), valid: true };
    };

    if (visibleChunks) {
      for (const chunk of visibleChunks) {
        const end = chunk.splatStart + chunk.splatCount;
        for (let i = chunk.splatStart; i < end; i++) {
          if (this.isSplatHidden(i)) {
            continue;
          }

          const proj = projectSplat(i);

          if (!proj.valid) {
            continue;
          }

          if (i === seedIndex) {
            seedDepth = proj.depth;
          }

          if (proj.depth < depthGrid[proj.pixelRow * cols + proj.pixelCol]) {
            depthGrid[proj.pixelRow * cols + proj.pixelCol] = proj.depth;
          }
        }
      }
    } else {
      for (const splatIndex of candidates.keys()) {
        const proj = projectSplat(splatIndex);

        if (!proj.valid) {
          candidates.delete(splatIndex);
          continue;
        }

        if (splatIndex === seedIndex) {
          seedDepth = proj.depth;
        }

        if (proj.depth < depthGrid[proj.pixelRow * cols + proj.pixelCol]) {
          depthGrid[proj.pixelRow * cols + proj.pixelCol] = proj.depth;
        }
      }
    }

    if (seedDepth === 0 && seedIndex !== undefined && candidates.has(seedIndex)) {
      const proj = projectSplat(seedIndex);

      if (proj.valid) {
        seedDepth = proj.depth;
      }
    }

    const maxDepth = seedDepth > 0 ? seedDepth * depthRangeFactor : Number.POSITIVE_INFINITY;
    const toDelete: number[] = [];

    for (const splatIndex of candidates.keys()) {
      const proj = projectSplat(splatIndex);

      if (!proj.valid) {
        toDelete.push(splatIndex);
        continue;
      }

      const frontDepth = depthGrid[proj.pixelRow * cols + proj.pixelCol];
      const occlusionTolerance = Math.max(0.01, frontDepth * 0.008);

      if (proj.depth > frontDepth + occlusionTolerance) {
        toDelete.push(splatIndex);
      } else if (proj.depth > maxDepth) {
        toDelete.push(splatIndex);
      }
    }

    for (const idx of toDelete) {
      candidates.delete(idx);
    }
  }

  public findNearestSplatAtScreenPos(
    screenX: number,
    screenY: number,
    viewportWidth: number,
    viewportHeight: number,
    viewProjectionMatrix: Float32Array,
    chunks: readonly SplatChunk[],
  ): number | null {
    if (!this.positions || !this.covariances) {
      return null;
    }

    let nearestIndex: number | null = null;
    let nearestDepth = Number.POSITIVE_INFINITY;
    const m = viewProjectionMatrix;

    for (const chunk of chunks) {
      const end = chunk.splatStart + chunk.splatCount;

      for (let i = chunk.splatStart; i < end; i++) {
        if (this.isSplatHidden(i)) {
          continue;
        }

        const base = i * 3;
        const x = this.positions[base];
        const y = this.positions[base + 1];
        const z = this.positions[base + 2];
        const clipX = m[0] * x + m[4] * y + m[8] * z + m[12];
        const clipY = m[1] * x + m[5] * y + m[9] * z + m[13];
        const clipW = m[3] * x + m[7] * y + m[11] * z + m[15];

        if (clipW <= 0.001) {
          continue;
        }

        const ndcX = clipX / clipW;
        const ndcY = clipY / clipW;
        const pixelX = (ndcX * 0.5 + 0.5) * viewportWidth;
        const pixelY = (0.5 - ndcY * 0.5) * viewportHeight;
        const dx = pixelX - screenX;
        const dy = pixelY - screenY;

        if (dx * dx > 2500 || dy * dy > 2500) {
          continue;
        }

        const screenRadius = estimateScreenRadius(this.covariances, i, clipW, viewportHeight);

        if (dx * dx + dy * dy <= screenRadius * screenRadius) {
          const depth = Math.abs(clipW);

          if (depth < nearestDepth) {
            nearestDepth = depth;
            nearestIndex = i;
          }
        }
      }
    }

    return nearestIndex;
  }

  private acceptTileBudget(splatIndex: number, state: TileBudgetState): boolean {
    if (!this.positions) {
      return true;
    }

    const base = splatIndex * 3;
    const x = this.positions[base];
    const y = this.positions[base + 1];
    const z = this.positions[base + 2];
    const m = state.options.viewProjectionMatrix;
    const clipX = m[0] * x + m[4] * y + m[8] * z + m[12];
    const clipY = m[1] * x + m[5] * y + m[9] * z + m[13];
    const clipW = m[3] * x + m[7] * y + m[11] * z + m[15];

    if (clipW <= 0.001) {
      return false;
    }

    const ndcX = clipX / clipW;
    const ndcY = clipY / clipW;

    if (ndcX < -1 || ndcX > 1 || ndcY < -1 || ndcY > 1) {
      return false;
    }

    const pixelX = (ndcX * 0.5 + 0.5) * state.options.viewportWidth;
    const pixelY = (0.5 - ndcY * 0.5) * state.options.viewportHeight;
    const tileX = Math.max(0, Math.min(state.tilesX - 1, Math.floor(pixelX / state.options.tileSize)));
    const tileY = Math.max(0, Math.min(state.tilesY - 1, Math.floor(pixelY / state.options.tileSize)));
    const tileIndex = tileY * state.tilesX + tileX;

    if (state.counts[tileIndex] >= state.options.maxSplatsPerTile) {
      return false;
    }

    state.counts[tileIndex]++;
    return true;
  }
}

interface TileBudgetState {
  options: TileBudgetOptions;
  tilesX: number;
  tilesY: number;
  counts: Uint32Array;
}

function createTileBudgetState(options: TileBudgetOptions): TileBudgetState {
  const tilesX = Math.max(1, Math.ceil(options.viewportWidth / options.tileSize));
  const tilesY = Math.max(1, Math.ceil(options.viewportHeight / options.tileSize));

  return {
    options,
    tilesX,
    tilesY,
    counts: new Uint32Array(tilesX * tilesY),
  };
}

function shouldApplyTileBudget(plan: ChunkRenderPlan, options: TileBudgetOptions): boolean {
  const depthDistance = Math.abs(plan.depthKey);
  return plan.screenRadius <= options.maxProtectedScreenRadius &&
    depthDistance >= options.protectedNearDepth;
}

function freezeIndexBuckets<Key>(source: Map<Key, number[]>): Map<Key, Uint32Array> {
  const frozen = new Map<Key, Uint32Array>();

  for (const [key, values] of source) {
    frozen.set(key, Uint32Array.from(values));
  }

  return frozen;
}

function colorBin(value: number): number {
  return Math.max(0, Math.min(COLOR_BIN_COUNT - 1, Math.floor(value * COLOR_BIN_COUNT)));
}

function packedColorBin(r: number, g: number, b: number): number {
  return r | (g << 5) | (b << 10);
}

function pointInPolygon(x: number, y: number, points: readonly ScreenPoint[]): boolean {
  let inside = false;

  for (let i = 0, j = points.length - 1; i < points.length; j = i++) {
    const xi = points[i].x;
    const yi = points[i].y;
    const xj = points[j].x;
    const yj = points[j].y;
    const intersects = (yi > y) !== (yj > y) &&
      x < ((xj - xi) * (y - yi)) / Math.max(1e-6, yj - yi) + xi;

    if (intersects) {
      inside = !inside;
    }
  }

  return inside;
}

function estimateScreenRadius(
  covariances: Float32Array | null,
  splatIndex: number,
  clipW: number,
  viewportHeight: number,
): number {
  if (!covariances) {
    return 1;
  }

  const base = splatIndex * 6;
  const sx = Math.sqrt(Math.max(0, covariances[base]));
  const sy = Math.sqrt(Math.max(0, covariances[base + 3]));
  const sz = Math.sqrt(Math.max(0, covariances[base + 5]));
  const worldRadius = Math.max(sx, sy, sz, 0.001);
  return Math.max(1, (worldRadius * viewportHeight) / Math.max(0.001, Math.abs(clipW)));
}

function rgbToHsv(r: number, g: number, b: number): [number, number, number] {
  const max = Math.max(r, g, b);
  const min = Math.min(r, g, b);
  const delta = max - min;
  let h = 0;

  if (delta > 1e-6) {
    if (max === r) {
      h = ((g - b) / delta) % 6;
    } else if (max === g) {
      h = (b - r) / delta + 2;
    } else {
      h = (r - g) / delta + 4;
    }
    h /= 6;
  }

  return [wrap01(h), max === 0 ? 0 : delta / max, max];
}

function hsvToRgb(h: number, s: number, v: number): [number, number, number] {
  const c = v * s;
  const x = c * (1 - Math.abs(((h * 6) % 2) - 1));
  const m = v - c;
  const sector = Math.floor(h * 6);
  let r = 0;
  let g = 0;
  let b = 0;

  if (sector === 0) {
    r = c; g = x;
  } else if (sector === 1) {
    r = x; g = c;
  } else if (sector === 2) {
    g = c; b = x;
  } else if (sector === 3) {
    g = x; b = c;
  } else if (sector === 4) {
    r = x; b = c;
  } else {
    r = c; b = x;
  }

  return [r + m, g + m, b + m];
}

function clamp01(value: number): number {
  return Math.max(0, Math.min(1, value));
}

function wrap01(value: number): number {
  return ((value % 1) + 1) % 1;
}

function shortestHueDelta(from: number, to: number): number {
  let delta = wrap01(to) - wrap01(from);

  if (delta > 0.5) {
    delta -= 1;
  } else if (delta < -0.5) {
    delta += 1;
  }

  return delta;
}

function normalizeQuaternion(value: [number, number, number, number]): [number, number, number, number] {
  const length = Math.hypot(value[0], value[1], value[2], value[3]) || 1;
  return [value[0] / length, value[1] / length, value[2] / length, value[3] / length];
}

function rotateVectorByQuaternion(
  x: number,
  y: number,
  z: number,
  qx: number,
  qy: number,
  qz: number,
  qw: number,
): Vector3Tuple {
  const ix = qw * x + qy * z - qz * y;
  const iy = qw * y + qz * x - qx * z;
  const iz = qw * z + qx * y - qy * x;
  const iw = -qx * x - qy * y - qz * z;

  return [
    ix * qw + iw * -qx + iy * -qz - iz * -qy,
    iy * qw + iw * -qy + iz * -qx - ix * -qz,
    iz * qw + iw * -qz + ix * -qy - iy * -qx,
  ];
}

function nextAnimationFrame(): Promise<void> {
  return new Promise((resolve) => {
    requestAnimationFrame(() => resolve());
  });
}
