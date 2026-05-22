import type { SplatData } from "../splats/splatBuffer";
import { packChunkGpuMetadata } from "./chunk-gpu-metadata";
import { Frustum } from "./frustum";
import {
  DEFAULT_CHUNK_BUILD_OPTIONS,
  type GpuRenderBackend,
  type ChunkRenderPlan,
  type ChunkBuildOptions,
  type ChunkDebugStats,
  type ChunkLookupResult,
  type ChunkVisibility,
  type RenderQualityMode,
  type SplatChunk,
  type Vector3Tuple,
  type WorldSplatData,
} from "./types";

interface BuildBucket {
  id: number;
  coord: Vector3Tuple;
  indices: number[];
}

interface ChunkBuild {
  buckets: BuildBucket[];
  cellSize: number;
  maxBucketSize: number;
}

export class SplatWorld {
  private readonly chunksById = new Map<number, SplatChunk>();
  private readonly splatIdToGlobalIndex = new Map<number, number>();
  private readonly frustum = new Frustum();
  private readonly chunks: SplatChunk[];
  private readonly splatData: WorldSplatData;
  private readonly boundsMin: Vector3Tuple;
  private readonly boundsMax: Vector3Tuple;
  private readonly cellSize: number;
  private visibleChunks: SplatChunk[] = [];
  private visibleSplatCount = 0;
  private readonly lastLodCounts = [0, 0, 0, 0];
  private lastChunkSortMs = 0;

  private constructor(
    splatData: WorldSplatData,
    chunks: SplatChunk[],
    boundsMin: Vector3Tuple,
    boundsMax: Vector3Tuple,
    cellSize: number,
  ) {
    this.splatData = splatData;
    this.chunks = chunks;
    this.boundsMin = boundsMin;
    this.boundsMax = boundsMax;
    this.cellSize = cellSize;

    for (const chunk of chunks) {
      this.chunksById.set(chunk.id, chunk);
    }

    for (let i = 0; i < splatData.count; i++) {
      this.splatIdToGlobalIndex.set(splatData.splatIds[i], i);
    }

    this.visibleChunks = chunks.slice();
    this.visibleSplatCount = splatData.count;
  }

  public static fromSplatData(
    data: SplatData,
    options: Partial<ChunkBuildOptions> = {},
  ): SplatWorld {
    const buildOptions = { ...DEFAULT_CHUNK_BUILD_OPTIONS, ...options };
    const bounds = computeBounds(data.positions, data.count);
    const build = buildGrid(data, bounds.min, bounds.max, buildOptions);
    const reordered = reorderSplatData(data, build.buckets);
    const chunks = createChunks(reordered, build.buckets);

    return new SplatWorld(reordered, chunks, bounds.min, bounds.max, build.cellSize);
  }

  public updateVisibility(viewProjection: Float32Array): ChunkVisibility {
    this.frustum.update(viewProjection);
    this.visibleChunks = [];
    this.visibleSplatCount = 0;

    for (const chunk of this.chunks) {
      if (this.frustum.intersectsChunk(chunk)) {
        this.visibleChunks.push(chunk);
        this.visibleSplatCount += chunk.splatCount;
      }
    }

    return this.getVisibility();
  }

  public getVisibility(): ChunkVisibility {
    return {
      chunks: this.visibleChunks,
      splatCount: this.visibleSplatCount,
    };
  }

  public createRenderPlans(
    viewMatrix: Float32Array,
    projectionMatrix: Float32Array,
    viewportHeight: number,
    qualityMode: RenderQualityMode,
    previousFrameMs: number,
  ): readonly ChunkRenderPlan[] {
    const start = performance.now();
    const focalLength = projectionMatrix[5] * viewportHeight * 0.5;
    const adaptivePressure = previousFrameMs > 16.6;
    const plans: ChunkRenderPlan[] = [];
    this.lastLodCounts.fill(0);

    for (const chunk of this.visibleChunks) {
      const depth =
        viewMatrix[2] * chunk.center[0] +
        viewMatrix[6] * chunk.center[1] +
        viewMatrix[10] * chunk.center[2] +
        viewMatrix[14];
      const safeDepth = Math.max(0.001, depth);
      const screenRadius = Math.max(0, (chunk.radius * focalLength) / safeDepth);
      const lodStep = chooseLodStep(screenRadius, qualityMode, adaptivePressure);

      chunk.depthKey = depth;
      chunk.screenRadius = screenRadius;
      chunk.lodStep = lodStep;
      chunk.lodLevel = Math.log2(lodStep);
      this.lastLodCounts[lodLevelIndex(lodStep)] += Math.ceil(chunk.splatCount / lodStep);
      plans.push({
        chunkId: chunk.id,
        depthKey: depth,
        lodStep,
        splatStart: chunk.splatStart,
        splatCount: chunk.splatCount,
        screenRadius,
        localOrderCacheVersion: chunk.localOrderCacheVersion,
      });
    }

    plans.sort((a, b) => b.depthKey - a.depthKey);
    this.lastChunkSortMs = performance.now() - start;
    return plans;
  }

  public getDebugStats(
    renderOrderSplatCount: number,
    telemetry: Partial<Omit<ChunkDebugStats, "backend" | "totalSplats" | "totalChunks" | "visibleChunks" | "visibleSplats" | "culledChunks" | "culledSplats" | "renderOrderSplatCount" | "lod0Splats" | "lod1Splats" | "lod2Splats" | "lod3Splats">> & { backend?: GpuRenderBackend } = {},
  ): ChunkDebugStats {
    return {
      backend: telemetry.backend ?? "cpuChunkBinned",
      totalSplats: this.splatData.count,
      totalChunks: this.chunks.length,
      visibleChunks: this.visibleChunks.length,
      visibleSplats: this.visibleSplatCount,
      culledChunks: this.chunks.length - this.visibleChunks.length,
      culledSplats: this.splatData.count - this.visibleSplatCount,
      renderOrderSplatCount,
      lod0Splats: this.lastLodCounts[0],
      lod1Splats: this.lastLodCounts[1],
      lod2Splats: this.lastLodCounts[2],
      lod3Splats: this.lastLodCounts[3],
      estimatedFps: telemetry.estimatedFps ?? 0,
      frameMs: telemetry.frameMs ?? 0,
      cpuCullMs: telemetry.cpuCullMs ?? 0,
      chunkSortMs: telemetry.chunkSortMs ?? this.lastChunkSortMs,
      localOrderRefreshMs: telemetry.localOrderRefreshMs ?? 0,
      visibleIndexBuildMs: telemetry.visibleIndexBuildMs ?? 0,
    };
  }

  public getSplatData(): WorldSplatData {
    return this.splatData;
  }

  public getChunks(): readonly SplatChunk[] {
    return this.chunks;
  }

  public getChunkById(chunkId: number): SplatChunk | undefined {
    return this.chunksById.get(chunkId);
  }

  public lookupSplat(splatId: number): ChunkLookupResult | undefined {
    const globalIndex = this.splatIdToGlobalIndex.get(splatId);

    if (globalIndex === undefined) {
      return undefined;
    }

    const chunkId = this.splatData.chunkIds[globalIndex];
    const chunk = this.chunksById.get(chunkId);

    if (!chunk) {
      return undefined;
    }

    return {
      splatId,
      chunkId,
      localIndex: this.splatData.localIndices[globalIndex],
      globalIndex,
      chunk,
    };
  }

  public findChunkAtWorldPosition(position: Vector3Tuple): SplatChunk | undefined {
    const coord = worldToGridCoord(position, this.boundsMin, this.cellSize);
    const id = chunkIdFromCoord(coord);
    return this.chunksById.get(id);
  }

  public packGpuMetadata(): ArrayBuffer {
    return packChunkGpuMetadata(this.chunks);
  }

  public getBoundsMin(): Vector3Tuple {
    return [...this.boundsMin];
  }

  public getBoundsMax(): Vector3Tuple {
    return [...this.boundsMax];
  }

  public getCellSize(): number {
    return this.cellSize;
  }
}

function computeBounds(positions: Float32Array, count: number): { min: Vector3Tuple; max: Vector3Tuple } {
  const min: Vector3Tuple = [Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY];
  const max: Vector3Tuple = [Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY];

  for (let i = 0; i < count; i++) {
    const base = i * 3;
    const x = positions[base];
    const y = positions[base + 1];
    const z = positions[base + 2];

    min[0] = Math.min(min[0], x);
    min[1] = Math.min(min[1], y);
    min[2] = Math.min(min[2], z);
    max[0] = Math.max(max[0], x);
    max[1] = Math.max(max[1], y);
    max[2] = Math.max(max[2], z);
  }

  if (count === 0) {
    return { min: [0, 0, 0], max: [0, 0, 0] };
  }

  return { min, max };
}

function buildGrid(
  data: SplatData,
  boundsMin: Vector3Tuple,
  boundsMax: Vector3Tuple,
  options: ChunkBuildOptions,
): ChunkBuild {
  const extentX = Math.max(boundsMax[0] - boundsMin[0], options.minCellSize);
  const extentY = Math.max(boundsMax[1] - boundsMin[1], options.minCellSize);
  const extentZ = Math.max(boundsMax[2] - boundsMin[2], options.minCellSize);
  const volume = extentX * extentY * extentZ;
  let cellSize = Math.max(
    options.minCellSize,
    Math.cbrt((volume * options.targetSplatsPerChunk) / Math.max(1, data.count)),
  );
  let build = bucketSplats(data.positions, data.count, boundsMin, cellSize);

  for (let pass = 0; pass < options.maxRefinementPasses && build.maxBucketSize > options.maxSplatsPerChunk; pass++) {
    cellSize = Math.max(options.minCellSize, cellSize * 0.5);
    build = bucketSplats(data.positions, data.count, boundsMin, cellSize);
  }

  return {
    buckets: build.buckets,
    cellSize,
    maxBucketSize: build.maxBucketSize,
  };
}

function bucketSplats(
  positions: Float32Array,
  count: number,
  boundsMin: Vector3Tuple,
  cellSize: number,
): { buckets: BuildBucket[]; maxBucketSize: number } {
  const bucketsById = new Map<number, BuildBucket>();
  let maxBucketSize = 0;

  for (let i = 0; i < count; i++) {
    const base = i * 3;
    const coord = worldToGridCoord(
      [positions[base], positions[base + 1], positions[base + 2]],
      boundsMin,
      cellSize,
    );
    const id = chunkIdFromCoord(coord);
    let bucket = bucketsById.get(id);

    if (!bucket) {
      bucket = { id, coord, indices: [] };
      bucketsById.set(id, bucket);
    }

    bucket.indices.push(i);
    maxBucketSize = Math.max(maxBucketSize, bucket.indices.length);
  }

  return {
    buckets: [...bucketsById.values()].sort((a, b) => a.id - b.id),
    maxBucketSize,
  };
}

function reorderSplatData(data: SplatData, buckets: readonly BuildBucket[]): WorldSplatData {
  const positions = new Float32Array(data.count * 3);
  const colors = new Float32Array(data.count * 3);
  const opacities = new Float32Array(data.count);
  const covariances = new Float32Array(data.count * 6);
  const shStride = data.count > 0 ? data.shCoefficients.length / data.count : 0;
  const shCoefficients = new Float32Array(data.count * shStride);
  const splatIds = new Uint32Array(data.count);
  const chunkIds = new Uint32Array(data.count);
  const localIndices = new Uint32Array(data.count);
  let writeIndex = 0;

  for (const bucket of buckets) {
    for (let localIndex = 0; localIndex < bucket.indices.length; localIndex++) {
      const sourceIndex = bucket.indices[localIndex];
      copyTuple(data.positions, positions, sourceIndex, writeIndex, 3);
      copyTuple(data.colors, colors, sourceIndex, writeIndex, 3);
      copyTuple(data.covariances, covariances, sourceIndex, writeIndex, 6);
      opacities[writeIndex] = data.opacities[sourceIndex];

      for (let sh = 0; sh < shStride; sh++) {
        shCoefficients[writeIndex * shStride + sh] = data.shCoefficients[sourceIndex * shStride + sh];
      }

      splatIds[writeIndex] = sourceIndex;
      chunkIds[writeIndex] = bucket.id;
      localIndices[writeIndex] = localIndex;
      writeIndex++;
    }
  }

  return {
    positions,
    colors,
    opacities,
    covariances,
    shCoefficients,
    splatIds,
    chunkIds,
    localIndices,
    count: data.count,
  };
}

function createChunks(data: WorldSplatData, buckets: readonly BuildBucket[]): SplatChunk[] {
  const chunks: SplatChunk[] = [];
  let splatStart = 0;

  for (const bucket of buckets) {
    const bounds = computeBoundsForRange(data.positions, splatStart, bucket.indices.length);
    const center: Vector3Tuple = [
      (bounds.min[0] + bounds.max[0]) * 0.5,
      (bounds.min[1] + bounds.max[1]) * 0.5,
      (bounds.min[2] + bounds.max[2]) * 0.5,
    ];
    const radius = Math.hypot(
      bounds.max[0] - center[0],
      bounds.max[1] - center[1],
      bounds.max[2] - center[2],
    );

    chunks.push({
      id: bucket.id,
      gridCoord: bucket.coord,
      boundsMin: bounds.min,
      boundsMax: bounds.max,
      center,
      radius,
      splatStart,
      splatCount: bucket.indices.length,
      gpuOffset: splatStart,
      lodLevel: 0,
      lodStep: 1,
      screenRadius: 0,
      depthKey: 0,
      lastSortDirection: null,
      localSortedIndicesOffset: splatStart,
      localSortedIndicesCount: bucket.indices.length,
      localOrderCacheVersion: 0,
      isDirty: false,
      editVersion: 0,
      selectionVersion: 0,
    });
    splatStart += bucket.indices.length;
  }

  return chunks;
}

function chooseLodStep(
  screenRadius: number,
  qualityMode: RenderQualityMode,
  adaptivePressure: boolean,
): number {
  if (qualityMode === "quality") {
    return screenRadius < 5 ? 2 : 1;
  }

  if (qualityMode === "performance") {
    if (screenRadius < 4) return 8;
    if (screenRadius < 9) return 4;
    if (screenRadius < 18) return 2;
    return adaptivePressure && screenRadius < 32 ? 2 : 1;
  }

  if (screenRadius < 3) return adaptivePressure ? 8 : 4;
  if (screenRadius < 8) return adaptivePressure ? 4 : 2;
  if (screenRadius < 18) return adaptivePressure ? 2 : 1;
  return 1;
}

function lodLevelIndex(lodStep: number): number {
  if (lodStep >= 8) return 3;
  if (lodStep >= 4) return 2;
  if (lodStep >= 2) return 1;
  return 0;
}

function computeBoundsForRange(
  positions: Float32Array,
  start: number,
  count: number,
): { min: Vector3Tuple; max: Vector3Tuple } {
  const min: Vector3Tuple = [Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY];
  const max: Vector3Tuple = [Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY];

  for (let i = start; i < start + count; i++) {
    const base = i * 3;
    min[0] = Math.min(min[0], positions[base]);
    min[1] = Math.min(min[1], positions[base + 1]);
    min[2] = Math.min(min[2], positions[base + 2]);
    max[0] = Math.max(max[0], positions[base]);
    max[1] = Math.max(max[1], positions[base + 1]);
    max[2] = Math.max(max[2], positions[base + 2]);
  }

  return { min, max };
}

function worldToGridCoord(position: Vector3Tuple, boundsMin: Vector3Tuple, cellSize: number): Vector3Tuple {
  return [
    Math.floor((position[0] - boundsMin[0]) / cellSize),
    Math.floor((position[1] - boundsMin[1]) / cellSize),
    Math.floor((position[2] - boundsMin[2]) / cellSize),
  ];
}

function chunkIdFromCoord(coord: Vector3Tuple): number {
  const x = coord[0] & 0x1fffff;
  const y = coord[1] & 0x1fffff;
  const z = coord[2] & 0x1fffff;
  return ((x * 73856093) ^ (y * 19349663) ^ (z * 83492791)) >>> 0;
}

function copyTuple(
  source: Float32Array,
  destination: Float32Array,
  sourceIndex: number,
  destinationIndex: number,
  stride: number,
): void {
  for (let i = 0; i < stride; i++) {
    destination[destinationIndex * stride + i] = source[sourceIndex * stride + i];
  }
}
