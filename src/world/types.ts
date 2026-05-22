import type { SplatData } from "../splats/splatBuffer";

export type Vector3Tuple = [number, number, number];

export interface StableSplatRef {
  splatId: number;
  chunkId: number;
  localIndex: number;
}

export interface SplatChunk {
  id: number;
  gridCoord: Vector3Tuple;
  boundsMin: Vector3Tuple;
  boundsMax: Vector3Tuple;
  center: Vector3Tuple;
  radius: number;
  splatStart: number;
  splatCount: number;
  gpuOffset: number;
  lodLevel: number;
  lodStep: number;
  screenRadius: number;
  depthKey: number;
  lastSortDirection: Vector3Tuple | null;
  localSortedIndicesOffset: number;
  localSortedIndicesCount: number;
  localOrderCacheVersion: number;
  isDirty: boolean;
  editVersion: number;
  selectionVersion: number;
}

export type RenderQualityMode = "quality" | "balanced" | "gpu-balanced" | "performance";
export type GpuRenderBackend = "cpuChunkBinned" | "gpuDepthBinned" | "gpuRadixSorted";

export interface ChunkRenderPlan {
  chunkId: number;
  depthKey: number;
  lodStep: number;
  splatStart: number;
  splatCount: number;
  screenRadius: number;
  localOrderCacheVersion: number;
}

export interface ChunkBuildOptions {
  targetSplatsPerChunk: number;
  maxSplatsPerChunk: number;
  minCellSize: number;
  maxRefinementPasses: number;
}

export interface ChunkVisibility {
  chunks: readonly SplatChunk[];
  splatCount: number;
}

export interface ChunkDebugStats {
  backend: GpuRenderBackend;
  totalSplats: number;
  totalChunks: number;
  visibleChunks: number;
  visibleSplats: number;
  culledChunks: number;
  culledSplats: number;
  renderOrderSplatCount: number;
  lod0Splats: number;
  lod1Splats: number;
  lod2Splats: number;
  lod3Splats: number;
  estimatedFps: number;
  frameMs: number;
  cpuCullMs: number;
  chunkSortMs: number;
  localOrderRefreshMs: number;
  visibleIndexBuildMs: number;
}

export interface WorldSplatData extends SplatData {
  splatIds: Uint32Array;
  chunkIds: Uint32Array;
  localIndices: Uint32Array;
}

export interface ChunkLookupResult extends StableSplatRef {
  globalIndex: number;
  chunk: SplatChunk;
}

export const DEFAULT_CHUNK_BUILD_OPTIONS: ChunkBuildOptions = {
  targetSplatsPerChunk: 32_000,
  maxSplatsPerChunk: 100_000,
  minCellSize: 1e-4,
  maxRefinementPasses: 6,
};
