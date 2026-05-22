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
  isDirty: boolean;
  editVersion: number;
  selectionVersion: number;
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

