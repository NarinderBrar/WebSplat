import type { SplatChunk } from "./types";

export const CHUNK_GPU_METADATA_STRIDE_BYTES = 64;

export enum ChunkDirtyFlags {
  None = 0,
  Attributes = 1 << 0,
  Selection = 1 << 1,
  Transform = 1 << 2,
  Lod = 1 << 3,
}

export function packChunkGpuMetadata(chunks: readonly SplatChunk[]): ArrayBuffer {
  const buffer = new ArrayBuffer(chunks.length * CHUNK_GPU_METADATA_STRIDE_BYTES);
  const view = new DataView(buffer);

  chunks.forEach((chunk, index) => {
    const base = index * CHUNK_GPU_METADATA_STRIDE_BYTES;
    const flags = chunk.isDirty ? ChunkDirtyFlags.Attributes : ChunkDirtyFlags.None;

    view.setFloat32(base, chunk.boundsMin[0], true);
    view.setFloat32(base + 4, chunk.boundsMin[1], true);
    view.setFloat32(base + 8, chunk.boundsMin[2], true);
    view.setFloat32(base + 12, chunk.radius, true);

    view.setFloat32(base + 16, chunk.boundsMax[0], true);
    view.setFloat32(base + 20, chunk.boundsMax[1], true);
    view.setFloat32(base + 24, chunk.boundsMax[2], true);
    view.setFloat32(base + 28, chunk.lodLevel, true);

    view.setFloat32(base + 32, chunk.center[0], true);
    view.setFloat32(base + 36, chunk.center[1], true);
    view.setFloat32(base + 40, chunk.center[2], true);
    view.setFloat32(base + 44, chunk.splatCount, true);

    view.setUint32(base + 48, chunk.id, true);
    view.setUint32(base + 52, chunk.splatStart, true);
    view.setUint32(base + 56, chunk.gpuOffset, true);
    view.setUint32(base + 60, flags, true);
  });

  return buffer;
}
