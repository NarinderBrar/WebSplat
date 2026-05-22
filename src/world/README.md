# World and Chunk Architecture

WebSplat uses a uniform spatial grid with fixed-size chunks as the world authority.
The renderer still consumes flat WebGPU buffers, but splats are reordered by chunk so
each chunk owns a contiguous range. That keeps the current draw path simple while
preparing the data model for GPU culling, indirect rendering, streaming, and editing.

## Folder Structure

- `types.ts`: editor-facing chunk, stable ID, lookup, and visibility contracts.
- `splat-world.ts`: chunk builder, uniform grid lookup, stable splat ID mapping, and per-frame visible chunk generation.
- `frustum.ts`: CPU frustum extraction and chunk sphere tests.
- `chunk-gpu-metadata.ts`: packed 64-byte chunk records for storage buffers and future compute passes.

## Class Responsibilities

- `SplatWorld`
  - Builds grid chunks from loaded splat data.
  - Reorders splat attributes into chunk-contiguous ranges.
  - Preserves stable `splatId`, `chunkId`, and `localIndex` arrays.
  - Answers `lookupSplat(splatId)` for picking and editor operations.
  - Answers `findChunkAtWorldPosition(position)` for brush and streaming queries.
  - Produces visible chunks each frame from camera frustum culling.
  - Packs chunk metadata for GPU storage buffers.

- `SplatBuffer`
  - Uploads flat splat attribute arrays.
  - Uploads stable ID buffers and chunk metadata buffers.
  - Builds the render order buffer from visible chunk ranges.

- `GaussianSplatViewer`
  - Owns the `SplatWorld`.
  - Updates camera matrices, asks the world for visible chunks, and renders only those ranges.

## GPU Buffer Strategy

Current buffers:

- `SplatPositions`: `array<f32>`, xyz per splat.
- `SplatColors`: `array<f32>`, rgb per splat.
- `SplatCovariances`: `array<f32>`, six covariance terms per splat.
- `SplatOpacities`: `array<f32>`, one opacity per splat.
- `SplatRenderOrder`: `array<u32>`, compact visible splat indices sorted by view depth.
- `StableSplatIds`: `array<u32>`, persistent editor IDs.
- `StableSplatChunkIds`: `array<u32>`, owning chunk ID per splat.
- `StableSplatLocalIndices`: `array<u32>`, index within the owning chunk.
- `SplatChunkMetadata`: 64-byte records with bounds, center, radius, LOD, ranges, offsets, and dirty flags.

Planned GPU-driven path:

1. Compute pass reads `SplatChunkMetadata` and camera planes.
2. Visible chunks are compacted into a chunk list.
3. Per-chunk LOD selects the active splat range or proxy.
4. Draw commands are written into an indirect buffer.
5. Render pass uses `drawIndirect` or chunk-batched draws.

Current GPU foundation:

- `GpuChunkCullPass` culls chunks on GPU and writes `visibleChunkFlags`, `visibleChunkIndices`, `visibleSplatIndices`, counters, and indirect draw args.
- The render shader reads `visibleSplatIndices[instanceIndex]`, so instance order is temporary and stable splat identity remains in the base ID buffers.
- `IdPickingPass` renders stable `splatId + 1` values into an `r32uint` texture and exposes readback for single-pixel picking.
- `SelectionMask` is allocated as a per-splat GPU buffer and the render shader can highlight selected splats.

## Update Flow

Load:

1. Parse splat source into flat attributes.
2. Build `SplatWorld.fromSplatData`.
3. Reorder splats by chunk while preserving stable IDs.
4. Upload splat attributes, stable ID buffers, and chunk metadata.

Frame:

1. Update camera matrices.
2. Frustum-cull chunks with `SplatWorld.updateVisibility`.
3. Fill and depth-sort `SplatRenderOrder` from visible chunk ranges.
4. Draw `visibleSplatCount * 6` vertices.

Edit:

1. Picking resolves a stable `splatId`.
2. `lookupSplat` maps the ID to `chunkId`, `localIndex`, and global buffer index.
3. Brush tools query grid chunks overlapping the brush volume.
4. Dirty chunks update only affected attribute, selection, color override, or transform buffers.
5. Undo/redo stores chunk-scoped deltas keyed by stable IDs.

## Extensibility Notes

- Chunk target size defaults to 32k splats and refines while any chunk exceeds 100k splats.
- Chunks carry `editVersion`, `selectionVersion`, and dirty flags so future tools can update GPU buffers incrementally.
- Stable IDs never depend on GPU instance index; render order can be compacted, sorted, culled, or LOD-filtered without breaking editor references.
- Connected color selection should flood through neighboring grid chunks instead of scanning the whole world.
- Move, paint, erase, layer, and override systems should write chunk-local deltas first, then rebuild bounds and GPU metadata only for dirty chunks.
