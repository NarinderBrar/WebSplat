# WebSplat High-Level Roadmap: Fast Custom WebGPU Gaussian Splatting Viewer

## Summary

Build WebSplat as a custom WebGPU-first Gaussian splatting viewer, with Babylon.js removed from the production render path. The long-term architecture should optimize for startup speed, streaming, GPU memory layout, compute-driven visibility/sorting, low-overhead draw submission, and measurable frame-time performance.

This is the high-level roadmap to pair with the step-by-step foundation plan in `docs/custom-webgpu-renderer-plan.md`.

## Phase 1: Custom WebGPU Foundation

- Own WebGPU directly: `GPUAdapter`, `GPUDevice`, `GPUCanvasContext`, swapchain format, resize, frame command submission.
- Replace Babylon render loop with `requestAnimationFrame`.
- Replace Babylon `Scene`/camera/mesh/material concepts with plain viewer state.
- Create a minimal custom renderer that clears the canvas and proves direct WebGPU ownership.
- Establish core modules: `viewer`, `renderer`, `passes`, `camera`, `splats`, `shaders`.

## Phase 2: Minimal Splat Rendering

- Define an internal packed splat data layout optimized for GPU upload.
- Implement a simple loader path for one splat source format.
- Upload splat attributes into GPU buffers.
- Add a first `splatRenderPass` using WGSL.
- Render visible splats without advanced sorting first, so the full data path works end to end.

## Phase 3: Camera, Projection, And Interaction

- Implement custom orbit/fly camera without Babylon.
- Maintain CPU camera state and GPU camera uniform buffers.
- Add frustum data for culling.
- Support resize, DPR changes, pointer controls, wheel zoom, and frame-stable camera updates.
- Keep camera math allocation-light and predictable.

## Phase 4: GPU Visibility And Sorting

- Add `cullPass` as a compute pass for frustum and optional screen-space culling.
- Add `computeSortPass` for camera-depth ordering.
- Start with a reliable GPU sort, then optimize toward radix/bitonic or hybrid strategies.
- Track visible splat count, culled count, sort time, render time, and total GPU frame time.
- Avoid CPU roundtrips in the hot path.

## Phase 5: Performance-Oriented Renderer Architecture

- Formalize `RenderPipeline` and `FrameGraph` around ordered GPU passes:
  `upload/update -> cull -> sort -> splat -> resolve`.
- Reuse command encoders, bind groups, pipelines, and buffers where possible.
- Separate static resources from per-frame resources.
- Add timing hooks using GPU timestamp queries where available.
- Keep renderer APIs data-oriented, not object-scene-oriented.

## Phase 6: Streaming And Large Scene Support

- Add progressive loading for large splat files.
- Use chunked GPU uploads and staging buffers.
- Add coarse spatial partitioning for streaming and culling.
- Support partial scene availability: render loaded chunks while more data arrives.
- Prepare for LOD, quantization, compression, and cache-aware memory layout.

## Phase 7: Quality And Advanced Rendering

- Add proper alpha blending strategy for splats.
- Add SH color evaluation if needed by the target format.
- Add covariance/screen-space ellipse projection.
- Add optional resolve/post pass for final composition.
- Keep visual quality modes explicit: fastest, balanced, quality.

## Phase 8: Benchmarking And Validation

- Add an in-app performance overlay for FPS, CPU frame time, GPU frame time, visible splats, sorted splats, and memory.
- Add repeatable benchmark scenes.
- Add browser/device capability reporting.
- Track bundle size and startup time.
- Define performance targets before optimizing each subsystem.

## Core Principle

The production viewer should never depend on Babylon rendering primitives. Babylon may exist only in isolated examples or debug experiments. The main path must remain:

`app -> GaussianSplatViewer -> GpuContext -> GaussianRenderer -> RenderPipeline -> WebGPU passes`

## Assumptions

- First priority is architecture correctness for a custom WebGPU pipeline.
- The initial renderer can be visually simple while the data and GPU ownership model are hardened.
- Performance work should be measured, not guessed.
- The project should evolve toward data-oriented GPU systems rather than scene-graph abstractions.
