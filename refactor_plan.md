# WebSplat Custom WebGPU Renderer Foundation Plan

## Summary

WebSplat now owns the production render path directly through WebGPU:

`app -> GaussianSplatViewer -> GpuContext -> GaussianRenderer -> RenderPipeline`

Babylon.js has been removed from the runtime path. The next foundation work is to harden this custom WebGPU renderer for a high-performance Gaussian splatting viewer: correct canvas/device ownership, low-allocation camera updates, validated splat ingestion, and a real WGSL render pipeline.

Current checks:

- `tsc -p tsconfig.json` passes
- `vite build` passes
- production JS bundle is about `3.5 kB`

## Completed Foundation Fixes

- `package.json` and `package-lock.json` now use the package name `websplat`.
- `webgpu-shims.d.ts` has been removed; `@webgpu/types` is the WebGPU type source.
- `GpuContext` calls `resize()` before first render so canvas pixel size is correct immediately.
- `GpuContext.dispose()` calls `context.unconfigure()` before destroying the device.
- `GaussianRenderer` routes frame rendering through `RenderPipeline`.
- `OrbitCamera.update()` uploads `view`, `projection`, and `viewProjection`.
- `CameraUniforms.upload()` reuses one packed `Float32Array(48)` instead of allocating every frame.
- `splatLoader.ts` validates source byte length and parses with `DataView`.
- `splatBuffer.ts` uploads complete typed-array byte ranges to GPU buffers.
- `splat.wgsl` is compiled into a real `GPURenderPipeline` for the debug triangle path.

## Revised Implementation Sequence

1. **Clean Naming And Stale Files**
   - Rename this file from `refactor_paln.md` to `refactor_plan.md`.
   - Rename package metadata to `websplat`.
   - Remove the stale WebGPU shim file and rely on `@webgpu/types`.

2. **Harden `GpuContext`**
   - Let `GpuContext` own adapter, device, canvas context, presentation format, resize, command encoders, and submission.
   - Configure once during creation and resize before the first frame.
   - Unconfigure the canvas context during dispose.

3. **Make Renderer Pipeline Real**
   - Move clear-pass and draw behavior into `RenderPipeline`.
   - Keep `GaussianRenderer.render()` as orchestration: resize, begin frame, run pipeline, submit.
   - Compile `splat.wgsl` into a `GPURenderPipeline` and draw a debug triangle through the real path.

4. **Wire Camera Correctly**
   - Instantiate `OrbitCamera` in `GaussianSplatViewer`.
   - Update camera uniforms before each render.
   - Upload `view`, `projection`, and `viewProjection`.
   - Avoid per-frame uniform packing allocations.

5. **Fix Splat Data Foundation**
   - Validate the 48-byte initial splat stride.
   - Parse with `DataView`.
   - Keep the initial binary layout explicit and narrow before adding broader format support.
   - Upload GPU buffers from typed arrays without element/byte length confusion.

## Test Plan

- Run `tsc -p tsconfig.json`; it must pass.
- Run `vite build`; it must pass.
- Confirm no `@babylonjs`, `Babylon`, `WebGPUEngine`, `Scene`, `scene.render`, or `runRenderLoop` imports/usages in `src`.
- Open `http://127.0.0.1:5173`; canvas must render through direct WebGPU.
- Resize the browser; canvas must keep correct pixel dimensions.
- Load an invalid splat buffer; loader must reject with a clear error.
- Load a synthetic valid splat buffer; renderer must accept it without CPU allocation spikes in the parse loop.

## Assumptions

- Babylon should remain absent from production runtime.
- The immediate goal is foundation correctness and low-overhead architecture, not full visual quality.
- Initial splat support can use one documented 48-byte binary layout before broader format support is added.
- Performance-sensitive code should avoid per-frame and per-splat temporary allocations.
