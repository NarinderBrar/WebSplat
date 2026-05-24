# WebSplat

WebSplat is a browser-based Gaussian Splat viewer and editor prototype built with WebGPU.

The goal is to become a Photoshop-like editor for splats: open a scene, inspect it, select parts of it, paint color, adjust color, hide details, and move selected splats directly in the browser.

## Features

- Open Gaussian Splat scenes in the browser
- Load `.ply` & `.sog` files
- Point selection
- Brush selection (color flood-fill, drag support, select-behind toggle, depth range slider)
- Circle selection
- Marquee selection
- Lasso selection
- Additive, subtractive, and normal selection modes
- Copy selected splats
- Paint brush
- HSV adjustment for selected splats
- Hide / unhide splats
- Move, rotate, scale transform tools
- Visualization modes: normal, particle cloud, random colors

## Getting Started

Install dependencies:

```bash
npm install
```

Start the development server:

```bash
npm run dev
```

Open:

```text
http://127.0.0.1:5173
```

## URL Flags

| Flag | Values | Default | Description |
|------|--------|---------|-------------|
| `?splat=<path>` | file path | demo scene | Load a `.ply` or `.sog` splat file |
| `?quality=quality\|balanced\|gpu-balanced\|performance` | one of 4 | `performance` | Render quality / fidelity level |
| `?optimized=1\|true\|yes\|on` | boolean | off | Enable GPU-optimized rendering (GPU chunk culling + depth binning) |

Example:

```text
http://127.0.0.1:5173/?splat=/supported/strawberry.ply&quality=balanced&optimized=1
```

## Visualization Modes

Three view modes toggleable via the circular buttons at the bottom center:

| Mode | Description |
|------|-------------|
| **Normal** | Full gaussian splat rendering with per-splat colors |
| **Particle cloud** (default) | Tiny fixed-size dots, no covariance scaling — lightweight overview |
| **Random colors** | Per-chunk random colors — visualizes the spatial grid partitioning |

## Scripts

```bash
npm run dev
npm run build
npm run preview
```
`npm run build` runs TypeScript and the Vite production build.

## Architecture

WebSplat is built around a chunked world model:

```text
World
  Grid Chunks
    Splats
```

The renderer still uses flat GPU buffers, but the scene is organized into uniform grid chunks with stable splat IDs. This makes future editing, selection, streaming, undo/redo, and GPU-driven rendering easier to build.

Core systems:

- `SplatWorld`: builds grid chunks, preserves stable IDs, performs chunk lookup, and produces visible chunks
- `SplatBuffer`: owns CPU/GPU splat attributes, visible index buffers, selection masks, hidden masks, and edit operations
- `GaussianSplatViewer`: coordinates camera, world visibility, rendering, picking, and editor APIs
- `RenderPipeline`: owns WebGPU render pipeline and splat draw bindings
- `IdPickingPass`: renders stable splat IDs for picking
- `GpuChunkCullPass`: GPU chunk-culling foundation
- `GpuDepthBinPass`: experimental GPU depth-binning foundation
- `GpuTilePressurePass`: experimental tile-pressure telemetry foundation

More detailed world/chunk notes are in:

```text
src/world/README.md
```

## Rendering and Optimization Notes

Several rendering optimization techniques have been explored. Some are enabled as part of the normal renderer, while others are experimental or currently disabled by default.

Enabled or active:

- Uniform grid chunking
- Stable splat IDs
- Chunk-contiguous splat ranges
- CPU frustum culling at chunk level
- Visible splat index buffer
- Indirect draw argument buffer
- Chunk depth ordering
- Cached per-chunk local ordering
- Selection mask buffer
- Hidden splat filtering
- ID picking pass

Available but disabled by default:

- GPU depth-binning path
- GPU tile-pressure pass
- CPU tile-budget thinning driven by tile pressure
- Optional optimization mode controlled by internal URL flag

Tried or prepared as foundations:

- GPU chunk culling
- GPU visible splat index generation
- GPU depth binning
- Tile-aware overdraw pressure telemetry
- Chunk-level LOD metadata
- Draw-indirect support

Future optimization work:

- Full GPU splat sorting
- Packed color buffers
- Packed or quantized covariance buffers
- Dirty-range GPU uploads for editing
- GPU-accelerated brush and lasso selection
- Better small-scene quality path with tighter per-splat ordering

## Current Limitations

- Full undo/redo is not implemented yet
- Full GPU splat sort is not implemented yet
- Some edit operations still upload full buffers
- Move tool is still a foundation, not a complete production transform system
- Some datasets may need better covariance/splat footprint calibration
- Very large files may hit WebGPU adapter limits on lower-end hardware

## Development Goal

This project is editor-first. Rendering order, selection, picking, painting, hiding, and movement all resolve through stable base splat indices so future tools can build on the same identity model instead of depending on temporary GPU instance order.
