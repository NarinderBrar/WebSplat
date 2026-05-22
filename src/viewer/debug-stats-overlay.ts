import type { ChunkDebugStats } from "../world/types";

export class DebugStatsOverlay {
  private readonly element: HTMLDivElement;

  constructor() {
    this.element = document.createElement("div");
    this.element.className = "debug-stats-overlay";
    document.body.append(this.element);
  }

  public update(stats: ChunkDebugStats): void {
    const isGpu = stats.backend !== "cpuChunkBinned";
    const renderText = isGpu ? "gpu indirect" : `${stats.renderOrderSplatCount}`;
    const timingText = isGpu
      ? "gpu cull/bin"
      : `cull ${stats.cpuCullMs.toFixed(2)}ms sort ${stats.chunkSortMs.toFixed(2)}ms ` +
        `cache ${stats.localOrderRefreshMs.toFixed(2)}ms build ${stats.visibleIndexBuildMs.toFixed(2)}ms`;

    this.element.textContent =
      `${stats.backend} | ` +
      `splats ${stats.visibleSplats}/${stats.totalSplats} | ` +
      `chunks ${stats.visibleChunks}/${stats.totalChunks} | ` +
      `render ${renderText} | ` +
      `tile ${stats.tileCulledSplats}/${stats.tileTestedSplats} skip/test protect ${stats.tileProtectedSplats} | ` +
      `lod ${stats.lod0Splats}/${stats.lod1Splats}/${stats.lod2Splats}/${stats.lod3Splats} | ` +
      `${stats.estimatedFps.toFixed(1)} fps | ` +
      timingText;
  }

  public dispose(): void {
    this.element.remove();
  }
}
