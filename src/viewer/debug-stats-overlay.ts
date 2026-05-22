import type { ChunkDebugStats } from "../world/types";

export class DebugStatsOverlay {
  private readonly element: HTMLDivElement;

  constructor() {
    this.element = document.createElement("div");
    this.element.className = "debug-stats-overlay";
    document.body.append(this.element);
  }

  public update(stats: ChunkDebugStats): void {
    this.element.textContent =
      `splats ${stats.visibleSplats}/${stats.totalSplats} | ` +
      `chunks ${stats.visibleChunks}/${stats.totalChunks} | ` +
      `culled ${stats.culledChunks} chunks, ${stats.culledSplats} splats | ` +
      `render ${stats.renderOrderSplatCount} | ` +
      `lod ${stats.lod0Splats}/${stats.lod1Splats}/${stats.lod2Splats}/${stats.lod3Splats} | ` +
      `${stats.estimatedFps.toFixed(1)} fps | ` +
      `cull ${stats.cpuCullMs.toFixed(2)}ms sort ${stats.chunkSortMs.toFixed(2)}ms ` +
      `cache ${stats.localOrderRefreshMs.toFixed(2)}ms build ${stats.visibleIndexBuildMs.toFixed(2)}ms`;
  }

  public dispose(): void {
    this.element.remove();
  }
}
