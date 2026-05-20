import { GpuContext } from "./gpu-context";
import RenderPipeline from "./render-pipeline";

export class GaussianRenderer {
  private readonly gpu: GpuContext;
  private readonly pipeline: RenderPipeline;

  constructor(gpu: GpuContext) {
    this.gpu = gpu;
    this.pipeline = new RenderPipeline(gpu);
  }

  public render(): void {
    this.gpu.resizeIfNeeded();

    const encoder = this.gpu.beginFrame();
    const textureView = this.gpu.getCurrentTextureView();

    this.pipeline.renderFrame(encoder, textureView);
    this.gpu.submit(encoder);
  }

  public dispose(): void {
    this.pipeline.dispose();
  }
}
