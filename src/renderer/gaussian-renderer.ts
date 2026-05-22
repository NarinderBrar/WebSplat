import { CameraUniforms } from "../camera/camera-uniforms";
import type { GpuChunkCullPass } from "../passes/gpuChunkCullPass";
import { SplatBuffer } from "../splats/splatBuffer";
import { GpuContext } from "./gpu-context";
import RenderPipeline from "./render-pipeline";

export class GaussianRenderer {
  private readonly gpu: GpuContext;
  private readonly pipeline: RenderPipeline;

  constructor(gpu: GpuContext) {
    this.gpu = gpu;
    this.pipeline = new RenderPipeline(gpu);
  }

  public setSplatBuffer(splatBuffer: SplatBuffer): void {
    this.pipeline.setSplatBuffer(splatBuffer);
  }

  public getCameraBindGroupLayout(): GPUBindGroupLayout {
    return this.pipeline.getCameraBindGroupLayout();
  }

  public setGpuChunkCullPass(pass: GpuChunkCullPass): void {
    this.pipeline.setGpuChunkCullPass(pass);
  }

  public render(cameraUniforms: CameraUniforms): void {
    this.gpu.resizeIfNeeded();

    const encoder = this.gpu.beginFrame();
    const textureView = this.gpu.getCurrentTextureView();

    this.pipeline.renderFrame(encoder, textureView, cameraUniforms);
    this.gpu.submit(encoder);
  }

  public dispose(): void {
    this.pipeline.dispose();
  }
}
