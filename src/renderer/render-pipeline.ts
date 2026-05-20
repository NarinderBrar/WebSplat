import splatShader from "../shaders/splat.wgsl?raw";
import { GpuContext } from "./gpu-context";

export default class RenderPipeline {
  private readonly pipeline: GPURenderPipeline;

  constructor(gpu: GpuContext) {
    const shaderModule = gpu.device.createShaderModule({
      label: "DebugSplatShader",
      code: splatShader,
    });

    this.pipeline = gpu.device.createRenderPipeline({
      label: "DebugSplatPipeline",
      layout: "auto",
      vertex: {
        module: shaderModule,
        entryPoint: "vsMain",
      },
      fragment: {
        module: shaderModule,
        entryPoint: "fsMain",
        targets: [
          {
            format: gpu.presentationFormat,
          },
        ],
      },
      primitive: {
        topology: "triangle-list",
      },
    });
  }

  renderFrame(encoder: GPUCommandEncoder, textureView: GPUTextureView): void {
    const renderPass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: textureView,
          clearValue: {
            r: 0.03,
            g: 0.05,
            b: 0.07,
            a: 1,
          },
          loadOp: "clear",
          storeOp: "store",
        },
      ],
    });

    renderPass.setPipeline(this.pipeline);
    renderPass.draw(3);
    renderPass.end();
  }

  dispose(): void {
    // GPURenderPipeline objects are owned by the device and do not need manual disposal.
  }
}
