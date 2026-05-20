import splatShader from "../shaders/splat.wgsl?raw";
import { CameraUniforms } from "../camera/camera-uniforms";
import { SplatBuffer } from "../splats/splatBuffer";
import { GpuContext } from "./gpu-context";

export default class RenderPipeline {
  private readonly device: GPUDevice;
  private readonly cameraBindGroupLayout: GPUBindGroupLayout;
  private readonly splatBindGroupLayout: GPUBindGroupLayout;
  private readonly pipeline: GPURenderPipeline;
  private splatBindGroup: GPUBindGroup | null = null;
  private splatCount = 0;

  constructor(gpu: GpuContext) {
    this.device = gpu.device;

    const shaderModule = gpu.device.createShaderModule({
      label: "SplatParticleShader",
      code: splatShader,
    });

    this.cameraBindGroupLayout = gpu.device.createBindGroupLayout({
      label: "CameraBindGroupLayout",
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.VERTEX,
          buffer: { type: "uniform" },
        },
      ],
    });

    this.splatBindGroupLayout = gpu.device.createBindGroupLayout({
      label: "SplatBindGroupLayout",
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.VERTEX,
          buffer: { type: "read-only-storage" },
        },
        {
          binding: 1,
          visibility: GPUShaderStage.VERTEX,
          buffer: { type: "read-only-storage" },
        },
      ],
    });

    this.pipeline = gpu.device.createRenderPipeline({
      label: "SplatParticlePipeline",
      layout: gpu.device.createPipelineLayout({
        label: "SplatParticlePipelineLayout",
        bindGroupLayouts: [
          this.cameraBindGroupLayout,
          this.splatBindGroupLayout,
        ],
      }),
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
        topology: "point-list",
      },
    });
  }

  setSplatBuffer(splatBuffer: SplatBuffer): void {
    const positionBuffer = splatBuffer.getPositionBuffer();
    const colorBuffer = splatBuffer.getColorBuffer();

    if (!positionBuffer || !colorBuffer) {
      throw new Error("Splat positions and colors must be uploaded before binding.");
    }

    this.splatBindGroup = this.device.createBindGroup({
      label: "SplatBindGroup",
      layout: this.splatBindGroupLayout,
      entries: [
        {
          binding: 0,
          resource: { buffer: positionBuffer },
        },
        {
          binding: 1,
          resource: { buffer: colorBuffer },
        },
      ],
    });
    this.splatCount = splatBuffer.getCount();
  }

  getCameraBindGroupLayout(): GPUBindGroupLayout {
    return this.cameraBindGroupLayout;
  }

  renderFrame(
    encoder: GPUCommandEncoder,
    textureView: GPUTextureView,
    cameraUniforms: CameraUniforms,
  ): void {
    const cameraBindGroup = cameraUniforms.getBindGroup();

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

    if (cameraBindGroup && this.splatBindGroup && this.splatCount > 0) {
      renderPass.setBindGroup(0, cameraBindGroup);
      renderPass.setBindGroup(1, this.splatBindGroup);
      renderPass.draw(this.splatCount);
    }

    renderPass.end();
  }

  dispose(): void {
    // GPURenderPipeline objects are owned by the device and do not need manual disposal.
  }
}
