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
  private splatBuffer: SplatBuffer | null = null;
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
        {
          binding: 2,
          visibility: GPUShaderStage.VERTEX,
          buffer: { type: "read-only-storage" },
        },
        {
          binding: 3,
          visibility: GPUShaderStage.VERTEX,
          buffer: { type: "read-only-storage" },
        },
        {
          binding: 4,
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
            blend: {
              color: {
                srcFactor: "one",
                dstFactor: "one-minus-src-alpha",
                operation: "add",
              },
              alpha: {
                srcFactor: "one",
                dstFactor: "one-minus-src-alpha",
                operation: "add",
              },
            },
          },
        ],
      },
      primitive: {
        topology: "triangle-list",
      },
    });

  }

  setSplatBuffer(splatBuffer: SplatBuffer): void {
    const positionBuffer = splatBuffer.getPositionBuffer();
    const colorBuffer = splatBuffer.getColorBuffer();
    const covarianceBuffer = splatBuffer.getCovarianceBuffer();
    const opacityBuffer = splatBuffer.getOpacityBuffer();
    const orderBuffer = splatBuffer.getOrderBuffer();

    if (!positionBuffer || !colorBuffer || !covarianceBuffer || !opacityBuffer || !orderBuffer) {
      throw new Error(
        "Splat positions, colors, covariances, opacities and render order must be uploaded before binding.",
      );
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
        {
          binding: 2,
          resource: { buffer: covarianceBuffer },
        },
        {
          binding: 3,
          resource: { buffer: opacityBuffer },
        },
        {
          binding: 4,
          resource: { buffer: orderBuffer },
        },
      ],
    });

    this.splatBuffer = splatBuffer;
    this.splatCount = splatBuffer.getRenderCount();
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

    this.splatCount = this.splatBuffer?.getRenderCount() ?? 0;

    if (cameraBindGroup && this.splatBindGroup && this.splatCount > 0) {
      renderPass.setPipeline(this.pipeline);
      renderPass.setBindGroup(0, cameraBindGroup);
      renderPass.setBindGroup(1, this.splatBindGroup);
      renderPass.draw(this.splatCount * 6);
    }

    renderPass.end();
  }

  dispose(): void {
    // GPURenderPipeline objects are owned by the device and do not need manual disposal.
  }
}
