import splatShader from "../shaders/splat.wgsl?raw";
import { CameraUniforms } from "../camera/camera-uniforms";
import type { GpuChunkCullPass } from "../passes/gpuChunkCullPass";
import type { GpuDepthBinPass } from "../passes/gpuDepthBinPass";
import type { GpuTilePressurePass } from "../passes/gpuTilePressurePass";
import type { ComputeSortPass } from "../passes/computeSortPass";
import { SplatBuffer } from "../splats/splatBuffer";
import { GpuContext } from "./gpu-context";

export default class RenderPipeline {
  private readonly device: GPUDevice;
  private readonly cameraBindGroupLayout: GPUBindGroupLayout;
  private readonly splatBindGroupLayout: GPUBindGroupLayout;
  private readonly pipeline: GPURenderPipeline;
  private readonly renderSettingsBuffer: GPUBuffer;
  private splatBindGroup: GPUBindGroup | null = null;
  private splatBuffer: SplatBuffer | null = null;
  private gpuChunkCullPass: GpuChunkCullPass | null = null;
  private gpuDepthBinPass: GpuDepthBinPass | null = null;
  private gpuTilePressurePass: GpuTilePressurePass | null = null;
  private computeSortPass: ComputeSortPass | null = null;
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
          visibility: GPUShaderStage.VERTEX | GPUShaderStage.COMPUTE,
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
        {
          binding: 5,
          visibility: GPUShaderStage.VERTEX,
          buffer: { type: "read-only-storage" },
        },
        {
          binding: 6,
          visibility: GPUShaderStage.VERTEX,
          buffer: { type: "uniform" },
        },
        {
          binding: 7,
          visibility: GPUShaderStage.VERTEX,
          buffer: { type: "read-only-storage" },
        },
      ],
    });

    this.renderSettingsBuffer = gpu.device.createBuffer({
      label: "SplatRenderSettings",
      size: 4 * Float32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.setQualityLevel(0);

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
    const visibleSplatIndicesBuffer = splatBuffer.getVisibleSplatIndicesBuffer();
    const selectionMaskBuffer = splatBuffer.getSelectionMaskBuffer();
    const chunkIdBuffer = splatBuffer.getChunkIdBuffer();

    if (!positionBuffer || !colorBuffer || !covarianceBuffer || !opacityBuffer || !visibleSplatIndicesBuffer || !selectionMaskBuffer) {
      throw new Error(
        "Splat positions, colors, covariances, opacities, visible indices and selection mask must be uploaded before binding.",
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
          resource: { buffer: visibleSplatIndicesBuffer },
        },
        {
          binding: 5,
          resource: { buffer: selectionMaskBuffer },
        },
        {
          binding: 6,
          resource: { buffer: this.renderSettingsBuffer },
        },
        {
          binding: 7,
          resource: { buffer: chunkIdBuffer! },
        },
      ],
    });

    this.splatBuffer = splatBuffer;
    this.splatCount = splatBuffer.getRenderCount();
  }

  getCameraBindGroupLayout(): GPUBindGroupLayout {
    return this.cameraBindGroupLayout;
  }

  setGpuChunkCullPass(pass: GpuChunkCullPass): void {
    this.gpuChunkCullPass = pass;
  }

  setGpuDepthBinPass(pass: GpuDepthBinPass): void {
    this.gpuDepthBinPass = pass;
  }

  setGpuTilePressurePass(pass: GpuTilePressurePass): void {
    this.gpuTilePressurePass = pass;
  }

  setComputeSortPass(pass: ComputeSortPass): void {
    this.computeSortPass = pass;
  }

  private vizMode = 0;
  private splatScale = 0.72;
  private maxSplatVariance = 0.0035;
  private quality = 0;

  setQualityLevel(quality: number): void {
    this.quality = Math.max(0, Math.min(1, quality));
    this.splatScale = 0.72 + this.quality * 0.95;
    this.maxSplatVariance = 0.0035 + this.quality * 0.032;
    this.flushRenderSettings();
  }

  setVisualizationMode(mode: number): void {
    this.vizMode = mode;
    this.flushRenderSettings();
  }

  private flushRenderSettings(): void {
    this.device.queue.writeBuffer(
      this.renderSettingsBuffer,
      0,
      new Float32Array([this.splatScale, this.maxSplatVariance, this.quality, this.vizMode]),
    );
  }

  renderFrame(
    encoder: GPUCommandEncoder,
    textureView: GPUTextureView,
    cameraUniforms: CameraUniforms,
    viewportWidth: number,
    viewportHeight: number,
  ): void {
    const cameraBindGroup = cameraUniforms.getBindGroup();

    if (this.computeSortPass) {
      const positionBuffer = this.splatBuffer?.getPositionBuffer();
      const visibleSplatIndicesBuffer = this.splatBuffer?.getVisibleSplatIndicesBuffer();
      const indirectArgsBuffer = this.splatBuffer?.getIndirectArgsBuffer();

      if (positionBuffer && visibleSplatIndicesBuffer && indirectArgsBuffer) {
        this.computeSortPass.encode(
          encoder,
          cameraUniforms,
          positionBuffer,
          visibleSplatIndicesBuffer,
          indirectArgsBuffer,
        );
      }
    }

    this.gpuChunkCullPass?.encode(encoder, cameraUniforms, viewportHeight);
    this.gpuDepthBinPass?.setViewportHeight(viewportHeight);
    this.gpuDepthBinPass?.encode(encoder, cameraUniforms);
    this.gpuTilePressurePass?.resize(viewportWidth, viewportHeight);
    this.gpuTilePressurePass?.encode(encoder, cameraUniforms, this.splatBuffer?.getRenderCount() ?? 0);

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
      const indirectArgsBuffer = this.splatBuffer?.getIndirectArgsBuffer();

      if (indirectArgsBuffer) {
        renderPass.drawIndirect(indirectArgsBuffer, 0);
      } else {
        renderPass.draw(6, this.splatCount);
      }
    }

    renderPass.end();
  }

  dispose(): void {
    // GPURenderPipeline objects are owned by the device and do not need manual disposal.
    this.renderSettingsBuffer.destroy();
  }
}
