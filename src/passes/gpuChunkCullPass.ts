import chunkCullShader from "../shaders/chunk-cull.wgsl?raw";
import type { CameraUniforms } from "../camera/camera-uniforms";
import type { GpuCullingBuffers } from "../splats/splatBuffer";

export interface GpuChunkCullPassOptions {
  chunkMetadataBuffer: GPUBuffer;
  cameraBindGroupLayout: GPUBindGroupLayout;
  chunkCount: number;
  splatCount: number;
}

export class GpuChunkCullPass {
  private readonly device: GPUDevice;
  private readonly chunkCount: number;
  private readonly splatCount: number;
  private readonly bindGroupLayout: GPUBindGroupLayout;
  private readonly pipeline: GPUComputePipeline;
  private readonly paramsBuffer: GPUBuffer;
  private readonly visibleChunkFlagsBuffer: GPUBuffer;
  private readonly visibleChunkIndicesBuffer: GPUBuffer;
  private readonly visibleChunkCounterBuffer: GPUBuffer;
  private readonly visibleSplatIndicesBuffer: GPUBuffer;
  private readonly visibleSplatCounterBuffer: GPUBuffer;
  private readonly indirectArgsBuffer: GPUBuffer;
  private bindGroup: GPUBindGroup | null = null;

  constructor(device: GPUDevice, options: GpuChunkCullPassOptions) {
    this.device = device;
    this.chunkCount = options.chunkCount;
    this.splatCount = options.splatCount;

    const shaderModule = device.createShaderModule({
      label: "GpuChunkCullShader",
      code: chunkCullShader,
    });

    this.bindGroupLayout = device.createBindGroupLayout({
      label: "GpuChunkCullBindGroupLayout",
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      ],
    });

    this.pipeline = device.createComputePipeline({
      label: "GpuChunkCullPipeline",
      layout: device.createPipelineLayout({
        label: "GpuChunkCullPipelineLayout",
        bindGroupLayouts: [
          options.cameraBindGroupLayout,
          this.bindGroupLayout,
        ],
      }),
      compute: {
        module: shaderModule,
        entryPoint: "main",
      },
    });

    this.paramsBuffer = device.createBuffer({
      label: "GpuChunkCullParams",
      size: 8 * 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.visibleChunkFlagsBuffer = this.createStorageBuffer("VisibleChunkFlags", Math.max(1, this.chunkCount));
    this.visibleChunkIndicesBuffer = this.createStorageBuffer("VisibleChunkIndices", Math.max(1, this.chunkCount));
    this.visibleChunkCounterBuffer = this.createStorageBuffer("VisibleChunkCounter", 1);
    this.visibleSplatIndicesBuffer = this.createStorageBuffer("GpuVisibleSplatIndices", Math.max(1, this.splatCount));
    this.visibleSplatCounterBuffer = this.createStorageBuffer("VisibleSplatCounter", 1);
    this.indirectArgsBuffer = device.createBuffer({
      label: "GpuSplatIndirectDrawArgs",
      size: 4 * Uint32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });

    this.bindGroup = device.createBindGroup({
      label: "GpuChunkCullBindGroup",
      layout: this.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: options.chunkMetadataBuffer } },
        { binding: 1, resource: { buffer: this.visibleChunkFlagsBuffer } },
        { binding: 2, resource: { buffer: this.visibleChunkIndicesBuffer } },
        { binding: 3, resource: { buffer: this.visibleChunkCounterBuffer } },
        { binding: 4, resource: { buffer: this.visibleSplatIndicesBuffer } },
        { binding: 5, resource: { buffer: this.visibleSplatCounterBuffer } },
        { binding: 6, resource: { buffer: this.indirectArgsBuffer } },
        { binding: 7, resource: { buffer: this.paramsBuffer } },
      ],
    });
  }

  public getBuffers(): GpuCullingBuffers {
    return {
      visibleChunkFlagsBuffer: this.visibleChunkFlagsBuffer,
      visibleChunkIndicesBuffer: this.visibleChunkIndicesBuffer,
      visibleChunkCounterBuffer: this.visibleChunkCounterBuffer,
      visibleSplatIndicesBuffer: this.visibleSplatIndicesBuffer,
      visibleSplatCounterBuffer: this.visibleSplatCounterBuffer,
      indirectArgsBuffer: this.indirectArgsBuffer,
    };
  }

  public encode(
    encoder: GPUCommandEncoder,
    cameraUniforms: CameraUniforms,
    viewportHeight: number,
  ): void {
    const cameraBindGroup = cameraUniforms.getBindGroup();

    if (!cameraBindGroup || !this.bindGroup) {
      return;
    }

    this.device.queue.writeBuffer(this.visibleChunkCounterBuffer, 0, new Uint32Array([0]));
    this.device.queue.writeBuffer(this.visibleSplatCounterBuffer, 0, new Uint32Array([0]));
    this.device.queue.writeBuffer(this.indirectArgsBuffer, 0, new Uint32Array([6, 0, 0, 0]));
    this.device.queue.writeBuffer(
      this.paramsBuffer,
      0,
      packCullParams(this.chunkCount, this.splatCount, viewportHeight),
    );

    const pass = encoder.beginComputePass({ label: "GpuChunkCullPass" });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, cameraBindGroup);
    pass.setBindGroup(1, this.bindGroup);
    pass.dispatchWorkgroups(Math.ceil(this.chunkCount / 64));
    pass.end();
  }

  public dispose(): void {
    this.paramsBuffer.destroy();
    this.visibleChunkFlagsBuffer.destroy();
    this.visibleChunkIndicesBuffer.destroy();
    this.visibleChunkCounterBuffer.destroy();
    this.visibleSplatIndicesBuffer.destroy();
    this.visibleSplatCounterBuffer.destroy();
    this.indirectArgsBuffer.destroy();
    this.bindGroup = null;
  }

  private createStorageBuffer(label: string, elementCount: number): GPUBuffer {
    return this.device.createBuffer({
      label,
      size: elementCount * Uint32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
  }

}

function packCullParams(
  chunkCount: number,
  splatCount: number,
  viewportHeight: number,
): ArrayBuffer {
  const buffer = new ArrayBuffer(8 * 4);
  const view = new DataView(buffer);

  view.setUint32(0, chunkCount, true);
  view.setUint32(4, splatCount, true);
  view.setFloat32(8, Math.max(1, viewportHeight), true);
  view.setUint32(12, 1, true);
  view.setFloat32(16, 2, true);
  view.setFloat32(20, 0, true);
  view.setUint32(24, 0, true);
  view.setUint32(28, 0, true);

  return buffer;
}
