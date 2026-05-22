import depthBinShader from "../shaders/depth-bin.wgsl?raw";
import type { CameraUniforms } from "../camera/camera-uniforms";
import type { RenderQualityMode } from "../world/types";

const BIN_COUNT = 512;
const WORKGROUP_SIZE = 128;

export interface GpuDepthBinPassOptions {
  cameraBindGroupLayout: GPUBindGroupLayout;
  positionBuffer: GPUBuffer;
  covarianceBuffer: GPUBuffer;
  opacityBuffer: GPUBuffer;
  splatCount: number;
  qualityMode: RenderQualityMode;
  viewportHeight: number;
}

export interface GpuDepthBinBuffers {
  visibleSplatIndicesBuffer: GPUBuffer;
  indirectArgsBuffer: GPUBuffer;
}

export class GpuDepthBinPass {
  private readonly device: GPUDevice;
  private readonly splatCount: number;
  private readonly qualityMode: RenderQualityMode;
  private viewportHeight: number;
  private readonly bindGroupLayout: GPUBindGroupLayout;
  private readonly clearPipeline: GPUComputePipeline;
  private readonly countPipeline: GPUComputePipeline;
  private readonly prefixPipeline: GPUComputePipeline;
  private readonly fillPipeline: GPUComputePipeline;
  private readonly paramsBuffer: GPUBuffer;
  private readonly binCountersBuffer: GPUBuffer;
  private readonly binOffsetsBuffer: GPUBuffer;
  private readonly binCursorsBuffer: GPUBuffer;
  private readonly visibleSplatIndicesBuffer: GPUBuffer;
  private readonly indirectArgsBuffer: GPUBuffer;
  private readonly bindGroup: GPUBindGroup;

  constructor(device: GPUDevice, options: GpuDepthBinPassOptions) {
    this.device = device;
    this.splatCount = options.splatCount;
    this.qualityMode = options.qualityMode;
    this.viewportHeight = options.viewportHeight;

    const module = device.createShaderModule({
      label: "GpuDepthBinShader",
      code: depthBinShader,
    });
    this.bindGroupLayout = device.createBindGroupLayout({
      label: "GpuDepthBinBindGroupLayout",
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      ],
    });
    const layout = device.createPipelineLayout({
      label: "GpuDepthBinPipelineLayout",
      bindGroupLayouts: [options.cameraBindGroupLayout, this.bindGroupLayout],
    });
    this.clearPipeline = this.createPipeline(module, layout, "clearBins");
    this.countPipeline = this.createPipeline(module, layout, "countBins");
    this.prefixPipeline = this.createPipeline(module, layout, "prefixBins");
    this.fillPipeline = this.createPipeline(module, layout, "fillBins");

    this.paramsBuffer = device.createBuffer({
      label: "GpuDepthBinParams",
      size: 12 * 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.binCountersBuffer = this.createStorageBuffer("DepthBinCounters", BIN_COUNT);
    this.binOffsetsBuffer = this.createStorageBuffer("DepthBinOffsets", BIN_COUNT);
    this.binCursorsBuffer = this.createStorageBuffer("DepthBinCursors", BIN_COUNT);
    this.visibleSplatIndicesBuffer = this.createStorageBuffer("BinnedVisibleSplatIndices", Math.max(1, this.splatCount));
    this.indirectArgsBuffer = device.createBuffer({
      label: "GpuDepthBinnedIndirectArgs",
      size: 4 * Uint32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    this.bindGroup = device.createBindGroup({
      label: "GpuDepthBinBindGroup",
      layout: this.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: options.positionBuffer } },
        { binding: 1, resource: { buffer: options.opacityBuffer } },
        { binding: 2, resource: { buffer: options.covarianceBuffer } },
        { binding: 3, resource: { buffer: this.binCountersBuffer } },
        { binding: 4, resource: { buffer: this.binOffsetsBuffer } },
        { binding: 5, resource: { buffer: this.binCursorsBuffer } },
        { binding: 6, resource: { buffer: this.visibleSplatIndicesBuffer } },
        { binding: 7, resource: { buffer: this.indirectArgsBuffer } },
        { binding: 8, resource: { buffer: this.paramsBuffer } },
      ],
    });
  }

  public getBuffers(): GpuDepthBinBuffers {
    return {
      visibleSplatIndicesBuffer: this.visibleSplatIndicesBuffer,
      indirectArgsBuffer: this.indirectArgsBuffer,
    };
  }

  public encode(encoder: GPUCommandEncoder, cameraUniforms: CameraUniforms): void {
    const cameraBindGroup = cameraUniforms.getBindGroup();

    if (!cameraBindGroup) {
      return;
    }

    this.device.queue.writeBuffer(
      this.paramsBuffer,
      0,
      packDepthBinParams(this.splatCount, this.qualityMode, this.viewportHeight),
    );

    const pass = encoder.beginComputePass({ label: "GpuDepthBinPass" });
    pass.setBindGroup(0, cameraBindGroup);
    pass.setBindGroup(1, this.bindGroup);

    pass.setPipeline(this.clearPipeline);
    pass.dispatchWorkgroups(Math.ceil(BIN_COUNT / WORKGROUP_SIZE));

    const splatWorkgroups = Math.ceil(this.splatCount / WORKGROUP_SIZE);
    pass.setPipeline(this.countPipeline);
    pass.dispatchWorkgroups(splatWorkgroups);

    pass.setPipeline(this.prefixPipeline);
    pass.dispatchWorkgroups(1);

    pass.setPipeline(this.fillPipeline);
    pass.dispatchWorkgroups(splatWorkgroups);
    pass.end();
  }

  public setViewportHeight(height: number): void {
    this.viewportHeight = Math.max(1, height);
  }

  public dispose(): void {
    this.paramsBuffer.destroy();
    this.binCountersBuffer.destroy();
    this.binOffsetsBuffer.destroy();
    this.binCursorsBuffer.destroy();
    this.visibleSplatIndicesBuffer.destroy();
    this.indirectArgsBuffer.destroy();
  }

  private createPipeline(
    module: GPUShaderModule,
    layout: GPUPipelineLayout,
    entryPoint: string,
  ): GPUComputePipeline {
    return this.device.createComputePipeline({
      label: `GpuDepthBin.${entryPoint}`,
      layout,
      compute: { module, entryPoint },
    });
  }

  private createStorageBuffer(label: string, elementCount: number): GPUBuffer {
    return this.device.createBuffer({
      label,
      size: elementCount * Uint32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
  }
}

function packDepthBinParams(
  splatCount: number,
  qualityMode: RenderQualityMode,
  viewportHeight: number,
): ArrayBuffer {
  const buffer = new ArrayBuffer(12 * 4);
  const view = new DataView(buffer);
  const mode = qualityMode === "performance" ? 3 : qualityMode === "gpu-balanced" ? 2 : 1;

  view.setUint32(0, splatCount, true);
  view.setUint32(4, mode, true);
  view.setFloat32(8, 0.05, true);
  view.setFloat32(12, 10000, true);
  view.setFloat32(16, qualityMode === "performance" ? 0.01 : 0.002, true);
  view.setFloat32(20, 80, true);
  view.setFloat32(24, 180, true);
  view.setFloat32(28, 360, true);
  view.setFloat32(32, viewportHeight, true);
  view.setFloat32(36, qualityMode === "performance" ? 0.45 : 0.2, true);
  view.setFloat32(40, qualityMode === "performance" ? 0.035 : 0.01, true);
  view.setUint32(44, 0, true);

  return buffer;
}
