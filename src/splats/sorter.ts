
import sortShader from "../shaders/sort.wgsl?raw";
import type { CameraUniforms } from "../camera/camera-uniforms";

const WORKGROUP_SIZE = 256;
const HISTOGRAM_BINS = 256;

export class GpuSorter {
  private readonly device: GPUDevice;
  private readonly module: GPUShaderModule;
  private readonly bindGroupLayout: GPUBindGroupLayout;
  private readonly buildKeysPipeline: GPUComputePipeline;
  private readonly clearHistogramPipeline: GPUComputePipeline;
  private readonly histogramPipeline: GPUComputePipeline;
  private readonly prefixPipeline: GPUComputePipeline;
  private readonly scatterPipeline: GPUComputePipeline;
  private readonly updateIndirectArgsPipeline: GPUComputePipeline;

  private keysBuffer: GPUBuffer | null = null;
  private valuesBuffer: GPUBuffer | null = null;
  private tempKeysBuffer: GPUBuffer | null = null;
  private tempValuesBuffer: GPUBuffer | null = null;
  private histogramBuffer: GPUBuffer | null = null;
  private configBuffer: GPUBuffer | null = null;
  private outputIndirectArgsBuffer: GPUBuffer | null = null;

  private bindGroup: GPUBindGroup | null = null;
  private lastPositionBuffer: GPUBuffer | null = null;
  private splatCount = 0;

  constructor(device: GPUDevice, cameraBindGroupLayout: GPUBindGroupLayout) {
    this.device = device;
    this.module = device.createShaderModule({
      label: "SortShader",
      code: sortShader,
    });

    this.bindGroupLayout = device.createBindGroupLayout({
      label: "SortBindGroupLayout",
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

    const layout = device.createPipelineLayout({
      label: "SortPipelineLayout",
      bindGroupLayouts: [cameraBindGroupLayout, this.bindGroupLayout],
    });

    this.buildKeysPipeline = this.createPipeline("csBuildKeys", layout);
    this.clearHistogramPipeline = this.createPipeline("csClearHistogram", layout);
    this.histogramPipeline = this.createPipeline("csHistogram", layout);
    this.prefixPipeline = this.createPipeline("csPrefix", layout);
    this.scatterPipeline = this.createPipeline("csScatter", layout);
    this.updateIndirectArgsPipeline = this.createPipeline("csUpdateIndirectArgs", layout);
  }

  public ensureBuffers(splatCount: number): void {
    if (this.splatCount === splatCount) { return; }
    this.splatCount = splatCount;
    this.destroyInternalBuffers();
    this.bindGroup = null;

    const size = Math.max(1, splatCount * Uint32Array.BYTES_PER_ELEMENT);
    this.keysBuffer = this.createStorageBuffer("SortKeys", size);
    this.valuesBuffer = this.createStorageBuffer("SortValues", size);
    this.tempKeysBuffer = this.createStorageBuffer("SortTempKeys", size);
    this.tempValuesBuffer = this.createStorageBuffer("SortTempValues", size);
    this.histogramBuffer = this.createStorageBuffer("SortHistogram", HISTOGRAM_BINS * Uint32Array.BYTES_PER_ELEMENT);
    this.configBuffer = this.device.createBuffer({
      label: "SortConfig",
      size: 4 * 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.outputIndirectArgsBuffer = this.device.createBuffer({
      label: "SortOutputIndirectArgs",
      size: 4 * Uint32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_SRC,
    });
  }

  public sort(
    encoder: GPUCommandEncoder,
    cameraUniforms: CameraUniforms,
    positionBuffer: GPUBuffer,
    visibleSplatIndicesBuffer: GPUBuffer,
    indirectArgsBuffer: GPUBuffer,
  ): void {
    if (this.splatCount === 0) { return; }

    if (!this.bindGroup || this.lastPositionBuffer !== positionBuffer) {
      this.lastPositionBuffer = positionBuffer;
      this.rebindGroup();
    }

    this.device.queue.writeBuffer(
      this.configBuffer!,
      0,
      new Uint32Array([this.splatCount, 0, 0, 0]),
    );

    const cameraBindGroup = cameraUniforms.getBindGroup();
    if (!cameraBindGroup || !this.bindGroup) { return; }

    const pass = encoder.beginComputePass({ label: "GpuSortPass" });
    pass.setBindGroup(0, cameraBindGroup);
    pass.setBindGroup(1, this.bindGroup);

    const workgroups = Math.ceil(this.splatCount / WORKGROUP_SIZE);

    pass.setPipeline(this.buildKeysPipeline);
    pass.dispatchWorkgroups(workgroups);

    for (let byteIndex = 0; byteIndex < 4; byteIndex++) {
      this.device.queue.writeBuffer(
        this.configBuffer!,
        0,
        new Uint32Array([this.splatCount, byteIndex, 0, 0]),
      );

      pass.setPipeline(this.clearHistogramPipeline);
      pass.dispatchWorkgroups(1);

      pass.setPipeline(this.histogramPipeline);
      pass.dispatchWorkgroups(workgroups);

      pass.setPipeline(this.prefixPipeline);
      pass.dispatchWorkgroups(1);

      pass.setPipeline(this.scatterPipeline);
      pass.dispatchWorkgroups(workgroups);

      this.swapBuffers();
      this.rebindGroup();
      pass.setBindGroup(1, this.bindGroup!);
    }

    pass.setPipeline(this.updateIndirectArgsPipeline);
    pass.dispatchWorkgroups(1);

    pass.end();

    encoder.copyBufferToBuffer(
      this.valuesBuffer!,
      0,
      visibleSplatIndicesBuffer,
      0,
      this.splatCount * Uint32Array.BYTES_PER_ELEMENT,
    );

    encoder.copyBufferToBuffer(
      this.outputIndirectArgsBuffer!,
      0,
      indirectArgsBuffer,
      0,
      4 * Uint32Array.BYTES_PER_ELEMENT,
    );
  }

  public setSplatCount(count: number): void {
    this.splatCount = count;
  }

  public dispose(): void {
    this.destroyInternalBuffers();
  }

  private rebindGroup(): void {
    this.bindGroup = this.device.createBindGroup({
      label: "SortBindGroup",
      layout: this.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.lastPositionBuffer! } },
        { binding: 1, resource: { buffer: this.keysBuffer! } },
        { binding: 2, resource: { buffer: this.valuesBuffer! } },
        { binding: 3, resource: { buffer: this.tempKeysBuffer! } },
        { binding: 4, resource: { buffer: this.tempValuesBuffer! } },
        { binding: 5, resource: { buffer: this.histogramBuffer! } },
        { binding: 6, resource: { buffer: this.outputIndirectArgsBuffer! } },
        { binding: 7, resource: { buffer: this.configBuffer! } },
      ],
    });
  }

  private swapBuffers(): void {
    const tempK = this.keysBuffer;
    const tempV = this.valuesBuffer;
    this.keysBuffer = this.tempKeysBuffer;
    this.valuesBuffer = this.tempValuesBuffer;
    this.tempKeysBuffer = tempK;
    this.tempValuesBuffer = tempV;
  }

  private createPipeline(entryPoint: string, layout: GPUPipelineLayout): GPUComputePipeline {
    return this.device.createComputePipeline({
      label: `Sort.${entryPoint}`,
      layout,
      compute: { module: this.module, entryPoint },
    });
  }

  private createStorageBuffer(label: string, size: number): GPUBuffer {
    return this.device.createBuffer({
      label,
      size,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
  }

  private destroyInternalBuffers(): void {
    this.keysBuffer?.destroy();
    this.valuesBuffer?.destroy();
    this.tempKeysBuffer?.destroy();
    this.tempValuesBuffer?.destroy();
    this.histogramBuffer?.destroy();
    this.configBuffer?.destroy();
    this.outputIndirectArgsBuffer?.destroy();
    this.keysBuffer = null;
    this.valuesBuffer = null;
    this.tempKeysBuffer = null;
    this.tempValuesBuffer = null;
    this.histogramBuffer = null;
    this.configBuffer = null;
    this.outputIndirectArgsBuffer = null;
    this.bindGroup = null;
    this.lastPositionBuffer = null;
  }
}
