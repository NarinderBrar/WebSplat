import type { CameraUniforms } from "../camera/camera-uniforms";
import tilePressureShader from "../shaders/tile-pressure.wgsl?raw";

const WORKGROUP_SIZE = 128;
const READBACK_STRIDE = 4 * Uint32Array.BYTES_PER_ELEMENT;

export interface GpuTilePressurePassOptions {
  cameraBindGroupLayout: GPUBindGroupLayout;
  positionBuffer: GPUBuffer;
  visibleSplatIndicesBuffer: GPUBuffer;
  maxVisibleSplatCount: number;
  tileSize: number;
  overloadThreshold: number;
}

export interface GpuTilePressureTelemetry {
  testedSplats: number;
  maxTileSplats: number;
  overloadedTiles: number;
}

interface ReadbackSlot {
  buffer: GPUBuffer;
  copyPending: boolean;
  mapPending: boolean;
}

export class GpuTilePressurePass {
  private readonly device: GPUDevice;
  private readonly maxVisibleSplatCount: number;
  private readonly tileSize: number;
  private readonly overloadThreshold: number;
  private positionBuffer: GPUBuffer;
  private visibleSplatIndicesBuffer: GPUBuffer;
  private readonly bindGroupLayout: GPUBindGroupLayout;
  private readonly clearPipeline: GPUComputePipeline;
  private readonly countPipeline: GPUComputePipeline;
  private readonly paramsBuffer: GPUBuffer;
  private readonly summaryBuffer: GPUBuffer;
  private tileCountsBuffer: GPUBuffer;
  private readonly readbackSlots: ReadbackSlot[];
  private bindGroup: GPUBindGroup;
  private tilesX = 1;
  private tilesY = 1;
  private tileCount = 1;
  private tileCapacity = 1;
  private viewportWidth = 1;
  private viewportHeight = 1;
  private frameIndex = 0;
  private latestTelemetry: GpuTilePressureTelemetry = {
    testedSplats: 0,
    maxTileSplats: 0,
    overloadedTiles: 0,
  };

  constructor(device: GPUDevice, options: GpuTilePressurePassOptions) {
    this.device = device;
    this.maxVisibleSplatCount = options.maxVisibleSplatCount;
    this.tileSize = options.tileSize;
    this.overloadThreshold = options.overloadThreshold;
    this.positionBuffer = options.positionBuffer;
    this.visibleSplatIndicesBuffer = options.visibleSplatIndicesBuffer;

    const module = device.createShaderModule({
      label: "GpuTilePressureShader",
      code: tilePressureShader,
    });

    this.bindGroupLayout = device.createBindGroupLayout({
      label: "GpuTilePressureBindGroupLayout",
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      ],
    });

    const pipelineLayout = device.createPipelineLayout({
      label: "GpuTilePressurePipelineLayout",
      bindGroupLayouts: [options.cameraBindGroupLayout, this.bindGroupLayout],
    });

    this.clearPipeline = device.createComputePipeline({
      label: "GpuTilePressure.clearTiles",
      layout: pipelineLayout,
      compute: { module, entryPoint: "clearTiles" },
    });
    this.countPipeline = device.createComputePipeline({
      label: "GpuTilePressure.countTiles",
      layout: pipelineLayout,
      compute: { module, entryPoint: "countTiles" },
    });
    this.paramsBuffer = device.createBuffer({
      label: "GpuTilePressureParams",
      size: 8 * 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.summaryBuffer = device.createBuffer({
      label: "GpuTilePressureSummary",
      size: READBACK_STRIDE,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    this.tileCountsBuffer = this.createTileCountsBuffer(4096);
    this.tileCapacity = 4096;
    this.readbackSlots = [
      { buffer: this.createReadbackBuffer("GpuTilePressureReadbackA"), copyPending: false, mapPending: false },
      { buffer: this.createReadbackBuffer("GpuTilePressureReadbackB"), copyPending: false, mapPending: false },
    ];
    this.bindGroup = this.createBindGroup(options.positionBuffer, options.visibleSplatIndicesBuffer);
  }

  public resize(viewportWidth: number, viewportHeight: number): void {
    const width = Math.max(1, Math.floor(viewportWidth));
    const height = Math.max(1, Math.floor(viewportHeight));
    const nextTilesX = Math.max(1, Math.ceil(width / this.tileSize));
    const nextTilesY = Math.max(1, Math.ceil(height / this.tileSize));
    const nextTileCount = nextTilesX * nextTilesY;

    this.viewportWidth = width;
    this.viewportHeight = height;

    this.tilesX = nextTilesX;
    this.tilesY = nextTilesY;
    this.tileCount = nextTileCount;

    if (nextTileCount <= this.tileCapacity) {
      return;
    }

    this.tileCountsBuffer.destroy();
    this.tileCapacity = Math.max(nextTileCount, this.tileCapacity * 2);
    this.tileCountsBuffer = this.createTileCountsBuffer(this.tileCapacity);
    this.bindGroup = this.createBindGroup(this.positionBuffer, this.visibleSplatIndicesBuffer);
  }

  public rebind(positionBuffer: GPUBuffer, visibleSplatIndicesBuffer: GPUBuffer): void {
    this.positionBuffer = positionBuffer;
    this.visibleSplatIndicesBuffer = visibleSplatIndicesBuffer;
    this.bindGroup = this.createBindGroup(positionBuffer, visibleSplatIndicesBuffer);
  }

  public encode(
    encoder: GPUCommandEncoder,
    cameraUniforms: CameraUniforms,
    visibleCount: number,
  ): void {
    const cameraBindGroup = cameraUniforms.getBindGroup();

    if (!cameraBindGroup || visibleCount <= 0) {
      return;
    }

    this.device.queue.writeBuffer(
      this.paramsBuffer,
      0,
      packTilePressureParams(
        Math.min(this.maxVisibleSplatCount, visibleCount),
        this.tileCount,
        this.tilesX,
        this.tilesY,
        this.tileSize,
        this.viewportWidth,
        this.viewportHeight,
        this.overloadThreshold,
      ),
    );

    const pass = encoder.beginComputePass({ label: "GpuTilePressurePass" });
    pass.setBindGroup(0, cameraBindGroup);
    pass.setBindGroup(1, this.bindGroup);
    pass.setPipeline(this.clearPipeline);
    pass.dispatchWorkgroups(Math.ceil(Math.max(this.tileCount, 4) / WORKGROUP_SIZE));
    pass.setPipeline(this.countPipeline);
    pass.dispatchWorkgroups(Math.ceil(visibleCount / WORKGROUP_SIZE));
    pass.end();

    const readbackSlot = this.readbackSlots[this.frameIndex % this.readbackSlots.length];

    if (!readbackSlot.copyPending && !readbackSlot.mapPending) {
      encoder.copyBufferToBuffer(this.summaryBuffer, 0, readbackSlot.buffer, 0, READBACK_STRIDE);
      readbackSlot.copyPending = true;
    }

    this.frameIndex++;
  }

  public pollTelemetry(): GpuTilePressureTelemetry {
    for (const slot of this.readbackSlots) {
      if (!slot.copyPending || slot.mapPending) {
        continue;
      }

      slot.copyPending = false;
      slot.mapPending = true;
      void slot.buffer.mapAsync(GPUMapMode.READ, 0, READBACK_STRIDE)
        .then(() => {
          const values = new Uint32Array(slot.buffer.getMappedRange(0, READBACK_STRIDE).slice(0));
          this.latestTelemetry = {
            testedSplats: values[0] ?? 0,
            maxTileSplats: values[1] ?? 0,
            overloadedTiles: values[2] ?? 0,
          };
          slot.buffer.unmap();
          slot.mapPending = false;
        })
        .catch(() => {
          if (slot.buffer.mapState === "mapped") {
            slot.buffer.unmap();
          }
          slot.mapPending = false;
        });
    }

    return this.latestTelemetry;
  }

  public dispose(): void {
    this.paramsBuffer.destroy();
    this.summaryBuffer.destroy();
    this.tileCountsBuffer.destroy();

    for (const slot of this.readbackSlots) {
      if (slot.buffer.mapState === "mapped") {
        slot.buffer.unmap();
      }

      slot.buffer.destroy();
    }
  }

  private createBindGroup(positionBuffer: GPUBuffer, visibleSplatIndicesBuffer: GPUBuffer): GPUBindGroup {
    return this.device.createBindGroup({
      label: "GpuTilePressureBindGroup",
      layout: this.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: positionBuffer } },
        { binding: 1, resource: { buffer: visibleSplatIndicesBuffer } },
        { binding: 2, resource: { buffer: this.tileCountsBuffer } },
        { binding: 3, resource: { buffer: this.summaryBuffer } },
        { binding: 4, resource: { buffer: this.paramsBuffer } },
      ],
    });
  }

  private createTileCountsBuffer(tileCount: number): GPUBuffer {
    return this.device.createBuffer({
      label: "GpuTilePressureCounts",
      size: Math.max(1, tileCount) * Uint32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
  }

  private createReadbackBuffer(label: string): GPUBuffer {
    return this.device.createBuffer({
      label,
      size: READBACK_STRIDE,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
  }
}

function packTilePressureParams(
  visibleCount: number,
  tileCount: number,
  tilesX: number,
  tilesY: number,
  tileSize: number,
  viewportWidth: number,
  viewportHeight: number,
  overloadThreshold: number,
): ArrayBuffer {
  const buffer = new ArrayBuffer(8 * 4);
  const view = new DataView(buffer);

  view.setUint32(0, visibleCount, true);
  view.setUint32(4, tileCount, true);
  view.setUint32(8, tilesX, true);
  view.setUint32(12, tilesY, true);
  view.setFloat32(16, tileSize, true);
  view.setFloat32(20, viewportWidth, true);
  view.setFloat32(24, viewportHeight, true);
  view.setUint32(28, overloadThreshold, true);

  return buffer;
}
