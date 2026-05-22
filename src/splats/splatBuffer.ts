import type { ChunkRenderPlan, SplatChunk, Vector3Tuple } from "../world/types";

export interface SplatData {
  positions: Float32Array;
  colors: Float32Array;
  opacities: Float32Array;
  covariances: Float32Array;
  shCoefficients: Float32Array;
  count: number;
}

export interface StableSplatData extends SplatData {
  splatIds: Uint32Array;
  chunkIds: Uint32Array;
  localIndices: Uint32Array;
}

export interface SplatRange {
  splatStart: number;
  splatCount: number;
}

export interface GpuCullingBuffers {
  visibleChunkFlagsBuffer: GPUBuffer;
  visibleChunkIndicesBuffer: GPUBuffer;
  visibleChunkCounterBuffer: GPUBuffer;
  visibleSplatIndicesBuffer: GPUBuffer;
  visibleSplatCounterBuffer: GPUBuffer;
  indirectArgsBuffer: GPUBuffer;
}

export interface VisibleIndexBuildTelemetry {
  localOrderRefreshMs: number;
  visibleIndexBuildMs: number;
  refreshedChunkCount: number;
}

const LOCAL_SORT_DIRECTION_EPSILON = 0.015;

function createStorageBuffer(device: GPUDevice, data: Float32Array, label: string): GPUBuffer {
  const buffer = device.createBuffer({
    label,
    size: data.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });

  device.queue.writeBuffer(
    buffer,
    0,
    data.buffer as ArrayBuffer,
    data.byteOffset,
    data.byteLength,
  );
  return buffer;
}

function createIndexBuffer(device: GPUDevice, data: Uint32Array, label: string): GPUBuffer {
  const buffer = device.createBuffer({
    label,
    size: data.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });

  device.queue.writeBuffer(
    buffer,
    0,
    data.buffer as ArrayBuffer,
    data.byteOffset,
    data.byteLength,
  );
  return buffer;
}

export class SplatBuffer {
  private positions: Float32Array | null = null;
  private colors: Float32Array | null = null;
  private opacities: Float32Array | null = null;
  private covariances: Float32Array | null = null;
  private shCoefficients: Float32Array | null = null;
  private order: Uint32Array | null = null;
  private visibleSplatIndices: Uint32Array | null = null;
  private chunkLocalSortedIndices: Uint32Array | null = null;
  private chunksById = new Map<number, SplatChunk>();
  private depths: Float32Array | null = null;
  private count: number = 0;
  private renderCount: number = 0;

  private positionBuffer: GPUBuffer | null = null;
  private colorBuffer: GPUBuffer | null = null;
  private opacityBuffer: GPUBuffer | null = null;
  private covarianceBuffer: GPUBuffer | null = null;
  private orderBuffer: GPUBuffer | null = null;
  private visibleSplatIndicesBuffer: GPUBuffer | null = null;
  private shBuffer: GPUBuffer | null = null;
  private splatIdBuffer: GPUBuffer | null = null;
  private chunkIdBuffer: GPUBuffer | null = null;
  private localIndexBuffer: GPUBuffer | null = null;
  private chunkMetadataBuffer: GPUBuffer | null = null;
  private selectionMaskBuffer: GPUBuffer | null = null;
  private indirectArgsBuffer: GPUBuffer | null = null;
  private ownsVisibleSplatIndicesBuffer = true;
  private ownsIndirectArgsBuffer = true;

  public setData(data: SplatData): void {
    this.positions = data.positions;
    this.colors = data.colors;
    this.opacities = data.opacities;
    this.covariances = data.covariances;
    this.shCoefficients = data.shCoefficients;
    this.count = data.count;
    this.renderCount = data.count;
    this.order = new Uint32Array(this.count);
    this.visibleSplatIndices = new Uint32Array(this.count);
    this.depths = new Float32Array(this.count);

    for (let i = 0; i < this.count; i++) {
      this.order[i] = i;
      this.visibleSplatIndices[i] = i;
    }
  }

  public createBuffers(device: GPUDevice): void {
    if (this.positions) {
      this.positionBuffer = createStorageBuffer(device, this.positions, "SplatPositions");
    }

    if (this.colors) {
      this.colorBuffer = createStorageBuffer(device, this.colors, "SplatColors");
    }

    if (this.opacities) {
      this.opacityBuffer = createStorageBuffer(device, this.opacities, "SplatOpacities");
    }

    if (this.covariances) {
      this.covarianceBuffer = createStorageBuffer(device, this.covariances, "SplatCovariances");
    }

    if (this.order) {
      this.orderBuffer = createIndexBuffer(device, this.order, "SplatRenderOrder");
      this.visibleSplatIndicesBuffer = createIndexBuffer(device, this.order, "VisibleSplatIndices");
      this.ownsVisibleSplatIndicesBuffer = true;
    }

    if (this.shCoefficients && this.shCoefficients.byteLength > 0) {
      this.shBuffer = createStorageBuffer(device, this.shCoefficients, "SplatSHCoefficients");
    }
  }

  public createStableIdBuffers(device: GPUDevice, data: StableSplatData): void {
    this.splatIdBuffer = createIndexBuffer(device, data.splatIds, "StableSplatIds");
    this.chunkIdBuffer = createIndexBuffer(device, data.chunkIds, "StableSplatChunkIds");
    this.localIndexBuffer = createIndexBuffer(device, data.localIndices, "StableSplatLocalIndices");
  }

  public createChunkMetadataBuffer(device: GPUDevice, metadata: ArrayBuffer): void {
    this.chunkMetadataBuffer?.destroy();
    this.chunkMetadataBuffer = device.createBuffer({
      label: "SplatChunkMetadata",
      size: metadata.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(this.chunkMetadataBuffer, 0, metadata);
  }

  public createSelectionMaskBuffer(device: GPUDevice): void {
    this.selectionMaskBuffer?.destroy();
    this.selectionMaskBuffer = device.createBuffer({
      label: "SelectionMask",
      size: Math.max(4, this.count * Uint32Array.BYTES_PER_ELEMENT),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
  }

  public createIndirectArgsBuffer(device: GPUDevice): void {
    this.indirectArgsBuffer?.destroy();
    this.indirectArgsBuffer = device.createBuffer({
      label: "SplatIndirectDrawArgs",
      size: 4 * Uint32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    this.ownsIndirectArgsBuffer = true;
    this.updateIndirectArgs(device);
  }

  public createChunkOrderCache(chunks: readonly SplatChunk[]): void {
    this.chunksById.clear();
    this.chunkLocalSortedIndices = new Uint32Array(this.count);

    for (const chunk of chunks) {
      this.chunksById.set(chunk.id, chunk);
      chunk.localSortedIndicesOffset = chunk.splatStart;
      chunk.localSortedIndicesCount = chunk.splatCount;
      chunk.localOrderCacheVersion = 0;
      chunk.lastSortDirection = null;

      for (let i = 0; i < chunk.splatCount; i++) {
        this.chunkLocalSortedIndices[chunk.localSortedIndicesOffset + i] = chunk.splatStart + i;
      }
    }
  }

  public getCount(): number {
    return this.count;
  }

  public getRenderCount(): number {
    return this.renderCount;
  }

  public getPositionBuffer(): GPUBuffer | null {
    return this.positionBuffer;
  }

  public getColorBuffer(): GPUBuffer | null {
    return this.colorBuffer;
  }

  public getOpacityBuffer(): GPUBuffer | null {
    return this.opacityBuffer;
  }

  public getCovarianceBuffer(): GPUBuffer | null {
    return this.covarianceBuffer;
  }

  public getOrderBuffer(): GPUBuffer | null {
    return this.orderBuffer;
  }

  public getVisibleSplatIndicesBuffer(): GPUBuffer | null {
    return this.visibleSplatIndicesBuffer;
  }

  public getShBuffer(): GPUBuffer | null {
    return this.shBuffer;
  }

  public getSplatIdBuffer(): GPUBuffer | null {
    return this.splatIdBuffer;
  }

  public getChunkIdBuffer(): GPUBuffer | null {
    return this.chunkIdBuffer;
  }

  public getLocalIndexBuffer(): GPUBuffer | null {
    return this.localIndexBuffer;
  }

  public getChunkMetadataBuffer(): GPUBuffer | null {
    return this.chunkMetadataBuffer;
  }

  public getSelectionMaskBuffer(): GPUBuffer | null {
    return this.selectionMaskBuffer;
  }

  public getIndirectArgsBuffer(): GPUBuffer | null {
    return this.indirectArgsBuffer;
  }

  public adoptGpuVisibleBuffers(
    visibleSplatIndicesBuffer: GPUBuffer,
    indirectArgsBuffer: GPUBuffer,
  ): void {
    if (this.ownsVisibleSplatIndicesBuffer) {
      this.visibleSplatIndicesBuffer?.destroy();
    }

    if (this.ownsIndirectArgsBuffer) {
      this.indirectArgsBuffer?.destroy();
    }

    this.visibleSplatIndicesBuffer = visibleSplatIndicesBuffer;
    this.indirectArgsBuffer = indirectArgsBuffer;
    this.ownsVisibleSplatIndicesBuffer = false;
    this.ownsIndirectArgsBuffer = false;
    this.renderCount = this.count;
  }

  public sortByView(
    viewMatrix: Float32Array,
    device: GPUDevice,
    visibleRanges?: readonly SplatRange[],
  ): void {
    if (!this.positions || !this.order || !this.visibleSplatIndices || !this.depths || !this.orderBuffer) {
      return;
    }

    let writeIndex = 0;
    const ranges = visibleRanges ?? [{ splatStart: 0, splatCount: this.count }];

    for (const range of ranges) {
      const end = Math.min(this.count, range.splatStart + range.splatCount);

      for (let i = range.splatStart; i < end; i++) {
        this.order[writeIndex] = i;
        writeIndex++;
      }
    }

    this.renderCount = writeIndex;

    for (let i = 0; i < this.renderCount; i++) {
      const splatIndex = this.order[i];
      const base = splatIndex * 3;
      this.depths[splatIndex] =
        viewMatrix[2] * this.positions[base] +
        viewMatrix[6] * this.positions[base + 1] +
        viewMatrix[10] * this.positions[base + 2] +
        viewMatrix[14];
    }

    this.order.subarray(0, this.renderCount).sort(
      (a, b) => this.getDepth(b) - this.getDepth(a),
    );
    this.visibleSplatIndices.set(this.order.subarray(0, this.renderCount), 0);

    device.queue.writeBuffer(
      this.orderBuffer,
      0,
      this.order.buffer as ArrayBuffer,
      this.order.byteOffset,
      this.renderCount * Uint32Array.BYTES_PER_ELEMENT,
    );
    this.uploadVisibleSplatIndices(device);
    this.updateIndirectArgs(device);
  }

  public sortAllByView(viewMatrix: Float32Array, device: GPUDevice): void {
    if (!this.positions || !this.order || !this.visibleSplatIndices || !this.depths || !this.orderBuffer) {
      return;
    }

    this.renderCount = this.count;

    for (let i = 0; i < this.renderCount; i++) {
      const base = i * 3;
      this.depths[i] =
        viewMatrix[2] * this.positions[base] +
        viewMatrix[6] * this.positions[base + 1] +
        viewMatrix[10] * this.positions[base + 2] +
        viewMatrix[14];
      this.order[i] = i;
    }

    this.order.sort((a, b) => this.depths![b] - this.depths![a]);
    this.visibleSplatIndices.set(this.order.subarray(0, this.renderCount), 0);
    device.queue.writeBuffer(
      this.orderBuffer,
      0,
      this.order.buffer as ArrayBuffer,
      this.order.byteOffset,
      this.renderCount * Uint32Array.BYTES_PER_ELEMENT,
    );
    this.uploadVisibleSplatIndices(device);
    this.updateIndirectArgs(device);
  }

  public buildVisibleSplatIndicesFromChunkPlans(
    plans: readonly ChunkRenderPlan[],
    viewMatrix: Float32Array,
    device: GPUDevice,
  ): VisibleIndexBuildTelemetry {
    const start = performance.now();
    let refreshMs = 0;
    let refreshedChunkCount = 0;

    if (!this.positions || !this.depths || !this.visibleSplatIndices || !this.chunkLocalSortedIndices) {
      return {
        localOrderRefreshMs: 0,
        visibleIndexBuildMs: 0,
        refreshedChunkCount: 0,
      };
    }

    const viewDirection: Vector3Tuple = [
      viewMatrix[2],
      viewMatrix[6],
      viewMatrix[10],
    ];
    let writeIndex = 0;

    for (const plan of plans) {
      const chunk = this.chunksById.get(plan.chunkId);

      if (!chunk) {
        continue;
      }

      if (this.shouldRefreshChunkOrder(chunk, viewDirection)) {
        const refreshStart = performance.now();
        this.refreshChunkOrder(chunk, viewMatrix, viewDirection);
        refreshMs += performance.now() - refreshStart;
        refreshedChunkCount++;
      }

      const sourceStart = chunk.localSortedIndicesOffset;
      const sourceEnd = sourceStart + chunk.localSortedIndicesCount;

      for (let i = sourceStart; i < sourceEnd; i += plan.lodStep) {
        this.visibleSplatIndices[writeIndex] = this.chunkLocalSortedIndices[i];
        writeIndex++;
      }
    }

    this.renderCount = writeIndex;
    this.uploadVisibleSplatIndices(device);
    this.updateIndirectArgs(device);

    return {
      localOrderRefreshMs: refreshMs,
      visibleIndexBuildMs: performance.now() - start,
      refreshedChunkCount,
    };
  }

  public adoptGpuCullingBuffers(buffers: GpuCullingBuffers): void {
    if (this.ownsVisibleSplatIndicesBuffer) {
      this.visibleSplatIndicesBuffer?.destroy();
    }

    if (this.ownsIndirectArgsBuffer) {
      this.indirectArgsBuffer?.destroy();
    }

    this.visibleSplatIndicesBuffer = buffers.visibleSplatIndicesBuffer;
    this.indirectArgsBuffer = buffers.indirectArgsBuffer;
    this.ownsVisibleSplatIndicesBuffer = false;
    this.ownsIndirectArgsBuffer = false;
  }

  public setGpuRenderCountEstimate(renderCount: number): void {
    this.renderCount = Math.max(0, Math.min(this.count, renderCount));
  }

  public dispose(): void {
    this.positionBuffer?.destroy();
    this.colorBuffer?.destroy();
    this.opacityBuffer?.destroy();
    this.covarianceBuffer?.destroy();
    this.orderBuffer?.destroy();
    if (this.ownsVisibleSplatIndicesBuffer) {
      this.visibleSplatIndicesBuffer?.destroy();
    }
    this.shBuffer?.destroy();
    this.splatIdBuffer?.destroy();
    this.chunkIdBuffer?.destroy();
    this.localIndexBuffer?.destroy();
    this.chunkMetadataBuffer?.destroy();
    this.selectionMaskBuffer?.destroy();
    if (this.ownsIndirectArgsBuffer) {
      this.indirectArgsBuffer?.destroy();
    }
    this.positionBuffer = null;
    this.colorBuffer = null;
    this.opacityBuffer = null;
    this.covarianceBuffer = null;
    this.orderBuffer = null;
    this.visibleSplatIndicesBuffer = null;
    this.shBuffer = null;
    this.splatIdBuffer = null;
    this.chunkIdBuffer = null;
    this.localIndexBuffer = null;
    this.chunkMetadataBuffer = null;
    this.selectionMaskBuffer = null;
    this.indirectArgsBuffer = null;
    this.ownsVisibleSplatIndicesBuffer = true;
    this.ownsIndirectArgsBuffer = true;
    this.positions = null;
    this.colors = null;
    this.opacities = null;
    this.covariances = null;
    this.shCoefficients = null;
    this.order = null;
    this.visibleSplatIndices = null;
    this.chunkLocalSortedIndices = null;
    this.chunksById.clear();
    this.depths = null;
    this.count = 0;
    this.renderCount = 0;
  }

  private getDepth(splatIndex: number): number {
    return this.depths?.[splatIndex] ?? 0;
  }

  private shouldRefreshChunkOrder(chunk: SplatChunk, viewDirection: Vector3Tuple): boolean {
    if (!chunk.lastSortDirection) {
      return true;
    }

    const dot =
      viewDirection[0] * chunk.lastSortDirection[0] +
      viewDirection[1] * chunk.lastSortDirection[1] +
      viewDirection[2] * chunk.lastSortDirection[2];

    return 1 - Math.abs(dot) > LOCAL_SORT_DIRECTION_EPSILON;
  }

  private refreshChunkOrder(
    chunk: SplatChunk,
    viewMatrix: Float32Array,
    viewDirection: Vector3Tuple,
  ): void {
    if (!this.positions || !this.depths || !this.chunkLocalSortedIndices) {
      return;
    }

    const start = chunk.localSortedIndicesOffset;
    const end = start + chunk.localSortedIndicesCount;

    for (let i = start; i < end; i++) {
      const splatIndex = this.chunkLocalSortedIndices[i];
      const base = splatIndex * 3;
      this.depths[splatIndex] =
        viewMatrix[2] * this.positions[base] +
        viewMatrix[6] * this.positions[base + 1] +
        viewMatrix[10] * this.positions[base + 2] +
        viewMatrix[14];
    }

    this.chunkLocalSortedIndices.subarray(start, end).sort(
      (a, b) => this.getDepth(b) - this.getDepth(a),
    );
    chunk.lastSortDirection = [...viewDirection];
    chunk.localOrderCacheVersion++;
  }

  private uploadVisibleSplatIndices(device: GPUDevice): void {
    if (!this.visibleSplatIndicesBuffer || !this.visibleSplatIndices) {
      return;
    }

    device.queue.writeBuffer(
      this.visibleSplatIndicesBuffer,
      0,
      this.visibleSplatIndices.buffer as ArrayBuffer,
      this.visibleSplatIndices.byteOffset,
      this.renderCount * Uint32Array.BYTES_PER_ELEMENT,
    );
  }

  private updateIndirectArgs(device: GPUDevice): void {
    if (!this.indirectArgsBuffer) {
      return;
    }

    device.queue.writeBuffer(
      this.indirectArgsBuffer,
      0,
      new Uint32Array([6, this.renderCount, 0, 0]),
    );
  }
}
