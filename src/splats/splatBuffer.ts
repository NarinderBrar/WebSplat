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
  private depths: Float32Array | null = null;
  private count: number = 0;
  private renderCount: number = 0;

  private positionBuffer: GPUBuffer | null = null;
  private colorBuffer: GPUBuffer | null = null;
  private opacityBuffer: GPUBuffer | null = null;
  private covarianceBuffer: GPUBuffer | null = null;
  private orderBuffer: GPUBuffer | null = null;
  private shBuffer: GPUBuffer | null = null;
  private splatIdBuffer: GPUBuffer | null = null;
  private chunkIdBuffer: GPUBuffer | null = null;
  private localIndexBuffer: GPUBuffer | null = null;
  private chunkMetadataBuffer: GPUBuffer | null = null;

  public setData(data: SplatData): void {
    this.positions = data.positions;
    this.colors = data.colors;
    this.opacities = data.opacities;
    this.covariances = data.covariances;
    this.shCoefficients = data.shCoefficients;
    this.count = data.count;
    this.renderCount = data.count;
    this.order = new Uint32Array(this.count);
    this.depths = new Float32Array(this.count);

    for (let i = 0; i < this.count; i++) {
      this.order[i] = i;
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

  public sortByView(
    viewMatrix: Float32Array,
    device: GPUDevice,
    visibleRanges?: readonly SplatRange[],
  ): void {
    if (!this.positions || !this.order || !this.depths || !this.orderBuffer) {
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

    device.queue.writeBuffer(
      this.orderBuffer,
      0,
      this.order.buffer as ArrayBuffer,
      this.order.byteOffset,
      this.renderCount * Uint32Array.BYTES_PER_ELEMENT,
    );
  }

  public sortAllByView(viewMatrix: Float32Array, device: GPUDevice): void {
    if (!this.positions || !this.order || !this.depths || !this.orderBuffer) {
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
    device.queue.writeBuffer(
      this.orderBuffer,
      0,
      this.order.buffer as ArrayBuffer,
      this.order.byteOffset,
      this.renderCount * Uint32Array.BYTES_PER_ELEMENT,
    );
  }

  public dispose(): void {
    this.positionBuffer?.destroy();
    this.colorBuffer?.destroy();
    this.opacityBuffer?.destroy();
    this.covarianceBuffer?.destroy();
    this.orderBuffer?.destroy();
    this.shBuffer?.destroy();
    this.splatIdBuffer?.destroy();
    this.chunkIdBuffer?.destroy();
    this.localIndexBuffer?.destroy();
    this.chunkMetadataBuffer?.destroy();
    this.positionBuffer = null;
    this.colorBuffer = null;
    this.opacityBuffer = null;
    this.covarianceBuffer = null;
    this.orderBuffer = null;
    this.shBuffer = null;
    this.splatIdBuffer = null;
    this.chunkIdBuffer = null;
    this.localIndexBuffer = null;
    this.chunkMetadataBuffer = null;
    this.positions = null;
    this.colors = null;
    this.opacities = null;
    this.covariances = null;
    this.shCoefficients = null;
    this.order = null;
    this.depths = null;
    this.count = 0;
    this.renderCount = 0;
  }

  private getDepth(splatIndex: number): number {
    return this.depths?.[splatIndex] ?? 0;
  }
}
