export interface SplatData {
  positions: Float32Array;
  colors: Float32Array;
  opacities: Float32Array;
  covariances: Float32Array;
  shCoefficients: Float32Array;
  count: number;
}

const DEBUG_RENDER_SPLAT_BUDGET = 500_000;
const SORT_EPSILON = 0.0001;

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
  private lastSortViewDirection: [number, number, number] | null = null;
  private count: number = 0;

  private positionBuffer: GPUBuffer | null = null;
  private colorBuffer: GPUBuffer | null = null;
  private opacityBuffer: GPUBuffer | null = null;
  private covarianceBuffer: GPUBuffer | null = null;
  private orderBuffer: GPUBuffer | null = null;
  private shBuffer: GPUBuffer | null = null;

  public setData(data: SplatData): void {
    const renderData = data.count > DEBUG_RENDER_SPLAT_BUDGET
      ? createRenderSample(data, DEBUG_RENDER_SPLAT_BUDGET)
      : data;

    this.positions = renderData.positions;
    this.colors = renderData.colors;
    this.opacities = renderData.opacities;
    this.covariances = renderData.covariances;
    this.shCoefficients = renderData.shCoefficients;
    this.count = renderData.count;
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

  public getCount(): number {
    return this.count;
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

  public sortByView(viewMatrix: Float32Array, device: GPUDevice): void {
    if (!this.positions || !this.order || !this.depths || !this.orderBuffer) {
      return;
    }

    const viewDirection: [number, number, number] = [
      viewMatrix[2],
      viewMatrix[6],
      viewMatrix[10],
    ];

    if (!this.shouldResort(viewDirection)) {
      return;
    }

    for (let i = 0; i < this.count; i++) {
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
      this.order.byteLength,
    );
    this.lastSortViewDirection = viewDirection;
  }

  public dispose(): void {
    this.positionBuffer?.destroy();
    this.colorBuffer?.destroy();
    this.opacityBuffer?.destroy();
    this.covarianceBuffer?.destroy();
    this.orderBuffer?.destroy();
    this.shBuffer?.destroy();
    this.positionBuffer = null;
    this.colorBuffer = null;
    this.opacityBuffer = null;
    this.covarianceBuffer = null;
    this.orderBuffer = null;
    this.shBuffer = null;
    this.positions = null;
    this.colors = null;
    this.opacities = null;
    this.covariances = null;
    this.shCoefficients = null;
    this.order = null;
    this.depths = null;
    this.lastSortViewDirection = null;
    this.count = 0;
  }

  private shouldResort(viewDirection: [number, number, number]): boolean {
    if (!this.lastSortViewDirection) {
      return true;
    }

    const dot =
      viewDirection[0] * this.lastSortViewDirection[0] +
      viewDirection[1] * this.lastSortViewDirection[1] +
      viewDirection[2] * this.lastSortViewDirection[2];

    return 1 - Math.abs(dot) > SORT_EPSILON;
  }
}

function createRenderSample(data: SplatData, budget: number): SplatData {
  const count = Math.min(data.count, budget);
  const stride = data.count / count;
  const positions = new Float32Array(count * 3);
  const colors = new Float32Array(count * 3);
  const opacities = new Float32Array(count);
  const covariances = new Float32Array(count * 6);

  for (let i = 0; i < count; i++) {
    const sourceIndex = Math.min(data.count - 1, Math.floor(i * stride));
    const sourceBase3 = sourceIndex * 3;
    const targetBase3 = i * 3;
    const sourceBase6 = sourceIndex * 6;
    const targetBase6 = i * 6;

    positions[targetBase3] = data.positions[sourceBase3];
    positions[targetBase3 + 1] = data.positions[sourceBase3 + 1];
    positions[targetBase3 + 2] = data.positions[sourceBase3 + 2];

    colors[targetBase3] = data.colors[sourceBase3];
    colors[targetBase3 + 1] = data.colors[sourceBase3 + 1];
    colors[targetBase3 + 2] = data.colors[sourceBase3 + 2];
    opacities[i] = data.opacities[sourceIndex];

    covariances[targetBase6] = data.covariances[sourceBase6];
    covariances[targetBase6 + 1] = data.covariances[sourceBase6 + 1];
    covariances[targetBase6 + 2] = data.covariances[sourceBase6 + 2];
    covariances[targetBase6 + 3] = data.covariances[sourceBase6 + 3];
    covariances[targetBase6 + 4] = data.covariances[sourceBase6 + 4];
    covariances[targetBase6 + 5] = data.covariances[sourceBase6 + 5];
  }

  return {
    positions,
    colors,
    opacities,
    covariances,
    shCoefficients: new Float32Array(0),
    count,
  };
}
