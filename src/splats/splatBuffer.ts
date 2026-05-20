export interface SplatData {
  positions: Float32Array;
  colors: Float32Array;
  covariances: Float32Array;
  shCoefficients: Float32Array;
  count: number;
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

export class SplatBuffer {
  private positions: Float32Array | null = null;
  private colors: Float32Array | null = null;
  private covariances: Float32Array | null = null;
  private shCoefficients: Float32Array | null = null;
  private count: number = 0;

  private positionBuffer: GPUBuffer | null = null;
  private colorBuffer: GPUBuffer | null = null;
  private covarianceBuffer: GPUBuffer | null = null;
  private shBuffer: GPUBuffer | null = null;

  public setData(data: SplatData): void {
    this.positions = data.positions;
    this.colors = data.colors;
    this.covariances = data.covariances;
    this.shCoefficients = data.shCoefficients;
    this.count = data.count;
  }

  public createBuffers(device: GPUDevice): void {
    if (this.positions) {
      this.positionBuffer = createStorageBuffer(device, this.positions, "SplatPositions");
    }

    if (this.colors) {
      this.colorBuffer = createStorageBuffer(device, this.colors, "SplatColors");
    }

    if (this.covariances) {
      this.covarianceBuffer = createStorageBuffer(device, this.covariances, "SplatCovariances");
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

  public getCovarianceBuffer(): GPUBuffer | null {
    return this.covarianceBuffer;
  }

  public getShBuffer(): GPUBuffer | null {
    return this.shBuffer;
  }

  public dispose(): void {
    this.positionBuffer?.destroy();
    this.colorBuffer?.destroy();
    this.covarianceBuffer?.destroy();
    this.shBuffer?.destroy();
    this.positionBuffer = null;
    this.colorBuffer = null;
    this.covarianceBuffer = null;
    this.shBuffer = null;
    this.positions = null;
    this.colors = null;
    this.covariances = null;
    this.shCoefficients = null;
    this.count = 0;
  }
}
