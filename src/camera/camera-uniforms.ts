export interface CameraMatrices {
  view: Float32Array;
  projection: Float32Array;
  viewProjection: Float32Array;
}

export class CameraUniforms {
  private buffer: GPUBuffer | null = null;
  private device: GPUDevice;
  private matrices: CameraMatrices;
  private packedMatrices = new Float32Array(48);
  private bindGroup: GPUBindGroup | null = null;
  private bindGroupLayout: GPUBindGroupLayout | null = null;

  constructor(device: GPUDevice) {
    this.device = device;
    this.matrices = {
      view: new Float32Array(16),
      projection: new Float32Array(16),
      viewProjection: new Float32Array(16),
    };
  }

  public createBuffers(): void {
    this.buffer = this.device.createBuffer({
      size: 48 * 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.bindGroupLayout = this.device.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
          buffer: { type: "uniform" },
        },
      ],
    });

    this.bindGroup = this.device.createBindGroup({
      layout: this.bindGroupLayout,
      entries: [
        {
          binding: 0,
          resource: { buffer: this.buffer },
        },
      ],
    });
  }

  public updateView(view: Float32Array): void {
    this.matrices.view.set(view);
  }

  public updateProjection(projection: Float32Array): void {
    this.matrices.projection.set(projection);
  }

  public updateViewProjection(viewProjection: Float32Array): void {
    this.matrices.viewProjection.set(viewProjection);
  }

  public upload(): void {
    if (!this.buffer) return;

    this.packedMatrices.set(this.matrices.view, 0);
    this.packedMatrices.set(this.matrices.projection, 16);
    this.packedMatrices.set(this.matrices.viewProjection, 32);

    this.device.queue.writeBuffer(this.buffer, 0, this.packedMatrices);
  }

  public getBindGroup(): GPUBindGroup | null {
    return this.bindGroup;
  }

  public getBindGroupLayout(): GPUBindGroupLayout | null {
    return this.bindGroupLayout;
  }

  public dispose(): void {
    this.buffer?.destroy();
    this.buffer = null;
    this.bindGroup = null;
    this.bindGroupLayout = null;
  }
}
