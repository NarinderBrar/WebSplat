import pickingShader from "../shaders/picking.wgsl?raw";
import type { CameraUniforms } from "../camera/camera-uniforms";
import type { SplatBuffer } from "../splats/splatBuffer";

export class IdPickingPass {
  private readonly device: GPUDevice;
  private readonly pipeline: GPURenderPipeline;
  private readonly bindGroupLayout: GPUBindGroupLayout;
  private bindGroup: GPUBindGroup | null = null;
  private idTexture: GPUTexture | null = null;
  private readbackBuffer: GPUBuffer | null = null;
  private width = 1;
  private height = 1;
  private copiedWidth = 1;
  private copiedHeight = 1;
  private copiedCenterX = 0;
  private copiedCenterY = 0;

  constructor(device: GPUDevice, cameraBindGroupLayout: GPUBindGroupLayout) {
    this.device = device;
    const shaderModule = device.createShaderModule({
      label: "IdPickingShader",
      code: pickingShader,
    });

    this.bindGroupLayout = device.createBindGroupLayout({
      label: "IdPickingBindGroupLayout",
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
        { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
        { binding: 3, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
      ],
    });

    this.pipeline = device.createRenderPipeline({
      label: "IdPickingPipeline",
      layout: device.createPipelineLayout({
        label: "IdPickingPipelineLayout",
        bindGroupLayouts: [cameraBindGroupLayout, this.bindGroupLayout],
      }),
      vertex: {
        module: shaderModule,
        entryPoint: "vsMain",
      },
      fragment: {
        module: shaderModule,
        entryPoint: "fsMain",
        targets: [{ format: "r32uint" }],
      },
      primitive: { topology: "triangle-list" },
    });
  }

  public setSplatBuffer(splatBuffer: SplatBuffer): void {
    const positionBuffer = splatBuffer.getPositionBuffer();
    const covarianceBuffer = splatBuffer.getCovarianceBuffer();
    const visibleSplatIndicesBuffer = splatBuffer.getVisibleSplatIndicesBuffer();
    const splatIdBuffer = splatBuffer.getSplatIdBuffer();

    if (!positionBuffer || !covarianceBuffer || !visibleSplatIndicesBuffer || !splatIdBuffer) {
      throw new Error("Picking buffers must be uploaded before binding.");
    }

    this.bindGroup = this.device.createBindGroup({
      label: "IdPickingBindGroup",
      layout: this.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: positionBuffer } },
        { binding: 1, resource: { buffer: covarianceBuffer } },
        { binding: 2, resource: { buffer: visibleSplatIndicesBuffer } },
        { binding: 3, resource: { buffer: splatIdBuffer } },
      ],
    });
  }

  public resize(width: number, height: number): void {
    const nextWidth = Math.max(1, Math.floor(width));
    const nextHeight = Math.max(1, Math.floor(height));

    if (nextWidth === this.width && nextHeight === this.height && this.idTexture) {
      return;
    }

    this.width = nextWidth;
    this.height = nextHeight;
    this.idTexture?.destroy();
    this.readbackBuffer?.destroy();
    this.idTexture = this.device.createTexture({
      label: "SplatIdPickingTexture",
      size: [this.width, this.height],
      format: "r32uint",
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
    });
    this.readbackBuffer = this.device.createBuffer({
      label: "SplatIdPickingReadback",
      size: 256 * 9,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
  }

  public encode(
    encoder: GPUCommandEncoder,
    cameraUniforms: CameraUniforms,
    indirectArgsBuffer: GPUBuffer,
  ): void {
    const cameraBindGroup = cameraUniforms.getBindGroup();

    if (!cameraBindGroup || !this.bindGroup || !this.idTexture) {
      return;
    }

    const pass = encoder.beginRenderPass({
      label: "IdPickingPass",
      colorAttachments: [
        {
          view: this.idTexture.createView(),
          clearValue: { r: 0, g: 0, b: 0, a: 0 },
          loadOp: "clear",
          storeOp: "store",
        },
      ],
    });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, cameraBindGroup);
    pass.setBindGroup(1, this.bindGroup);
    pass.drawIndirect(indirectArgsBuffer, 0);
    pass.end();
  }

  public copyPixelToReadback(encoder: GPUCommandEncoder, x: number, y: number): void {
    if (!this.idTexture || !this.readbackBuffer) {
      return;
    }

    const radius = 4;
    const centerX = Math.max(0, Math.min(this.width - 1, Math.floor(x)));
    const centerY = Math.max(0, Math.min(this.height - 1, Math.floor(y)));
    const originX = Math.max(0, centerX - radius);
    const originY = Math.max(0, centerY - radius);
    const endX = Math.min(this.width, centerX + radius + 1);
    const endY = Math.min(this.height, centerY + radius + 1);

    this.copiedWidth = Math.max(1, endX - originX);
    this.copiedHeight = Math.max(1, endY - originY);
    this.copiedCenterX = centerX - originX;
    this.copiedCenterY = centerY - originY;

    encoder.copyTextureToBuffer(
      {
        texture: this.idTexture,
        origin: {
          x: originX,
          y: originY,
        },
      },
      {
        buffer: this.readbackBuffer,
        bytesPerRow: 256,
      },
      [this.copiedWidth, this.copiedHeight],
    );
  }

  public async readCopiedSplatId(): Promise<number | null> {
    if (!this.readbackBuffer) {
      return null;
    }

    await this.readbackBuffer.mapAsync(GPUMapMode.READ, 0, 256 * this.copiedHeight);
    const data = new DataView(this.readbackBuffer.getMappedRange(0, 256 * this.copiedHeight));
    let encodedId = 0;
    let bestDistanceSq = Number.POSITIVE_INFINITY;

    for (let row = 0; row < this.copiedHeight; row++) {
      for (let column = 0; column < this.copiedWidth; column++) {
        const candidateId = data.getUint32(row * 256 + column * Uint32Array.BYTES_PER_ELEMENT, true);

        if (candidateId === 0) {
          continue;
        }

        const dx = column - this.copiedCenterX;
        const dy = row - this.copiedCenterY;
        const distanceSq = dx * dx + dy * dy;

        if (distanceSq < bestDistanceSq) {
          encodedId = candidateId;
          bestDistanceSq = distanceSq;
        }
      }
    }

    this.readbackBuffer.unmap();
    return encodedId === 0 ? null : encodedId - 1;
  }

  public dispose(): void {
    this.idTexture?.destroy();
    this.readbackBuffer?.destroy();
    this.bindGroup = null;
    this.idTexture = null;
    this.readbackBuffer = null;
  }
}
