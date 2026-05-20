import { GpuContext } from "./gpu-context";

export interface GaussianRendererOptions {
  canvas: HTMLCanvasElement;
}

export class GaussianRenderer {
  private readonly gpu: GpuContext;

  constructor(gpu: GpuContext) {
    this.gpu = gpu;
  }

  public render(): void {
    const encoder = this.gpu.beginFrame();

    const textureView = this.gpu.getCurrentTextureView();

    const renderPass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: textureView,
          clearValue: {
            r: 0.03,
            g: 0.05,
            b: 0.07,
            a: 1,
          },
          loadOp: "clear",
          storeOp: "store",
        },
      ],
    });

    renderPass.end();

    this.gpu.submit(encoder);
  }
}
