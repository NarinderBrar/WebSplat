export class GpuContext {
  public readonly adapter: GPUAdapter;
  public readonly device: GPUDevice;
  public readonly canvas: HTMLCanvasElement;
  public readonly context: GPUCanvasContext;
  public readonly presentationFormat: GPUTextureFormat;
  private renderScale = 1;

  private constructor(
    adapter: GPUAdapter,
    device: GPUDevice,
    canvas: HTMLCanvasElement,
    context: GPUCanvasContext,
    presentationFormat: GPUTextureFormat,
  ) {
    this.adapter = adapter;
    this.device = device;
    this.canvas = canvas;
    this.context = context;
    this.presentationFormat = presentationFormat;
  }

  static async create(canvas: HTMLCanvasElement): Promise<GpuContext> {
    if (!navigator.gpu) {
      throw new Error("WebGPU is not available in this browser.");
    }

    const adapter = (await navigator.gpu.requestAdapter()) as unknown as GPUAdapter | null;

    if (!adapter) {
      throw new Error("WebGPU adapter could not be created.");
    }

    const device = await adapter.requestDevice();

    if (!device) {
      throw new Error("WebGPU device could not be created.");
    }

    const context = canvas.getContext("webgpu");

    if (!context) {
      throw new Error("WebGPU context could not be created.");
    }

    const presentationFormat = navigator.gpu.getPreferredCanvasFormat();

    const gpu = new GpuContext(adapter, device, canvas, context, presentationFormat);
    gpu.configure();
    gpu.resize();

    return gpu;
  }

  private configure(): void {
    this.context.configure({
      device: this.device,
      format: this.presentationFormat,
      alphaMode: "opaque",
    });
  }

  public resize(): boolean {
    const devicePixelRatio = window.devicePixelRatio || 1;
    const width = Math.max(1, Math.floor(this.canvas.clientWidth * devicePixelRatio * this.renderScale));
    const height = Math.max(1, Math.floor(this.canvas.clientHeight * devicePixelRatio * this.renderScale));

    if (this.canvas.width !== width || this.canvas.height !== height) {
      this.canvas.width = width;
      this.canvas.height = height;
      this.configure();
      return true;
    }

    return false;
  }

  public resizeIfNeeded(): boolean {
    return this.resize();
  }

  public setRenderScale(scale: number): void {
    const nextScale = Math.max(0.4, Math.min(1, scale));

    if (Math.abs(this.renderScale - nextScale) < 0.001) {
      return;
    }

    this.renderScale = nextScale;
    this.resize();
  }

  public beginFrame(): GPUCommandEncoder {
    return this.device.createCommandEncoder({
      label: "MainCommandEncoder",
    });
  }

  public submit(encoder: GPUCommandEncoder): void {
    this.device.queue.submit([encoder.finish()]);
  }

  public getCurrentTextureView(): GPUTextureView {
    return this.context.getCurrentTexture().createView();
  }

  public dispose(): void {
    this.context.unconfigure();
    this.device.destroy();
  }
}
