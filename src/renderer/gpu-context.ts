import type { WebGPUEngine } from "@babylonjs/core/Engines/webgpuEngine";

export class GpuContext {
  public readonly engine: WebGPUEngine;

  public readonly device: GPUDevice;

  public readonly canvas: HTMLCanvasElement;

  public readonly context: GPUCanvasContext;

  public readonly presentationFormat: GPUTextureFormat;

  constructor(engine: WebGPUEngine) {
    const canvas = engine.getRenderingCanvas();

    if (!canvas) {
      throw new Error("Rendering canvas was not found.");
    }

    const context = canvas.getContext("webgpu");

    if (!context) {
      throw new Error("WebGPU context could not be created.");
    }

    this.engine = engine;

    this.canvas = canvas;

    this.context = context;

    this.device = engine._device;

    this.presentationFormat = navigator.gpu.getPreferredCanvasFormat();

    this.configure();
  }

  private configure(): void {
    this.context.configure({
      device: this.device,
      format: this.presentationFormat,
      alphaMode: "opaque",
    });
  }

  public resize(): void {
    const devicePixelRatio = window.devicePixelRatio || 1;

    const width = Math.floor(this.canvas.clientWidth * devicePixelRatio);
    const height = Math.floor(this.canvas.clientHeight * devicePixelRatio);

    if (this.canvas.width !== width || this.canvas.height !== height) {
      this.canvas.width = width;
      this.canvas.height = height;

      this.configure();
    }
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
}
