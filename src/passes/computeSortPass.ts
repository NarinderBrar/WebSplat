
import type { CameraUniforms } from "../camera/camera-uniforms";
import { GpuSorter } from "../splats/sorter";

export class ComputeSortPass {
  private readonly sorter: GpuSorter;

  constructor(device: GPUDevice, cameraBindGroupLayout: GPUBindGroupLayout) {
    this.sorter = new GpuSorter(device, cameraBindGroupLayout);
  }

  public ensureBuffers(splatCount: number): void {
    this.sorter.ensureBuffers(splatCount);
  }

  public encode(
    encoder: GPUCommandEncoder,
    cameraUniforms: CameraUniforms,
    positionBuffer: GPUBuffer,
    visibleSplatIndicesBuffer: GPUBuffer,
    indirectArgsBuffer: GPUBuffer,
  ): void {
    this.sorter.sort(
      encoder,
      cameraUniforms,
      positionBuffer,
      visibleSplatIndicesBuffer,
      indirectArgsBuffer,
    );
  }

  public dispose(): void {
    this.sorter.dispose();
  }
}
