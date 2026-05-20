import { CameraUniforms } from "./camera-uniforms";

export interface OrbitCameraState {
  alpha: number;
  beta: number;
  radius: number;
  target: [number, number, number];
}

export class OrbitCamera {
  public readonly uniforms: CameraUniforms;

  private alpha: number;
  private beta: number;
  private radius: number;
  private target: [number, number, number];
  private minRadius: number;
  private maxRadius: number;
  private canvas: HTMLCanvasElement;

  constructor(device: GPUDevice, canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    this.uniforms = new CameraUniforms(device);
    this.alpha = Math.PI / 4;
    this.beta = Math.PI / 3;
    this.radius = 6;
    this.target = [0, 0, 0];
    this.minRadius = 3;
    this.maxRadius = 10;

    this.uniforms.createBuffers();
  }

  public getState(): OrbitCameraState {
    return {
      alpha: this.alpha,
      beta: this.beta,
      radius: this.radius,
      target: [...this.target],
    };
  }

  public setState(state: Partial<OrbitCameraState>): void {
    if (state.alpha !== undefined) this.alpha = state.alpha;
    if (state.beta !== undefined) this.beta = state.beta;
    if (state.radius !== undefined) this.radius = state.radius;
    if (state.target !== undefined) this.target = state.target;

    this.radius = Math.max(this.minRadius, Math.min(this.maxRadius, this.radius));
    this.beta = Math.max(0.1, Math.min(Math.PI - 0.1, this.beta));
  }

  public update(): void {
    const position = this.getPosition();

    const viewMatrix = this.lookAt(position, this.target);
    const projectionMatrix = this.perspective();

    const viewProjection = new Float32Array(16);
    this.multiplyMatrices(projectionMatrix, viewMatrix, viewProjection);

    this.uniforms.updateView(viewMatrix);
    this.uniforms.updateProjection(projectionMatrix);
    this.uniforms.updateViewProjection(viewProjection);
    this.uniforms.upload();
  }

  private getPosition(): [number, number, number] {
    const x = this.target[0] + this.radius * Math.sin(this.beta) * Math.cos(this.alpha);
    const y = this.target[1] + this.radius * Math.cos(this.beta);
    const z = this.target[2] + this.radius * Math.sin(this.beta) * Math.sin(this.alpha);
    return [x, y, z];
  }

  private lookAt(eye: [number, number, number], target: [number, number, number]): Float32Array {
    const matrix = new Float32Array(16);

    const zAxis = this.normalize([
      eye[0] - target[0],
      eye[1] - target[1],
      eye[2] - target[2],
    ]);
    const xAxis = this.normalize(this.cross([0, 1, 0], zAxis));
    const yAxis = this.cross(zAxis, xAxis);

    matrix[0] = xAxis[0];
    matrix[1] = yAxis[0];
    matrix[2] = zAxis[0];
    matrix[3] = 0;
    matrix[4] = xAxis[1];
    matrix[5] = yAxis[1];
    matrix[6] = zAxis[1];
    matrix[7] = 0;
    matrix[8] = xAxis[2];
    matrix[9] = yAxis[2];
    matrix[10] = zAxis[2];
    matrix[11] = 0;
    matrix[12] = -this.dot(xAxis, eye);
    matrix[13] = -this.dot(yAxis, eye);
    matrix[14] = -this.dot(zAxis, eye);
    matrix[15] = 1;

    return matrix;
  }

  private perspective(): Float32Array {
    const matrix = new Float32Array(16);
    const aspect = this.canvas.width / this.canvas.height;
    const fov = Math.PI / 4;
    const near = 0.1;
    const far = 100;

    const f = 1 / Math.tan(fov / 2);
    const rangeInv = 1 / (near - far);

    matrix[0] = f / aspect;
    matrix[5] = f;
    matrix[10] = far * rangeInv;
    matrix[11] = -1;
    matrix[14] = far * near * rangeInv;

    return matrix;
  }

  private multiplyMatrices(a: Float32Array, b: Float32Array, out: Float32Array): void {
    for (let i = 0; i < 4; i++) {
      for (let j = 0; j < 4; j++) {
        out[i * 4 + j] = 0;
        for (let k = 0; k < 4; k++) {
          out[i * 4 + j] += a[k * 4 + j] * b[i * 4 + k];
        }
      }
    }
  }

  private normalize(v: [number, number, number]): [number, number, number] {
    const len = Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    return len > 0 ? [v[0] / len, v[1] / len, v[2] / len] : [0, 0, 0];
  }

  private cross(a: [number, number, number], b: [number, number, number]): [number, number, number] {
    return [
      a[1] * b[2] - a[2] * b[1],
      a[2] * b[0] - a[0] * b[2],
      a[0] * b[1] - a[1] * b[0],
    ];
  }

  private dot(a: [number, number, number], b: [number, number, number]): number {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  }

  public dispose(): void {
    this.uniforms.dispose();
  }
}
