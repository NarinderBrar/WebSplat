import { ArcRotateCamera } from "@babylonjs/core/Cameras/arcRotateCamera";
import { Engine } from "@babylonjs/core/Engines/engine";
import { Color4 } from "@babylonjs/core/Maths/math.color";
import { Matrix, Vector3 } from "@babylonjs/core/Maths/math.vector";
import { Scene } from "@babylonjs/core/scene";
import { CameraUniforms } from "./camera-uniforms";

export interface OrbitCameraState {
  alpha: number;
  beta: number;
  radius: number;
  target: [number, number, number];
}

export class OrbitCamera {
  public readonly uniforms: CameraUniforms;

  private readonly engine: Engine;
  private readonly scene: Scene;
  private readonly camera: ArcRotateCamera;
  private readonly canvas: HTMLCanvasElement;
  private readonly overlayCanvas: HTMLCanvasElement;
  private readonly viewMatrix = new Float32Array(16);
  private readonly projectionMatrix = new Float32Array(16);
  private readonly viewProjectionMatrix = new Float32Array(16);
  private readonly babylonProjection = Matrix.Identity();
  private controlsEnabled = true;

  constructor(
    device: GPUDevice,
    canvas: HTMLCanvasElement,
    cameraBindGroupLayout: GPUBindGroupLayout,
  ) {
    this.canvas = canvas;
    this.overlayCanvas = createBabylonOverlayCanvas(canvas);
    this.engine = new Engine(
      this.overlayCanvas,
      true,
      {
        alpha: true,
        premultipliedAlpha: false,
        preserveDrawingBuffer: true,
      },
      true,
    );
    this.scene = new Scene(this.engine);
    this.scene.clearColor = new Color4(0, 0, 0, 0);
    this.scene.autoClear = true;
    this.camera = new ArcRotateCamera(
      "viewerCamera",
      -Math.PI / 2,
      Math.PI / 2.1,
      1.25,
      Vector3.Zero(),
      this.scene,
    );
    this.camera.lowerRadiusLimit = 0.02;
    this.camera.upperRadiusLimit = 10000;
    this.camera.minZ = 0.1;
    this.camera.maxZ = 10000;
    this.camera.fov = Math.PI / 4;
    this.camera.wheelPrecision = 20;
    this.camera.panningSensibility = 80;
    this.camera.attachControl(this.overlayCanvas, true);

    this.uniforms = new CameraUniforms(device);
    this.uniforms.createBuffers(cameraBindGroupLayout);
  }

  public getState(): OrbitCameraState {
    return {
      alpha: this.camera.alpha,
      beta: this.camera.beta,
      radius: this.camera.radius,
      target: [
        this.camera.target.x,
        this.camera.target.y,
        this.camera.target.z,
      ],
    };
  }

  public setState(state: Partial<OrbitCameraState>): void {
    if (state.alpha !== undefined) this.camera.alpha = state.alpha;
    if (state.beta !== undefined) this.camera.beta = state.beta;
    if (state.radius !== undefined) this.camera.radius = state.radius;

    if (state.target !== undefined) {
      this.camera.target.set(
        state.target[0],
        state.target[1],
        state.target[2],
      );
    }
  }

  public getBabylonCamera(): ArcRotateCamera {
    return this.camera;
  }

  public getBabylonScene(): Scene {
    return this.scene;
  }

  public setControlsEnabled(enabled: boolean): void {
    if (this.controlsEnabled === enabled) {
      return;
    }

    this.controlsEnabled = enabled;

    if (enabled) {
      this.camera.attachControl(this.overlayCanvas, true);
    } else {
      this.camera.detachControl();
    }
  }

  public setGizmoPointerEnabled(enabled: boolean): void {
    this.overlayCanvas.style.pointerEvents = enabled ? "auto" : "none";
  }

  public getViewMatrix(): Float32Array {
    return this.viewMatrix;
  }

  public getProjectionMatrix(): Float32Array {
    return this.projectionMatrix;
  }

  public getViewProjectionMatrix(): Float32Array {
    return this.viewProjectionMatrix;
  }

  public update(): void {
    this.engine.resize();
    this.camera.update();
    this.camera.getViewMatrix(true).copyToArray(this.viewMatrix, 0);

    Matrix.PerspectiveFovLHToRef(
      this.camera.fov,
      this.canvas.width / this.canvas.height,
      this.camera.minZ,
      this.camera.maxZ,
      this.babylonProjection,
      true,
      true,
    );
    this.babylonProjection.copyToArray(this.projectionMatrix, 0);
    this.multiplyMatrices(
      this.projectionMatrix,
      this.viewMatrix,
      this.viewProjectionMatrix,
    );

    this.uniforms.updateView(this.viewMatrix);
    this.uniforms.updateProjection(this.projectionMatrix);
    this.uniforms.updateViewProjection(this.viewProjectionMatrix);
    this.uniforms.upload();
    this.scene.render();
  }

  public dispose(): void {
    this.camera.detachControl();
    this.uniforms.dispose();
    this.scene.dispose();
    this.engine.dispose();
    this.overlayCanvas.remove();
  }

  private multiplyMatrices(
    a: Float32Array,
    b: Float32Array,
    out: Float32Array,
  ): void {
    for (let i = 0; i < 4; i++) {
      for (let j = 0; j < 4; j++) {
        out[i * 4 + j] = 0;
        for (let k = 0; k < 4; k++) {
          out[i * 4 + j] += a[k * 4 + j] * b[i * 4 + k];
        }
      }
    }
  }
}

function createBabylonOverlayCanvas(baseCanvas: HTMLCanvasElement): HTMLCanvasElement {
  const overlayCanvas = document.createElement("canvas");
  overlayCanvas.className = "babylon-gizmo-overlay";
  overlayCanvas.setAttribute("aria-hidden", "true");
  overlayCanvas.style.position = "fixed";
  overlayCanvas.style.inset = "0";
  overlayCanvas.style.width = "100%";
  overlayCanvas.style.height = "100%";
  overlayCanvas.style.zIndex = "8";
  overlayCanvas.style.pointerEvents = "none";
  overlayCanvas.style.background = "transparent";
  baseCanvas.insertAdjacentElement("afterend", overlayCanvas);
  return overlayCanvas;
}
