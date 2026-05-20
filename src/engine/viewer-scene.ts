import { ArcRotateCamera } from "@babylonjs/core/Cameras/arcRotateCamera";
import type { AbstractEngine } from "@babylonjs/core/Engines/abstractEngine";
import { DirectionalLight } from "@babylonjs/core/Lights/directionalLight";
import { HemisphericLight } from "@babylonjs/core/Lights/hemisphericLight";
import { Color3, Color4 } from "@babylonjs/core/Maths/math.color";
import { Vector3 } from "@babylonjs/core/Maths/math.vector";
import { StandardMaterial } from "@babylonjs/core/Materials/standardMaterial";
import { CreateGround } from "@babylonjs/core/Meshes/Builders/groundBuilder";
import { CreateSphere } from "@babylonjs/core/Meshes/Builders/sphereBuilder";
import type { Mesh } from "@babylonjs/core/Meshes/mesh";
import { Scene } from "@babylonjs/core/scene";

export default class ViewerScene {
  public readonly scene: Scene;

  public readonly camera: ArcRotateCamera;

  public readonly placeholder: Mesh;

  private readonly engine: AbstractEngine;

  constructor(engine: AbstractEngine, canvas: HTMLCanvasElement) {
    this.engine = engine;

    this.scene = new Scene(engine);
    this.scene.clearColor = new Color4(0.03, 0.05, 0.07, 1);

    this.camera = this.createCamera(canvas);

    this.createLights();

    this.placeholder = this.createPlaceholder();

    this.createGround();

    this.registerBeforeRender();
  }

  private createCamera(canvas: HTMLCanvasElement): ArcRotateCamera {
    const camera = new ArcRotateCamera(
      "viewerCamera",
      Math.PI / 4,
      Math.PI / 3,
      6,
      Vector3.Zero(),
      this.scene,
    );

    camera.attachControl(canvas, true);

    camera.lowerRadiusLimit = 3;
    camera.upperRadiusLimit = 10;

    return camera;
  }

  private createLights(): void {
    const keyLight = new DirectionalLight(
      "keyLight",
      new Vector3(-0.5, -1, -0.7),
      this.scene,
    );

    keyLight.intensity = 1.5;

    const fillLight = new HemisphericLight(
      "fillLight",
      new Vector3(0, 1, 0),
      this.scene,
    );

    fillLight.intensity = 0.45;
  }

  private createPlaceholder(): Mesh {
    const placeholder = CreateSphere(
      "splatPlaceholder",
      {
        diameter: 1.6,
        segments: 48,
      },
      this.scene,
    );

    placeholder.position.y = 1.05;

    const material = new StandardMaterial(
      "splatPlaceholderMaterial",
      this.scene,
    );

    material.diffuseColor = new Color3(0.2, 0.8, 0.95);
    material.specularColor = new Color3(0.9, 0.9, 0.9);

    placeholder.material = material;

    return placeholder;
  }

  private createGround(): void {
    const ground = CreateGround(
      "referenceGround",
      {
        width: 8,
        height: 8,
      },
      this.scene,
    );

    const material = new StandardMaterial(
      "referenceGroundMaterial",
      this.scene,
    );

    material.diffuseColor = new Color3(0.18, 0.2, 0.22);

    ground.material = material;
  }

  private registerBeforeRender(): void {
    this.scene.onBeforeRenderObservable.add(() => {
      this.placeholder.rotation.y += this.engine.getDeltaTime() * 0.001;
    });
  }

  public dispose(): void {
    this.scene.dispose();
  }
}
