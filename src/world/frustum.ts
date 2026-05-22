import type { SplatChunk } from "./types";

export interface FrustumPlane {
  x: number;
  y: number;
  z: number;
  w: number;
}

export class Frustum {
  private readonly planes: FrustumPlane[] = [];

  public update(viewProjection: Float32Array): void {
    const m = viewProjection;
    this.planes.length = 0;

    this.pushNormalizedPlane(m[3] + m[0], m[7] + m[4], m[11] + m[8], m[15] + m[12]);
    this.pushNormalizedPlane(m[3] - m[0], m[7] - m[4], m[11] - m[8], m[15] - m[12]);
    this.pushNormalizedPlane(m[3] + m[1], m[7] + m[5], m[11] + m[9], m[15] + m[13]);
    this.pushNormalizedPlane(m[3] - m[1], m[7] - m[5], m[11] - m[9], m[15] - m[13]);
    this.pushNormalizedPlane(m[3] + m[2], m[7] + m[6], m[11] + m[10], m[15] + m[14]);
    this.pushNormalizedPlane(m[3] - m[2], m[7] - m[6], m[11] - m[10], m[15] - m[14]);
  }

  public intersectsChunk(chunk: SplatChunk): boolean {
    return this.intersectsSphere(chunk.center[0], chunk.center[1], chunk.center[2], chunk.radius);
  }

  public intersectsSphere(x: number, y: number, z: number, radius: number): boolean {
    for (const plane of this.planes) {
      const distance = plane.x * x + plane.y * y + plane.z * z + plane.w;

      if (distance < -radius) {
        return false;
      }
    }

    return true;
  }

  private pushNormalizedPlane(x: number, y: number, z: number, w: number): void {
    const length = Math.hypot(x, y, z);

    if (length <= 0) {
      return;
    }

    this.planes.push({
      x: x / length,
      y: y / length,
      z: z / length,
      w: w / length,
    });
  }
}

