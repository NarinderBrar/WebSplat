const DEMO_SPLAT_STRIDE_BYTES = 32;
const DEMO_SPLAT_COUNT = 8192;

export function createDemoSplatSource(): ArrayBuffer {
  const buffer = new ArrayBuffer(DEMO_SPLAT_COUNT * DEMO_SPLAT_STRIDE_BYTES);
  const view = new DataView(buffer);
  const goldenAngle = Math.PI * (3 - Math.sqrt(5));

  for (let i = 0; i < DEMO_SPLAT_COUNT; i++) {
    const t = i / (DEMO_SPLAT_COUNT - 1);
    const y = 1 - 2 * t;
    const ringRadius = Math.sqrt(Math.max(0, 1 - y * y));
    const theta = i * goldenAngle;
    const nx = Math.cos(theta) * ringRadius;
    const ny = y;
    const nz = Math.sin(theta) * ringRadius;
    const offset = i * DEMO_SPLAT_STRIDE_BYTES;
    const shade = 0.62 + 0.38 * Math.max(0, nx * -0.35 + ny * 0.55 + nz * 0.35);
    const band = 0.5 + 0.5 * Math.sin(theta * 3.0 + y * 8.0);
    const r = Math.round(255 * Math.min(1, (0.2 + 0.5 * band) * shade));
    const g = Math.round(255 * Math.min(1, (0.45 + 0.35 * ny + 0.15 * band) * shade));
    const b = Math.round(255 * Math.min(1, (0.95 - 0.2 * band) * shade));

    view.setFloat32(offset, nx * 1.15, true);
    view.setFloat32(offset + 4, ny * 1.15, true);
    view.setFloat32(offset + 8, nz * 1.15, true);
    view.setFloat32(offset + 12, 0.026, true);
    view.setFloat32(offset + 16, 0.026, true);
    view.setFloat32(offset + 20, 0.026, true);
    view.setUint8(offset + 24, r);
    view.setUint8(offset + 25, g);
    view.setUint8(offset + 26, b);
    view.setUint8(offset + 27, 225);
    view.setUint8(offset + 28, 128);
    view.setUint8(offset + 29, 128);
    view.setUint8(offset + 30, 128);
    view.setUint8(offset + 31, 255);
  }

  return buffer;
}
