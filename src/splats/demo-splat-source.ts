const DEMO_SPLAT_STRIDE_BYTES = 48;
const DEMO_SPLAT_COUNT = 2048;

export function createDemoSplatSource(): ArrayBuffer {
  const buffer = new ArrayBuffer(DEMO_SPLAT_COUNT * DEMO_SPLAT_STRIDE_BYTES);
  const view = new DataView(buffer);
  const goldenAngle = Math.PI * (3 - Math.sqrt(5));

  for (let i = 0; i < DEMO_SPLAT_COUNT; i++) {
    const t = i / (DEMO_SPLAT_COUNT - 1);
    const y = 1 - 2 * t;
    const radius = Math.sqrt(Math.max(0, 1 - y * y));
    const theta = i * goldenAngle;
    const x = Math.cos(theta) * radius;
    const z = Math.sin(theta) * radius;
    const offset = i * DEMO_SPLAT_STRIDE_BYTES;

    view.setFloat32(offset, x * 1.6, true);
    view.setFloat32(offset + 4, y * 1.6, true);
    view.setFloat32(offset + 8, z * 1.6, true);

    view.setFloat32(offset + 32, Math.log(0.2 + 0.8 * t), true);
    view.setFloat32(offset + 36, Math.log(0.75), true);
    view.setFloat32(offset + 40, Math.log(1 - 0.7 * t), true);
  }

  return buffer;
}
