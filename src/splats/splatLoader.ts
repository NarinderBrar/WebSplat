import type { SplatData } from "./splatBuffer";

const SPLAT_STRIDE_BYTES = 48;

export async function loadSplatSource(
  source: string | ArrayBuffer,
): Promise<SplatData> {
  const buffer = typeof source === "string" ? await fetchSplatBuffer(source) : source;

  if (buffer.byteLength % SPLAT_STRIDE_BYTES !== 0) {
    throw new Error(
      `Invalid splat buffer size ${buffer.byteLength}. Expected a multiple of ${SPLAT_STRIDE_BYTES} bytes.`,
    );
  }

  const view = new DataView(buffer);
  const count = buffer.byteLength / SPLAT_STRIDE_BYTES;

  const positions = new Float32Array(count * 3);
  const colors = new Float32Array(count * 3);
  const covariances = new Float32Array(count * 6);
  const shCoefficients = new Float32Array(count * 0);

  for (let i = 0; i < count; i++) {
    const offset = i * SPLAT_STRIDE_BYTES;

    positions[i * 3] = view.getFloat32(offset, true);
    positions[i * 3 + 1] = view.getFloat32(offset + 4, true);
    positions[i * 3 + 2] = view.getFloat32(offset + 8, true);

    const dc0 = view.getFloat32(offset + 32, true);
    const dc1 = view.getFloat32(offset + 36, true);
    const dc2 = view.getFloat32(offset + 40, true);

    colors[i * 3] = Math.exp(dc0);
    colors[i * 3 + 1] = Math.exp(dc1);
    colors[i * 3 + 2] = Math.exp(dc2);
  }

  return {
    positions,
    colors,
    covariances,
    shCoefficients,
    count,
  };
}

async function fetchSplatBuffer(source: string): Promise<ArrayBuffer> {
  const response = await fetch(source);

  if (!response.ok) {
    throw new Error(`Failed to load splat source "${source}": ${response.status} ${response.statusText}`);
  }

  return response.arrayBuffer();
}
