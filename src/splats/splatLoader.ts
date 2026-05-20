import type { SplatData } from "./splatBuffer";

const INTERNAL_SPLAT_STRIDE_BYTES = 48;
const STANDARD_SPLAT_STRIDE_BYTES = 32;

export async function loadSplatSource(
  source: string | ArrayBuffer,
): Promise<SplatData> {
  const buffer = typeof source === "string" ? await fetchSplatBuffer(source) : source;

  if (buffer.byteLength % STANDARD_SPLAT_STRIDE_BYTES === 0) {
    return parseStandardSplatBuffer(buffer);
  }

  if (buffer.byteLength % INTERNAL_SPLAT_STRIDE_BYTES === 0) {
    return parseInternalSplatBuffer(buffer);
  }

  throw new Error(
    `Invalid splat buffer size ${buffer.byteLength}. Expected a multiple of ${STANDARD_SPLAT_STRIDE_BYTES} or ${INTERNAL_SPLAT_STRIDE_BYTES} bytes.`,
  );
}

function parseInternalSplatBuffer(buffer: ArrayBuffer): SplatData {
  const view = new DataView(buffer);
  const count = buffer.byteLength / INTERNAL_SPLAT_STRIDE_BYTES;

  const positions = new Float32Array(count * 3);
  const colors = new Float32Array(count * 3);
  const covariances = new Float32Array(count * 6);
  const shCoefficients = new Float32Array(count * 0);

  for (let i = 0; i < count; i++) {
    const offset = i * INTERNAL_SPLAT_STRIDE_BYTES;

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

function parseStandardSplatBuffer(buffer: ArrayBuffer): SplatData {
  const view = new DataView(buffer);
  const count = buffer.byteLength / STANDARD_SPLAT_STRIDE_BYTES;

  const positions = new Float32Array(count * 3);
  const colors = new Float32Array(count * 3);
  const covariances = new Float32Array(count * 6);
  const shCoefficients = new Float32Array(0);

  for (let i = 0; i < count; i++) {
    const offset = i * STANDARD_SPLAT_STRIDE_BYTES;
    const base3 = i * 3;
    const base6 = i * 6;

    positions[base3] = view.getFloat32(offset, true);
    positions[base3 + 1] = view.getFloat32(offset + 4, true);
    positions[base3 + 2] = view.getFloat32(offset + 8, true);

    const sx = view.getFloat32(offset + 12, true);
    const sy = view.getFloat32(offset + 16, true);
    const sz = view.getFloat32(offset + 20, true);

    covariances[base6] = sx;
    covariances[base6 + 1] = sy;
    covariances[base6 + 2] = sz;

    colors[base3] = view.getUint8(offset + 24) / 255;
    colors[base3 + 1] = view.getUint8(offset + 25) / 255;
    colors[base3 + 2] = view.getUint8(offset + 26) / 255;
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
