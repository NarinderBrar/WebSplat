import type { SplatData } from "./splatBuffer";

const INTERNAL_SPLAT_STRIDE_BYTES = 48;
const STANDARD_SPLAT_STRIDE_BYTES = 32;
const SH_C0 = 0.28209479177387814;

type PlyFormat = "ascii" | "binary_little_endian";

interface PlyProperty {
  name: string;
  type: PlyScalarType;
}

interface PlyHeader {
  format: PlyFormat;
  vertexCount: number;
  vertexProperties: PlyProperty[];
  dataOffset: number;
}

type PlyScalarType =
  | "char"
  | "uchar"
  | "short"
  | "ushort"
  | "int"
  | "uint"
  | "float"
  | "double";

export async function loadSplatSource(
  source: string | ArrayBuffer,
): Promise<SplatData> {
  const buffer =
    typeof source === "string" ? await fetchSplatBuffer(source) : source;

  if (isPlyBuffer(buffer)) {
    return parsePlyBuffer(buffer);
  }

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

function isPlyBuffer(buffer: ArrayBuffer): boolean {
  if (buffer.byteLength < 3) {
    return false;
  }

  const bytes = new Uint8Array(buffer, 0, 3);
  return bytes[0] === 0x70 && bytes[1] === 0x6c && bytes[2] === 0x79;
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

    const sourceX = view.getFloat32(offset, true);
    const sourceY = view.getFloat32(offset + 4, true);
    const sourceZ = view.getFloat32(offset + 8, true);

    positions[base3] = sourceX;
    positions[base3 + 1] = -sourceY;
    positions[base3 + 2] = sourceZ;

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
    throw new Error(
      `Failed to load splat source "${source}": ${response.status} ${response.statusText}`,
    );
  }

  return response.arrayBuffer();
}

function parsePlyBuffer(buffer: ArrayBuffer): SplatData {
  const header = parsePlyHeader(buffer);

  if (header.format === "ascii") {
    return parseAsciiPly(buffer, header);
  }

  return parseBinaryLittleEndianPly(buffer, header);
}

function parsePlyHeader(buffer: ArrayBuffer): PlyHeader {
  const bytes = new Uint8Array(buffer);
  const endHeaderNeedle = new TextEncoder().encode("end_header");
  const headerEndStart = findBytes(bytes, endHeaderNeedle);

  if (headerEndStart === -1) {
    throw new Error("Invalid PLY: missing end_header.");
  }

  let dataOffset = headerEndStart + endHeaderNeedle.length;

  if (bytes[dataOffset] === 0x0d) {
    dataOffset++;
  }

  if (bytes[dataOffset] === 0x0a) {
    dataOffset++;
  }

  const headerText = new TextDecoder().decode(bytes.subarray(0, dataOffset));
  const lines = headerText.split(/\r?\n/);
  let format: PlyFormat | null = null;
  let vertexCount = 0;
  let inVertexElement = false;
  const vertexProperties: PlyProperty[] = [];

  for (const line of lines) {
    const parts = line.trim().split(/\s+/);

    if (parts[0] === "format") {
      if (parts[1] !== "ascii" && parts[1] !== "binary_little_endian") {
        throw new Error(`Unsupported PLY format "${parts[1]}".`);
      }

      format = parts[1];
    } else if (parts[0] === "element") {
      inVertexElement = parts[1] === "vertex";

      if (inVertexElement) {
        vertexCount = Number.parseInt(parts[2], 10);
      }
    } else if (parts[0] === "property" && inVertexElement) {
      if (parts[1] === "list") {
        throw new Error(
          "Unsupported PLY: list properties in vertex element are not supported.",
        );
      }

      const type = normalizePlyScalarType(parts[1]);
      vertexProperties.push({ type, name: parts[2] });
    }
  }

  if (!format) {
    throw new Error("Invalid PLY: missing format.");
  }

  if (!Number.isFinite(vertexCount) || vertexCount <= 0) {
    throw new Error("Invalid PLY: missing vertex count.");
  }

  if (vertexProperties.length === 0) {
    throw new Error("Invalid PLY: missing vertex properties.");
  }

  return {
    format,
    vertexCount,
    vertexProperties,
    dataOffset,
  };
}

function parseBinaryLittleEndianPly(
  buffer: ArrayBuffer,
  header: PlyHeader,
): SplatData {
  const view = new DataView(buffer, header.dataOffset);
  const propertyOffsets = new Map<string, number>();
  const propertyTypes = new Map<string, PlyScalarType>();
  let stride = 0;

  for (const property of header.vertexProperties) {
    propertyOffsets.set(property.name, stride);
    propertyTypes.set(property.name, property.type);
    stride += getPlyScalarByteSize(property.type);
  }

  const expectedByteLength = header.dataOffset + stride * header.vertexCount;

  if (expectedByteLength > buffer.byteLength) {
    throw new Error(
      "Invalid PLY: vertex data is shorter than the header declares.",
    );
  }

  const positions = new Float32Array(header.vertexCount * 3);
  const colors = new Float32Array(header.vertexCount * 3);
  const covariances = new Float32Array(header.vertexCount * 6);
  const shCoefficients = new Float32Array(0);

  for (let i = 0; i < header.vertexCount; i++) {
    const rowOffset = i * stride;
    writePlyVertex(
      view,
      rowOffset,
      propertyOffsets,
      propertyTypes,
      i,
      positions,
      colors,
      covariances,
    );
  }

  return {
    positions,
    colors,
    covariances,
    shCoefficients,
    count: header.vertexCount,
  };
}

function parseAsciiPly(buffer: ArrayBuffer, header: PlyHeader): SplatData {
  const text = new TextDecoder().decode(buffer.slice(header.dataOffset));
  const lines = text.trim().split(/\r?\n/);
  const propertyIndices = new Map<string, number>();

  header.vertexProperties.forEach((property, index) => {
    propertyIndices.set(property.name, index);
  });

  const positions = new Float32Array(header.vertexCount * 3);
  const colors = new Float32Array(header.vertexCount * 3);
  const covariances = new Float32Array(header.vertexCount * 6);
  const shCoefficients = new Float32Array(0);

  for (let i = 0; i < header.vertexCount; i++) {
    const values = lines[i]?.trim().split(/\s+/).map(Number);

    if (!values || values.length < header.vertexProperties.length) {
      throw new Error(`Invalid ASCII PLY: missing vertex row ${i}.`);
    }

    writeAsciiPlyVertex(
      values,
      propertyIndices,
      i,
      positions,
      colors,
      covariances,
    );
  }

  return {
    positions,
    colors,
    covariances,
    shCoefficients,
    count: header.vertexCount,
  };
}

function writePlyVertex(
  view: DataView,
  rowOffset: number,
  propertyOffsets: Map<string, number>,
  propertyTypes: Map<string, PlyScalarType>,
  index: number,
  positions: Float32Array,
  colors: Float32Array,
  covariances: Float32Array,
): void {
  const get = (name: string, fallback = 0): number => {
    const offset = propertyOffsets.get(name);
    const type = propertyTypes.get(name);

    return offset === undefined || !type
      ? fallback
      : readPlyScalar(view, rowOffset + offset, type);
  };

  writeParsedVertex(get, index, positions, colors, covariances);
}

function writeAsciiPlyVertex(
  values: number[],
  propertyIndices: Map<string, number>,
  index: number,
  positions: Float32Array,
  colors: Float32Array,
  covariances: Float32Array,
): void {
  const get = (name: string, fallback = 0): number => {
    const propertyIndex = propertyIndices.get(name);
    return propertyIndex === undefined ? fallback : values[propertyIndex];
  };

  writeParsedVertex(get, index, positions, colors, covariances);
}

function writeParsedVertex(
  get: (name: string, fallback?: number) => number,
  index: number,
  positions: Float32Array,
  colors: Float32Array,
  covariances: Float32Array,
): void {
  const base3 = index * 3;
  const base6 = index * 6;
  const sourceX = get("x");
  const sourceY = get("y");
  const sourceZ = get("z");

  positions[base3] = sourceX;
  positions[base3 + 1] = -sourceY;
  positions[base3 + 2] = sourceZ;

  if (hasColorFields(get)) {
    colors[base3] = normalizeColor(get("red", get("r")));
    colors[base3 + 1] = normalizeColor(get("green", get("g")));
    colors[base3 + 2] = normalizeColor(get("blue", get("b")));
  } else {
    colors[base3] = clamp01(0.5 + SH_C0 * get("f_dc_0"));
    colors[base3 + 1] = clamp01(0.5 + SH_C0 * get("f_dc_1"));
    colors[base3 + 2] = clamp01(0.5 + SH_C0 * get("f_dc_2"));
  }

  covariances[base6] = Math.exp(get("scale_0"));
  covariances[base6 + 1] = Math.exp(get("scale_1"));
  covariances[base6 + 2] = Math.exp(get("scale_2"));
}

function hasColorFields(
  get: (name: string, fallback?: number) => number,
): boolean {
  return (
    get("red", Number.NaN) === get("red", Number.NaN) ||
    get("r", Number.NaN) === get("r", Number.NaN)
  );
}

function normalizeColor(value: number): number {
  return value > 1 ? value / 255 : value;
}

function clamp01(value: number): number {
  return Math.max(0, Math.min(1, value));
}

function findBytes(haystack: Uint8Array, needle: Uint8Array): number {
  for (let i = 0; i <= haystack.length - needle.length; i++) {
    let found = true;

    for (let j = 0; j < needle.length; j++) {
      if (haystack[i + j] !== needle[j]) {
        found = false;
        break;
      }
    }

    if (found) {
      return i;
    }
  }

  return -1;
}

function normalizePlyScalarType(type: string): PlyScalarType {
  switch (type) {
    case "int8":
      return "char";
    case "uint8":
      return "uchar";
    case "int16":
      return "short";
    case "uint16":
      return "ushort";
    case "int32":
      return "int";
    case "uint32":
      return "uint";
    case "float32":
      return "float";
    case "float64":
      return "double";
    case "char":
    case "uchar":
    case "short":
    case "ushort":
    case "int":
    case "uint":
    case "float":
    case "double":
      return type;
    default:
      throw new Error(`Unsupported PLY scalar type "${type}".`);
  }
}

function getPlyScalarByteSize(type: PlyScalarType): number {
  switch (type) {
    case "char":
    case "uchar":
      return 1;
    case "short":
    case "ushort":
      return 2;
    case "int":
    case "uint":
    case "float":
      return 4;
    case "double":
      return 8;
  }
}

function readPlyScalar(
  view: DataView,
  byteOffset: number,
  type: PlyScalarType,
): number {
  switch (type) {
    case "char":
      return view.getInt8(byteOffset);
    case "uchar":
      return view.getUint8(byteOffset);
    case "short":
      return view.getInt16(byteOffset, true);
    case "ushort":
      return view.getUint16(byteOffset, true);
    case "int":
      return view.getInt32(byteOffset, true);
    case "uint":
      return view.getUint32(byteOffset, true);
    case "float":
      return view.getFloat32(byteOffset, true);
    case "double":
      return view.getFloat64(byteOffset, true);
  }
}
