import type { SplatData } from "./splatBuffer";

const INTERNAL_SPLAT_STRIDE_BYTES = 48;
const STANDARD_SPLAT_STRIDE_BYTES = 32;
const SH_C0 = 0.28209479177387814;
const ZIP_LOCAL_FILE_HEADER_SIGNATURE = 0x04034b50;
const ZIP_CENTRAL_DIRECTORY_SIGNATURE = 0x02014b50;
const ZIP_END_OF_CENTRAL_DIRECTORY_SIGNATURE = 0x06054b50;

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

interface SogMeta {
  version: number;
  count: number;
  means: {
    mins: [number, number, number];
    maxs: [number, number, number];
    files: [string, string];
  };
  scales: {
    codebook: number[];
    files: [string];
  };
  quats: {
    files: [string];
  };
  sh0: {
    codebook: number[];
    files: [string];
  };
}

interface DecodedImage {
  width: number;
  height: number;
  data: Uint8ClampedArray;
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

  if (isZipBuffer(buffer)) {
    return parseSogBuffer(buffer);
  }

  if (buffer.byteLength % STANDARD_SPLAT_STRIDE_BYTES === 0) {
    return parseSplatBuffer(buffer);
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

function isZipBuffer(buffer: ArrayBuffer): boolean {
  if (buffer.byteLength < 4) {
    return false;
  }

  return new DataView(buffer).getUint32(0, true) === ZIP_LOCAL_FILE_HEADER_SIGNATURE;
}

function parseInternalSplatBuffer(buffer: ArrayBuffer): SplatData {
  const view = new DataView(buffer);
  const count = buffer.byteLength / INTERNAL_SPLAT_STRIDE_BYTES;

  const positions = new Float32Array(count * 3);
  const colors = new Float32Array(count * 3);
  const opacities = new Float32Array(count);
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
    opacities[i] = 1;
  }

  return {
    positions,
    colors,
    opacities,
    covariances,
    shCoefficients,
    count,
  };
}

async function parseSogBuffer(buffer: ArrayBuffer): Promise<SplatData> {
  const files = await unzipFiles(buffer);
  const metaBytes = files.get("meta.json");

  if (!metaBytes) {
    throw new Error("Invalid SOG: missing meta.json.");
  }

  const meta = JSON.parse(new TextDecoder().decode(metaBytes)) as SogMeta;

  if (meta.version !== 2) {
    throw new Error(`Unsupported SOG version "${meta.version}". Expected version 2.`);
  }

  const requiredFiles = [
    ...meta.means.files,
    ...meta.scales.files,
    ...meta.quats.files,
    ...meta.sh0.files,
  ];

  for (const fileName of requiredFiles) {
    if (!files.has(fileName)) {
      throw new Error(`Invalid SOG: missing "${fileName}".`);
    }
  }

  const meansL = await decodeImage(files.get(meta.means.files[0])!, meta.means.files[0]);
  const meansU = await decodeImage(files.get(meta.means.files[1])!, meta.means.files[1]);
  const scalesImage = await decodeImage(files.get(meta.scales.files[0])!, meta.scales.files[0]);
  const quatsImage = await decodeImage(files.get(meta.quats.files[0])!, meta.quats.files[0]);
  const sh0Image = await decodeImage(files.get(meta.sh0.files[0])!, meta.sh0.files[0]);
  const count = meta.count;

  if (
    meansL.width * meansL.height < count ||
    meansU.width * meansU.height < count ||
    scalesImage.width * scalesImage.height < count ||
    quatsImage.width * quatsImage.height < count ||
    sh0Image.width * sh0Image.height < count
  ) {
    throw new Error("Invalid SOG: one or more attribute images are smaller than meta.count.");
  }

  const positions = new Float32Array(count * 3);
  const colors = new Float32Array(count * 3);
  const opacities = new Float32Array(count);
  const covariances = new Float32Array(count * 6);
  const shCoefficients = new Float32Array(0);

  for (let i = 0; i < count; i++) {
    const imageOffset = i * 4;
    const base3 = i * 3;
    const base6 = i * 6;
    const px = unlogSogCoordinate(unpackSogCoordinate(meansL.data[imageOffset], meansU.data[imageOffset], meta.means.mins[0], meta.means.maxs[0]));
    const py = unlogSogCoordinate(unpackSogCoordinate(meansL.data[imageOffset + 1], meansU.data[imageOffset + 1], meta.means.mins[1], meta.means.maxs[1]));
    const pz = unlogSogCoordinate(unpackSogCoordinate(meansL.data[imageOffset + 2], meansU.data[imageOffset + 2], meta.means.mins[2], meta.means.maxs[2]));
    positions[base3] = px;
    positions[base3 + 1] = -py;
    positions[base3 + 2] = pz;

    const sx = decodeSogScale(meta.scales.codebook[scalesImage.data[imageOffset]]);
    const sy = decodeSogScale(meta.scales.codebook[scalesImage.data[imageOffset + 1]]);
    const sz = decodeSogScale(meta.scales.codebook[scalesImage.data[imageOffset + 2]]);
    const [qx, qy, qz, qw] = unpackSogQuaternion(
      quatsImage.data[imageOffset],
      quatsImage.data[imageOffset + 1],
      quatsImage.data[imageOffset + 2],
      quatsImage.data[imageOffset + 3],
    );
    writeCovariance(covariances, base6, sx, sy, sz, qx, qy, qz, qw, true);

    colors[base3] = clamp01(0.5 + SH_C0 * (meta.sh0.codebook[sh0Image.data[imageOffset]] ?? 0));
    colors[base3 + 1] = clamp01(0.5 + SH_C0 * (meta.sh0.codebook[sh0Image.data[imageOffset + 1]] ?? 0));
    colors[base3 + 2] = clamp01(0.5 + SH_C0 * (meta.sh0.codebook[sh0Image.data[imageOffset + 2]] ?? 0));
    opacities[i] = sh0Image.data[imageOffset + 3] / 255;
  }

  return {
    positions,
    colors,
    opacities,
    covariances,
    shCoefficients,
    count,
  };
}

function parseSplatBuffer(buffer: ArrayBuffer): SplatData {
  const view = new DataView(buffer);
  const count = buffer.byteLength / STANDARD_SPLAT_STRIDE_BYTES;

  const positions = new Float32Array(count * 3);
  const colors = new Float32Array(count * 3);
  const opacities = new Float32Array(count);
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
    positions[base3 + 1] = sourceY;
    positions[base3 + 2] = sourceZ;

    const sx = view.getFloat32(offset + 12, true);
    const sy = view.getFloat32(offset + 16, true);
    const sz = view.getFloat32(offset + 20, true);
    const qx = view.getUint8(offset + 28) / 127.5 - 1;
    const qy = view.getUint8(offset + 29) / 127.5 - 1;
    const qz = view.getUint8(offset + 30) / 127.5 - 1;
    const qw = view.getUint8(offset + 31) / 127.5 - 1;

    writeCovariance(covariances, base6, sx, sy, sz, qx, qy, qz, qw, true);

    colors[base3] = view.getUint8(offset + 24) / 255;
    colors[base3 + 1] = view.getUint8(offset + 25) / 255;
    colors[base3 + 2] = view.getUint8(offset + 26) / 255;
    opacities[i] = view.getUint8(offset + 27) / 255;
  }

  return {
    positions,
    colors,
    opacities,
    covariances,
    shCoefficients,
    count,
  };
}

async function unzipFiles(buffer: ArrayBuffer): Promise<Map<string, Uint8Array>> {
  const view = new DataView(buffer);
  const bytes = new Uint8Array(buffer);
  const eocdOffset = findEndOfCentralDirectory(view);
  const centralDirectoryOffset = view.getUint32(eocdOffset + 16, true);
  const entryCount = view.getUint16(eocdOffset + 10, true);
  const files = new Map<string, Uint8Array>();
  let offset = centralDirectoryOffset;

  for (let i = 0; i < entryCount; i++) {
    if (view.getUint32(offset, true) !== ZIP_CENTRAL_DIRECTORY_SIGNATURE) {
      throw new Error("Invalid ZIP/SOG: central directory entry is malformed.");
    }

    const compressionMethod = view.getUint16(offset + 10, true);
    const compressedSize = view.getUint32(offset + 20, true);
    const fileNameLength = view.getUint16(offset + 28, true);
    const extraLength = view.getUint16(offset + 30, true);
    const commentLength = view.getUint16(offset + 32, true);
    const localHeaderOffset = view.getUint32(offset + 42, true);
    const fileName = decodeZipFileName(bytes.subarray(offset + 46, offset + 46 + fileNameLength));
    const localNameLength = view.getUint16(localHeaderOffset + 26, true);
    const localExtraLength = view.getUint16(localHeaderOffset + 28, true);
    const dataOffset = localHeaderOffset + 30 + localNameLength + localExtraLength;
    const compressed = bytes.slice(dataOffset, dataOffset + compressedSize);

    if (view.getUint32(localHeaderOffset, true) !== ZIP_LOCAL_FILE_HEADER_SIGNATURE) {
      throw new Error(`Invalid ZIP/SOG: local file header for "${fileName}" is malformed.`);
    }

    if (!fileName.endsWith("/")) {
      files.set(fileName, await decompressZipEntry(compressed, compressionMethod));
    }

    offset += 46 + fileNameLength + extraLength + commentLength;
  }

  return files;
}

async function decompressZipEntry(compressed: Uint8Array, compressionMethod: number): Promise<Uint8Array> {
  if (compressionMethod === 0) {
    return compressed;
  }

  if (compressionMethod !== 8) {
    throw new Error(`Unsupported ZIP compression method "${compressionMethod}" in SOG.`);
  }

  if (typeof DecompressionStream === "undefined") {
    throw new Error("This browser does not support DecompressionStream, which is required for compressed .sog files.");
  }

  const stream = new Blob([toArrayBuffer(compressed)]).stream().pipeThrough(new DecompressionStream("deflate-raw"));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

function findEndOfCentralDirectory(view: DataView): number {
  const minOffset = Math.max(0, view.byteLength - 0xffff - 22);

  for (let offset = view.byteLength - 22; offset >= minOffset; offset--) {
    if (view.getUint32(offset, true) === ZIP_END_OF_CENTRAL_DIRECTORY_SIGNATURE) {
      return offset;
    }
  }

  throw new Error("Invalid ZIP/SOG: missing end of central directory.");
}

function decodeZipFileName(bytes: Uint8Array): string {
  return new TextDecoder().decode(bytes);
}

function toArrayBuffer(bytes: Uint8Array): ArrayBuffer {
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer;
}

async function decodeImage(bytes: Uint8Array, label: string): Promise<DecodedImage> {
  const image = await createImageBitmap(new Blob([toArrayBuffer(bytes)], { type: getImageMimeType(label) }), {
    colorSpaceConversion: "none",
  });
  const canvas = new OffscreenCanvas(image.width, image.height);
  const context = canvas.getContext("2d", {
    colorSpace: "srgb",
    willReadFrequently: true,
  });

  if (!context) {
    throw new Error(`Could not decode SOG image "${label}".`);
  }

  context.drawImage(image, 0, 0);
  const data = context.getImageData(0, 0, image.width, image.height).data;
  image.close();
  return {
    width: canvas.width,
    height: canvas.height,
    data,
  };
}

function getImageMimeType(label: string): string {
  if (label.toLowerCase().endsWith(".png")) {
    return "image/png";
  }

  return "image/webp";
}

function unpackSogCoordinate(lowByte: number, highByte: number, min: number, max: number): number {
  const quantized = (highByte << 8) | lowByte;
  return min + (max - min) * (quantized / 65535);
}

function unlogSogCoordinate(value: number): number {
  return Math.sign(value) * (Math.exp(Math.abs(value)) - 1);
}

function decodeSogScale(value: number | undefined): number {
  if (!Number.isFinite(value)) {
    return 1e-6;
  }

  const scale = value as number;
  return scale > 0 ? scale : Math.exp(scale);
}

function unpackSogQuaternion(a: number, b: number, c: number, mode: number): [number, number, number, number] {
  const values = [
    (a / 255 - 0.5) * 2 / Math.SQRT2,
    (b / 255 - 0.5) * 2 / Math.SQRT2,
    (c / 255 - 0.5) * 2 / Math.SQRT2,
  ];
  const missingComponent = mode - 252;

  if (missingComponent < 0 || missingComponent > 3) {
    throw new Error(`Invalid SOG quaternion mode "${mode}".`);
  }

  const sumSq = values[0] * values[0] + values[1] * values[1] + values[2] * values[2];
  const missingValue = Math.sqrt(Math.max(0, 1 - sumSq));
  const quat = [0, 0, 0, 0];
  let sourceIndex = 0;

  for (let i = 0; i < 4; i++) {
    if (i === missingComponent) {
      quat[i] = missingValue;
    } else {
      quat[i] = values[sourceIndex++];
    }
  }

  return [quat[0], quat[1], quat[2], quat[3]];
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
  const opacities = new Float32Array(header.vertexCount);
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
      opacities,
      covariances,
    );
  }

  return {
    positions,
    colors,
    opacities,
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
  const opacities = new Float32Array(header.vertexCount);
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
      opacities,
      covariances,
    );
  }

  return {
    positions,
    colors,
    opacities,
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
  opacities: Float32Array,
  covariances: Float32Array,
): void {
  const get = (name: string, fallback = 0): number => {
    const offset = propertyOffsets.get(name);
    const type = propertyTypes.get(name);

    return offset === undefined || !type
      ? fallback
      : readPlyScalar(view, rowOffset + offset, type);
  };

  writeParsedVertex(get, index, positions, colors, opacities, covariances);
}

function writeAsciiPlyVertex(
  values: number[],
  propertyIndices: Map<string, number>,
  index: number,
  positions: Float32Array,
  colors: Float32Array,
  opacities: Float32Array,
  covariances: Float32Array,
): void {
  const get = (name: string, fallback = 0): number => {
    const propertyIndex = propertyIndices.get(name);
    return propertyIndex === undefined ? fallback : values[propertyIndex];
  };

  writeParsedVertex(get, index, positions, colors, opacities, covariances);
}

function writeParsedVertex(
  get: (name: string, fallback?: number) => number,
  index: number,
  positions: Float32Array,
  colors: Float32Array,
  opacities: Float32Array,
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

  const opacity = get("opacity", Number.NaN);
  opacities[index] = Number.isFinite(opacity) ? sigmoid(opacity) : 1;

  writeCovariance(
    covariances,
    base6,
    Math.exp(get("scale_0")),
    Math.exp(get("scale_1")),
    Math.exp(get("scale_2")),
    get("rot_1"),
    get("rot_2"),
    get("rot_3"),
    get("rot_0", 1),
    true,
  );
}

function writeCovariance(
  covariances: Float32Array,
  base: number,
  scaleX: number,
  scaleY: number,
  scaleZ: number,
  quatX: number,
  quatY: number,
  quatZ: number,
  quatW: number,
  flipY: boolean,
): void {
  const sx = clampSplatScale(scaleX);
  const sy = clampSplatScale(scaleY);
  const sz = clampSplatScale(scaleZ);
  const length = Math.hypot(quatX, quatY, quatZ, quatW) || 1;
  const x = quatX / length;
  const y = quatY / length;
  const z = quatZ / length;
  const w = quatW / length;

  const xx = x * x;
  const yy = y * y;
  const zz = z * z;
  const xy = x * y;
  const xz = x * z;
  const yz = y * z;
  const wx = w * x;
  const wy = w * y;
  const wz = w * z;

  const r00 = 1 - 2 * (yy + zz);
  const r01 = 2 * (xy - wz);
  const r02 = 2 * (xz + wy);
  const r10 = 2 * (xy + wz);
  const r11 = 1 - 2 * (xx + zz);
  const r12 = 2 * (yz - wx);
  const r20 = 2 * (xz - wy);
  const r21 = 2 * (yz + wx);
  const r22 = 1 - 2 * (xx + yy);

  const vx = sx * sx;
  const vy = sy * sy;
  const vz = sz * sz;

  covariances[base] = r00 * r00 * vx + r01 * r01 * vy + r02 * r02 * vz;
  covariances[base + 1] = signedCovariance(
    r00 * r10 * vx + r01 * r11 * vy + r02 * r12 * vz,
    flipY,
  );
  covariances[base + 2] = r00 * r20 * vx + r01 * r21 * vy + r02 * r22 * vz;
  covariances[base + 3] = r10 * r10 * vx + r11 * r11 * vy + r12 * r12 * vz;
  covariances[base + 4] = signedCovariance(
    r10 * r20 * vx + r11 * r21 * vy + r12 * r22 * vz,
    flipY,
  );
  covariances[base + 5] = r20 * r20 * vx + r21 * r21 * vy + r22 * r22 * vz;
}

function signedCovariance(value: number, flipY: boolean): number {
  return flipY ? -value : value;
}

function clampSplatScale(value: number): number {
  return Math.max(1e-6, value || 1e-6);
}

function hasColorFields(
  get: (name: string, fallback?: number) => number,
): boolean {
  return (
    Number.isFinite(get("red", Number.NaN)) ||
    Number.isFinite(get("r", Number.NaN))
  );
}

function normalizeColor(value: number): number {
  return value > 1 ? value / 255 : value;
}

function clamp01(value: number): number {
  return Math.max(0, Math.min(1, value));
}

function sigmoid(value: number): number {
  return 1 / (1 + Math.exp(-value));
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
