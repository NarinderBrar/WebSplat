struct CameraUniforms {
  view: mat4x4<f32>,
  projection: mat4x4<f32>,
  viewProjection: mat4x4<f32>,
}

struct ChunkCullParams {
  chunkCount: u32,
  splatCount: u32,
  viewportHeight: f32,
  lodEnabled: u32,
  minScreenRadius: f32,
  contributionThreshold: f32,
  padding0: u32,
  padding1: u32,
}

struct ChunkMetadata {
  boundsMinRadius: vec4<f32>,
  boundsMaxLod: vec4<f32>,
  centerCount: vec4<f32>,
  idStartOffsetFlags: vec4<u32>,
}

@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(1) @binding(0) var<storage, read> chunks: array<ChunkMetadata>;
@group(1) @binding(1) var<storage, read_write> visibleChunkFlags: array<u32>;
@group(1) @binding(2) var<storage, read_write> visibleChunkIndices: array<u32>;
@group(1) @binding(3) var<storage, read_write> visibleChunkCounter: atomic<u32>;
@group(1) @binding(4) var<storage, read_write> visibleSplatIndices: array<u32>;
@group(1) @binding(5) var<storage, read_write> visibleSplatCounter: atomic<u32>;
@group(1) @binding(6) var<storage, read_write> indirectArgs: array<atomic<u32>>;
@group(1) @binding(7) var<uniform> params: ChunkCullParams;

fn is_chunk_visible(chunk: ChunkMetadata) -> bool {
  let center = chunk.centerCount.xyz;
  let radius = chunk.boundsMinRadius.w;
  let clip = camera.viewProjection * vec4<f32>(center, 1.0);
  let extent = radius * max(abs(camera.projection[0][0]), abs(camera.projection[1][1]));

  if (clip.w <= 0.001) {
    return false;
  }

  return abs(clip.x) <= clip.w + extent &&
    abs(clip.y) <= clip.w + extent &&
    clip.z >= -extent &&
    clip.z <= clip.w + extent;
}

fn lod_step_for_chunk(chunk: ChunkMetadata) -> u32 {
  if (params.lodEnabled == 0u) {
    return 1u;
  }

  let viewPos = camera.view * vec4<f32>(chunk.centerCount.xyz, 1.0);
  let depth = max(viewPos.z, 0.001);
  let focalLength = camera.projection[1][1] * params.viewportHeight * 0.5;
  let screenRadius = chunk.boundsMinRadius.w * focalLength / depth;

  if (screenRadius < params.minScreenRadius * 2.0) {
    return 8u;
  }

  if (screenRadius < params.minScreenRadius * 4.0) {
    return 4u;
  }

  if (screenRadius < params.minScreenRadius * 8.0) {
    return 2u;
  }

  return 1u;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) globalId: vec3<u32>) {
  let chunkIndex = globalId.x;

  if (chunkIndex >= params.chunkCount) {
    return;
  }

  let chunk = chunks[chunkIndex];

  if (!is_chunk_visible(chunk)) {
    visibleChunkFlags[chunkIndex] = 0u;
    return;
  }

  visibleChunkFlags[chunkIndex] = 1u;
  let visibleChunkSlot = atomicAdd(&visibleChunkCounter, 1u);
  visibleChunkIndices[visibleChunkSlot] = chunkIndex;

  let splatStart = chunk.idStartOffsetFlags.y;
  let splatCount = u32(chunk.centerCount.w);
  let lodStep = lod_step_for_chunk(chunk);

  var localIndex = 0u;
  loop {
    if (localIndex >= splatCount) {
      break;
    }

    if (localIndex % lodStep == 0u) {
      let splatIndex = splatStart + localIndex;

      if (splatIndex < params.splatCount) {
        let visibleSplatSlot = atomicAdd(&visibleSplatCounter, 1u);
        visibleSplatIndices[visibleSplatSlot] = splatIndex;
        atomicAdd(&indirectArgs[1], 1u);
      }
    }

    localIndex += 1u;
  }
}

