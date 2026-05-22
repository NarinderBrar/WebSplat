const BIN_COUNT = 512u;
const WORKGROUP_SIZE = 128u;

struct CameraUniforms {
  view: mat4x4<f32>,
  projection: mat4x4<f32>,
  viewProjection: mat4x4<f32>,
}

struct DepthBinParams {
  splatCount: u32,
  qualityMode: u32,
  nearDepth: f32,
  farDepth: f32,
  opacityThreshold: f32,
  lodNearDepth: f32,
  lodMidDepth: f32,
  lodFarDepth: f32,
  viewportHeight: f32,
  minPixelRadius: f32,
  contributionThreshold: f32,
  padding0: u32,
}

@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(1) @binding(0) var<storage, read> positions: array<f32>;
@group(1) @binding(1) var<storage, read> opacities: array<f32>;
@group(1) @binding(2) var<storage, read> covariances: array<f32>;
@group(1) @binding(3) var<storage, read_write> binCounters: array<atomic<u32>>;
@group(1) @binding(4) var<storage, read_write> binOffsets: array<u32>;
@group(1) @binding(5) var<storage, read_write> binCursors: array<atomic<u32>>;
@group(1) @binding(6) var<storage, read_write> visibleSplatIndices: array<u32>;
@group(1) @binding(7) var<storage, read_write> indirectArgs: array<atomic<u32>>;
@group(1) @binding(8) var<uniform> params: DepthBinParams;

fn splat_position(index: u32) -> vec3<f32> {
  let base = index * 3u;
  return vec3<f32>(positions[base], positions[base + 1u], positions[base + 2u]);
}

fn projected_radius(index: u32, depth: f32) -> f32 {
  let base = index * 6u;
  let maxVariance = max(max(abs(covariances[base]), abs(covariances[base + 3u])), abs(covariances[base + 5u]));
  let worldRadius = sqrt(max(maxVariance, 1e-8));
  let focalLength = camera.projection[1][1] * params.viewportHeight * 0.5;
  return worldRadius * focalLength / max(depth, 0.001);
}

fn depth_bin(depth: f32) -> u32 {
  let normalized = clamp((depth - params.nearDepth) / max(0.001, params.farDepth - params.nearDepth), 0.0, 1.0);
  let nearToFar = u32(normalized * f32(BIN_COUNT - 1u));
  return (BIN_COUNT - 1u) - nearToFar;
}

fn lod_step(depth: f32, radiusPx: f32) -> u32 {
  if (params.qualityMode == 3u) {
    if (radiusPx < 1.2) {
      return 8u;
    }

    if (radiusPx < 2.0) {
      return 4u;
    }

    if (depth > params.lodFarDepth) {
      return 8u;
    }

    if (depth > params.lodMidDepth) {
      return 4u;
    }

    if (depth > params.lodNearDepth) {
      return 2u;
    }
  }

  if (params.qualityMode == 2u) {
    if (radiusPx < 0.8) {
      return 4u;
    }

    if (radiusPx < 1.4) {
      return 2u;
    }

    if (depth > params.lodFarDepth) {
      return 4u;
    }

    if (depth > params.lodMidDepth) {
      return 2u;
    }
  }

  return 1u;
}

fn is_visible_splat(index: u32, depth: f32, clip: vec4<f32>) -> bool {
  if (clip.w <= 0.001) {
    return false;
  }

  if (abs(clip.x) > clip.w || abs(clip.y) > clip.w || clip.z < 0.0 || clip.z > clip.w) {
    return false;
  }

  let opacity = opacities[index];
  if (opacity < params.opacityThreshold) {
    return false;
  }

  let radiusPx = projected_radius(index, depth);
  if (radiusPx < params.minPixelRadius) {
    return false;
  }

  if (opacity * radiusPx * radiusPx < params.contributionThreshold) {
    return false;
  }

  let step = lod_step(depth, radiusPx);
  return step == 1u || (index % step) == 0u;
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn clearBins(@builtin(global_invocation_id) globalId: vec3<u32>) {
  let index = globalId.x;

  if (index < BIN_COUNT) {
    atomicStore(&binCounters[index], 0u);
    atomicStore(&binCursors[index], 0u);
    binOffsets[index] = 0u;
  }

  if (index == 0u) {
    atomicStore(&indirectArgs[0], 6u);
    atomicStore(&indirectArgs[1], 0u);
    atomicStore(&indirectArgs[2], 0u);
    atomicStore(&indirectArgs[3], 0u);
  }
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn countBins(@builtin(global_invocation_id) globalId: vec3<u32>) {
  let index = globalId.x;

  if (index >= params.splatCount) {
    return;
  }

  let world = splat_position(index);
  let view = camera.view * vec4<f32>(world, 1.0);
  let clip = camera.viewProjection * vec4<f32>(world, 1.0);
  let depth = max(view.z, 0.0);

  if (!is_visible_splat(index, depth, clip)) {
    return;
  }

  atomicAdd(&binCounters[depth_bin(depth)], 1u);
}

@compute @workgroup_size(1)
fn prefixBins() {
  var cursor = 0u;

  for (var bin = 0u; bin < BIN_COUNT; bin = bin + 1u) {
    binOffsets[bin] = cursor;
    let count = atomicLoad(&binCounters[bin]);
    cursor = cursor + count;
  }

  atomicStore(&indirectArgs[0], 6u);
  atomicStore(&indirectArgs[1], cursor);
  atomicStore(&indirectArgs[2], 0u);
  atomicStore(&indirectArgs[3], 0u);
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn fillBins(@builtin(global_invocation_id) globalId: vec3<u32>) {
  let index = globalId.x;

  if (index >= params.splatCount) {
    return;
  }

  let world = splat_position(index);
  let view = camera.view * vec4<f32>(world, 1.0);
  let clip = camera.viewProjection * vec4<f32>(world, 1.0);
  let depth = max(view.z, 0.0);

  if (!is_visible_splat(index, depth, clip)) {
    return;
  }

  let bin = depth_bin(depth);
  let slot = atomicAdd(&binCursors[bin], 1u);
  let dst = binOffsets[bin] + slot;
  let step = lod_step(depth, projected_radius(index, depth));
  visibleSplatIndices[dst] = index;
}
