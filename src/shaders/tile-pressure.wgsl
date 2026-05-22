struct CameraMatrices {
  view: mat4x4<f32>,
  projection: mat4x4<f32>,
  viewProjection: mat4x4<f32>,
};

struct TilePressureParams {
  visibleCount: u32,
  tileCount: u32,
  tilesX: u32,
  tilesY: u32,
  tileSize: f32,
  viewportWidth: f32,
  viewportHeight: f32,
  overloadThreshold: u32,
};

@group(0) @binding(0) var<uniform> camera: CameraMatrices;

@group(1) @binding(0) var<storage, read> positions: array<vec3<f32>>;
@group(1) @binding(1) var<storage, read> visibleSplatIndices: array<u32>;
@group(1) @binding(2) var<storage, read_write> tileCounts: array<atomic<u32>>;
@group(1) @binding(3) var<storage, read_write> summary: array<atomic<u32>>;
@group(1) @binding(4) var<uniform> params: TilePressureParams;

@compute @workgroup_size(128)
fn clearTiles(@builtin(global_invocation_id) globalId: vec3<u32>) {
  let index = globalId.x;

  if (index < params.tileCount) {
    atomicStore(&tileCounts[index], 0u);
  }

  if (index < 4u) {
    atomicStore(&summary[index], 0u);
  }
}

@compute @workgroup_size(128)
fn countTiles(@builtin(global_invocation_id) globalId: vec3<u32>) {
  let visibleIndex = globalId.x;

  if (visibleIndex >= params.visibleCount) {
    return;
  }

  let splatIndex = visibleSplatIndices[visibleIndex];
  let position = vec4<f32>(positions[splatIndex], 1.0);
  let clip = camera.viewProjection * position;

  if (clip.w <= 0.001) {
    return;
  }

  let ndc = clip.xy / clip.w;

  if (ndc.x < -1.0 || ndc.x > 1.0 || ndc.y < -1.0 || ndc.y > 1.0) {
    return;
  }

  let pixelX = (ndc.x * 0.5 + 0.5) * params.viewportWidth;
  let pixelY = (0.5 - ndc.y * 0.5) * params.viewportHeight;
  let tileX = min(params.tilesX - 1u, u32(pixelX / params.tileSize));
  let tileY = min(params.tilesY - 1u, u32(pixelY / params.tileSize));
  let tileIndex = tileY * params.tilesX + tileX;
  let previous = atomicAdd(&tileCounts[tileIndex], 1u);
  let nextCount = previous + 1u;

  atomicAdd(&summary[0], 1u);
  atomicMax(&summary[1], nextCount);

  if (nextCount == params.overloadThreshold + 1u) {
    atomicAdd(&summary[2], 1u);
  }
}
