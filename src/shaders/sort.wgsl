
struct CameraUniforms {
  view: mat4x4<f32>,
  projection: mat4x4<f32>,
  viewProjection: mat4x4<f32>,
}

struct SortConfig {
  count: u32,
  byteIndex: u32,
  _pad: vec2<u32>,
}

@group(0) @binding(0) var<uniform> camera: CameraUniforms;

@group(1) @binding(0) var<storage, read> positions: array<f32>;
@group(1) @binding(1) var<storage, read_write> keys: array<u32>;
@group(1) @binding(2) var<storage, read_write> values: array<u32>;
@group(1) @binding(3) var<storage, read_write> tempKeys: array<u32>;
@group(1) @binding(4) var<storage, read_write> tempValues: array<u32>;
@group(1) @binding(5) var<storage, read_write> histogram: array<atomic<u32>>;
@group(1) @binding(6) var<storage, read_write> indirectArgs: array<u32>;
@group(1) @binding(7) var<uniform> config: SortConfig;

@compute @workgroup_size(256)
fn csBuildKeys(@builtin(global_invocation_id) globalId: vec3<u32>) {
  let id = globalId.x;
  if (id >= config.count) { return; }

  let base = id * 3u;
  let worldPos = vec3<f32>(positions[base], positions[base + 1u], positions[base + 2u]);
  let viewPos = camera.view * vec4<f32>(worldPos, 1.0);
  let depth = max(-viewPos.z, 0.0);

  keys[id] = bitcast<u32>(depth);
  values[id] = id;
}

@compute @workgroup_size(256)
fn csClearHistogram(@builtin(global_invocation_id) globalId: vec3<u32>) {
  let id = globalId.x;
  if (id >= 256u) { return; }
  atomicStore(&histogram[id], 0u);
}

@compute @workgroup_size(256)
fn csHistogram(@builtin(global_invocation_id) globalId: vec3<u32>) {
  let id = globalId.x;
  if (id >= config.count) { return; }

  let key = keys[id];
  let digit = (key >> (config.byteIndex * 8u)) & 0xFFu;
  atomicAdd(&histogram[digit], 1u);
}

@compute @workgroup_size(1)
fn csPrefix() {
  var sum = 0u;
  for (var i = 0u; i < 256u; i = i + 1u) {
    let count = atomicLoad(&histogram[i]);
    atomicStore(&histogram[i], sum);
    sum = sum + count;
  }
}

@compute @workgroup_size(256)
fn csScatter(@builtin(global_invocation_id) globalId: vec3<u32>) {
  let id = globalId.x;
  if (id >= config.count) { return; }

  let key = keys[id];
  let digit = (key >> (config.byteIndex * 8u)) & 0xFFu;
  let dst = atomicAdd(&histogram[digit], 1u);

  tempKeys[dst] = key;
  tempValues[dst] = values[id];
}

@compute @workgroup_size(1)
fn csUpdateIndirectArgs() {
  indirectArgs[0] = 6u;
  indirectArgs[1] = config.count;
  indirectArgs[2] = 0u;
  indirectArgs[3] = 0u;
}
