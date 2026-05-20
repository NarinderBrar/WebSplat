struct CameraUniforms {
  view: mat4x4<f32>,
  projection: mat4x4<f32>,
  viewProjection: mat4x4<f32>,
}

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) color: vec3<f32>,
}

@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(1) @binding(0) var<storage, read> splatPositions: array<f32>;
@group(1) @binding(1) var<storage, read> splatColors: array<f32>;

@vertex
fn vsMain(@builtin(vertex_index) splatIndex: u32) -> VertexOutput {
  let base = splatIndex * 3u;
  let worldPosition = vec4<f32>(
    splatPositions[base],
    splatPositions[base + 1u],
    splatPositions[base + 2u],
    1.0,
  );

  var output: VertexOutput;
  output.position = camera.viewProjection * worldPosition;
  output.color = vec3<f32>(
    splatColors[base],
    splatColors[base + 1u],
    splatColors[base + 2u],
  );
  return output;
}

@fragment
fn fsMain(input: VertexOutput) -> @location(0) vec4<f32> {
  return vec4<f32>(input.color, 1.0);
}
