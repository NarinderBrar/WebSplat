struct CameraUniforms {
  view: mat4x4<f32>,
  projection: mat4x4<f32>,
  viewProjection: mat4x4<f32>,
}

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) localPosition: vec2<f32>,
}

@group(0) @binding(0) var<uniform> camera: CameraUniforms;

@vertex
fn vsMain(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
  let planeSize = 80.0;
  let vertices = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>( 1.0, -1.0),
    vec2<f32>(-1.0,  1.0),
    vec2<f32>(-1.0,  1.0),
    vec2<f32>( 1.0, -1.0),
    vec2<f32>( 1.0,  1.0),
  );
  let local = vertices[vertexIndex] * planeSize;
  let worldPosition = vec4<f32>(local.x, 0.0, local.y, 1.0);

  var output: VertexOutput;
  output.position = camera.viewProjection * worldPosition;
  output.localPosition = local;
  return output;
}

@fragment
fn fsMain(input: VertexOutput) -> @location(0) vec4<f32> {
  let gridX = 1.0 - smoothstep(0.0, 0.035, abs(fract(input.localPosition.x * 0.1) - 0.5));
  let gridY = 1.0 - smoothstep(0.0, 0.035, abs(fract(input.localPosition.y * 0.1) - 0.5));
  let grid = max(gridX, gridY);
  let base = vec3<f32>(0.07, 0.095, 0.105);
  let line = vec3<f32>(0.18, 0.36, 0.34);

  return vec4<f32>(mix(base, line, grid), 1.0);
}
