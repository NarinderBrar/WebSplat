struct SplatVertex {
  @builtin(vertex_index) vertexIndex: u32,
}

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
}

@vertex
fn vsMain(input: SplatVertex) -> VertexOutput {
  var output: VertexOutput;
  let triangle = array<vec2<f32>, 3>(
    vec2<f32>( 0.0,  0.5),
    vec2<f32>(-0.5, -0.5),
    vec2<f32>( 0.5, -0.5),
  );
  output.position = vec4<f32>(triangle[input.vertexIndex], 0.0, 1.0);
  return output;
}

@fragment
fn fsMain() -> @location(0) vec4<f32> {
  return vec4<f32>(0.2, 0.85, 1.0, 1.0);
}
