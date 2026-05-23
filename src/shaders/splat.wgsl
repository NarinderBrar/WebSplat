
struct CameraUniforms {
  view:           mat4x4<f32>,
  projection:     mat4x4<f32>,
  viewProjection: mat4x4<f32>,
}

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0)       color:    vec3<f32>,
  @location(1)       localPos: vec2<f32>,
  @location(2)       opacity:  f32,
}

@group(0) @binding(0) var<uniform>          camera:          CameraUniforms;
@group(1) @binding(0) var<storage, read>    splatPositions:  array<f32>;
@group(1) @binding(1) var<storage, read>    splatColors:     array<f32>;
@group(1) @binding(2) var<storage, read>    splatCovariances: array<f32>;
@group(1) @binding(3) var<storage, read>    splatOpacities:  array<f32>;
@group(1) @binding(4) var<storage, read>    visibleSplatIndices: array<u32>;
@group(1) @binding(5) var<storage, read>    selectionMask: array<u32>;
@group(1) @binding(6) var<uniform>          renderSettings: vec4<f32>;

const EXP4 = exp(-4.0);
const INV_EXP4 = 1.0 / (1.0 - EXP4);

fn project_covariance_2d(
  worldCenter: vec3<f32>,
  cov3d: array<f32, 6>
) -> vec3<f32> {

  let viewPos = camera.view * vec4<f32>(worldCenter, 1.0);
  let tx = viewPos.x;
  let ty = viewPos.y;
  let tz = max(viewPos.z, 0.05);

  let fx = camera.projection[0][0];
  let fy = camera.projection[1][1];

  let itz  = 1.0 / tz;
  let itz2 = itz * itz;

  let j00 =  fx * itz;
  let j02 = -fx * tx * itz2;
  let j11 =  fy * itz;
  let j12 = -fy * ty * itz2;

  let w00 = camera.view[0][0]; let w01 = camera.view[0][1]; let w02 = camera.view[0][2];
  let w10 = camera.view[1][0]; let w11 = camera.view[1][1]; let w12 = camera.view[1][2];
  let w20 = camera.view[2][0]; let w21 = camera.view[2][1]; let w22 = camera.view[2][2];

  let s00 = cov3d[0]; let s01 = cov3d[1]; let s02 = cov3d[2];
  let s11 = cov3d[3]; let s12 = cov3d[4]; let s22 = cov3d[5];

  let a00 = w00*s00 + w10*s01 + w20*s02;
  let a01 = w00*s01 + w10*s11 + w20*s12;
  let a02 = w00*s02 + w10*s12 + w20*s22;

  let a10 = w01*s00 + w11*s01 + w21*s02;
  let a11 = w01*s01 + w11*s11 + w21*s12;
  let a12 = w01*s02 + w11*s12 + w21*s22;

  let a20 = w02*s00 + w12*s01 + w22*s02;
  let a21 = w02*s01 + w12*s11 + w22*s12;
  let a22 = w02*s02 + w12*s12 + w22*s22;

  let b00 = a00*w00 + a01*w10 + a02*w20;
  let b01 = a00*w01 + a01*w11 + a02*w21;
  let b02 = a00*w02 + a01*w12 + a02*w22;

  let b10 = a10*w00 + a11*w10 + a12*w20;
  let b11 = a10*w01 + a11*w11 + a12*w21;
  let b12 = a10*w02 + a11*w12 + a12*w22;

  let b20 = a20*w00 + a21*w10 + a22*w20;
  let b21 = a20*w01 + a21*w11 + a22*w21;
  let b22 = a20*w02 + a21*w12 + a22*w22;

  var c00 = j00*j00*b00 + 2.0*j00*j02*b02 + j02*j02*b22;
  let c01 = j00*j11*b01 + j00*j12*b02 + j02*j11*b20 + j02*j12*b22;
  var c11 = j11*j11*b11 + 2.0*j11*j12*b21 + j12*j12*b22;

  c00 += 0.000001;
  c11 += 0.000001;

  return vec3<f32>(c00, c01, c11);
}

fn ellipse_offset(cov2d: vec3<f32>, corner: vec2<f32>) -> vec2<f32> {
  let c00 = cov2d.x;
  let c01 = cov2d.y;
  let c11 = cov2d.z;


  let traceHalf = 0.5 * (c00 + c11);
  let delta      = sqrt(max(0.0, traceHalf*traceHalf - (c00*c11 - c01*c01)));

  let maxSplatVariance = renderSettings.y;
  let lambda0 = clamp(traceHalf + delta, 1e-8, maxSplatVariance);
  let lambda1 = clamp(traceHalf - delta, 1e-8, maxSplatVariance);

  var axis0 = vec2<f32>(1.0, 0.0);
  if (abs(c01) > 1e-6 || abs(lambda0 - c00) > 1e-6) {
    axis0 = normalize(vec2<f32>(c01, lambda0 - c00));
  }
  let axis1 = vec2<f32>(-axis0.y, axis0.x);

  let sigmaExtent = 3.0;
  let ellipseX = axis0 * sqrt(lambda0) * sigmaExtent;
  let ellipseY = axis1 * sqrt(lambda1) * sigmaExtent;

  return ellipseX * corner.x + ellipseY * corner.y;
}

@vertex
fn vsMain(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32
) -> VertexOutput {

  let splatIndex = visibleSplatIndices[instanceIndex];
  let quadVertex = vertexIndex;

  let corners = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>( 1.0, -1.0),
    vec2<f32>(-1.0,  1.0),
    vec2<f32>(-1.0,  1.0),
    vec2<f32>( 1.0, -1.0),
    vec2<f32>( 1.0,  1.0),
  );
  let corner = corners[quadVertex];

  let base3 = splatIndex * 3u;
  let worldCenter = vec3<f32>(
    splatPositions[base3],
    splatPositions[base3 + 1u],
    splatPositions[base3 + 2u],
  );

  let base6 = splatIndex * 6u;
  let covariance = array<f32, 6>(
    splatCovariances[base6],
    splatCovariances[base6 + 1u],
    splatCovariances[base6 + 2u],
    splatCovariances[base6 + 3u],
    splatCovariances[base6 + 4u],
    splatCovariances[base6 + 5u],
  );

  let clipCenter = camera.viewProjection * vec4<f32>(worldCenter, 1.0);

  if (clipCenter.w <= 0.001) {
    var discarded: VertexOutput;
    discarded.position = vec4<f32>(0.0, 0.0, 2.0, 1.0);
    discarded.color = vec3<f32>(0.0);
    discarded.localPos = vec2<f32>(2.0);
    discarded.opacity = 0.0;
    return discarded;
  }

  let cov2d = project_covariance_2d(worldCenter, covariance);

  let ndcOffset = ellipse_offset(cov2d, corner) * renderSettings.x;

  var output: VertexOutput;
  output.position = vec4<f32>(
    clipCenter.x + ndcOffset.x * clipCenter.w,
    clipCenter.y + ndcOffset.y * clipCenter.w,
    clipCenter.z,
    clipCenter.w,
  );
  let baseColor = vec3<f32>(
    splatColors[base3],
    splatColors[base3 + 1u],
    splatColors[base3 + 2u],
  );
  let selected = selectionMask[splatIndex] != 0u;
  output.color = select(baseColor, mix(baseColor, vec3<f32>(1.0, 0.86, 0.18), 0.65), selected);
  output.localPos = corner;
  output.opacity  = clamp(splatOpacities[splatIndex], 0.0, 1.0);

  return output;
}

@fragment
fn fsMain(input: VertexOutput) -> @location(0) vec4<f32> {
  let r2 = dot(input.localPos, input.localPos);


  if (r2 > 1.0) {
    discard;
  }

  let gaussian = (exp(-4.0 * r2) - EXP4) * INV_EXP4;

  let alpha = gaussian * input.opacity;

  if (alpha < 1.0 / 255.0) {
    discard;
  }

  return vec4<f32>(input.color * alpha, alpha);
}
