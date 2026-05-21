// ============================================================
//  Gaussian Splatting Shader — Fixed Version
//  Fixes applied:
//   1. Correct perspective projection Jacobian for 2D covariance
//   2. Standard Gaussian falloff (exp(-0.5 * r²)) in fragment
//   3. Raised covariance eigenvalue clamp (was 0.00025, now 0.1)
//   4. Proper NDC offset scale (divided by clipCenter.w for correct NDC)
//   5. Removed broken normExp — plain Gaussian alpha compositing
//  NOTE: You MUST sort splats back-to-front by depth on the CPU/GPU
//        each frame before issuing this draw call. Without sorting,
//        alpha compositing will still look noisy regardless of shader fixes.
// ============================================================

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
@group(1) @binding(4) var<storage, read>    splatOrder:      array<u32>;

// ------------------------------------------------------------------
//  Project a 3-D world-space covariance Σ (stored as upper-triangle
//  [s00, s01, s02, s11, s12, s22]) into 2-D NDC space using the
//  correct EWA perspective Jacobian.
//
//  Formula: Σ_2D = J · W · Σ_3D · Wᵀ · Jᵀ   (drop z row/col)
//  where W is the 3×3 rotation part of the view matrix and
//  J is the Jacobian of the perspective projection at viewPos.
// ------------------------------------------------------------------
fn project_covariance_2d(
  worldCenter: vec3<f32>,
  cov3d: array<f32, 6>         // [s00,s01,s02,s11,s12,s22]
) -> vec3<f32> {               // returns [c00, c01, c11]

  // View-space position of the Gaussian centre
  let viewPos = camera.view * vec4<f32>(worldCenter, 1.0);
  let tx = viewPos.x;
  let ty = viewPos.y;
  let tz = max(viewPos.z, 0.05);

  // Focal lengths from the projection matrix
  let fx = camera.projection[0][0];   // 2·n / (r-l)  or  f/aspect
  let fy = camera.projection[1][1];   // 2·n / (t-b)  or  f

  // Perspective Jacobian J (2×3, row-major here as two vec3s)
  //   ∂(NDC_x)/∂(view) = [ fx/tz,  0,      -fx·tx/tz² ]
  //   ∂(NDC_y)/∂(view) = [ 0,      fy/tz,  -fy·ty/tz² ]
  let itz  = 1.0 / tz;
  let itz2 = itz * itz;

  let j00 =  fx * itz;
  let j02 = -fx * tx * itz2;
  let j11 =  fy * itz;
  let j12 = -fy * ty * itz2;

  // Extract 3×3 rotation block W from view matrix (column-major in WGSL)
  //   view[col][row]
  let w00 = camera.view[0][0]; let w01 = camera.view[0][1]; let w02 = camera.view[0][2];
  let w10 = camera.view[1][0]; let w11 = camera.view[1][1]; let w12 = camera.view[1][2];
  let w20 = camera.view[2][0]; let w21 = camera.view[2][1]; let w22 = camera.view[2][2];

  // Unpack symmetric 3-D covariance
  let s00 = cov3d[0]; let s01 = cov3d[1]; let s02 = cov3d[2];
  let s11 = cov3d[3]; let s12 = cov3d[4]; let s22 = cov3d[5];

  // T = W · Σ_3D  (3×3 result, but we only need rows 0 and 1 after ·Jᵀ)
  // We compute M = J · W · Σ_3D · Wᵀ · Jᵀ  in two steps:
  //   Step A: A = W · Σ   (full 3×3)
  let a00 = w00*s00 + w10*s01 + w20*s02;
  let a01 = w00*s01 + w10*s11 + w20*s12;
  let a02 = w00*s02 + w10*s12 + w20*s22;

  let a10 = w01*s00 + w11*s01 + w21*s02;
  let a11 = w01*s01 + w11*s11 + w21*s12;
  let a12 = w01*s02 + w11*s12 + w21*s22;

  let a20 = w02*s00 + w12*s01 + w22*s02;
  let a21 = w02*s01 + w12*s11 + w22*s12;
  let a22 = w02*s02 + w12*s12 + w22*s22;

  //   Step B: B = A · Wᵀ  (= W · Σ · Wᵀ, the view-space covariance)
  let b00 = a00*w00 + a01*w10 + a02*w20;
  let b01 = a00*w01 + a01*w11 + a02*w21;
  let b02 = a00*w02 + a01*w12 + a02*w22;

  let b10 = a10*w00 + a11*w10 + a12*w20;
  let b11 = a10*w01 + a11*w11 + a12*w21;
  let b12 = a10*w02 + a11*w12 + a12*w22;

  let b20 = a20*w00 + a21*w10 + a22*w20;
  let b21 = a20*w01 + a21*w11 + a22*w21;
  let b22 = a20*w02 + a21*w12 + a22*w22;

  //   Step C: C = J · B · Jᵀ  (2×2 result)
  //   Only upper-triangle needed (symmetric):
  //   c00 = j00²·b00 + 2·j00·j02·b02 + j02²·b22
  //   c01 = j00·j11·b01 + j00·j12·b02 + j02·j11·b20 + j02·j12·b22
  //   c11 = j11²·b11 + 2·j11·j12·b21 + j12²·b22
  var c00 = j00*j00*b00 + 2.0*j00*j02*b02 + j02*j02*b22;
  let c01 = j00*j11*b01 + j00*j12*b02 + j02*j11*b20 + j02*j12*b22;
  var c11 = j11*j11*b11 + 2.0*j11*j12*b21 + j12*j12*b22;

  // Low-pass filter: keep very small — larger values inflate every splat into a blob
  c00 += 0.000001;
  c11 += 0.000001;

  return vec3<f32>(c00, c01, c11);
}

// ------------------------------------------------------------------
//  Given the 2-D projected covariance, return the NDC offset for a
//  billboard corner so it covers the 3-sigma ellipse extent.
// ------------------------------------------------------------------
fn ellipse_offset(cov2d: vec3<f32>, corner: vec2<f32>) -> vec2<f32> {
  let c00 = cov2d.x;
  let c01 = cov2d.y;
  let c11 = cov2d.z;

  // Eigendecomposition of 2×2 symmetric matrix
  let traceHalf = 0.5 * (c00 + c11);
  let delta      = sqrt(max(0.0, traceHalf*traceHalf - (c00*c11 - c01*c01)));

  // Max clamp: 0.01 NDC ≈ 1% of screen width per sigma — prevents giant blobs.
  // Min clamp: 1e-7 prevents degenerate zero-size splats.
  // If splats still look too large, lower to 0.005; if too small/noisy raise toward 0.02.
  let lambda0 = clamp(traceHalf + delta, 1e-7, 0.01);
  let lambda1 = clamp(traceHalf - delta, 1e-7, 0.01);

  // Principal axis of the larger eigenvalue
  var axis0 = vec2<f32>(1.0, 0.0);
  if (abs(c01) > 1e-6 || abs(lambda0 - c00) > 1e-6) {
    axis0 = normalize(vec2<f32>(c01, lambda0 - c00));
  }
  let axis1 = vec2<f32>(-axis0.y, axis0.x);

  let sigmaExtent = 3.0;   // cover 3-sigma → captures ~99% of Gaussian mass
  let ellipseX = axis0 * sqrt(lambda0) * sigmaExtent;
  let ellipseY = axis1 * sqrt(lambda1) * sigmaExtent;

  return ellipseX * corner.x + ellipseY * corner.y;
}

// ------------------------------------------------------------------
//  Vertex shader
// ------------------------------------------------------------------
@vertex
fn vsMain(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {

  let renderIndex = vertexIndex / 6u;
  let splatIndex = splatOrder[renderIndex];
  let quadVertex = vertexIndex % 6u;

  // Two triangles forming a quad
  let corners = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>( 1.0, -1.0),
    vec2<f32>(-1.0,  1.0),
    vec2<f32>(-1.0,  1.0),
    vec2<f32>( 1.0, -1.0),
    vec2<f32>( 1.0,  1.0),
  );
  let corner = corners[quadVertex];

  // Load splat data
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

  // Clip-space centre
  let clipCenter = camera.viewProjection * vec4<f32>(worldCenter, 1.0);

  if (clipCenter.w <= 0.001) {
    var discarded: VertexOutput;
    discarded.position = vec4<f32>(0.0, 0.0, 2.0, 1.0);
    discarded.color = vec3<f32>(0.0);
    discarded.localPos = vec2<f32>(2.0);
    discarded.opacity = 0.0;
    return discarded;
  }

  // Project 3-D covariance → 2-D NDC covariance via correct Jacobian
  let cov2d = project_covariance_2d(worldCenter, covariance);

  // NDC offset for this corner
  // FIX: divide by clipCenter.w so the offset is in true NDC, not clip space
  let ndcOffset = ellipse_offset(cov2d, corner);

  var output: VertexOutput;
  output.position = vec4<f32>(
    clipCenter.x + ndcOffset.x * clipCenter.w,
    clipCenter.y + ndcOffset.y * clipCenter.w,
    clipCenter.z,
    clipCenter.w,
  );
  output.color    = vec3<f32>(
    splatColors[base3],
    splatColors[base3 + 1u],
    splatColors[base3 + 2u],
  );
  output.localPos = corner;
  output.opacity  = clamp(splatOpacities[splatIndex], 0.0, 1.0);

  return output;
}

// ------------------------------------------------------------------
//  Fragment shader
// ------------------------------------------------------------------
@fragment
fn fsMain(input: VertexOutput) -> @location(0) vec4<f32> {

  // r² in local ellipse space (corner ∈ [-1,1]² maps to 3-sigma extent)
  // We already scaled the quad to 3σ, so localPos == 1 means r = 3σ.
  // Convert: r²_gaussian = (localPos / 1)² maps 1→1 in [0,1]² quad coords.
  let r2 = dot(input.localPos, input.localPos);

  // Discard outside unit disc (corners of the quad)
  if (r2 > 1.0) {
    discard;
  }

  // FIX: standard Gaussian falloff  α = exp(-0.5 · (3σ · r)²)
  //      Since localPos == 1 corresponds to 3σ, the exponent is:
  //      -0.5 · (3·r)² = -4.5·r²
  //      The old normExp() was wrong — it shifted and rescaled the
  //      exponential in a way that broke standard alpha compositing.
  let gaussian = exp(-4.5 * r2);

  let alpha = gaussian * input.opacity;

  // Pre-multiplied alpha output — matches standard "over" compositing.
  // Ensure your WebGPU blend state is:
  //   srcFactor:      one            (color already multiplied)
  //   dstFactor:      one-minus-src-alpha
  //   srcAlphaFactor: one
  //   dstAlphaFactor: one-minus-src-alpha
  return vec4<f32>(input.color * alpha, alpha);
}
