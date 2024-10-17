@group(0) @binding(0) var mySampler : sampler;
@group(0) @binding(1) var myTexture : texture_2d<f32>;

struct VertexOutput {
  @builtin(position) Position : vec4f,
  @location(0) fragUV : vec2f,
  @location(1) @interpolate(flat) instance : u32,
}

struct VertexInput {
  @location(0) fragUV : vec2f,
  @location(1) @interpolate(flat) instance : u32,
}

@vertex
fn vertexMain(@builtin(vertex_index) VertexIndex : u32, @builtin(instance_index) instance: u32) -> VertexOutput {
  const pos = array(
    vec2( 1.0,  1.0),
    vec2( 1.0, -1.0),
    vec2(-1.0, -1.0),
    vec2( 1.0,  1.0),
    vec2(-1.0, -1.0),
    vec2(-1.0,  1.0),
  );

  const uv = array(
    vec2(1.0, 0.0),
    vec2(1.0, 1.0),
    vec2(0.0, 1.0),
    vec2(1.0, 0.0),
    vec2(0.0, 1.0),
    vec2(0.0, 0.0),
  );

  var output : VertexOutput;
  output.Position = vec4(pos[VertexIndex], 0.0, 1.0);
  output.fragUV = uv[VertexIndex];
  output.instance = instance;
  return output;
}

@fragment
fn fragmentMain(input: VertexInput) -> @location(0) vec4f {
  let r = f32(input.instance + 1);
  let color = textureSample(myTexture, mySampler, input.fragUV);
  return color;
  // return color / vec4(r, 1.0, 1.0, 1.0);
}