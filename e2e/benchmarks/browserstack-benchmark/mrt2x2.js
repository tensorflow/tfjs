'use strict';

/* Parameters */
// Square packing & continuous 2x2 MRT


function getMaxDrawBuffers() {
  const { gl, program, getRes } = mrt2x2Program(4, 4, 4);
  program();
  const res = getRes(-1).join();
  const expectedRes
    = '84,94,124,142,124,134,196,214,244,286,284,334,412,454,484,534';

  if (res !== expectedRes) {
    throw new Error(`Got res ${res}.`);
  }
  return gl.getParameter(gl.MAX_DRAW_BUFFERS);
}

const vs = `#version 300 es
#define POSITION_LOCATION 0

precision highp float;
precision highp int;

layout(location = POSITION_LOCATION) in vec2 position;

void main()
{
    gl_Position = vec4(position, 0.0, 1.0);
}`;

const fs2x2 = `#version 300 es
layout(location = 0) out highp vec4 result_00;
layout(location = 1) out highp vec4 result_01;
layout(location = 2) out highp vec4 result_10;
layout(location = 3) out highp vec4 result_11;
uniform highp sampler2DArray src_tensor_tex2d;
uniform highp sampler2DArray weights_tex2d;
uniform highp int src_tensor_logical_tex2d_width;
uniform highp int output_logical_tex2d_width;
uniform highp int src_tensor_tex2d_width;
uniform highp int weights_tex2d_width;
uniform highp int output_tex2d_width;
precision highp int;
#define MOD_MACRO(a, b) ((a) % (b))
#define FLOAT4 highp vec4
#define INIT_ACCUM_FLOAT4(value) vec4(value)
#define ACCUM_FLOAT4 highp vec4

void main() {
  int oLinearIndex = int(gl_FragCoord.y) * output_tex2d_width + int(gl_FragCoord.x);
  int y = oLinearIndex / output_logical_tex2d_width;
  int x = oLinearIndex - y * output_logical_tex2d_width;

  ACCUM_FLOAT4 res_00 = INIT_ACCUM_FLOAT4(0.0);
  ACCUM_FLOAT4 res_01 = INIT_ACCUM_FLOAT4(0.0);
  ACCUM_FLOAT4 res_10 = INIT_ACCUM_FLOAT4(0.0);
  ACCUM_FLOAT4 res_11 = INIT_ACCUM_FLOAT4(0.0);

  for (int ic = 0; ic < src_tensor_logical_tex2d_width; ++ic) {  // params as iC/4
    // Logical texture coords to physical texture coords.
    int aLinearIndex = y * src_tensor_logical_tex2d_width + ic;
    int ay = aLinearIndex / src_tensor_tex2d_width;
    int ax = aLinearIndex - ay * src_tensor_tex2d_width;
    FLOAT4 a_00 = texelFetch(src_tensor_tex2d, ivec3(ax, ay, 0), 0);
    FLOAT4 a_01 = texelFetch(src_tensor_tex2d, ivec3(ax, ay, 1), 0);
    FLOAT4 a_10 = texelFetch(src_tensor_tex2d, ivec3(ax, ay, 2), 0);
    FLOAT4 a_11 = texelFetch(src_tensor_tex2d, ivec3(ax, ay, 3), 0);

    // weights_logical_tex2d_width === output_logical_tex2d_width
    int bLinearIndex = ic * output_logical_tex2d_width + x;
    int by = bLinearIndex / weights_tex2d_width;
    int bx = bLinearIndex - by * weights_tex2d_width;
    FLOAT4 b_00 = texelFetch(weights_tex2d, ivec3(bx, by, 0), 0);
    FLOAT4 b_01 = texelFetch(weights_tex2d, ivec3(bx, by, 1), 0);
    FLOAT4 b_10 = texelFetch(weights_tex2d, ivec3(bx, by, 2), 0);
    FLOAT4 b_11 = texelFetch(weights_tex2d, ivec3(bx, by, 3), 0);

    FLOAT4 a_row_0 = vec4(a_00.xy, a_01.xy);
    FLOAT4 a_row_1 = vec4(a_00.zw, a_01.zw);
    FLOAT4 a_row_2 = vec4(a_10.xy, a_11.xy);
    FLOAT4 a_row_3 = vec4(a_10.zw, a_11.zw);

    FLOAT4 b_col_0 = vec4(b_00.xz, b_10.xz);
    FLOAT4 b_col_1 = vec4(b_00.yw, b_10.yw);
    FLOAT4 b_col_2 = vec4(b_01.xz, b_11.xz);
    FLOAT4 b_col_3 = vec4(b_01.yw, b_11.yw);

    res_00.x += dot(a_row_0, b_col_0);
    res_00.y += dot(a_row_0, b_col_1);
    res_00.z += dot(a_row_1, b_col_0);
    res_00.w += dot(a_row_1, b_col_1);

    res_01.x += dot(a_row_0, b_col_2);
    res_01.y += dot(a_row_0, b_col_3);
    res_01.z += dot(a_row_1, b_col_2);
    res_01.w += dot(a_row_1, b_col_3);

    res_10.x += dot(a_row_2, b_col_0);
    res_10.y += dot(a_row_2, b_col_1);
    res_10.z += dot(a_row_3, b_col_0);
    res_10.w += dot(a_row_3, b_col_1);

    res_11.x += dot(a_row_2, b_col_2);
    res_11.y += dot(a_row_2, b_col_3);
    res_11.z += dot(a_row_3, b_col_2);
    res_11.w += dot(a_row_3, b_col_3);
  }
  result_00 = res_00;
  result_01 = res_01;
  result_10 = res_10;
  result_11 = res_11;
}`;

function mrt2x2Program(SHARED_DIM, OUTPUT_HEIGHT, OUTPUT_WIDTH) {
  function getTextureshape(logicalShape) {
    if (Math.max(...logicalShape) > 40000) {
      const texelNum = Math.ceil(size(logicalShape) / 16);
      const width = 100;
      return [width, Math.ceil(texelNum / width)];
    } else {
      // Align with the logical shape
      return logicalShape
        .map(e => Math.ceil(e / 4)) // MRT/2, pack/2
        .reverse(); // width first
    }
  }

  const OUTPUT_SHAPE = [OUTPUT_HEIGHT, OUTPUT_WIDTH];
  const A_SHAPE = [OUTPUT_HEIGHT, SHARED_DIM];
  const B_SHAPE = [SHARED_DIM, OUTPUT_WIDTH];
  const OUTPUT_TEXTURE_SHAPE = getTextureshape(OUTPUT_SHAPE);
  const A_TEXTURE_SHAPE = getTextureshape(A_SHAPE);
  const B_TEXTURE_SHAPE = getTextureshape(B_SHAPE);

  // Uniforms
  const SRC_TENSOR_LOGICAL_TEX2D_WIDTH = Math.ceil(SHARED_DIM / 4);
  const OUTPUT_LOGICAL_TEX2D_WIDTH = Math.ceil(OUTPUT_WIDTH / 4);
  const SRC_TENSOR_TEX2D_WIDTH = A_TEXTURE_SHAPE[0];
  const WEIGHTS_TEX2D_WIDTH = B_TEXTURE_SHAPE[0];
  const OUTPUT_TEX2D_WIDTH = OUTPUT_TEXTURE_SHAPE[0];

  const canvas = document.createElement('canvas');

  const gl = canvas.getContext('webgl2', { antialias: false });
  if (gl == null) {
    throw new Error('WebGL2 is unavailable!');
  }

  // enable float for color frame buffer
  gl.getExtension('EXT_color_buffer_float');

  // -- Init program
  const webglProgram = createProgram(gl, vs, fs2x2);
  // map inputs and params for fragment shader
  const srcLocation = gl.getUniformLocation(webglProgram, 'src_tensor_tex2d');
  const weightLocation = gl.getUniformLocation(webglProgram, 'weights_tex2d');
  const shared0Location = gl.getUniformLocation(webglProgram, 'src_tensor_logical_tex2d_width');
  const shared1Location = gl.getUniformLocation(webglProgram, 'output_logical_tex2d_width');
  const shared2Location = gl.getUniformLocation(webglProgram, 'src_tensor_tex2d_width');
  const shared3Location = gl.getUniformLocation(webglProgram, 'weights_tex2d_width');
  const shared4Location = gl.getUniformLocation(webglProgram, 'output_tex2d_width');

  // -- Init buffers
  const positions = new Float32Array([
    -1.0, -1.0,
    1.0, -1.0,
    1.0, 1.0,
    1.0, 1.0,
    -1.0, 1.0,
    -1.0, -1.0
  ]);
  const vertexPosBuffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, vertexPosBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);
  gl.bindBuffer(gl.ARRAY_BUFFER, null);

  // -- Init VertexArray
  const vertexArray = gl.createVertexArray();
  gl.bindVertexArray(vertexArray);

  const vertexPosLocation = 0; // set with GLSL layout qualifier
  gl.enableVertexAttribArray(vertexPosLocation);
  gl.bindBuffer(gl.ARRAY_BUFFER, vertexPosBuffer);
  gl.vertexAttribPointer(vertexPosLocation, 2, gl.FLOAT, false, 0, 0);
  gl.bindBuffer(gl.ARRAY_BUFFER, null);

  gl.bindVertexArray(null);

  // create input data array
  const tensor = new Float32Array(Array.from(Array(size(A_SHAPE)).keys()));
  const filter = new Float32Array(Array.from(Array(size(B_SHAPE)).keys()));
  // -- Init input Textures
  const inputTexture = createTextureArray(gl, ...A_TEXTURE_SHAPE, 4, 0, tensor);
  const filterTexture = createTextureArray(gl, ...B_TEXTURE_SHAPE, 4, 1, filter);
  // gl.bindTexture(gl.TEXTURE_2D, null);

  // init output texture
  gl.activeTexture(gl.TEXTURE2);
  var output = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D_ARRAY, output);
  // gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_BASE_LEVEL, 0);
  // gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MAX_LEVEL, 0);
  gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texStorage3D(gl.TEXTURE_2D_ARRAY, 1, gl.RGBA32F, ...OUTPUT_TEXTURE_SHAPE, 4);
  // gl.bindTexture(gl.TEXTURE_2D_ARRAY, null);

  // attach output textures to color buffers
  var frameBuffer = gl.createFramebuffer();
  gl.bindFramebuffer(gl.FRAMEBUFFER, frameBuffer);

  gl.framebufferTextureLayer(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, output, 0, 0);
  gl.framebufferTextureLayer(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT1, output, 0, 1);
  gl.framebufferTextureLayer(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT2, output, 0, 2);
  gl.framebufferTextureLayer(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT3, output, 0, 3);
  var status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
  if (status != gl.FRAMEBUFFER_COMPLETE) {
    throw new Error('fb status: ' + status);
  }

  gl.drawBuffers([
    gl.COLOR_ATTACHMENT0,
    gl.COLOR_ATTACHMENT1,
    gl.COLOR_ATTACHMENT2,
    gl.COLOR_ATTACHMENT3
  ]);
  //gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, null);

  // run rendering multiple times to benchmark

  function program() {
    // -- Render
    // gl.clearColor(1.0, 1.0, 1.0, 1.0);
    // gl.clear(gl.COLOR_BUFFER_BIT);
    gl.useProgram(webglProgram);
    gl.bindVertexArray(vertexArray);
    gl.viewport(0, 0, ...OUTPUT_TEXTURE_SHAPE);
    gl.scissor(0, 0, ...OUTPUT_TEXTURE_SHAPE);

    // init uniforms
    gl.uniform1i(shared0Location, SRC_TENSOR_LOGICAL_TEX2D_WIDTH);
    gl.uniform1i(shared1Location, OUTPUT_LOGICAL_TEX2D_WIDTH);
    gl.uniform1i(shared2Location, SRC_TENSOR_TEX2D_WIDTH);
    gl.uniform1i(shared3Location, WEIGHTS_TEX2D_WIDTH);
    gl.uniform1i(shared4Location, OUTPUT_TEX2D_WIDTH);
    gl.uniform1i(srcLocation, 0);
    gl.uniform1i(weightLocation, 1);

    // start the computation
    gl.drawArrays(gl.TRIANGLES, 0, 6);
  }

  function getRes(num) {
    if (num === -1) {
      return [
        readDataFromRGBATexture2DArray(gl, output, ...OUTPUT_TEXTURE_SHAPE, 0, frameBuffer),
        readDataFromRGBATexture2DArray(gl, output, ...OUTPUT_TEXTURE_SHAPE, 1, frameBuffer),
        readDataFromRGBATexture2DArray(gl, output, ...OUTPUT_TEXTURE_SHAPE, 2, frameBuffer),
        readDataFromRGBATexture2DArray(gl, output, ...OUTPUT_TEXTURE_SHAPE, 3, frameBuffer),
      ];
    } else {
      return readDataFromRGBATexture2DArray(gl, output, num, num, 0, frameBuffer);
    }
  }
  return { gl, program, getRes };
}

/* ******************** Helper functions ******************** */
function size(shape) {
  return shape.reduce((c, x) => c *= x, 1);
}

function createTextureArray(gl, width, height, layers, unit, inputs) {
  var texture = gl.createTexture();
  gl.activeTexture(gl.TEXTURE0 + unit);
  gl.bindTexture(gl.TEXTURE_2D_ARRAY, texture);
  gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texImage3D(
    gl.TEXTURE_2D_ARRAY,
    0,
    gl.RGBA32F,
    width,
    height,
    layers,
    0,
    gl.RGBA,
    gl.FLOAT,
    inputs
  );
  // gl.bindTexture(gl.TEXTURE_2D_ARRAY, null);
  return texture;
}

function readDataFromRGBATexture2DArray(gl, texture, width, height, layer, buffer) {
  gl.bindFramebuffer(gl.FRAMEBUFFER, buffer);
  gl.readBuffer(gl.COLOR_ATTACHMENT0 + layer);
  const data = new Float32Array(width * height * 4);
  gl.readPixels(0, 0, width, height, gl.RGBA, gl.FLOAT, data);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  return data;
}


function createShader(gl, source, type) {
  var shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  return shader;
}

function createProgram(gl, vertexShaderSource, fragmentShaderSource) {
  var program = gl.createProgram();
  var vshader = createShader(gl, vertexShaderSource, gl.VERTEX_SHADER);
  var fshader = createShader(gl, fragmentShaderSource, gl.FRAGMENT_SHADER);
  gl.attachShader(program, vshader);
  gl.deleteShader(vshader);
  gl.attachShader(program, fshader);
  gl.deleteShader(fshader);
  gl.linkProgram(program);

  var log = gl.getProgramInfoLog(program);
  if (log) {
    console.log(log);
  }

  log = gl.getShaderInfoLog(vshader);
  if (log) {
    console.log(log);
  }

  log = gl.getShaderInfoLog(fshader);
  if (log) {
    console.log(log);
  }

  return program;
};
