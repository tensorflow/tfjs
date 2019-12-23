/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

interface Dimensions {
  x?: number;
  y?: number;
  width: number;
  height: number;
  depth: number;
}

export function vertexShaderSource() {
  return `#version 300 es
precision highp float;
precision highp int;

in vec2 position;
in vec2 texCoords;
out vec2 uv;

void main() {
  uv = texCoords;
  gl_Position = vec4(position, 0, 1);
}`;
}

export function fragmentShaderSource(
    sourceDims: Dimensions, targetDims: Dimensions, alignCorners: boolean) {
  const outputFragType = targetDims.depth === 3 ? 'vec3' : 'vec4';
  const outputFragColor = targetDims.depth === 3 ?
      'vec3(texSample.r,texSample.g,texSample.b)' :
      'texSample';

  const sWidth = sourceDims.height;
  const sHeight = sourceDims.height;
  const tWidth = targetDims.height;
  const tHeight = targetDims.height;

  const effectiveInSize: [number, number] = [
    (alignCorners && tWidth > 1) ? sWidth - 1 : sWidth,
    (alignCorners && tHeight > 1) ? sHeight - 1 : sHeight,
  ];

  const effectiveOutSize: [number, number] = [
    (alignCorners && tWidth > 1) ? tWidth - 1 : tWidth,
    (alignCorners && tHeight > 1) ? tHeight - 1 : tHeight,
  ];

  const roundBase = alignCorners ? '0.5' : '0.0';
  const mainFunc =
      getResizeNearestNeighborMain(targetDims, roundBase, outputFragColor);

  const source = `#version 300 es
precision highp float;
precision highp int;

uniform sampler2D inputTexture;
in vec2 uv;
out ${outputFragType} fragColor;

vec2 effectiveInputOverOutputRatioRC = vec2(
          ${effectiveInSize[0] / effectiveOutSize[0]},
          ${effectiveInSize[1] / effectiveOutSize[1]});
vec2 inputShapeRC = vec2(${sourceDims.width}.0, ${sourceDims.height}.0);
${mainFunc}
`;

  return source;
}


// vec4 texSample =
//     texture(inputTexture, vec2(sourceNearestRC.x, sourceNearestRC.y));

// // Fractional source index.
// vec2 sourceFracIndexRC = uv * effectiveInputOverOutputRatioRC;

// // Compute the coordinates of nearest neighbor point.
// ivec2 sourceNearestRC =
//     ivec2(min(inputShapeRC - 1.0, floor(sourceFracIndexRC + ${roundBase})));

// vec2 sourceFracIndexRC = uv * effectiveInputOverOutputRatioRC;

// vec2 texelLoc = inputShapeRC * st;
// // vec2 sourceNearestRC =
// //   min(inputShapeRC - 1.0, floor(sourceFracIndexRC + ${roundBase}));

// // vec4 texSample = texelFetch(inputTexture, ivec2(texelLoc), 0);

function getResizeNearestNeighborMain(
    targetDims: Dimensions, roundBase: string, outputFragColor: string) {
  const tWidth = targetDims.width;
  const tHeight = targetDims.height;

  return `
void main() {
  vec2 texUV = gl_FragCoord.xy / vec2(${tWidth}.0, ${tHeight}.0);
  // vec4 texSampleRR = texture(inputTexture, texUV);
  // vec4 texSample = vec4(texUV, 0.5, texSampleRR[3]);

  // vec4 texSample = vec4(uv.x, uv.x, uv.x, 1.0);
  // vec4 texSample = vec4(uv.y, uv.y, uv.y, 1.0);
  vec4 texSample = vec4(0.8, 0.8, 0.8, 1.0);

  fragColor = ${outputFragColor};
}`;
}

export function vertices() {
  return new Float32Array([
    // x, y,
    -1, 1,   // upper left
    1, 1,    // upper right
    -1, -1,  // lower left
    1, -1,   // lower right
  ]);
}

export function texCoords() {
  return new Float32Array([
    // u, v
    0, 1,  // upper left
    1, 1,  // upper right
    0, 0,  // lower left
    1, 0,  // lower right
  ]);
}
