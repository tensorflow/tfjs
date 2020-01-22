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
  const sWidth = sourceDims.width;
  const sHeight = sourceDims.height;
  const tWidth = targetDims.width;
  const tHeight = targetDims.height;

  const effectiveInSize: [number, number] = [
    (alignCorners && tWidth > 1) ? sWidth - 1 : sWidth,
    (alignCorners && tHeight > 1) ? sHeight - 1 : sHeight,
  ];

  const effectiveOutSize: [number, number] = [
    (alignCorners && tWidth > 1) ? tWidth - 1 : tWidth,
    (alignCorners && tHeight > 1) ? tHeight - 1 : tHeight,
  ];

  const outputFragType = targetDims.depth === 3 ? 'vec3' : 'vec4';
  const outputFragColor = targetDims.depth === 3 ?
      'vec3(texSample.r,texSample.g,texSample.b)' :
      'texSample';

  const source = `#version 300 es
precision highp float;

uniform sampler2D inputTexture;
in vec2 uv;
out ${outputFragType} fragColor;

vec2 sourceDims = vec2(${sWidth}, ${sHeight});
vec2 targetDims = vec2(${tWidth}, ${tHeight});

const vec2 inputToOutputRatio = vec2(
    ${effectiveInSize[0] / effectiveOutSize[0]},
    ${effectiveInSize[1] / effectiveOutSize[1]});

void main() {
  ivec2 targetCoords = ivec2(uv * targetDims);

  vec2 sourceCoords = vec2(targetCoords) * inputToOutputRatio;

  ivec2 sourceFloor = ivec2(sourceCoords);
  ivec2 sourceCeil = ivec2(min(sourceDims - 1.0, ceil(sourceCoords)));

  vec4 topLeft = texelFetch(inputTexture,
    ivec2(sourceFloor.x, sourceFloor.y), 0);
  vec4 bottomLeft = texelFetch(inputTexture,
    ivec2(sourceCeil.x, sourceFloor.y), 0);
  vec4 topRight = texelFetch(inputTexture,
    ivec2(sourceFloor.x, sourceCeil.y), 0);
  vec4 bottomRight = texelFetch(inputTexture,
    ivec2(sourceCeil.x, sourceCeil.y), 0);

  vec2 fracRC = sourceCoords - vec2(sourceFloor);

  vec4 top = topLeft + (topRight - topLeft) * fracRC.y;
  vec4 bottom = bottomLeft + (bottomRight - bottomLeft) * fracRC.y;
  vec4 newValue = top + (bottom - top) * fracRC.x;

  vec4 texSample = newValue;
  fragColor = ${outputFragColor};
}
`;

  return source;
}

export function vertices() {
  return new Float32Array([
    // clang-format off
    -1, -1,
    -1, 1,
    1, 1,
    1, 1,
    -1, -1,
    1, -1,
    // clang-format on
  ]);
}

export function texCoords() {
  return new Float32Array([
    // clang-format off
    0, 0,
    0, 1,
    1, 1,
    1, 1,
    0, 0,
    1, 0,
    // clang-format on
  ]);
}
