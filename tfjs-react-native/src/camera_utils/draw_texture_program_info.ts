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

export function vertexShaderSource() {
  return `#version 300 es
precision highp float;

in vec2 position;
uniform mat4 u_matrix;

in vec2 texCoords;
out vec2 uv;

void main() {
  gl_Position = vec4(u_matrix * vec4(position, 0, 1));
  uv = texCoords;
}`;
}

export function fragmentShaderSource() {
  return `#version 300 es
precision highp float;
uniform sampler2D inputTexture;
in vec2 uv;
out vec4 fragColor;
void main() {
  // vec4 texSample = vec4(uv.x, uv.x, uv.x, 1.0);
  // vec4 texSample = vec4(uv.y, uv.y, uv.y, 1.0);

  vec4 texSample = texture(inputTexture, uv);
  fragColor = texSample;
}`;
}

export function vertices() {
  return new Float32Array([
    // clang-format off
    0, 0,
    0, 1,
    1, 0,
    1, 0,
    0, 1,
    1, 1,
    // clang-format on
  ]);
}

export function texCoords() {
  return new Float32Array([
    // clang-format off
    0, 0,
    0, 1,
    1, 0,
    1, 0,
    0, 1,
    1, 1,
    // clang-format on
  ]);
}
