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

import {Rotation} from './types';

export function vertexShaderSource(
    flipHorizontal: boolean, flipVertical: boolean, rotation: Rotation) {
  const horizontalScale = flipHorizontal ? -1 : 1;
  const verticalScale = flipVertical ? -1 : 1;
  const rotateAngle = rotation === 0 ? '0.' : rotation * (Math.PI / 180);
  return `#version 300 es
precision highp float;

in vec2 position;
in vec2 texCoords;

out vec2 uv;

vec2 rotate(vec2 uvCoods, vec2 pivot, float rotation) {
  float cosa = cos(rotation);
  float sina = sin(rotation);
  uvCoods -= pivot;
  return vec2(
      cosa * uvCoods.x - sina * uvCoods.y,
      cosa * uvCoods.y + sina * uvCoods.x
  ) + pivot;
}

void main() {
  uv = rotate(texCoords, vec2(0.5), ${rotateAngle});

  // Invert geometry to match the image orientation from the camera.
  gl_Position = vec4(position * vec2(${horizontalScale}., ${
      verticalScale}. * -1.), 0, 1);
}`;
}

export function fragmentShaderSource() {
  return `#version 300 es
precision highp float;
uniform sampler2D inputTexture;
in vec2 uv;
out vec4 fragColor;
void main() {
  vec4 texSample = texture(inputTexture, uv);
  fragColor = texSample;
}`;
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
