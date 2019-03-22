/**
 * @license
 * Copyright 2018 Google Inc. All Rights Reserved.
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

import {ENV} from '../../environment';

export type GLSL = {
  version: string,
  attribute: string,
  varyingVs: string,
  varyingFs: string,
  texture2D: string,
  output: string,
  defineOutput: string,
  defineSpecialNaN: string,
  defineSpecialInf: string,
  defineRound: string
};

export function getGlslDifferences(): GLSL {
  let version: string;
  let attribute: string;
  let varyingVs: string;
  let varyingFs: string;
  let texture2D: string;
  let output: string;
  let defineOutput: string;
  let defineSpecialNaN: string;
  let defineSpecialInf: string;
  let defineRound: string;

  if (ENV.get('WEBGL_VERSION') === 2) {
    version = '#version 300 es';
    attribute = 'in';
    varyingVs = 'out';
    varyingFs = 'in';
    texture2D = 'texture';
    output = 'outputColor';
    defineOutput = 'out vec4 outputColor;';
    defineSpecialNaN = `
      const float NAN = uintBitsToFloat(uint(0x7fc00000));
    `;
    defineSpecialInf = `
      const float INFINITY = uintBitsToFloat(uint(0x7f800000));
    `;
    defineRound = `
      #define round(value) newRound(value)
      int newRound(float value) {
        return int(floor(value + 0.5));
      }

      ivec4 newRound(vec4 value) {
        return ivec4(floor(value + vec4(0.5)));
      }
    `;
  } else {
    version = '';
    attribute = 'attribute';
    varyingVs = 'varying';
    varyingFs = 'varying';
    texture2D = 'texture2D';
    output = 'gl_FragColor';
    defineOutput = '';
    defineSpecialNaN = `
      uniform float NAN;

      bool isnan(float val) {
        return (val < 1.0 || 0.0 < val || val == 0.0) ? false : true;
      }
      bvec4 isnan(vec4 val) {
        return bvec4(isnan(val.x), isnan(val.y), isnan(val.z), isnan(val.w));
      }
    `;
    defineSpecialInf = `
      uniform float INFINITY;

      bool isinf(float val) {
        return abs(val) == INFINITY;
      }
      bvec4 isinf(vec4 val) {
        return equal(abs(val), vec4(INFINITY));
      }
    `;
    defineRound = `
      int round(float value) {
        return int(floor(value + 0.5));
      }

      ivec4 round(vec4 value) {
        return ivec4(floor(value + vec4(0.5)));
      }
    `;
  }

  return {
    version,
    attribute,
    varyingVs,
    varyingFs,
    texture2D,
    output,
    defineOutput,
    defineSpecialNaN,
    defineSpecialInf,
    defineRound
  };
}
