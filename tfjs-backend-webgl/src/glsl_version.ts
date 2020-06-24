/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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
import {env} from '@tensorflow/tfjs-core';

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

  if (env().getNumber('WEBGL_VERSION') === 2) {
    version = '#version 300 es';
    attribute = 'in';
    varyingVs = 'out';
    varyingFs = 'in';
    texture2D = 'texture';
    output = 'outputColor';
    defineOutput = 'out vec4 outputColor;';

    // Use custom isnan definition to work across differences between
    // implementations on various platforms. While this should happen in ANGLE
    // we still see differences between android and windows (on chrome) when
    // using isnan directly.
    defineSpecialNaN = `
      bool isnan_custom(float val) {
        return (val > 0.0 || val < 0.0) ? false : val != 0.0;
      }

      bvec4 isnan_custom(vec4 val) {
        return bvec4(isnan_custom(val.x),
          isnan_custom(val.y), isnan_custom(val.z), isnan_custom(val.w));
      }

      #define isnan(value) isnan_custom(value)
    `;
    // In webgl 2 we do not need to specify a custom isinf so there is no
    // need for a special INFINITY constant.
    defineSpecialInf = ``;
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
    // WebGL1 has no built in isnan so we define one here.
    defineSpecialNaN = `
      #define isnan(value) isnan_custom(value)
      bool isnan_custom(float val) {
        return (val > 0. || val < 1. || val == 0.) ? false : true;
      }
      bvec4 isnan_custom(vec4 val) {
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
