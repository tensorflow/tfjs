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

import {getGlslDifferences} from './glsl_version';
import {GPGPUProgram} from './gpgpu_math';
import {ENCODE_FLOAT_SNIPPET} from './shader_compiler_util';
import {TextureUsage} from './tex_util';

export class EncodeFloatPackedProgram implements GPGPUProgram {
  variableNames = ['A'];
  userCode: string;
  outputShape: number[];
  packedInputs = true;
  packedOutput = false;
  outTexUsage = TextureUsage.DOWNLOAD;

  constructor(outputShape: [number, number, number]) {
    const glsl = getGlslDifferences();
    this.outputShape = outputShape;
    this.userCode = `
      ${ENCODE_FLOAT_SNIPPET}

      void main() {
        ivec3 coords = getOutputCoords();
        float x = getChannel(getAAtOutCoords(), vec2(coords.y, coords.z));
        ${glsl.output} = encode_float(x);
      }
    `;
  }
}
