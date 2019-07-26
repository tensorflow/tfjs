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

import {getGlslDifferences} from './glsl_version';
import {GPGPUProgram} from './gpgpu_math';
import * as shader_util from './shader_compiler_util';

export class EncodeMatrixProgram implements GPGPUProgram {
  variableNames = ['A'];
  userCode: string;
  outputShape: number[];

  constructor(
      outputShape: [number, number, number], texShape: [number, number],
      inputIsUnsignedByte = false) {
    const glsl = getGlslDifferences();
    const [height, width] = texShape;
    this.outputShape = outputShape;

    let output = `result`;
    if (inputIsUnsignedByte) {
      output = `floor(result * 255. + 0.5)`;
    }

    this.userCode = `
      ${shader_util.getFlatIndexFrom3D(outputShape)}

      void main() {
        ivec3 coords = getOutputCoords();

        int flatIndex = getFlatIndex(coords);
        int offset = imod(flatIndex, 4);

        flatIndex = idiv(flatIndex, 4, 1.);
        
        int r = flatIndex / ${width};
        int c = imod(flatIndex, ${width});
        vec2 uv = (vec2(c, r) + halfCR) / vec2(${width}.0, ${height}.0);
        vec4 values = ${glsl.texture2D}(A, uv);

        float result;

        if(offset == 0) {
          result = values[0];
        } else if(offset == 1) {
          result = values[1];
        } else if(offset == 2) {
          result = values[2];
        } else {
          result = values[3];
        }

        ${glsl.output} = vec4(${output}, 0., 0., 0.);
      }
    `;
  }
}
