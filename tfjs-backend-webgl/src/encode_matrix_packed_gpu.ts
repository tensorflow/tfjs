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
import * as shader_util from './shader_compiler_util';

/*
This is how the shader encodes a tensor with shape = [2, 3, 5]
(indices are [batch, row, col]).

000|001   002|003   004|xxx   020|021   022|023   024|xxx
-------   -------   -------   -------   -------   -------
010|011   012|013   014|xxx   xxx|xxx   xxx|xxx   xxx|xxx

100|101   102|103   104|xxx   120|121   122|123   124|xxx
-------   -------   -------   -------   -------   -------
110|111   112|113   114|xxx   xxx|xxx   xxx|xxx   xxx|xxx

Single texels contain only values from the same batch, and from adjacent rows
and columns.
 */

export class EncodeMatrixPackedProgram implements GPGPUProgram {
  variableNames = ['A'];
  userCode: string;
  outputShape: number[];
  packedInputs = false;
  packedOutput = true;

  constructor(
      outputShape: [number, number, number], texShape: [number, number],
      inputIsUnsignedByte = false) {
    const glsl = getGlslDifferences();
    const [height, width] = texShape;
    this.outputShape = outputShape;

    let mainLoop = '';
    let output = 'result';
    if (inputIsUnsignedByte) {
      output = 'floor(result * 255. + 0.5)';
    }

    for (let row = 0; row <= 1; row++) {
      for (let col = 0; col <= 1; col++) {
        const channel = row * 2 + col;

        mainLoop += `
          localCoords = coords;
          if(localCoords[2] + ${col} < ${outputShape[2]}) {
            localCoords[2] += ${col};
            if(localCoords[1] + ${row} < ${outputShape[1]}) {
              localCoords[1] += ${row};

              flatIndex = getFlatIndex(localCoords);
              offset = imod(flatIndex, 4);

              flatIndex = idiv(flatIndex, 4, 1.);

              r = flatIndex / ${width};
              c = imod(flatIndex, ${width});
              uv = (vec2(c, r) + halfCR) / vec2(${width}.0, ${height}.0);
              values = ${glsl.texture2D}(A, uv);

              if(offset == 0) {
                result[${channel}] = values[0];
              } else if(offset == 1) {
                result[${channel}] = values[1];
              } else if(offset == 2) {
                result[${channel}] = values[2];
              } else {
                result[${channel}] = values[3];
              }
            }
          }
        `;
      }
    }

    this.userCode = `
      ${shader_util.getFlatIndexFrom3D(outputShape)}

      void main() {
        ivec3 coords = getOutputCoords();

        vec4 result = vec4(0.);
        int flatIndex, r, c, offset;
        ivec3 localCoords;
        vec2 uv;
        vec4 values;

        ${mainLoop}

        ${glsl.output} = ${output};
      }
    `;
  }
}
