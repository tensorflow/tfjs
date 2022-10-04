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
import {GPGPUProgram, useShapeUniforms} from './gpgpu_math';
import * as shader_util from './shader_compiler_util';

const CHANNEL_CHAR_TO_INDEX_MAP: Record<string, number> = {
  'R': 0,
  'G': 1,
  'B': 2,
  'A': 3
};

export class EncodeMatrixProgram implements GPGPUProgram {
  variableNames = ['A'];
  userCode: string;
  outputShape: number[];
  enableShapeUniforms: boolean;
  customUniforms = [{name: 'texShape', type: 'ivec2' as const }];

  constructor(
      outputShape: [number, number, number], inputIsUnsignedByte = false,
      usedChannels = 'RGBA') {
    const glsl = getGlslDifferences();
    this.outputShape = outputShape;
    this.enableShapeUniforms = useShapeUniforms(this.outputShape.length);

    let output = `result`;
    if (inputIsUnsignedByte) {
      output = `floor(result * 255. + 0.5)`;
    }

    let mainLoop = '';
    for (let usedChannelIndex = 0; usedChannelIndex < usedChannels.length;
         usedChannelIndex++) {
      const curChannel = usedChannels[usedChannelIndex];
      mainLoop += `
          if(offset == ${usedChannelIndex}) {
            result = values[${CHANNEL_CHAR_TO_INDEX_MAP[curChannel]}];
          }`;
    }

    this.userCode = `
      ${
        this.enableShapeUniforms ? shader_util.getFlatIndexFrom3DOutput() :
                                   shader_util.getFlatIndexFrom3D(outputShape)}

      void main() {
        ivec3 coords = getOutputCoords();
        int flatIndex = getFlatIndex(coords);
        float result = 0.;
        int offset = imod(flatIndex, ${usedChannels.length});

        flatIndex = idiv(flatIndex, ${usedChannels.length}, 1.);

        int r = flatIndex / texShape[1];
        if (r < texShape[0]) {
          int c = imod(flatIndex, texShape[1]);
          vec2 uv = (vec2(c, r) + halfCR) / vec2(texShape[1], texShape[0]);
          vec4 values = ${glsl.texture2D}(A, uv);
          ${mainLoop}
        }
        ${glsl.output} = vec4(${output}, 0., 0., 0.);
      }
    `;
  }
}
