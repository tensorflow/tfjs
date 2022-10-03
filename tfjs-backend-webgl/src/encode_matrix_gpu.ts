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

export const CHANNELS_TO_NUM_MAP: Record < string, number >= {
  'A': 0b1,
  'B': 0b10,
  'BA': 0b11,
  'G': 0b100,
  'GA': 0b101,
  'GB': 0b110,
  'GBA': 0b111,
  'R': 0b1000,
  'RA': 0b1001,
  'RB': 0b1010,
  'RBA': 0b1011,
  'RG': 0b1100,
  'RGA': 0b1101,
  'RGB': 0b1110,
  'RGBA': 0b1111
};

function isColorBitIncluded(colorString: string, colorBit: string) {
  return CHANNELS_TO_NUM_MAP[colorString] & CHANNELS_TO_NUM_MAP[colorBit]
}

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
    let usedChannelsNum = 0;
    const channels = ['R', 'G', 'B', 'A'];
    channels.forEach((channel, channelIndex) => {
      if (isColorBitIncluded(usedChannels, channel)) {
        mainLoop += `
          if(offset == ${usedChannelsNum}) {
            result = values[${channelIndex}];
          }`;
        usedChannelsNum++;
      }
    });

    this.userCode = `
      ${
        this.enableShapeUniforms ? shader_util.getFlatIndexFrom3DOutput() :
                                   shader_util.getFlatIndexFrom3D(outputShape)}

      void main() {
        ivec3 coords = getOutputCoords();

        int flatIndex = getFlatIndex(coords);
        int offset = imod(flatIndex, ${usedChannels.length});

        flatIndex = idiv(flatIndex, ${usedChannels.length}, 1.);

        int r = flatIndex / texShape[1];
        int c = imod(flatIndex, texShape[1]);
        vec2 uv = (vec2(c, r) + halfCR) / vec2(texShape[1], texShape[0]);
        vec4 values = ${glsl.texture2D}(A, uv);

        float result;
        ${mainLoop}
        ${glsl.output} = vec4(${output}, 0., 0., 0.);
      }
    `;
  }
}
