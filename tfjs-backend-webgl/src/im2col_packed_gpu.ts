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

import {backend_util} from '@tensorflow/tfjs-core';
import {getGlslDifferences} from './glsl_version';
import {GPGPUProgram} from './gpgpu_math';

export class Im2ColPackedProgram implements GPGPUProgram {
  variableNames = ['A'];
  packedInputs = true;
  packedOutput = true;
  outputShape: number[];
  userCode: string;

  constructor(
      outputShape: number[], inputShape: number[],
      convInfo: backend_util.Conv2DInfo) {
    this.outputShape = outputShape;

    const {
      filterWidth,
      inChannels,
      strideWidth,
      strideHeight,
      padInfo,
      outWidth,
      dilationWidth,
      dilationHeight,
      dataFormat
    } = convInfo;
    const {left, top} = padInfo;
    const itemsPerBlockRow = inChannels * filterWidth;
    const glsl = getGlslDifferences();
    const isChannelsLast = dataFormat === 'channelsLast';
    const rowDim = isChannelsLast ? 0 : 1;
    const colDim = isChannelsLast ? 1 : 2;

    let unrolled = ``;

    for (let row = 0; row <= 1; row++) {
      for (let col = 0; col <= 1; col++) {
        unrolled += `
          blockIndex = rc.y + ${col};
          pos = rc.x + ${row};

          if(blockIndex < ${outputShape[1]} && pos < ${outputShape[0]}) {
            offsetY = int(blockIndex / (${outWidth})) * ${strideHeight} - ${
            top};
            d0 = offsetY + ${dilationHeight} * (pos / ${itemsPerBlockRow});

            if(d0 < ${inputShape[rowDim]} && d0 >= 0) {

              offsetX = int(mod(float(blockIndex), ${outWidth}.) * ${
            strideWidth}. - ${left}.);
              d1 = offsetX + ${dilationWidth} * (int(mod(float(pos), ${
            itemsPerBlockRow}.) / ${inChannels}.));

              if(d1 < ${inputShape[colDim]} && d1 >= 0) {

                ch = int(mod(float(pos), ${inChannels}.));

                if (${isChannelsLast}) {
                  innerDims = vec2(d1, ch);
                  result[${row * 2 + col}] = getChannel(
                    getA(d0, int(innerDims.x),
                    int(innerDims.y)), innerDims);
                } else {
                  innerDims = vec2(d0, d1);
                  result[${row * 2 + col}] = getChannel(
                    getA(ch, int(innerDims.x),
                    int(innerDims.y)), innerDims);
                }
              }
            }
          }
        `;
      }
    }

    this.userCode = `
      void main() {
        ivec2 rc = getOutputCoords();

        vec4 result = vec4(0);

        int blockIndex, pos, offsetY, d0, offsetX, d1, ch;
        vec2 innerDims;

        ${unrolled}

        ${glsl.output} = result;
      }
    `;
  }
}
