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
import {GPGPUProgram, useShapeUniforms} from './gpgpu_math';

export class Im2ColPackedProgram implements GPGPUProgram {
  variableNames = ['A'];
  packedInputs = true;
  packedOutput = true;
  outputShape: number[];
  userCode: string;
  enableShapeUniforms: boolean;
  customUniforms = [
    {name: 'inputShape', type: 'ivec3' as const },
    {name: 'pad', type: 'ivec2' as const },
    {name: 'stride', type: 'ivec2' as const },
    {name: 'dilation', type: 'ivec2' as const },
    {name: 'inChannels', type: 'int' as const },
    {name: 'itemsPerBlockRow', type: 'int' as const },
    {name: 'outWidth', type: 'int' as const },
  ];

  constructor(outputShape: number[], convInfo: backend_util.Conv2DInfo) {
    this.outputShape = outputShape;
    this.enableShapeUniforms = useShapeUniforms(this.outputShape.length);
    const {dataFormat} = convInfo;
    const glsl = getGlslDifferences();
    const isChannelsLast = dataFormat === 'channelsLast';
    const rowDim = isChannelsLast ? 0 : 1;
    const colDim = isChannelsLast ? 1 : 2;

    const boundsCheckingSnippet = this.enableShapeUniforms ?
        'if(blockIndex < outShape[1] && pos < outShape[0]) {' :
        `if(blockIndex < ${outputShape[1]} && pos < ${outputShape[0]}) {`;
    let unrolled = ``;

    for (let row = 0; row <= 1; row++) {
      for (let col = 0; col <= 1; col++) {
        unrolled += `
          blockIndex = rc.y + ${col};
          pos = rc.x + ${row};

          ${boundsCheckingSnippet}
            offsetY = int(blockIndex / outWidth) * stride[0] - pad[0];
            d0 = offsetY + dilation[0] * (pos / itemsPerBlockRow);

            if(d0 < inputShape[${rowDim}] && d0 >= 0) {
              // Use custom imod instead mod. On Intel GPU, mod may generate
              // unexpected value.
              // https://github.com/tensorflow/tfjs/issues/5447
              offsetX = imod(blockIndex, outWidth) * stride[1] - pad[1];
              d1 = offsetX + dilation[1] * (imod(pos, itemsPerBlockRow) /
                  inChannels);

              if(d1 < inputShape[${colDim}] && d1 >= 0) {

                ch = imod(pos, inChannels);

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
