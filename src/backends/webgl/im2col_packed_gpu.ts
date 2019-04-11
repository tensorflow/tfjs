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

import {Conv2DInfo} from '../../ops/conv_util';
import {getGlslDifferences} from './glsl_version';
import {GPGPUProgram} from './gpgpu_math';

export class Im2ColPackedProgram implements GPGPUProgram {
  variableNames = ['A'];
  usesPackedTextures = true;
  outputShape: number[];
  userCode: string;

  constructor(
      outputShape: number[], inputShape: number[], convInfo: Conv2DInfo) {
    this.outputShape = outputShape;

    const {
      filterWidth,
      inChannels,
      strideWidth,
      strideHeight,
      padInfo,
      outWidth,
      dilationWidth,
      dilationHeight
    } = convInfo;
    const {left, top} = padInfo;
    const itemsPerBlockRow = inChannels * filterWidth;
    const glsl = getGlslDifferences();

    this.userCode = `
      void main() {
        ivec2 rc = getOutputCoords();

        vec4 result = vec4(0);

        for(int row=0; row<=1; row++) {
          for(int col=0; col<=1; col++) {
            int blockIndex = rc.y + col;
            int pos = rc.x + row;

            if(blockIndex >= ${outputShape[1]} || pos >= ${
        outputShape[0]}) continue;

            int offsetY = int(blockIndex / (${outWidth})) * ${strideHeight} - ${
        top};
            int d0 = offsetY + ${dilationHeight} * (pos / ${itemsPerBlockRow});

            if(d0 >= ${inputShape[0]} || d0 < 0) continue;

            int offsetX = int(mod(float(blockIndex), ${outWidth}.) * ${
        strideWidth}. - ${left}.);
            int d1 = offsetX + ${dilationWidth} * (int(mod(float(pos), ${
        itemsPerBlockRow}.) / ${inChannels}.));

            if(d1 >= ${inputShape[1]} || d1 < 0) continue;

            vec2 innerDims = vec2(d1, int(mod(float(pos), ${inChannels}.)));
            result[row * 2 + col] = getChannel(getA(d0, int(innerDims.x),
                                              int(innerDims.y)), innerDims);
          }
        }

        ${glsl.output} = result;
      }
    `;
  }
}
