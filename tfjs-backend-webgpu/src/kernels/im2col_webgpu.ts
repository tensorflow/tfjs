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

import {backend_util, util} from '@tensorflow/tfjs-core';
import {computeDispatch, flatDispatchLayout} from '../webgpu_util';

import {WebGPUProgram} from './webgpu_program';

export class Im2ColProgram implements WebGPUProgram {
  variableNames = ['A'];
  outputShape: number[];
  shaderKey: string;
  dispatchLayout: {x: number[]};
  dispatch: [number, number, number];
  rank: number;
  workPerThread = 4;
  workGroupSize: [number, number, number] = [64, 1, 1];
  inputShape: number[];
  convInfo: backend_util.Conv2DInfo;

  constructor(
      outputShape: number[], inputShape: number[],
      convInfo: backend_util.Conv2DInfo) {
    this.outputShape = outputShape;
    this.rank = outputShape.length;

    this.dispatchLayout = flatDispatchLayout(this.outputShape);
    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workGroupSize,
        [this.workPerThread, 1, 1]);
    this.inputShape = inputShape;
    this.convInfo = convInfo;
    this.shaderKey = `im2col_${convInfo}`;
  }

  getUserCode(): string {
    const size = util.sizeFromShape(this.outputShape);
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
    } = this.convInfo;
    const {left, top} = padInfo;
    const itemsPerBlockRow = inChannels * filterWidth;

    const isChannelsLast = dataFormat === 'channelsLast';
    const rowDim = isChannelsLast ? 0 : 1;
    const colDim = isChannelsLast ? 1 : 2;

    const userCode = `
      void main() {
        int index = int(gl_GlobalInvocationID.x);

        for(int i=0; i<${this.workPerThread}; i++) {
          int flatIndex = index * ${this.workPerThread} + i;

          ivec2 rc = getCoordsFromFlatIndex(flatIndex);

          if(flatIndex < ${size}) {
            int blockIndex = rc[0];
            int pos = rc[1];

            int offsetY = int(blockIndex / ${outWidth}) * ${strideHeight} -
              ${top};
            int d0 = offsetY + ${dilationHeight} * (pos / ${itemsPerBlockRow});
            float value = 0.0;
            if(d0 < ${this.inputShape[rowDim]} && d0 >= 0) {
              int offsetX = int(mod(float(blockIndex), ${outWidth}.) *
                ${strideWidth}. - ${left}.);
              int d1 = offsetX + ${dilationWidth} * (int(mod(float(pos),
                ${itemsPerBlockRow}.) / ${inChannels}.));
              int ch = int(mod(float(pos), ${inChannels}.));
              if(d1 < ${this.inputShape[colDim]} && d1 >= 0) {
                value = getA(d0, d1, ch);
              }
            }
            setOutput(flatIndex, value);
          }
        }
      }
    `;
    return userCode;
  }
}
