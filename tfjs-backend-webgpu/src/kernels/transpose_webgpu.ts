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

import {util} from '@tensorflow/tfjs-core';
import {getCoordsDataType} from '../shader_preprocessor';
import {computeDispatch, flatDispatchLayout} from '../webgpu_util';

import {WebGPUProgram} from './webgpu_program';

export class TransposeProgram implements WebGPUProgram {
  variableNames = ['A'];
  shaderKey: string;
  outputShape: number[];
  userCode: string;
  dispatchLayout: {x: number[]};
  dispatch: [number, number, number];
  rank: number;
  workPerThread = 4;
  workGroupSize: [number, number, number] = [64, 1, 1];

  constructor(aShape: number[], newDim: number[]) {
    const outputShape: number[] = new Array(aShape.length);
    for (let i = 0; i < outputShape.length; i++) {
      outputShape[i] = aShape[newDim[i]];
    }
    this.outputShape = outputShape;
    this.rank = outputShape.length;
    const dtype = getCoordsDataType(this.rank);
    const size = util.sizeFromShape(this.outputShape);
    this.dispatchLayout = flatDispatchLayout(this.outputShape);
    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workGroupSize,
        [this.workPerThread, 1, 1]);

    const switched = getSwitchedCoords(newDim);

    this.userCode = `
      void main() {
        int index = int(gl_GlobalInvocationID.x);

        for(int i = 0; i < ${this.workPerThread}; i++) {
          int flatIndex = index * ${this.workPerThread} + i;
          if(flatIndex < ${size}) {
            ${dtype} resRC = getCoordsFromFlatIndex(flatIndex);
            setOutput(flatIndex, A[getFlatIndex(
              ${dtype}(${switched}), aShape)]);
          }
        }
      }
    `;
    this.shaderKey = `tranpose${size}${dtype}${newDim.join(',')}`;
  }
}

function getSwitchedCoords(newDim: number[]): string {
  const rank = newDim.length;
  if (rank > 4) {
    throw Error(`Transpose for rank ${rank} is not yet supported`);
  }
  const switchedCoords = new Array(rank);
  for (let i = 0; i < newDim.length; i++) {
    switchedCoords[newDim[i]] = `resRC[${i}]`;
  }

  return switchedCoords.join();
}
