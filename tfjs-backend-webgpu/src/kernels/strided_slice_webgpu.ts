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

import {getCoordsDataType} from '../shader_preprocessor';
import {computeDispatch, flatDispatchLayout} from '../webgpu_util';

import {WebGPUProgram} from './webgpu_program';

export class StridedSliceProgram implements WebGPUProgram {
  variableNames = ['x'];
  outputShape: number[];
  shaderKey: string;
  dispatchLayout: {x: number[]};
  dispatch: [number, number, number];
  // TODO(xing.xu): Increase the workPerThread.
  workPerThread = 1;
  workGroupSize: [number, number, number] = [16, 1, 1];
  begin: number[];
  strides: number[];

  constructor(begin: number[], strides: number[], destSize: number[]) {
    this.outputShape = destSize;
    this.dispatchLayout = flatDispatchLayout(this.outputShape);
    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workGroupSize,
        [this.workPerThread, 1, 1]);

    this.begin = begin;
    this.strides = strides;
    this.shaderKey = `stridedSlice_${begin}_${strides}`;
  }

  getUserCode(): string {
    const rank = this.outputShape.length;
    const inputDtype = getCoordsDataType(this.outputShape.length);
    const dtype = getCoordsDataType(this.outputShape.length);
    let newCoords = '';
    if (rank === 1) {
      newCoords = 'coords * strides + begin';
    } else {
      let outputAxis = 0;
      newCoords =
          this.outputShape
              .map((_, i) => {
                outputAxis++;
                return this.outputShape.length === 1 ?
                    `coords * strides[${i}] + begin[${i}]` :
                    `coords[${outputAxis - 1}] * strides[${i}] + begin[${i}]`;
              })
              .join(',');
    }

    const userCode = `
      ${inputDtype} begin = ${inputDtype}(${this.begin});
      ${inputDtype} strides = ${inputDtype}(${this.strides});

      void main() {
        ${dtype} coords = getOutputCoords();
        int index = int(gl_GlobalInvocationID.x);
        setOutput(index, getX(${newCoords}));
      }
    `;
    return userCode;
  }
}
