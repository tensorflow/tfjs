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
import {getCoordsDataTypeWgsl, getGlobalIndexStringWgsl, getMainHeaderStringWgsl} from '../shader_preprocessor_wgsl';
import {computeDispatch, flatDispatchLayout} from '../webgpu_util';

import {getUseWgsl, WebGPUProgram} from './webgpu_program';

export class StridedSliceProgram implements WebGPUProgram {
  variableNames = ['x'];
  uniforms: string;
  uniformsWgsl: string;
  outputShape: number[];
  shaderKey: string;
  dispatchLayout: {x: number[]};
  dispatch: [number, number, number];
  // TODO(xing.xu): Increase the workPerThread.
  workPerThread = 1;
  workGroupSize: [number, number, number] = [64, 1, 1];
  dtype: string;
  dtypeWgsl: string;
  size: number;
  useWgsl: boolean;

  constructor(destSize: number[]) {
    this.outputShape = destSize;
    this.dispatchLayout = flatDispatchLayout(this.outputShape);
    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workGroupSize,
        [this.workPerThread, 1, 1]);

    this.dtype = getCoordsDataType(this.outputShape.length);
    this.dtypeWgsl = getCoordsDataTypeWgsl(this.outputShape.length, 'i32');
    this.uniforms = `${this.dtype} begin; ${this.dtype} strides; `;
    this.uniformsWgsl =
        `begin : ${this.dtypeWgsl};  strides : ${this.dtypeWgsl}; `;
    this.shaderKey = 'stridedSlice';
    this.size = util.sizeFromShape(this.outputShape);
    this.useWgsl = getUseWgsl();
  }

  getUserCode(): string {
    const rank = this.outputShape.length;
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
       void main() {
         int index = getGlobalIndex();
         if (index < size)
         {
           ${this.dtype} coords = getOutputCoords();
           setOutput(index, getX(${newCoords}));
         }
       }
     `;
    return userCode;
  }

  getUserCodeWgsl(): string {
    const rank = this.outputShape.length;
    let newCoords = '';
    if (rank === 1) {
      newCoords = 'coords * u32(uniforms.strides) + u32(uniforms.begin)';
    } else {
      let outputAxis = 0;
      newCoords =
          this.outputShape
              .map((_, i) => {
                outputAxis++;
                return this.outputShape.length === 1 ?
                    `coords * u32(uniforms.strides[${i}]) + u32(uniforms.begin[${i}])` :
                    `coords[${outputAxis - 1}] * u32(uniforms.strides[${
                        i}]) + u32(uniforms.begin[${i}])`;
              })
              .join(',');
    }

    const userCode = `
       ${getMainHeaderStringWgsl()} {
         ${getGlobalIndexStringWgsl()}
         if (index < uniforms.size)
         {
           let coords = getOutputCoords(globalId, index);
           setOutputFlat(index, getX(${newCoords}));
         }
       }
     `;
    return userCode;
  }
}
