/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
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

export class GatherNDProgram implements WebGPUProgram {
  outputShape: number[];
  shaderKey: string;
  dispatchLayout: {x: number[]};
  dispatch: [number, number, number];
  variableNames: string[] = ['A', 'indices'];
  workGroupSize: [number, number, number] = [64, 1, 1];
  sliceDim: number;
  strides: number[];
  shape: number[];
  userCode: string;
  size: number;
  constructor(sliceDim: number, strides: number[], shape: number[]) {
    this.sliceDim = sliceDim;
    this.strides = strides;
    this.shape = shape;
    this.outputShape = shape;
    this.dispatchLayout = flatDispatchLayout(this.outputShape);
    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workGroupSize);
    this.shaderKey = `gathernd_${sliceDim}_${strides}_${shape}`;
    this.size = util.sizeFromShape(this.outputShape);
  }
  getUserCode(): string {
    const stridesType = getCoordsDataType(this.strides.length);
    const dtype = getCoordsDataType(this.shape.length);
    const strideString = this.sliceDim > 1 ? 'strides[j]' : 'strides';
    const userCode = `
        ${stridesType} strides = ${stridesType}(${this.strides});
         void main() {
          int currentIndex = int(gl_GlobalInvocationID.x);
          ${dtype} coords = getOutputCoords();
          int flattenIndex = 0;
          for (int j = 0; j < ${this.sliceDim}; j++) {
            int index = int(round(getIndices(coords[0], j)));
            flattenIndex += index * ${strideString};
          }
          if (currentIndex < ${this.size}) {
            setOutput(currentIndex, getA(flattenIndex, coords[1]));
          }
        }
      `;
    return userCode;
  }
}
