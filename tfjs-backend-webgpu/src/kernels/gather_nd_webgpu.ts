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
  uniforms: string;
  workGroupSize: [number, number, number] = [64, 1, 1];
  size: number;
  sliceDim: number;
  constructor(sliceDim: number, shape: number[]) {
    this.outputShape = shape;
    this.dispatchLayout = flatDispatchLayout(this.outputShape);
    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workGroupSize);
    this.shaderKey = `gathernd_${sliceDim}`;
    this.size = util.sizeFromShape(this.outputShape);
    this.sliceDim = sliceDim;
    this.uniforms = `int sliceDim; ${getCoordsDataType(sliceDim)} strides;`;
  }
  getUserCode(): string {
    const dtype = getCoordsDataType(this.outputShape.length);
    let strideString;
    if (this.sliceDim > 1) {
      strideString = 'strides[j]';
    } else {
      strideString = 'strides';
    }
    const userCode = `
         void main() {
          int currentIndex = int(gl_GlobalInvocationID.x);
          ${dtype} coords = getOutputCoords();
          int flattenIndex = 0;
          for (int j = 0; j < sliceDim; j++) {
            int index = int(round(getIndices(coords[0], j)));
            int strideNum = ${strideString};
            flattenIndex += index * strideNum;
          }
          if (currentIndex < size) {
            setOutput(currentIndex, getA(flattenIndex, coords[1]));
          }
        }
      `;
    return userCode;
  }
}
