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

import {getCoordsDataTypeWgsl, getGlobalIndexStringWgsl, getMainHeaderStringWgsl} from '../shader_preprocessor_wgsl';
import {computeDispatch, flatDispatchLayout} from '../webgpu_util';

import {WebGPUProgram} from './webgpu_program';

export class GatherNDProgram implements WebGPUProgram {
  outputShape: number[];
  shaderKey: string;
  dispatchLayout: {x: number[]};
  dispatch: [number, number, number];
  variableNames: string[] = ['A', 'indices'];
  uniformsWgsl: string;
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
    this.uniformsWgsl =
        `sliceDim : i32; strides : ${getCoordsDataTypeWgsl(sliceDim)};`;
  }

  getUserCodeWgsl(): string {
    let strideString;
    if (this.sliceDim > 1) {
      strideString = 'uniforms.strides[j]';
    } else {
      strideString = 'uniforms.strides';
    }
    const userCode = `
        ${getMainHeaderStringWgsl()} {
          ${getGlobalIndexStringWgsl()}
          let coords = getOutputCoords(globalId, index);
          var flattenIndex = 0;
          for (var j = 0; j < uniforms.sliceDim; j = j + 1) {
            let indexTemp = i32(round(getIndices(coords[0], j)));
            let strideNum = ${strideString};
            flattenIndex = flattenIndex + indexTemp * strideNum;
          }
          if (index < uniforms.size) {
            setOutputFlat(index, getA(flattenIndex, coords[1]));
          }
        }
      `;
    return userCode;
  }
}
