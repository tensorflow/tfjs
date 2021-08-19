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
import {getCoordsDataTypeWgsl, getWorkGroupSizeStringWgsl} from '../shader_preprocessor_wgsl';
import {computeDispatch, flatDispatchLayout} from '../webgpu_util';

import {getUseWgsl, WebGPUProgram} from './webgpu_program';

export class GatherNDProgram implements WebGPUProgram {
  outputShape: number[];
  shaderKey: string;
  dispatchLayout: {x: number[]};
  dispatch: [number, number, number];
  variableNames: string[] = ['A', 'indices'];
  uniforms: string;
  uniformsWgsl: string;
  workGroupSize: [number, number, number] = [64, 1, 1];
  size: number;
  sliceDim: number;
  useWgsl: boolean;
  constructor(sliceDim: number, shape: number[]) {
    this.outputShape = shape;
    this.dispatchLayout = flatDispatchLayout(this.outputShape);
    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workGroupSize);
    this.shaderKey = `gathernd_${sliceDim}`;
    this.size = util.sizeFromShape(this.outputShape);
    this.sliceDim = sliceDim;
    this.uniforms = `int sliceDim; ${getCoordsDataType(sliceDim)} strides;`;
    this.uniformsWgsl =
        `sliceDim : u32; strides : ${getCoordsDataTypeWgsl(sliceDim)};`;
    this.useWgsl = getUseWgsl();
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

  getUserCodeWgsl(): string {
    let strideString;
    if (this.sliceDim > 1) {
      strideString = 'uniforms.strides[j]';
    } else {
      strideString = 'uniforms.strides';
    }
    const userCode = `
        ${getWorkGroupSizeStringWgsl(this.workGroupSize)}
        fn main([[builtin(global_invocation_id)]] globalId : vec3<u32>) {
          let currentIndex = globalId.x;
          let coords = getOutputCoords(globalId);
          var flattenIndex = 0u;
          for (var j = 0u; j < uniforms.sliceDim; j = j + 1u) {
            let index = u32(round(getIndices(coords[0], j)));
            let strideNum = ${strideString};
            flattenIndex = flattenIndex + index * strideNum;
          }
          if (currentIndex < uniforms.size) {
            setOutputFlat(currentIndex, getA(flattenIndex, coords[1]));
          }
        }
      `;
    return userCode;
  }
}
