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

import {DataType} from '@tensorflow/tfjs-core';
import {getCoordsDataType, getMainHeaderAndGlobalIndexString, WebGPUProgram} from './webgpu_program';
import {computeDispatch, flatDispatchLayout} from './webgpu_util';

export class ScatterOptimizedProgram implements WebGPUProgram {
  variableNames = ['updates', 'indices'];
  uniforms: string;
  outputShape: number[];
  shaderKey: string;
  dispatchLayout: {x: number[]};
  dispatch: [number, number, number];
  workGroupSize: [number, number, number] = [64, 1, 1];
  updatesRank: number;
  indicesRank: number;
  sliceDimGreaterThanOne: boolean;
  atomic = true;
  type: DataType;

  constructor(
      flattenXShape: number[], sliceDim: number, indicesRank: number,
      updatesRank: number, strides: number[], shape: number[],
      outputDtype: DataType) {
    this.outputShape = shape;
    this.type = outputDtype;
    this.dispatchLayout = flatDispatchLayout(flattenXShape);
    // Dispatching based on |updates| shape instead of output shape.
    this.dispatch =
        computeDispatch(this.dispatchLayout, flattenXShape, this.workGroupSize);
    this.sliceDimGreaterThanOne = sliceDim > 1;
    this.shaderKey = `scatter_${indicesRank}_${updatesRank}_${
        this.sliceDimGreaterThanOne}_${outputDtype}`;
    const stridesType = getCoordsDataType(strides.length);
    this.uniforms = `sliceDim : i32, strides: ${stridesType}, size: i32,`;
    this.updatesRank = updatesRank;
    this.indicesRank = indicesRank;
  }

  getUserCode(): string {
    let indicesString = '';
    if (this.indicesRank === 1) {
      indicesString = 'coords[0]';
    } else if (this.indicesRank === 2) {
      indicesString = 'coords[0], j';
    }
    const indicesSnippet = `getIndices(${indicesString})`;

    const strideString = this.sliceDimGreaterThanOne ? 'uniforms.strides[j]' :
                                                       'uniforms.strides';

    let updatesString = '';
    let outCoordsString = '';
    let getUpdatesCoordsFromFlatIndex = '';
    if (this.updatesRank === 1) {
      updatesString = 'coords[0]';
      outCoordsString = 'flattenedIndex';
      getUpdatesCoordsFromFlatIndex = `
      fn getUpdatesCoordsFromFlatIndex(index : i32) -> i32 {
        return index;
      }
      `;
    } else if (this.updatesRank === 2) {
      updatesString = 'coords[0], coords[1]';
      outCoordsString = 'vec2<i32>(flattenedIndex, coords[1])';
      getUpdatesCoordsFromFlatIndex = `
      fn getUpdatesCoordsFromFlatIndex(index : i32) -> vec2<i32> {
        let d0 = index / uniforms.updatesShape[1];
        let d1 = index - d0 * uniforms.updatesShape[1];
        return vec2<i32>(d0, d1);
      }
      `;
    }
    const updatesSnippet = `getUpdates(${updatesString})`;

    // atomicAdd only supports uint/int type. For float, we use
    // atomicCompareExchangeWeak to simulate.
    const atomicAddSnippet = this.type === 'int32' ?
        `atomicAdd(&(result[flatIndex]), i32(updateValue));` :
        `
     var oldValue = atomicLoad(&(result[flatIndex]));
     var exchanged = false;
     for (; !exchanged;) {
       let newValueF32 = bitcast<f32>(oldValue) + updateValue;
       let newValue = bitcast<i32>(newValueF32);
       let res = atomicCompareExchangeWeak(&(result[flatIndex]), oldValue, newValue);
       oldValue = res.old_value;
       exchanged = res.exchanged;
     }
     `;

    const userCode = `
    ${getUpdatesCoordsFromFlatIndex}

      ${getMainHeaderAndGlobalIndexString()}

        if (index < uniforms.size) {
          let coords = getUpdatesCoordsFromFlatIndex(index);
          var flattenedIndex = 0;
          for (var j = 0; j < uniforms.sliceDim; j = j + 1) {
            let indexInside = i32(round(${indicesSnippet}));
            flattenedIndex = flattenedIndex + indexInside * ${strideString};
          }
          let updateValue = ${updatesSnippet};
          let flatIndex = getOutputIndexFromCoords(${outCoordsString});

         ${atomicAddSnippet}
        }
      }`;
    return userCode;
  }
}
