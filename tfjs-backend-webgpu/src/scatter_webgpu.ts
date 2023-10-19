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

import {atomicAddSnippet} from './shader_util';
import {dataTypeToGPUType, getCoordsDataType, getMainHeaderString as main, WebGPUProgram} from './webgpu_program';
import {computeDispatch, flatDispatchLayout} from './webgpu_util';

export class ScatterProgram implements WebGPUProgram {
  variableNames = ['updates', 'indices'];
  uniforms: string;
  outputShape: number[];
  sumDupeIndices: boolean;
  shaderKey: string;
  dispatchLayout: {x: number[]};
  dispatch: [number, number, number];
  workgroupSize: [number, number, number] = [64, 1, 1];
  updatesRank: number;
  indicesRank: number;
  sliceDimGreaterThanOne: boolean;
  atomic = true;
  type: DataType;

  constructor(
      flattenXShape: number[], sliceDim: number, indicesRank: number,
      updatesRank: number, strides: number[], shape: number[],
      outputDtype: DataType, sumDupeIndices = true) {
    this.outputShape = shape;
    this.type = outputDtype;
    this.sumDupeIndices = sumDupeIndices;
    this.dispatchLayout = flatDispatchLayout(flattenXShape);
    // Dispatching based on |updates| shape instead of output shape.
    this.dispatch =
        computeDispatch(this.dispatchLayout, flattenXShape, this.workgroupSize);
    this.sliceDimGreaterThanOne = sliceDim > 1;
    this.shaderKey =
        `scatter_${indicesRank}_${updatesRank}_${this.sliceDimGreaterThanOne}_${
            outputDtype}_${sumDupeIndices}_${strides.length}`;
    const stridesType = getCoordsDataType(strides.length);
    this.uniforms =
        `sliceDim : i32, strides: ${stridesType}, updatesSize: i32,`;
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

    let outCoordsString = '';
    let getUpdatesCoordsFromFlatIndex = '';
    if (this.dispatchLayout.x.length === 1) {
      outCoordsString = 'flattenedIndex';
      getUpdatesCoordsFromFlatIndex = `
      fn getUpdatesCoordsFromFlatIndex(index : i32) -> i32 {
        return index;
      }
      `;
    } else if (this.dispatchLayout.x.length === 2) {
      outCoordsString = 'vec2<i32>(flattenedIndex, coords[1])';
      getUpdatesCoordsFromFlatIndex = `
      fn getUpdatesCoordsFromFlatIndex(index : i32) -> vec2<i32> {
        // N.B. |updates| could be a scalar tensor, conceptually representing a
        // 2D tensor with all values equal to that. By design, its size must be
        // the same as |outShape[1]| in one dimension, and |indicesShape[0]|
        // gives the other.
        let sliceSize = uniforms.outShape[1];
        let d0 = index / sliceSize;
        let d1 = index - d0 * sliceSize;
        return vec2<i32>(d0, d1);
      }
      `;
    }
    const updatesString =
        Array.from({length: this.updatesRank}, (_, idx) => `coords[${idx}]`);
    const updatesSnippet = `getUpdates(${updatesString.join(', ')})`;

    const userCode = `
    ${getUpdatesCoordsFromFlatIndex}
      ${main('index')} {
        if (index < uniforms.updatesSize) {
          let coords = getUpdatesCoordsFromFlatIndex(index);
          var flattenedIndex = 0;
          for (var j = 0; j < uniforms.sliceDim; j = j + 1) {
            let indexInside = i32(round(${indicesSnippet}));
            flattenedIndex = flattenedIndex + indexInside * ${strideString};
          }
          let updateValue =
              ${dataTypeToGPUType(this.type)}(${updatesSnippet});
          let flatIndex = getOutputIndexFromCoords(${outCoordsString});

          ${
        this.sumDupeIndices ?
            atomicAddSnippet(
                '&result[flatIndex]', 'updateValue',
                this.type as 'float32' | 'int32') :
            `atomicStore(&result[flatIndex], bitcast<i32>(updateValue));`}
        }
      }`;
    return userCode;
  }
}
