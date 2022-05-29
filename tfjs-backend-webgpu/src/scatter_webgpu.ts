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

import {getCoordsDataType, getMainHeaderAndGlobalIndexString, WebGPUProgram} from './webgpu_program';
import {computeDispatch, flatDispatchLayout} from './webgpu_util';

export class ScatterProgram implements WebGPUProgram {
  variableNames = ['updates', 'indices', 'defaultValue'];
  uniforms: string;
  outputShape: number[];
  shaderKey: string;
  dispatchLayout: {x: number[]};
  dispatch: [number, number, number];
  workGroupSize: [number, number, number] = [64, 1, 1];
  workPerThread = 4;
  size = true;
  indicesSnippet: string;
  strideString: string;
  updatesSnippet: string;

  constructor(
      updateSize: number, sliceDim: number, indicesRank: number,
      updatesRank: number, strides: number[], shape: number[],
      summingDupeIndex = true) {
    this.outputShape = shape;
    this.dispatchLayout = flatDispatchLayout(this.outputShape);
    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workGroupSize,
        [this.workPerThread, 1, 1]);
    const sliceDimGreaterThanOne = sliceDim > 1;
    this.shaderKey =
        `scatter_${indicesRank}_${updatesRank}_${sliceDimGreaterThanOne}`;
    const stridesType = getCoordsDataType(strides.length);
    this.uniforms =
        `updateSize : i32, sliceDim : i32, strides: ${stridesType},`;
    let indicesString = '';
    if (indicesRank === 1) {
      indicesString = 'i';
    } else if (indicesRank === 2) {
      indicesString = 'i, j';
    }
    this.indicesSnippet = `getIndices(${indicesString})`;

    let updatesString = '';
    if (updatesRank === 1) {
      updatesString = 'i';
    } else if (updatesRank === 2) {
      updatesString = 'i, coords[1]';
    }
    this.updatesSnippet = `getUpdates(${updatesString})`;

    this.strideString =
        sliceDimGreaterThanOne ? 'uniforms.strides[j]' : 'uniforms.strides';
  }

  getUserCode(): string {
    const userCode = `
      ${getMainHeaderAndGlobalIndexString()}

        let globalIndex = index * ${this.workPerThread};
        if (globalIndex < uniforms.size) {
          var sum = vec4<f32>(0.0);
          var found = vec4<bool>(false);
          for (var i = 0; i < uniforms.updateSize; i = i + 1) {
            var flattenedIndex = 0;
            for (var j = 0; j < uniforms.sliceDim; j = j + 1) {
              let indexInside = i32(round(${this.indicesSnippet}));
              flattenedIndex = flattenedIndex + indexInside * ${
        this.strideString};
            }
            for (var innerIndex = 0; innerIndex < ${
        this.workPerThread}; innerIndex = innerIndex + 1) {
              let curIndex = globalIndex + innerIndex;
              let coords = getCoordsFromIndex(curIndex);
              if (flattenedIndex == coords[0]) {
                sum[innerIndex] = sum[innerIndex] + ${this.updatesSnippet};
                found[innerIndex] = true;
              }
            }
          }
          for (var innerIndex = 0; innerIndex < ${
        this.workPerThread}; innerIndex = innerIndex + 1) {
            let curIndex = globalIndex + innerIndex;
            if (curIndex < uniforms.size)
            {
              setOutputAtIndex(curIndex, mix(getDefaultValue(), sum[innerIndex], f32(found[innerIndex])));
            }
          }
        }
      }`;
    return userCode;
  }
}
