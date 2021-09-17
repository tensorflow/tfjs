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
import {getCoordsDataTypeWgsl, getGlobalIndexStringWgsl, getMainHeaderStringWgsl} from '../shader_preprocessor_wgsl';
import {computeDispatch, flatDispatchLayout} from '../webgpu_util';

import {getUseWgsl, WebGPUProgram} from './webgpu_program';

export class ScatterProgram implements WebGPUProgram {
  variableNames = ['updates', 'indices', 'defaultValue'];
  outputShape: number[];
  userCode: string;
  shaderKey: string;
  dispatchLayout: {x: number[]};
  dispatch: [number, number, number];
  workGroupSize: [number, number, number] = [64, 1, 1];
  size: number;
  useWgsl: boolean;
  updateSize: number;
  sliceDim: number;
  strides: number[];
  shape: number[];
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
        this.dispatchLayout, this.outputShape, this.workGroupSize);
    this.shaderKey = `scatter`;
    this.size = util.sizeFromShape(this.outputShape);
    this.useWgsl = getUseWgsl();
    this.updateSize = updateSize;
    this.strides = strides;
    this.sliceDim = sliceDim;
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

    this.strideString = sliceDim > 1 ? 'strides[j]' : 'strides';
  }

  getUserCode(): string {
    const stridesType = getCoordsDataType(this.strides.length);
    const dtype = getCoordsDataType(this.outputShape.length);
    const userCode = `
        ${stridesType} strides = ${stridesType}(${this.strides});

        void main() {
          int gIndex = getGlobalIndex();
          if (gIndex < size) {
            ${dtype} coords = getOutputCoords();
            float sum = 0.0;
            bool found = false;
            for (int i = 0; i < ${this.updateSize}; i++) {
              int flattenedIndex = 0;
              for (int j = 0; j < ${this.sliceDim}; j++) {
                int index = int(round(${this.indicesSnippet}));
                flattenedIndex += index * ${this.strideString};
              }
              if (flattenedIndex == coords[0]) {
                sum += ${this.updatesSnippet};
                found = true;
              }
            }
            setOutput(gIndex, mix(getDefaultValue(), sum, float(found)));
          }
        }
    `;
    return userCode;
  }

  getUserCodeWgsl(): string {
    const stridesType = getCoordsDataTypeWgsl(this.strides.length);
    const userCode = `
      ${getMainHeaderStringWgsl()} {
        ${getGlobalIndexStringWgsl()}
        let strides = ${stridesType}(${this.strides});

        if (index < uniforms.size) {
          let coords = getOutputCoords(globalId, index);
          var sum = 0.0;
          var found = false;
          for (var i = 0; i < ${this.updateSize}; i = i + 1) {
            var flattenedIndex = 0;
            for (var j = 0; j < ${this.sliceDim}; j = j + 1) {
              let indexInside = i32(round(${this.indicesSnippet}));
              flattenedIndex = flattenedIndex + indexInside * ${
        this.strideString};
            }
            if (flattenedIndex == coords[0]) {
              sum = sum + ${this.updatesSnippet};
              found = true;
            }
          }
          setOutputFlat(index, mix(getDefaultValue(), sum, f32(found)));
        }
      }`;
    return userCode;
  }
}
