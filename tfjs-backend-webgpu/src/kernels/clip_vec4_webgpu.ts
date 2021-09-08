/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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

import {getGlobalIndexStringWgsl, getMainHeaderStringWgsl} from '../shader_preprocessor_wgsl';
import {computeDispatch, flatDispatchLayout} from '../webgpu_util';

import {getUseWgsl, WebGPUProgram} from './webgpu_program';

export class ClipVec4Program implements WebGPUProgram {
  outputShape: number[];
  shaderKey: string;
  variableNames = ['A'];
  uniforms = 'float minVal; float maxVal;';
  uniformsWgsl = 'minVal : f32; maxVal : f32;';
  dispatchLayout: {x: number[]};
  dispatch: [number, number, number];
  workPerThread = 4;
  workGroupSize: [number, number, number] = [64, 1, 1];
  isVec4 = true;
  size: number;
  useWgsl: boolean;

  constructor(outputShape: number[]) {
    this.outputShape = outputShape;
    this.dispatchLayout = flatDispatchLayout(this.outputShape);
    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workGroupSize,
        [this.workPerThread, 1, 1]);
    this.shaderKey = 'clipVec4';
    this.size = util.sizeFromShape(this.outputShape) / 4;
    this.useWgsl = getUseWgsl();
  }

  getUserCode(): string {
    const userCode = `
      void main() {
        int index = getGlobalIndex();
          if(index < size) {
            vec4 value = getAAtOutCoords();
            vec4 clampedValue;
            for (int i = 0; i < 4; ++i) {
              if (isnan(value[i])) {
                clampedValue[i] = value[i];
              } else {
                clampedValue[i] = clamp(value[i], minVal, maxVal);
              }
            }

            setOutput(index, clampedValue);
          }
      }
    `;
    return userCode;
  }

  getUserCodeWgsl(): string {
    const userCode = `
      ${getMainHeaderStringWgsl(this.workGroupSize)} {
        ${getGlobalIndexStringWgsl(this.workGroupSize)}
        if(index < uniforms.size) {
          let value = getAAtOutCoordsByGlobalId(globalId, index);
          var clampedValue : vec4<f32>;
          for (var i = 0u; i < 4u; i = i + 1u) {
            if (isNanCustom(value[i])) {
              clampedValue[i] = value[i];
            } else {
              clampedValue[i] = clamp(value[i], uniforms.minVal, uniforms.maxVal);
            }
          }

          setOutputFlat(index, clampedValue);
        }
      }
    `;
    return userCode;
  }
}
