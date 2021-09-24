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

import {getGlobalIndexString, getMainHeaderString} from '../shader_preprocessor';
import {computeDispatch, flatDispatchLayout} from '../webgpu_util';

import {WebGPUProgram} from './webgpu_program';

export class FillProgram implements WebGPUProgram {
  variableNames: string[] = [];
  outputShape: number[] = [];
  shaderKey: string;
  dispatchLayout: {x: number[]};
  dispatch: [number, number, number];
  uniforms = 'value : f32;';
  workPerThread = 4;
  workGroupSize: [number, number, number] = [16, 1, 1];
  size: number;

  constructor(shape: number[]) {
    this.outputShape = shape;
    this.dispatchLayout = flatDispatchLayout(this.outputShape);
    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workGroupSize,
        [this.workPerThread, 1, 1]);

    this.shaderKey = 'fill';
    this.size = util.sizeFromShape(this.outputShape);
  }

  getUserCode(): string {
    const userCode = `
    ${getMainHeaderString()} {
      ${getGlobalIndexString()}
      for (var i = 0; i < ${this.workPerThread}; i = i + 1) {
        let flatIndex = index * ${this.workPerThread} + i;
        if (flatIndex < uniforms.size) {
          setOutputFlat(flatIndex, uniforms.value);
        }
      }
    }
  `;
    return userCode;
  }
}
