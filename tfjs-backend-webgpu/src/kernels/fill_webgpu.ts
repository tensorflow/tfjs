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
import {computeDispatch, flatDispatchLayout} from '../webgpu_util';
import {WebGPUProgram} from './webgpu_program';

export class FillProgram implements WebGPUProgram {
  variableNames: string[] = [];
  outputShape: number[] = [];
  shaderKey: string;
  dispatchLayout: {x: number[]};
  dispatch: [number, number, number];
  uniforms = 'float value;';
  workPerThread = 4;
  workGroupSize: [number, number, number] = [16, 1, 1];

  constructor(shape: number[]) {
    this.outputShape = shape;
    this.dispatchLayout = flatDispatchLayout(this.outputShape);
    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workGroupSize,
        [this.workPerThread, 1, 1]);

    this.shaderKey = 'fill';
  }

  getUserCode(): string {
    const size = util.sizeFromShape(this.outputShape);
    const userCode = `
    void main() {
      int index = int(gl_GlobalInvocationID.x);
      for (int i = 0; i < ${this.workPerThread}; i++) {
        int flatIndex = index * ${this.workPerThread} + i;
        if (flatIndex < ${size}) {
          setOutput(flatIndex, value);
        }
      }
    }
  `;
    return userCode;
  }
}
