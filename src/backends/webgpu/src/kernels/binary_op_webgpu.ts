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
import {WebGPUProgram} from './webgpu_program';

export const MUL = 'return a * b;';
export const ADD = 'return a + b;';

export class BinaryOpProgram implements WebGPUProgram {
  outputShape: number[];
  userCode: string;
  dispatch: [number, number, number];
  variableNames = ['A', 'B'];

  constructor(op: string, outputShape: number[]) {
    this.outputShape = outputShape;
    this.dispatch = [util.sizeFromShape(this.outputShape), 1, 1];

    this.userCode = `
      float binaryOperation(float a, float b) {
        ${op}
      }

      void main() {
        uint index = gl_GlobalInvocationID.x;
        float a = A[index];
        float b = B[index];
        setOutput(index, binaryOperation(a, b));
      }
    `;
  }
}
