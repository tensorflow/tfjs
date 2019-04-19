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

export const RELU = 'return max(a, 0.0);';

export class UnaryOpProgram implements WebGPUProgram {
  outputShape: number[];
  userCode: string;
  dispatch: [number, number, number];
  variableNames = ['A'];

  constructor(op: string, outputShape: number[]) {
    this.outputShape = outputShape;
    this.dispatch = [util.sizeFromShape(this.outputShape), 1, 1];

    this.userCode = `
      float unaryOperation(float a) {
        ${op}
      }

      void main() {
        uint index = gl_GlobalInvocationID.x;
        float a = A[index];
        setOutput(index, unaryOperation(a));
      }
    `;
  }
}
