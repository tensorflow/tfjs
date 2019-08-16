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

import {backend_util} from '@tensorflow/tfjs-core';
import {getCoordsDataType} from '../shader_preprocessor';

import {computeDispatch} from '../webgpu_util';

import {WebGPUProgram} from './webgpu_program';

export const MUL = 'return a * b;';
export const ADD = 'return a + b;';
export const SUB = 'return a - b;';

export const INT_DIV = `
  float s = sign(a) * sign(b);
  int ia = int(round(a));
  int ib = int(round(b));
  return float(idiv(ia, ib, s));
`;

export class BinaryOpProgram implements WebGPUProgram {
  outputShape: number[];
  userCode: string;
  dispatchLayout: {x: number[]};
  dispatch: [number, number, number];
  variableNames = ['A', 'B'];

  constructor(op: string, aShape: number[], bShape: number[]) {
    this.outputShape = backend_util.assertAndGetBroadcastShape(aShape, bShape);

    this.dispatchLayout = {x: this.outputShape.map((d, i) => i)};
    this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape);
    const type = getCoordsDataType(this.outputShape.length);
    const workPerThread = 2;

    this.userCode = `
      float binaryOperation(float a, float b) {
        ${op}
      }

      void main() {
        uint index = gl_GlobalInvocationID.x;

        if(mod(index, ${workPerThread}) == 0) {
          for(uint i = 0; i < ${workPerThread}; i++) {
            if(index + i < ${this.dispatch[0]}) {
              ${type} coords = getCoords(index + i);

              float a = getA(coords);
              float b = getB(coords);

              setOutput(index + i, binaryOperation(a, b));
            }
          }
        }
      }
    `;
  }
}
