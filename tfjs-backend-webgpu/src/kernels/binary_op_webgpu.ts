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

import {backend_util, util} from '@tensorflow/tfjs-core';
import {getCoordsDataType} from '../shader_preprocessor';

import {computeDispatch, flatDispatchLayout} from '../webgpu_util';

import {WebGPUProgram} from './webgpu_program';

export const MUL = 'return a * b;';
export const ADD = 'return a + b;';
export const SUB = 'return a - b;';
export const DIV = 'return a / b;';
export const GREATER = 'return float(a > b);';
export const GREATER_EQUAL = 'return float(a >= b);';
export const LESS = `return float(a < b);`;
export const LESS_EQUAL = `return float(a <= b);`;

export const INT_DIV = `
  float s = sign(a) * sign(b);
  int ia = int(round(a));
  int ib = int(round(b));
  return float(idiv(ia, ib, s));
`;

export const PRELU = `return (a < 0.) ? b * a : a;`;

export class BinaryOpProgram implements WebGPUProgram {
  outputShape: number[];
  shaderKey: string;
  userCode: string;
  dispatchLayout: {x: number[]};
  dispatch: [number, number, number];
  variableNames = ['A', 'B'];
  workPerThread = 4;
  workGroupSize: [number, number, number] = [16, 1, 1];

  constructor(op: string, aShape: number[], bShape: number[]) {
    this.outputShape = backend_util.assertAndGetBroadcastShape(aShape, bShape);
    const size = util.sizeFromShape(this.outputShape);

    this.dispatchLayout = flatDispatchLayout(this.outputShape);
    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workGroupSize,
        [this.workPerThread, 1, 1]);
    const type = getCoordsDataType(this.outputShape.length);

    this.userCode = `
      float binaryOperation(float a, float b) {
        ${op}
      }

      void main() {
        int index = int(gl_GlobalInvocationID.x);

        for(int i = 0; i < ${this.workPerThread}; i++) {
          int flatIndex = index * ${this.workPerThread} + i;

          if(flatIndex < ${size}) {
            ${type} coords = getCoordsFromFlatIndex(flatIndex);

            float a = getAAtOutCoords(coords);
            float b = getBAtOutCoords(coords);
            setOutput(flatIndex, binaryOperation(a, b));
          }
        }
      }
    `;
    this.shaderKey = `binary${op}${type}${size}`;
  }
}
