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
export const SQUARED_DIFFERENCE = 'return (a - b) * (a - b);';

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
  workPerThread: number;
  workGroupSize: [number, number, number];

  constructor(op: string, aShape: number[], bShape: number[]) {
    // This is an experimental value when using shared memory.
    const workGroupSizeX = 512;
    this.workGroupSize = [workGroupSizeX, 1, 1];
    this.outputShape = backend_util.assertAndGetBroadcastShape(aShape, bShape);
    this.dispatchLayout = flatDispatchLayout(this.outputShape);
    const size = util.sizeFromShape(this.outputShape);
    const fit = util.arraysEqual(aShape, bShape) && size % workGroupSizeX === 0;
    this.workPerThread = fit ? 1 : 2;
    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workGroupSize,
        [this.workPerThread, 1, 1]);

    // TODO(jiajia.qin@intel.com): More tuning on this. Maybe split the shared
    // memory shader into a new class.
    const useSharedMemoryWithA =
        aShape.length === 1 && bShape.length > 1 && aShape[0] < 2048;
    const useSharedMemoryWithB =
        bShape.length === 1 && aShape.length > 1 && bShape[0] < 2048;

    if (fit) {
      this.userCode = `
          float binaryOperation(float a, float b) {
            ${op}
          }

          void main() {
            int index = int(gl_GlobalInvocationID.x);

            float a = A[index];
            float b = B[index];
            setOutput(index, binaryOperation(a, b));
          }
        `;
      this.shaderKey = `binary2${op}`;
    } else if (useSharedMemoryWithA || useSharedMemoryWithB) {
      const type = getCoordsDataType(this.outputShape.length);
      const sharedMemorySize = useSharedMemoryWithB ? bShape[0] : aShape[0];
      const sharedIndexSnippet =
          sharedMemorySize > 1 ? `coords[${this.outputShape.length - 1}]` : '0';
      const accessDataSnippet = useSharedMemoryWithB ?
          `            float a = getAAtOutCoords(coords);
                       float b = sharedBuf[${sharedIndexSnippet}];` :
          `            float a = sharedBuf[${sharedIndexSnippet}];
                       float b = getBAtOutCoords(coords);
      `;
      this.userCode = `
      float binaryOperation(float a, float b) {
        ${op}
      }

      shared float sharedBuf[${sharedMemorySize}];
      void main() {
        int index = int(gl_GlobalInvocationID.x);
        int localIndex = int(gl_LocalInvocationIndex);

        // Fill in the shared memory buffer. Here we need a loop to make sure
        // that all data in A|B are uploaded when |sharedMemorySize| is larger
        // than work group size.
        while(localIndex < ${sharedMemorySize})
        {
          sharedBuf[localIndex] = ${
          useSharedMemoryWithB ? 'B' : 'A'}[localIndex];
          localIndex += int(gl_WorkGroupSize.x);
        }
        barrier();

        for(int i = 0; i < ${this.workPerThread}; i++) {
          int flatIndex = index * ${this.workPerThread} + i;

          if(flatIndex < ${size}) {
            ${type} coords = getCoordsFromFlatIndex(flatIndex);

            ${accessDataSnippet}
            setOutput(flatIndex, binaryOperation(a, b));
          }
        }
      }
      `;
    } else {
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
}
