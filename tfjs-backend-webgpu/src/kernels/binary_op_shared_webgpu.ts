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

import {backend_util, util} from '@tensorflow/tfjs-core';
import {getCoordsDataType} from '../shader_preprocessor';

import {computeDispatch, flatDispatchLayout} from '../webgpu_util';

import {WebGPUProgram} from './webgpu_program';

export class BinaryOpSharedProgram implements WebGPUProgram {
  outputShape: number[];
  shaderKey: string;
  dispatchLayout: {x: number[]};
  dispatch: [number, number, number];
  variableNames = ['A', 'B'];
  workPerThread: number;
  workGroupSize: [number, number, number];
  useSharedMemoryWithB: boolean;
  lastDimensionSize: number;
  op: string;

  constructor(
      op: string, aShape: number[], bShape: number[],
      useSharedMemoryWithB: boolean) {
    // This is an experimental value when using shared memory.
    const workGroupSizeX = 512;
    this.workGroupSize = [workGroupSizeX, 1, 1];
    this.outputShape = backend_util.assertAndGetBroadcastShape(aShape, bShape);
    this.dispatchLayout = flatDispatchLayout(this.outputShape);
    this.lastDimensionSize = useSharedMemoryWithB ? bShape[0] : aShape[0];
    if (this.lastDimensionSize < 512) {
      this.workPerThread = 1;
    } else if (this.lastDimensionSize < 1024) {
      this.workPerThread = 2;
    } else {
      this.workPerThread = 4;
    }
    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workGroupSize,
        [this.workPerThread, 1, 1]);
    this.shaderKey = `binaryShared_${op}`;
    this.useSharedMemoryWithB = useSharedMemoryWithB;
    this.op = op;
  }

  getUserCode(): string {
    const type = getCoordsDataType(this.outputShape.length);
    const size = util.sizeFromShape(this.outputShape);
    const sharedIndexSnippet = this.lastDimensionSize > 1 ?
        `coords[${this.outputShape.length - 1}]` :
        '0';
    const accessDataSnippet = this.useSharedMemoryWithB ?
        `float a = getAAtOutCoords(coords);
         float b = sharedBuf[${sharedIndexSnippet}];` :
        `float a = sharedBuf[${sharedIndexSnippet}];
         float b = getBAtOutCoords(coords);`;
    const sizeFit = size % (this.workGroupSize[0] * this.workPerThread) === 0;
    const writeDataSnippet = sizeFit ?
        `${type} coords = getCoordsFromFlatIndex(flatIndex);

         ${accessDataSnippet}
         setOutput(flatIndex, binaryOperation(a, b));` :
        `if(flatIndex < ${size}) {
            ${type} coords = getCoordsFromFlatIndex(flatIndex);

            ${accessDataSnippet}
            setOutput(flatIndex, binaryOperation(a, b));
          }`;
    const userCode = `
        float binaryOperation(float a, float b) {
          ${this.op}
        }

        shared float sharedBuf[${this.lastDimensionSize}];
        void main() {
          int index = int(gl_GlobalInvocationID.x);
          int localIndex = int(gl_LocalInvocationIndex);

          // Fill in the shared memory buffer. Here we need a loop to make sure
          // that all data in A|B are uploaded when |sharedMemorySize| is larger
          // than work group size.
          while(localIndex < ${this.lastDimensionSize})
          {
            sharedBuf[localIndex] = ${
        this.useSharedMemoryWithB ? 'B' : 'A'}[localIndex];
            localIndex += int(gl_WorkGroupSize.x);
          }
          barrier();

          for(int i = 0; i < ${this.workPerThread}; i++) {
            int flatIndex = index * ${this.workPerThread} + i;

            ${writeDataSnippet}
          }
        }
        `;
    return userCode;
  }
}
