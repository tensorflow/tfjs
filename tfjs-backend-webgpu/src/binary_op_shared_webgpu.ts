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

import {backend_util} from '@tensorflow/tfjs-core';
import {BinaryOpType, getBinaryOpString} from './binary_op_util';
import {getMainHeaderAndGlobalIndexString, WebGPUProgram} from './webgpu_program';
import {computeDispatch, flatDispatchLayout} from './webgpu_util';

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
  op: BinaryOpType;
  size = true;

  constructor(
      op: BinaryOpType, aShape: number[], bShape: number[],
      useSharedMemoryWithB: boolean) {
    // This is an experimental value when using shared memory.
    // Note that the maximum of workgroup X dimension is 256.
    const workGroupSizeX = 256;
    this.workGroupSize = [workGroupSizeX, 1, 1];
    this.outputShape = backend_util.assertAndGetBroadcastShape(aShape, bShape);
    this.dispatchLayout = flatDispatchLayout(this.outputShape);
    this.lastDimensionSize = useSharedMemoryWithB ? bShape[0] : aShape[0];
    if (this.lastDimensionSize < 256) {
      this.workPerThread = 1;
    } else if (this.lastDimensionSize < 512) {
      this.workPerThread = 2;
    } else {
      this.workPerThread = 4;
    }
    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workGroupSize,
        [this.workPerThread, 1, 1]);

    this.useSharedMemoryWithB = useSharedMemoryWithB;
    this.op = op;
    // this.lastDimensionSize is used as sharedBuf array size, so can not be
    // used as uniform.
    this.shaderKey = `binaryShared_${op}_${this.lastDimensionSize}_${
        this.useSharedMemoryWithB}`;
  }

  getUserCode(): string {
    const sharedIndexSnippet = this.lastDimensionSize > 1 ?
        `coords[${this.outputShape.length - 1}]` :
        '0';
    const accessDataSnippet = this.useSharedMemoryWithB ?
        `let a = getAByOutputCoords(coords);
         let b = sharedBuf[${sharedIndexSnippet}];` :
        `let a = sharedBuf[${sharedIndexSnippet}];
         let b = getBByOutputCoords(coords);`;

    const opStr = getBinaryOpString(this.op, false);
    const userCode = `
        fn binaryOperation(a : f32, b : f32) -> f32 {
          ${opStr}
        }
        var<workgroup> sharedBuf : array<f32, ${this.lastDimensionSize}>;
        ${getMainHeaderAndGlobalIndexString()}

          // Fill in the shared memory buffer. Here we need a loop to make sure
          // that all data in A|B are uploaded when |sharedMemorySize| is larger
          // than work group size.
          for(var localIndex = i32(localId.x); localIndex < ${
        this.lastDimensionSize}; localIndex = localIndex + ${
        this.workGroupSize[0]}) {
            sharedBuf[localIndex] = f32(${
        this.useSharedMemoryWithB ? 'B' : 'A'}[localIndex]);
          }
          workgroupBarrier();

          for(var i = 0; i < ${this.workPerThread}; i = i + 1) {
            let flatIndex = index * ${this.workPerThread} + i;
            if(flatIndex < uniforms.size) {
              let coords = getCoordsFromIndex(flatIndex);

              ${accessDataSnippet}
              setOutputAtIndex(flatIndex, binaryOperation(a, b));
            }
          }
        }
        `;
    return userCode;
  }
}
