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

import {getMainHeaderAndGlobalIndexString} from '../shader_preprocessor';
import {computeDispatch, flatDispatchLayout} from '../webgpu_util';
import {BinaryOpType, getBinaryOpString} from './binary_op_util';

import {WebGPUProgram} from './webgpu_program';

export class BinaryOpSharedProgram implements WebGPUProgram {
  outputShape: number[];
  shaderKey: string;
  dispatchLayout: {x: number[]};
  dispatch: [number, number, number];
  variableNames = ['A', 'B'];
  workPerThread = 4;
  workGroupSize: [number, number, number];
  useSharedMemoryWithB: boolean;
  isScater: boolean;
  op: BinaryOpType;
  size = true;

  constructor(
      op: BinaryOpType, outputShape: number[], useSharedMemoryWithB: boolean,
      isScater: boolean) {
    // This is an experimental value when using shared memory.
    // Note that the maximum of workgroup X dimension is 256.
    const workGroupSizeX = 256;
    this.workGroupSize = [workGroupSizeX, 1, 1];
    this.outputShape = outputShape;
    this.dispatchLayout = flatDispatchLayout(this.outputShape);
    this.isScater = isScater;
    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workGroupSize,
        [this.workPerThread, 1, 1]);

    this.useSharedMemoryWithB = useSharedMemoryWithB;
    this.op = op;
    this.shaderKey =
        `binaryShared_${op}_${this.useSharedMemoryWithB}_${isScater}`;
  }

  getUserCode(): string {
    const sharedIndexSnippet =
        this.isScater ? '0' : `coords[${this.outputShape.length - 1}]`;
    const accessDataSnippet = this.useSharedMemoryWithB ?
        `let a = getAAtOutCoordsByGlobalIndex(flatIndex);
         let b = sharedBuf[${sharedIndexSnippet}];` :
        `let a = sharedBuf[${sharedIndexSnippet}];
         let b = getBAtOutCoordsByGlobalIndex(flatIndex);`;

    const userCode = `
        fn binaryOperation(a : f32, b : f32) -> f32 {
          ${getBinaryOpString(this.op, false)}
        }

        var<workgroup> sharedBuf : array<f32, ${
        this.workGroupSize[0] * this.workPerThread}>;
        ${getMainHeaderAndGlobalIndexString()}

          // Fill in the shared memory buffer. Here we need a loop to make sure
          // that all data in A|B are uploaded when |sharedMemorySize| is larger
          // than work group size.
          for(var localIndex = i32(localId.x); localIndex < ${
        this.useSharedMemoryWithB ? 'uniforms.bShape' : 'uniforms.aShape'};
              localIndex = localIndex + ${this.workGroupSize[0]}) {
            sharedBuf[localIndex] = f32(${
        this.useSharedMemoryWithB ? 'B' : 'A'}.numbers[localIndex]);
          }
          workgroupBarrier();

          for(var i = 0; i < ${this.workPerThread}; i = i + 1) {
            let flatIndex = index * ${this.workPerThread} + i;
            if(flatIndex < uniforms.size) {
              let coords = getCoordsFromFlatIndex(flatIndex);

              ${accessDataSnippet}
              setOutputFlat(flatIndex, binaryOperation(a, b));
            }
          }
        }
        `;
    return userCode;
  }
}
