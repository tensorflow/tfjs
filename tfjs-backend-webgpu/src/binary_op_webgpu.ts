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

import {BinaryOpType, getBinaryOpString} from './binary_op_util';
import {getMainHeaderString as main, WebGPUProgram} from './webgpu_program';
import {computeDispatch, flatDispatchLayout} from './webgpu_util';

export class BinaryOpProgram implements WebGPUProgram {
  dispatch: [number, number, number];
  dispatchLayout: {x: number[]};
  isVec4: boolean;
  op: BinaryOpType;
  outputShape: number[];
  shaderKey: string;
  size = true;
  variableNames = ['A', 'B'];
  workgroupSize: [number, number, number];
  workPerThread: number;

  private lastDimensionSize: number;
  private useSharedMemoryWithA: boolean;
  private useSharedMemoryWithB: boolean;
  private type: string;

  constructor(op: BinaryOpType, aShape: number[], bShape: number[]) {
    this.outputShape = backend_util.assertAndGetBroadcastShape(aShape, bShape);
    this.dispatchLayout = flatDispatchLayout(this.outputShape);
    this.op = op;

    this.useSharedMemoryWithA =
        aShape.length <= 1 && bShape.length > 1 && aShape[0] < 128;
    this.useSharedMemoryWithB =
        bShape.length <= 1 && aShape.length > 1 && bShape[0] < 128;

    if (this.useSharedMemoryWithA || this.useSharedMemoryWithB) {
      this.isVec4 = false;
      // lastDimensionSize is used as sharedBuf array size, so can not be
      // used as uniform.
      this.lastDimensionSize =
          this.useSharedMemoryWithB ? bShape[0] : aShape[0];
      this.shaderKey = `binary_${this.type}_${op}_${this.lastDimensionSize}_${
          this.useSharedMemoryWithB}`;
      this.type = 'shared';
      // This is an experimental value when using shared memory.
      // Note that the maximum of workgroup X dimension is 256.
      this.workgroupSize = [256, 1, 1];
      this.workPerThread = 1;
    } else {
      if (util.arraysEqual(aShape, bShape) &&
          util.sizeFromShape(aShape) % 4 === 0) {
        this.isVec4 = true;
        this.type = 'vec4';
        this.workPerThread = 4;
      } else {
        this.isVec4 = false;
        this.type = 'plain';
        this.workPerThread = 1;
      }
      this.shaderKey = `binary_${this.type}_${op}`;
      // TODO(jiajia.qin@intel.com): Heuristically select a good work group
      // size.
      this.workgroupSize = [128, 1, 1];
    }
    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workgroupSize,
        [this.workPerThread, 1, 1]);
  }

  getUserCode(): string {
    let userCode;
    const dType = this.isVec4 ? 'vec4<f32>' : 'f32';
    const opFnStr = `
    fn binaryOperation(a : ${dType}, b : ${dType}) -> ${dType} {
      ${getBinaryOpString(this.op, this.isVec4)}
    };
    `;

    if (this.type === 'shared') {
      const sharedIndexSnippet = this.lastDimensionSize > 1 ?
          `coords[${this.outputShape.length - 1}]` :
          '0';
      const accessDataSnippet = this.useSharedMemoryWithB ?
          `let a = getAByOutputIndex(index);
          let b = sharedBuf[${sharedIndexSnippet}];` :
          `let a = sharedBuf[${sharedIndexSnippet}];
          let b = getBByOutputIndex(index);`;
      userCode = `
        ${opFnStr}
        var<workgroup> sharedBuf : array<f32, ${this.lastDimensionSize}>;
        ${main('index')} {
          // Fill in the shared memory buffer.
          let localIndex = i32(localId.x);
          if(localIndex < ${this.lastDimensionSize}) {
            sharedBuf[localIndex] = f32(${
          this.useSharedMemoryWithB ? 'B' : 'A'}[localIndex]);
          }
          workgroupBarrier();

          if(index < uniforms.size) {
            let coords = getCoordsFromIndex(index);
            ${accessDataSnippet}
            setOutputAtIndex(index, binaryOperation(a, b));
          }
        }
        `;
    } else {
      userCode = `
       ${opFnStr}
       ${main('index')} {
         if (index < uniforms.size) {
           let a = getAByOutputIndex(index);
           let b = getBByOutputIndex(index);
           setOutputAtIndex(index, binaryOperation(a, b));
         }
       }
       `;
    }

    return userCode;
  }
}
