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
import {getMainHeaderAndGlobalIndexString, WebGPUProgram} from './webgpu_program';
import {computeDispatch, flatDispatchLayout} from './webgpu_util';

export class BinaryOpProgram implements WebGPUProgram {
  dispatch: [number, number, number];
  dispatchLayout: {x: number[]};
  isVec4: boolean;
  op: BinaryOpType;
  outputShape: number[];
  shaderKey: string;
  size = true;
  userCode: string;
  variableNames = ['A', 'B'];
  workGroupSize: [number, number, number];
  workPerThread: number;

  constructor(op: BinaryOpType, aShape: number[], bShape: number[]) {
    this.outputShape = backend_util.assertAndGetBroadcastShape(aShape, bShape);
    this.dispatchLayout = flatDispatchLayout(this.outputShape);
    this.op = op;

    const useSharedMemoryWithA =
        aShape.length === 1 && bShape.length > 1 && aShape[0] < 1024;
    const useSharedMemoryWithB =
        bShape.length === 1 && aShape.length > 1 && bShape[0] < 1024;

    if (useSharedMemoryWithA || useSharedMemoryWithB) {
      this.isVec4 = false;
      // This is an experimental value when using shared memory.
      // Note that the maximum of workgroup X dimension is 256.
      this.workGroupSize = [256, 1, 1];

      // lastDimensionSize is used as sharedBuf array size, so can not be
      // used as uniform.
      const lastDimensionSize = useSharedMemoryWithB ? bShape[0] : aShape[0];
      if (lastDimensionSize < 256) {
        this.workPerThread = 1;
      } else if (lastDimensionSize < 512) {
        this.workPerThread = 2;
      } else {
        this.workPerThread = 4;
      }
      this.dispatch = computeDispatch(
          this.dispatchLayout, this.outputShape, this.workGroupSize,
          [this.workPerThread, 1, 1]);
      this.shaderKey =
          `binaryShared_${op}_${lastDimensionSize}_${useSharedMemoryWithB}`;

      const sharedIndexSnippet = lastDimensionSize > 1 ?
          `coords[${this.outputShape.length - 1}]` :
          '0';
      const accessDataSnippet = useSharedMemoryWithB ?
          `let a = getAByOutputCoords(coords);
          let b = sharedBuf[${sharedIndexSnippet}];` :
          `let a = sharedBuf[${sharedIndexSnippet}];
          let b = getBByOutputCoords(coords);`;
      const opStr = getBinaryOpString(this.op, this.isVec4);
      this.userCode = `
         fn binaryOperation(a : f32, b : f32) -> f32 {
           ${opStr}
         }
         var<workgroup> sharedBuf : array<f32, ${lastDimensionSize}>;
         ${getMainHeaderAndGlobalIndexString()}

           // Fill in the shared memory buffer. Here we need a loop to make sure
           // that all data in A|B are uploaded when |sharedMemorySize| is larger
           // than work group size.
           for(var localIndex = i32(localId.x); localIndex < ${
          lastDimensionSize}; localIndex = localIndex + ${
          this.workGroupSize[0]}) {
             sharedBuf[localIndex] = f32(${
          useSharedMemoryWithB ? 'B' : 'A'}[localIndex]);
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
    } else {
      let dType;

      if (util.arraysEqual(aShape, bShape) &&
          util.sizeFromShape(aShape) % 4 === 0) {
        dType = 'vec4<f32>';
        this.isVec4 = true;
        this.shaderKey = `binaryVec4_${op}`;
      } else {
        dType = 'f32';
        this.isVec4 = false;
        this.shaderKey = `binary_${op}`;
      }
      // TODO(jiajia.qin@intel.com): Heuristically select a good work group
      // size.
      this.workGroupSize = [128, 1, 1];
      this.workPerThread = 4;
      this.dispatch = computeDispatch(
          this.dispatchLayout, this.outputShape, this.workGroupSize,
          [this.workPerThread, 1, 1]);
      const opStr = getBinaryOpString(this.op, this.isVec4);
      this.userCode = `
       fn binaryOperation(a : ${dType}, b : ${dType}) -> ${dType} {
         ${opStr}
       }
       ${getMainHeaderAndGlobalIndexString()}
         if (index < uniforms.size) {
           let a = getAByOutputIndex(index);
           let b = getBByOutputIndex(index);
           setOutputAtIndex(index, binaryOperation(a, b));
         }
       }
       `;
    }
  }

  getUserCode(): string {
    return this.userCode;
  }
}
