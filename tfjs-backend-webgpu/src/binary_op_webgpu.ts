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
  outputComponent: number;
  op: BinaryOpType;
  outputShape: number[];
  shaderKey: string;
  size = true;
  variableNames = ['A', 'B'];
  workgroupSize: [number, number, number];
  variableComponents: number[];

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
      this.outputComponent = 1;
      this.variableComponents = [1, 1];
      // lastDimensionSize is used as sharedBuf array size, so can not be
      // used as uniform.
      this.lastDimensionSize =
          this.useSharedMemoryWithB ? bShape[0] : aShape[0];
      this.shaderKey = `binary_${op}_${this.lastDimensionSize}`;
      this.type = 'shared';
      // This is an experimental value when using shared memory.
      // Note that the maximum of workgroup X dimension is 256.
      this.workgroupSize = [256, 1, 1];
    } else {
      const aDivisibleBy4 =
          aShape.length > 0 && aShape[aShape.length - 1] % 4 === 0;
      const bDivisibleBy4 =
          bShape.length > 0 && bShape[bShape.length - 1] % 4 === 0;
      if (aDivisibleBy4 && bDivisibleBy4) {
        this.outputComponent = 4;
        this.variableComponents = [4, 4];
      } else if (
          (aDivisibleBy4 &&
           (util.isScalarShape(bShape) || bShape[bShape.length - 1] === 1)) ||
          (bDivisibleBy4 &&
           (util.isScalarShape(aShape) || aShape[aShape.length - 1] === 1))) {
        this.outputComponent = 4;
        this.variableComponents = aDivisibleBy4 ? [4, 1] : [1, 4];
      } else {
        this.outputComponent = 1;
        this.variableComponents = [1, 1];
      }
      this.type = 'nonshared';
      this.shaderKey = `binary_${op}_${this.variableComponents}`;
      // TODO(jiajia.qin@intel.com): Heuristically select a good work group
      // size.
      this.workgroupSize = [128, 1, 1];
    }
    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workgroupSize,
        [this.outputComponent, 1, 1]);
  }

  getUserCode(): string {
    let userCode;
    const dType = this.outputComponent === 4 ? 'vec4<f32>' : 'f32';
    const opFnStr = `
    fn binaryOperation(a : ${dType}, b : ${dType}) -> ${dType} {
      ${getBinaryOpString(this.op, this.outputComponent === 4)}
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
           let coords = getCoordsFromIndex(index * ${this.outputComponent});
           let a = ${dType}(getAByOutputCoords(coords));
           let b = ${dType}(getBByOutputCoords(coords));
           setOutputAtIndex(index, binaryOperation(a, b));
         }
       }
       `;
    }

    return userCode;
  }
}
