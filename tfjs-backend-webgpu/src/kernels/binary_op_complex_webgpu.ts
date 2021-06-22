/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
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

import {computeDispatch, flatDispatchLayout} from '../webgpu_util';
import {BinaryOpType, getBinaryOpString} from './binary_op_util';

import {WebGPUProgram} from './webgpu_program';

export class BinaryOpComplexProgram implements WebGPUProgram {
  variableNames = ['AReal', 'AImag', 'BReal', 'BImag'];
  outputShape: number[];
  shaderKey: string;
  dispatchLayout: {x: number[]};
  dispatch: [number, number, number];
  workGroupSize: [number, number, number] = [128, 1, 1];
  op: BinaryOpType;
  size: number;

  constructor(op: BinaryOpType, aShape: number[], bShape: number[]) {
    this.outputShape = backend_util.assertAndGetBroadcastShape(aShape, bShape);
    this.dispatchLayout = flatDispatchLayout(this.outputShape);
    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workGroupSize);

    this.shaderKey = `binaryOpComplex_${op}`;
    this.op = op;
    this.size = util.sizeFromShape(this.outputShape);
  }

  getUserCode(): string {
    const opStr = getBinaryOpString(this.op);
    const userCode = `
      float binaryOpComplex(
          float areal, float aimag, float breal, float bimag) {
        ${opStr}
      }

      void main() {
        int index = int(gl_GlobalInvocationID.x);
        if(index < size) {
          float areal = getARealAtOutCoords();
          float aimag = getAImagAtOutCoords();
          float breal = getBRealAtOutCoords();
          float bimag = getBImagAtOutCoords();
          setOutput(index, binaryOpComplex(areal, aimag, breal, bimag));
        }
      }
    `;
    return userCode;
  }
}
