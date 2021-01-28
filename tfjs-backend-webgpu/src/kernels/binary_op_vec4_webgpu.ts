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

import {computeDispatch, flatDispatchLayout} from '../webgpu_util';

import {WebGPUProgram} from './webgpu_program';

export class BinaryOpVec4Program implements WebGPUProgram {
  outputShape: number[];
  shaderKey: string;
  dispatchLayout: {x: number[]};
  dispatch: [number, number, number];
  variableNames = ['A', 'B'];
  workPerThread = 4;
  workGroupSize: [number, number, number];
  isVec4 = true;
  op: string;

  constructor(op: string, aShape: number[], bShape: number[]) {
    // TODO(jiajia.qin@intel.com): Heuristically select a good work group size.
    const workGroupSizeX = 128;
    this.workGroupSize = [workGroupSizeX, 1, 1];
    this.outputShape = backend_util.assertAndGetBroadcastShape(aShape, bShape);
    this.dispatchLayout = flatDispatchLayout(this.outputShape);
    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workGroupSize,
        [this.workPerThread, 1, 1]);
    this.op = op;
    this.shaderKey = `binaryVec4_${op}`;
  }

  getUserCode(): string {
    const size = util.sizeFromShape(this.outputShape) / this.workPerThread;
    const fitShape = size % this.workGroupSize[0] === 0;
    let userCode: string;
    if (fitShape) {
      userCode = `
      vec4 binaryOperation(vec4 a, vec4 b) {
        ${this.op}
      }

      void main() {
        int index = int(gl_GlobalInvocationID.x);
        vec4 a = A[index];
        vec4 b = B[index];
        setOutput(index, binaryOperation(a, b));
      }
    `;
    } else {
      userCode = `
      vec4 binaryOperation(vec4 a, vec4 b) {
        ${this.op}
      }

      void main() {
        int index = int(gl_GlobalInvocationID.x);
        if (index < ${size})
        {
          vec4 a = A[index];
          vec4 b = B[index];
          setOutput(index, binaryOperation(a, b));
        }
      }
    `;
    }
    return userCode;
  }
}
