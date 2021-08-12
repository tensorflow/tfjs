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
import {getWorkGroupSizeStringWgsl} from '../shader_preprocessor_wgsl';
import {computeDispatch, flatDispatchLayout} from '../webgpu_util';
import {BinaryOpType, getBinaryOpString} from './binary_op_util';

import {getUseWgsl, WebGPUProgram} from './webgpu_program';

export class BinaryOpVec4Program implements WebGPUProgram {
  outputShape: number[];
  shaderKey: string;
  dispatchLayout: {x: number[]};
  dispatch: [number, number, number];
  variableNames = ['A', 'B'];
  workPerThread = 4;
  workGroupSize: [number, number, number];
  isVec4 = true;
  useWgsl: boolean;
  op: BinaryOpType;
  size: number;
  fitShape: boolean;

  constructor(op: BinaryOpType, aShape: number[], bShape: number[]) {
    // TODO(jiajia.qin@intel.com): Heuristically select a good work group size.
    const workGroupSizeX = 128;
    this.workGroupSize = [workGroupSizeX, 1, 1];
    this.outputShape = backend_util.assertAndGetBroadcastShape(aShape, bShape);
    this.dispatchLayout = flatDispatchLayout(this.outputShape);
    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workGroupSize,
        [this.workPerThread, 1, 1]);
    this.op = op;
    this.fitShape = this.size % this.workGroupSize[0] === 0;
    this.shaderKey = `binaryVec4_${op}_${this.fitShape}`;
    this.size = util.sizeFromShape(this.outputShape) / this.workPerThread;
    this.useWgsl = getUseWgsl();
  }

  getUserCode(): string {
    let userCode: string;
    const opStr = getBinaryOpString(this.op, this.isVec4);
    if (this.fitShape) {
      userCode = `
      vec4 binaryOperation(vec4 a, vec4 b) {
        ${opStr}
      }

      void main() {
        int index = int(gl_GlobalInvocationID.x);
        vec4 a = vec4(A[index]);
        vec4 b = vec4(B[index]);
        setOutput(index, binaryOperation(a, b));
      }
    `;
    } else {
      userCode = `
      vec4 binaryOperation(vec4 a, vec4 b) {
        ${opStr}
      }

      void main() {
        int index = int(gl_GlobalInvocationID.x);
        if (index < size)
        {
          vec4 a = vec4(A[index]);
          vec4 b = vec4(B[index]);
          setOutput(index, binaryOperation(a, b));
        }
      }
    `;
    }
    return userCode;
  }

  getUserCodeWgsl(): string {
    let userCode: string;
    const opStr = getBinaryOpString(this.op, this.isVec4, this.useWgsl);
    const miscStr =
        `fn binaryOperation(a : vec4<f32>, b : vec4<f32>) -> vec4<f32> {
          ${opStr}
        }`;

    if (this.fitShape) {
      userCode = `
      ${miscStr}
      ${getWorkGroupSizeStringWgsl(this.workGroupSize)}
      fn main([[builtin(global_invocation_id)]] global_id : vec3<u32>) {
        let index = global_id.x;
        let a = vec4<f32>(A.numbers[index]);
        let b = vec4<f32>(B.numbers[index]);
        setOutputFlat(index, binaryOperation(a, b));
      }
    `;
    } else {
      userCode = `
      ${miscStr}
      ${getWorkGroupSizeStringWgsl(this.workGroupSize)}
      fn main([[builtin(global_invocation_id)]] global_id : vec3<u32>) {
        let index = global_id.x;
        if (index < uniforms.size) {
          let a = vec4<f32>(A.numbers[index]);
          let b = vec4<f32>(B.numbers[index]);
          setOutputFlat(index, binaryOperation(a, b));
        }
      }
    `;
    }
    return userCode;
  }
}
