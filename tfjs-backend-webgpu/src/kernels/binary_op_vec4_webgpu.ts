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
import {getGlobalIndexStringWgsl, getMainHeaderStringWgsl} from '../shader_preprocessor_wgsl';
import {computeDispatch, flatDispatchLayout} from '../webgpu_util';
import {BinaryOpType, getBinaryOpString} from './binary_op_util';

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
  }

  getUserCodeWgsl(): string {
    let userCode: string;
    const opStr = getBinaryOpString(this.op, this.isVec4);
    const miscStr =
        `fn binaryOperation(a : vec4<f32>, b : vec4<f32>) -> vec4<f32> {
          ${opStr}
        }`;

    if (this.fitShape) {
      userCode = `
      ${miscStr}
      ${getMainHeaderStringWgsl()} {
        ${getGlobalIndexStringWgsl()}
        let a = vec4<f32>(A.numbers[index]);
        let b = vec4<f32>(B.numbers[index]);
        setOutputFlat(index, binaryOperation(a, b));
      }
    `;
    } else {
      userCode = `
      ${miscStr}
      ${getMainHeaderStringWgsl()} {
        ${getGlobalIndexStringWgsl()}
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
