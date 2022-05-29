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

import {backend_util} from '@tensorflow/tfjs-core';
import {BinaryOpType, getBinaryOpString} from './binary_op_util';
import {getMainHeaderAndGlobalIndexString, WebGPUProgram} from './webgpu_program';
import {computeDispatch, flatDispatchLayout} from './webgpu_util';

export class BinaryOpProgram implements WebGPUProgram {
  outputShape: number[];
  shaderKey: string;
  dispatchLayout: {x: number[]};
  dispatch: [number, number, number];
  variableNames = ['A', 'B'];
  workGroupSize: [number, number, number];
  op: BinaryOpType;
  size = true;

  constructor(op: BinaryOpType, aShape: number[], bShape: number[]) {
    // TODO(jiajia.qin@intel.com): Heuristically select a good work group size.
    const workGroupSizeX = 128;
    this.workGroupSize = [workGroupSizeX, 1, 1];
    this.outputShape = backend_util.assertAndGetBroadcastShape(aShape, bShape);
    this.dispatchLayout = flatDispatchLayout(this.outputShape);

    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workGroupSize);
    this.shaderKey = `binary_${op}`;
    this.op = op;
  }

  getUserCode(): string {
    const opStr = getBinaryOpString(this.op, false);
    const userCode = `
      fn binaryOperation(a : f32, b : f32) -> f32 {
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
    return userCode;
  }
}
