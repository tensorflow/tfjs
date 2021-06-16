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
import {util} from '@tensorflow/tfjs-core';

import {computeDispatch, flatDispatchLayout} from '../webgpu_util';
import {getUnaryOpString, UnaryOpType} from './unary_op_util';

import {WebGPUProgram} from './webgpu_program';

export class UnaryOpProgram implements WebGPUProgram {
  outputShape: number[];
  shaderKey: string;
  dispatchLayout: {x: number[]};
  dispatch: [number, number, number];
  variableNames = ['A'];
  workGroupSize: [number, number, number];
  op: UnaryOpType;
  size: number;

  constructor(outputShape: number[], op: UnaryOpType) {
    // TODO(jiajia.qin@intel.com): Heuristically select a good work group size.
    const workGroupSizeX = 128;
    this.workGroupSize = [workGroupSizeX, 1, 1];
    this.outputShape = outputShape;
    this.dispatchLayout = flatDispatchLayout(this.outputShape);
    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workGroupSize);
    this.op = op;
    this.shaderKey = `unary_${op}`;
    this.size = util.sizeFromShape(this.outputShape);
  }

  getUserCode(): string {
    const opStr = getUnaryOpString(this.op);
    return `
      float unaryOperation(float a) {
        ${opStr}
      }

      void main() {
        int index = int(gl_GlobalInvocationID.x);
        if (index < size)
        {
          float a = getAAtOutCoords();
          setOutput(index, unaryOperation(a));
        }
      }
      `;
  }
}
