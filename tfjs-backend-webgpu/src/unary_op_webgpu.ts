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

import {getUnaryOpString, UnaryOpType} from './unary_op_util';
import {getMainHeaderString as main, WebGPUProgram} from './webgpu_program';
import {computeDispatch, flatDispatchLayout} from './webgpu_util';

export class UnaryOpProgram implements WebGPUProgram {
  outputShape: number[];
  shaderKey: string;
  dispatchLayout: {x: number[]};
  dispatch: [number, number, number];
  variableNames = ['A'];
  workgroupSize: [number, number, number];
  op: UnaryOpType;
  uniforms?: string;
  size = true;

  constructor(outputShape: number[], op: UnaryOpType, uniforms = '') {
    // TODO(jiajia.qin@intel.com): Heuristically select a good work group size.
    const workgroupSizeX = 128;
    this.workgroupSize = [workgroupSizeX, 1, 1];
    this.outputShape = outputShape;
    this.dispatchLayout = flatDispatchLayout(this.outputShape);
    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workgroupSize);
    this.op = op;
    if (uniforms !== '') {
      this.uniforms = uniforms;
    }
    this.shaderKey = `unary_${op}`;
  }

  getUserCode(): string {
    return `
      fn unaryOperation(a : f32) -> f32 {
        ${getUnaryOpString(this.op, false)}
      }
      ${main('index')} {
        if (index < uniforms.size) {
          let a = getAByOutputIndex(index);
          setOutputAtIndex(index, unaryOperation(a));
        }
      }
      `;
  }
}
