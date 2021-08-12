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

import {getWorkGroupSizeStringWgsl} from '../shader_preprocessor_wgsl';
import {computeDispatch, flatDispatchLayout} from '../webgpu_util';

import {getUnaryOpString, UnaryOpType} from './unary_op_util';
import {getUseWgsl, WebGPUProgram} from './webgpu_program';

export class UnaryOpProgram implements WebGPUProgram {
  outputShape: number[];
  shaderKey: string;
  dispatchLayout: {x: number[]};
  dispatch: [number, number, number];
  variableNames = ['A'];
  workGroupSize: [number, number, number];
  useWgsl: boolean;
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
    this.useWgsl = getUseWgsl();
    this.op = op;
    this.shaderKey = `unary_${op}`;
    this.size = util.sizeFromShape(this.outputShape);
  }

  getUserCode(): string {
    return `
      float unaryOperation(float a) {
        ${getUnaryOpString(this.op)}
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

  getUserCodeWgsl(): string {
    return `
      fn unaryOperation(a : f32) -> f32 {
        ${getUnaryOpString(this.op, false, true)}
      }
      ${getWorkGroupSizeStringWgsl(this.workGroupSize)}
      fn main([[builtin(global_invocation_id)]] globalId  : vec3<u32>) {
        let index = globalId.x;
        if (index < uniforms.size) {
          let a = getAAtOutCoordsByGlobalId(globalId);
          setOutputFlat(index, unaryOperation(a));
        }
      }
      `;
  }
}
