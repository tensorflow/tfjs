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
import {getWorkGroupSizeStringWgsl} from '../shader_preprocessor_wgsl';
import {computeDispatch, flatDispatchLayout} from '../webgpu_util';
import {BinaryOpType, getBinaryOpString} from './binary_op_util';

import {getUseWgsl, WebGPUProgram} from './webgpu_program';

export class BinaryOpComplexProgram implements WebGPUProgram {
  variableNames = ['AReal', 'AImag', 'BReal', 'BImag'];
  outputShape: number[];
  shaderKey: string;
  dispatchLayout: {x: number[]};
  dispatch: [number, number, number];
  workGroupSize: [number, number, number] = [128, 1, 1];
  op: BinaryOpType;
  size: number;
  useWgsl: boolean;

  constructor(op: BinaryOpType, aShape: number[], bShape: number[]) {
    this.outputShape = backend_util.assertAndGetBroadcastShape(aShape, bShape);
    this.dispatchLayout = flatDispatchLayout(this.outputShape);
    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workGroupSize);

    this.shaderKey = `binaryOpComplex_${op}`;
    this.op = op;
    this.size = util.sizeFromShape(this.outputShape);
    this.useWgsl = getUseWgsl();
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

  getUserCodeWgsl(): string {
    const opStr = getBinaryOpString(this.op, false, true);
    const userCode = `
      fn binaryOpComplex(
          areal : f32, aimag : f32, breal : f32, bimag : f32) -> f32 {
        ${opStr}
      }

      ${getWorkGroupSizeStringWgsl(this.workGroupSize)}
      fn main([[builtin(global_invocation_id)]] global_id : vec3<u32>) {
        let index = global_id.x;
        if(index < uniforms.size) {
          let areal = getARealAtOutCoordsByGlobalId(global_id);
          let aimag = getAImagAtOutCoordsByGlobalId(global_id);
          let breal = getBRealAtOutCoordsByGlobalId(global_id);
          let bimag = getBImagAtOutCoordsByGlobalId(global_id);
          setOutputFlat(index, binaryOpComplex(areal, aimag, breal, bimag));
        }
      }
    `;
    return userCode;
  }
}
