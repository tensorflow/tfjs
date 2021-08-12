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
import {getCoordsDataType} from '../shader_preprocessor';
import {getWorkGroupSizeStringWgsl} from '../shader_preprocessor_wgsl';
import {computeDispatch, flatDispatchLayout} from '../webgpu_util';
import {BinaryOpType, getBinaryOpString} from './binary_op_util';

import {getUseWgsl, WebGPUProgram} from './webgpu_program';

export class BinaryOpProgram implements WebGPUProgram {
  outputShape: number[];
  shaderKey: string;
  dispatchLayout: {x: number[]};
  dispatch: [number, number, number];
  variableNames = ['A', 'B'];
  workPerThread: number;
  workGroupSize: [number, number, number];
  useWgsl: boolean;
  op: BinaryOpType;
  sizeFit: boolean;
  shapesFit: boolean;
  size: number;

  constructor(op: BinaryOpType, aShape: number[], bShape: number[]) {
    // TODO(jiajia.qin@intel.com): Heuristically select a good work group size.
    const workGroupSizeX = 128;
    this.workGroupSize = [workGroupSizeX, 1, 1];
    this.outputShape = backend_util.assertAndGetBroadcastShape(aShape, bShape);
    this.dispatchLayout = flatDispatchLayout(this.outputShape);
    this.size = util.sizeFromShape(this.outputShape);
    this.sizeFit = this.size % workGroupSizeX === 0;
    this.shapesFit = util.arraysEqual(aShape, bShape) && this.sizeFit;
    this.workPerThread = this.sizeFit || this.shapesFit ? 1 : 2;

    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workGroupSize,
        [this.workPerThread, 1, 1]);
    this.shaderKey = `binary_${op}_${this.sizeFit}_${this.shapesFit}`;
    this.useWgsl = getUseWgsl();
    this.op = op;
  }

  getUserCode(): string {
    let userCode: string;
    const opStr = getBinaryOpString(this.op);
    if (this.shapesFit) {
      userCode = `
          float binaryOperation(float a, float b) {
            ${opStr}
          }

          void main() {
            int index = int(gl_GlobalInvocationID.x);

            float a = float(A[index]);
            float b = float(B[index]);
            setOutput(index, binaryOperation(a, b));
          }
        `;
    } else if (this.sizeFit) {
      const type = getCoordsDataType(this.outputShape.length);
      userCode = `
      float binaryOperation(float a, float b) {
        ${opStr}
      }

      void main() {
        int index = int(gl_GlobalInvocationID.x);

        ${type} coords = getCoordsFromFlatIndex(index);

        float a = getAAtOutCoords(coords);
        float b = getBAtOutCoords(coords);
        setOutput(index, binaryOperation(a, b));
      }
      `;
    } else {
      const type = getCoordsDataType(this.outputShape.length);
      userCode = `
      float binaryOperation(float a, float b) {
        ${opStr}
      }

      void main() {
        int index = int(gl_GlobalInvocationID.x);

        for(int i = 0; i < ${this.workPerThread}; i++) {
          int flatIndex = index * ${this.workPerThread} + i;

          if(flatIndex < size) {
            ${type} coords = getCoordsFromFlatIndex(flatIndex);

            float a = getAAtOutCoords(coords);
            float b = getBAtOutCoords(coords);
            setOutput(flatIndex, binaryOperation(a, b));
          }
        }
      }
      `;
    }
    return userCode;
  }

  getUserCodeWgsl(): string {
    let userCode: string;
    const opStr = getBinaryOpString(this.op, false, this.useWgsl);
    const miscStr = `          fn binaryOperation(a : f32, b : f32) -> f32 {
      ${opStr}
    }`;
    if (this.shapesFit) {
      userCode = `
          ${miscStr}
          ${getWorkGroupSizeStringWgsl(this.workGroupSize)}
          fn main([[builtin(global_invocation_id)]] global_id : vec3<u32>) {
            let index = global_id.x;

            let a = f32(A[index]);
            let b = f32(B[index]);
            setOutputFlat(index, binaryOperation(a, b));
          }
        `;
    } else if (this.sizeFit) {
      userCode = `
      ${miscStr}
      ${getWorkGroupSizeStringWgsl(this.workGroupSize)}
      fn main([[builtin(global_invocation_id)]] global_id : vec3<u32>) {
        let index = global_id.x;

        let coords = getCoordsFromFlatIndex(index);

        let a = getAAtOutCoordsByCoords(coords);
        let b = getBAtOutCoordsByCoords(coords);
        setOutputFlat(index, binaryOperation(a, b));
      }
      `;
    } else {
      userCode = `
      ${miscStr}
      ${getWorkGroupSizeStringWgsl(this.workGroupSize)}
      fn main([[builtin(global_invocation_id)]] global_id : vec3<u32>) {
        let index = global_id.x;
        for (var i = 0u; i < ${this.workPerThread}u; i = i + 1u ) {
          let flatIndex = index * ${this.workPerThread}u + i;

          if(flatIndex < uniforms.size) {
            let coords = getCoordsFromFlatIndex(flatIndex);

            let a = getAAtOutCoordsByCoords(coords);
            let b = getBAtOutCoordsByCoords(coords);
            setOutputFlat(flatIndex, binaryOperation(a, b));
          }
        }
      }
      `;
    }
    return userCode;
  }
}
