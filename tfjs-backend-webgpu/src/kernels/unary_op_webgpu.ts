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

import {getCoordsDataType} from '../shader_preprocessor';
import {computeDispatch, flatDispatchLayout} from '../webgpu_util';

import {WebGPUProgram} from './webgpu_program';

export const RELU = 'return max(a, 0.0);';
export const RELU6 = 'return clamp(a, 0.0, 6.0);';
export const LINEAR = `return a;`;
export const ELU = `return (a >= 0.0) ? a : (exp(a) - 1.0);`;
export const PRELU = `return (a < 0.) ? b * a : a;`;

export const ELU_VEC4 = `
  vec4 result;

  result.r = (a.r >= 0.0) ? a.r : (exp(a.r) - 1.0);
  result.g = (a.g >= 0.0) ? a.g : (exp(a.g) - 1.0);
  result.b = (a.b >= 0.0) ? a.b : (exp(a.b) - 1.0);
  result.a = (a.a >= 0.0) ? a.a : (exp(a.a) - 1.0);

  return result;
`;

export const RELU_VEC4 = `
  vec4 result = a * vec4(greaterThanEqual(a, vec4(0.0)));
  bvec4 isNaN = isnan(a);

  result.r = isNaN.r ? a.r : result.r;
  result.g = isNaN.g ? a.g : result.g;
  result.b = isNaN.b ? a.b : result.b;
  result.a = isNaN.a ? a.a : result.a;

  return result;
`;

export const SIGMOID = `return 1.0 / (1.0 + exp(-1.0 * a));`;
export const ABS = `return abs(a);`;
export const SQUARE = `return a * a;`;
export const NEG = `return -a;`;
export const TANH = `
  float e2x = exp(-2.0 * abs(a));
  return sign(a) * (1.0 - e2x) / (1.0 + e2x);
`;
export const EXP = `return exp(a);`;
export const LOG = `if (a < 0.0) return 1.0/0.0;
  return log(a);`;
export const TO_INT = `return float(int(a));`;

export class UnaryOpProgram implements WebGPUProgram {
  outputShape: number[];
  shaderKey: string;
  dispatchLayout: {x: number[]};
  dispatch: [number, number, number];
  variableNames = ['A'];
  workPerThread: number;
  workGroupSize: [number, number, number];
  op: string;

  constructor(outputShape: number[], op: string) {
    // TODO(jiajia.qin@intel.com): Heuristically select a good work group size.
    const workGroupSizeX = 128;
    this.workGroupSize = [workGroupSizeX, 1, 1];
    this.outputShape = outputShape;
    this.dispatchLayout = flatDispatchLayout(this.outputShape);
    const size = util.sizeFromShape(this.outputShape);
    const fit = size % this.workGroupSize[0] === 0;
    this.workPerThread = fit ? 1 : 2;
    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workGroupSize,
        [this.workPerThread, 1, 1]);
    this.op = op;
    this.shaderKey = `unary_${op}`;
  }

  getUserCode(): string {
    const size = util.sizeFromShape(this.outputShape);
    const fit = size % this.workGroupSize[0] === 0;
    let userCode: string;
    if (fit) {
      userCode = `
      float unaryOperation(float a) {
        ${this.op}
      }

      void main() {
        int index = int(gl_GlobalInvocationID.x);
        float a = A[index];
        setOutput(index, unaryOperation(a));;
      }
      `;
    } else {
      const type = getCoordsDataType(this.outputShape.length);
      userCode = `
      float unaryOperation(float a) {
        ${this.op}
      }

      void main() {
        int index = int(gl_GlobalInvocationID.x);

        for(int i = 0; i < ${this.workPerThread}; i++) {
          int flatIndex = index * ${this.workPerThread} + i;

          if(flatIndex < ${size}) {
            ${type} coords = getCoordsFromFlatIndex(flatIndex);

            float a = getAAtOutCoords(coords);
            setOutput(flatIndex, unaryOperation(a));
          }
        }
      }
      `;
    }
    return userCode;
  }
}
