/**
 * @license
 * Copyright 2018 Google Inc. All Rights Reserved.
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

import * as broadcast_util from '../../ops/broadcast_util';
import {GPGPUContext} from './gpgpu_context';
import {GPGPUProgram} from './gpgpu_math';

// We do the same as in ./binaryop_gpu, with vec4 and ivec4.
export const PACKED_DIV = `
  vec4 one = vec4(equal(a, b));
  return one + (vec4(1.0) - one) * a / b;
`;

export const PACKED_INT_DIV = `
  vec4 resultSign = sign(a) * sign(b);
  ivec4 ia = round(a);
  ivec4 ib = round(b);
  ivec4 result = ia / ib;
  ivec4 amodb = ia - ib * result;

  // Vectorize INT_DIV
  // if (resultSign < 0.0 && amodb != 0) result -= 1;
  // return float(result);
  return vec4(result -
     ivec4(lessThan(resultSign, vec4(0.0))) * ivec4(notEqual(amodb, ivec4(0))));
`;

export const PACKED_POW = `
  // isModRound1 has 1 for components with round(mod(b, 2.0)) == 1, 0 otherwise.
  vec4 isModRound1 = vec4(equal(round(mod(b, 2.0)), ivec4(1)));
  vec4 multiplier = sign(a) * isModRound1 + (vec4(1.0) - isModRound1);
  vec4 result = multiplier * pow(abs(a), b);

  vec4 isNaN = vec4(lessThan(a, vec4(0.0))) * vec4(lessThan(floor(b), b));
  result.r = isNaN.r == 1.0 ? NAN : result.r;
  result.g = isNaN.g == 1.0 ? NAN : result.g;
  result.b = isNaN.b == 1.0 ? NAN : result.b;
  result.a = isNaN.a == 1.0 ? NAN : result.a;

  return result;
`;

export class BinaryOpPackedProgram implements GPGPUProgram {
  variableNames = ['A', 'B'];
  outputShape: number[];
  userCode: string;
  supportsBroadcasting = true;
  usesPackedTextures = true;

  // Caching uniform location for speed.
  startLoc: WebGLUniformLocation;

  constructor(op: string, aShape: number[], bShape: number[]) {
    this.outputShape =
        broadcast_util.assertAndGetBroadcastShape(aShape, bShape);
    this.userCode = `
      uniform float NAN;
      vec4 binaryOperation(vec4 a, vec4 b) {
        ${op}
      }

      void main() {
        vec4 a = getAAtOutCoords();
        vec4 b = getBAtOutCoords();
        setOutput(binaryOperation(a, b));
      }
    `;
  }

  getCustomSetupFunc() {
    return (gpgpu: GPGPUContext, webGLProgram: WebGLProgram) => {
      if (this.startLoc == null) {
        this.startLoc = gpgpu.getUniformLocationNoThrow(webGLProgram, 'NAN');
        if (this.startLoc == null) {
          // This means the compiler has optimized and realized it doesn't need
          // the uniform.
          return;
        }
      }
      gpgpu.gl.uniform1f(this.startLoc, NaN);
    };
  }
}
