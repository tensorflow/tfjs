/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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

import {GPGPUContext} from './gpgpu_context';
import {GPGPUProgram} from './gpgpu_math';

export const LINEAR = `return x;`;

export const LOG = `
  vec4 result = log(x);
  vec4 isNaN = vec4(lessThan(x, vec4(0.0)));
  result.r = isNaN.r == 1.0 ? NAN : result.r;
  result.g = isNaN.g == 1.0 ? NAN : result.g;
  result.b = isNaN.b == 1.0 ? NAN : result.b;
  result.a = isNaN.a == 1.0 ? NAN : result.a;

  return result;
`;

export const RELU = `
  vec4 result = x * vec4(greaterThanEqual(x, vec4(0.0)));

  result.r = isNaN(x.r) ? x.r : result.r;
  result.g = isNaN(x.g) ? x.g : result.g;
  result.b = isNaN(x.b) ? x.b : result.b;
  result.a = isNaN(x.a) ? x.a : result.a;

  return result;
`;

export class UnaryOpPackedProgram implements GPGPUProgram {
  variableNames = ['A'];
  userCode: string;
  outputShape: number[];
  usesPackedTextures = true;

  // Caching uniform location for speed.
  startLoc: WebGLUniformLocation;

  constructor(aShape: number[], opSnippet: string) {
    this.outputShape = aShape;
    this.userCode = `
      uniform float NAN;
      vec4 unaryOperation(vec4 x) {
        ${opSnippet}
      }

      void main() {
        vec4 x = getAAtOutCoords();
        vec4 y = unaryOperation(x);

        setOutput(y);
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
