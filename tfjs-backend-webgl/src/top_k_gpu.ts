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
import {GPGPUContext} from './gpgpu_context';
import {GPGPUProgram} from './gpgpu_math';

// Based on Algorithm 2 of Bitonic Top K, ref:
// https://anilshanbhag.in/static/papers/gputopk_sigmod18.pdf
export class SwapProgram implements GPGPUProgram {
  variableNames = ['x'];
  outputShape: number[];
  userCode: string;

  // Caching uniform location for speed.
  negativeInf: WebGLUniformLocation;
  dir: WebGLUniformLocation;
  inc: WebGLUniformLocation;

  constructor(n: number, shape: number[], firstPass: boolean) {
    if (!firstPass) {
      this.variableNames.push('indices');
    }

    this.outputShape = shape;

    this.userCode = `
       uniform float negativeInf;
       uniform int dir;
       // Swaps pairs of indices (0, inc), (1, inc + 1), (2, inc + 2) ...
       uniform int inc;
       void main() {
         ivec2 coords = getOutputCoords();
         int batch = coords[0];
         int elemIdx = coords[1];

         // Maps both indices in a pair to the first index
         bool isFirstInPair = imod(elemIdx, 2 * inc) < inc;
         int i = isFirstInPair ? elemIdx : elemIdx - inc;

         int i0 = ${firstPass ? 'i' : 'int(getIndices(batch, i))'};
         int i1 = ${firstPass ? 'i + inc' : 'int(getIndices(batch, i + inc))'};
         float x0 = i0 < ${n} ? getX(batch, i0) : negativeInf;
         float x1 = i1 < ${n} ? getX(batch, i1) : negativeInf;

         bool reverse = imod(elemIdx, 2 * dir) >= dir;
         bool isGreater = x0 > x1 || (x0 == x1 && i1 > i0);
         if (reverse == isGreater) { // reverse ^ (x0 < x1)
           int iTemp = i0;
           i0 = i1;
           i1 = iTemp;
         }
         if (isFirstInPair) {
            setOutput(float(i0));
         } else {
            setOutput(float(i1));
         }
       }
     `;
  }
  getCustomSetupFunc(dir: number, inc: number) {
    return (gpgpu: GPGPUContext, webGLProgram: WebGLProgram) => {
      if (this.negativeInf == null) {
        this.negativeInf =
            gpgpu.getUniformLocation(webGLProgram, 'negativeInf');
      }
      if (this.dir == null) {
        this.dir = gpgpu.getUniformLocation(webGLProgram, 'dir');
      }
      if (this.inc == null) {
        this.inc = gpgpu.getUniformLocation(webGLProgram, 'inc');
      }
      gpgpu.gl.uniform1f(this.negativeInf, -Infinity);
      gpgpu.gl.uniform1i(this.dir, dir);
      gpgpu.gl.uniform1i(this.inc, inc);
    };
  }
}

export class MergeProgram implements GPGPUProgram {
  variableNames = ['x'];
  outputShape: number[];
  userCode: string;

  constructor(n: number, k: number, shape: number[], firstPass: boolean) {
    if (!firstPass) {
      this.variableNames.push('indices');
    }

    this.outputShape = shape;

    this.userCode = `
       void main() {
         // Takes max of indices (0, k), (1, k + 1), (2, k + 2) ...
         ivec2 coords = getOutputCoords();
         int batch = coords[0];
         int elemIdx = coords[1];

         int i = elemIdx < ${k} ? elemIdx : (elemIdx * 2 - imod(elemIdx, ${k}));
         int i0 = ${firstPass ? 'i' : 'int(getIndices(batch, i))'};
         int i1 = ${firstPass ? `i +${k}` : `int(getIndices(batch, i + ${k}))`};

         float x0 = getX(batch, i0);
         float x1 = i1 < ${n} ? getX(batch, i1) : x0;

         setOutput(x0 >= x1 ? float(i0) : float(i1));
       }
     `;
  }
}
