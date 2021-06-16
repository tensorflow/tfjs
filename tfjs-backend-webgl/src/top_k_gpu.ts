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

// Based on Algorithm 2 of Bitonic Top K, ref:
// https://anilshanbhag.in/static/papers/gputopk_sigmod18.pdf
export class SwapProgram implements GPGPUProgram {
  variableNames = ['x'];
  outputShape: number[];
  userCode: string;

  // Caching uniform location for speed.
  dir: WebGLUniformLocation;
  inc: WebGLUniformLocation;

  constructor(shape: number[], inputDim: 2 | 3) {
    this.outputShape = shape;

    const firstPass = inputDim === 2;
    this.userCode = `
       uniform int dir;
       // Swaps pairs of indices (0, inc), (1, inc + 1), (2, inc + 2) ...
       uniform int inc;
       void main() {
         ivec3 coords = getOutputCoords();
         bool isIndexOutput = coords[0] == 1;
         int batch = coords[1];
         int elemIdx = coords[2];

         // Maps both indices in a pair to the first index
         bool isFirstInPair = imod(elemIdx, 2 * inc) < inc;
         int i = isFirstInPair ? elemIdx : elemIdx - inc;

         float i0 = ${firstPass ? 'float(i)' : 'getX(1, batch, i)'};
         float i1 = ${firstPass ? 'float(i + inc)' : 'getX(1, batch, i + inc)'};
         float x0 = ${firstPass ? 'getX(batch, i)' : 'getX(0, batch, i)'};
         float x1 = ${firstPass ? 'getX(batch, i + inc)' : 'getX(0, batch, i + inc)'};

         bool reverse = imod(elemIdx, 2 * dir) >= dir;
         if (reverse && x0 >= x1 || !reverse && x0 < x1) { // reverse ^ (x0 < x1)
           float xTemp = x0;
           x0 = x1;
           x1 = xTemp;

           float iTemp = i0;
           i0 = i1;
           i1 = iTemp;
         }
         if (isFirstInPair) {
            setOutput(isIndexOutput ? i0 : x0);
         } else {
            setOutput(isIndexOutput ? i1 : x1);
         }
       }
     `;
  }
  getCustomSetupFunc(dir: number, inc: number) {
    return (gpgpu: GPGPUContext, webGLProgram: WebGLProgram) => {
      if (this.dir == null) {
        this.dir = gpgpu.getUniformLocation(webGLProgram, 'dir');
      }
      if (this.inc == null) {
        this.inc = gpgpu.getUniformLocation(webGLProgram, 'inc');
      }
      gpgpu.gl.uniform1i(this.dir, dir);
      gpgpu.gl.uniform1i(this.inc, inc);
    };
  }
}

export class MergeProgram implements GPGPUProgram {
  variableNames = ['x'];
  outputShape: number[];
  userCode: string;

  // Caching uniform location for speed.
  k: WebGLUniformLocation;

  constructor(shape: number[], inputDim: 2 | 3) {
    this.outputShape = shape;

    const firstPass = inputDim === 2;
    this.userCode = `
       uniform int k;
       void main() {
         // Takes max of indices (0, k), (1, k + 1), (2, k + 2) ...
         ivec3 coords = getOutputCoords();
         bool isIndexOutput = coords[0] == 1;
         int batch = coords[1];
         int elemIdx = coords[2];

         int i = elemIdx < k ? elemIdx : (elemIdx * 2 - imod(elemIdx, k));
         float i0 = ${firstPass ? 'float(i)' : 'getX(1, batch, i)'};
         float i1 = ${firstPass ? 'float(i + k)' : 'getX(1, batch, i + k)'};

         float x0 = ${firstPass ? 'getX(batch, i)' : 'getX(0, batch, i)'};
         float x1 = ${firstPass ? 'getX(batch, i + k)' : 'getX(0, batch, i + k)'};

         float xMax = x0 >= x1 ? x0 : x1;
         float iMax = x0 >= x1 ? i0 : i1;

         setOutput(isIndexOutput ? iMax : xMax);
       }
     `;
  }
  getCustomSetupFunc(k: number) {
    return (gpgpu: GPGPUContext, webGLProgram: WebGLProgram) => {
      if (this.k == null) {
        this.k = gpgpu.getUniformLocation(webGLProgram, 'k');
      }
      gpgpu.gl.uniform1i(this.k, k);
    };
  }
}
