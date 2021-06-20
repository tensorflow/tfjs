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
  n: WebGLUniformLocation;
  negativeInf: WebGLUniformLocation;
  dir: WebGLUniformLocation;
  inc: WebGLUniformLocation;

  /**
   * @param shape desired output shape (can be larger than input shape, output
   *                                    will be padded with -Infinity)
   * @param firstPass indicates if this is the first time swap is begin used
   *                  which means no indices input containing the top K is
   *                  present yet.
   */
  constructor(shape: number[], firstPass: boolean) {
    if (!firstPass) {
      this.variableNames.push('indices');
    }

    this.outputShape = shape;

    this.userCode = `
       // Size of the original input of TopK
       uniform int n;
       uniform float negativeInf;
       uniform int dir;
       // Swaps pairs of indices (0, inc), (1, inc + 1), (2, inc + 2) ...
       uniform int inc;
       void main() {
         ivec2 coords = getOutputCoords();
         int batch = coords[0];
         int elemIdx = coords[1];

         // We compare elements pair-wise within a group of size 2 * inc.
         // The comparing rule for each group alternates between ascending
         // and descending. Within each group, we compare each pair at
         // positions i and i+inc. To decide whether an element at position i
         // is x0 or x1, we mod it by 2 * inc, if the result is smaller than
         // inc, it is in the first half of the group, we denote it as x0,
         // otherwise we denote it as x1.
         // For example, as shown in the Bitonic top K paper referenced above,
         // Figure5(a) shows that element[1] is in the
         // second half of the group when group size is 2, but it is in the
         // first half of the group when group size is 4.

         bool isFirstInPair = imod(elemIdx, 2 * inc) < inc;
         int i = isFirstInPair ? elemIdx : elemIdx - inc;

         int i0 = ${firstPass ? 'i' : 'int(getIndices(batch, i))'};
         int i1 = ${firstPass ? 'i + inc' : 'int(getIndices(batch, i + inc))'};
         float x0 = i0 < n ? getX(batch, i0) : negativeInf;
         float x1 = i1 < n ? getX(batch, i1) : negativeInf;

         // Denotes which direction indices are in (ascending or descending).
         bool reverse = imod(elemIdx, 2 * dir) >= dir;
         bool isGreater = x0 > x1 || (x0 == x1 && i1 > i0);
         if (reverse == isGreater) { // Elements in opposite order of direction
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

  getCustomSetupFunc(n: number, dir: number, inc: number) {
    return (gpgpu: GPGPUContext, webGLProgram: WebGLProgram) => {
      if (this.n == null) {
        this.n = gpgpu.getUniformLocation(webGLProgram, 'n');
      }
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
      gpgpu.gl.uniform1i(this.n, n);
      gpgpu.gl.uniform1f(this.negativeInf, Number.NEGATIVE_INFINITY);
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
  n: WebGLUniformLocation;
  k: WebGLUniformLocation;

  /**
   * @param shape desired output shape (must be half of the input size)
   * @param firstPass indicates if this is the first time merge is being used
   *                  which means no indices input containing the top K is
   *                  present yet.
   */
  constructor(shape: number[], firstPass: boolean) {
    if (!firstPass) {
      this.variableNames.push('indices');
    }

    this.outputShape = shape;

    this.userCode = `
    // Size of the original input of TopK
    uniform int n;
    // Top k elements desired
    uniform int k;
    void main() {
         // Takes max of indices (0, k), (1, k + 1), (2, k + 2) ...
         ivec2 coords = getOutputCoords();
         int batch = coords[0];
         int elemIdx = coords[1];

         // The output size is half of the previous size.
         // If the previous sequence is | | | | _ _ _ _  | | | |  _ _ _ _ (k=4),
         // we only need to output the indices at positions |, the indices at
         // positions _ can be thrown away, see Figure5(b) After Phase 2
         // (Merge phase) in the Bitonic Top K paper referenced above.
         // For example, the paper shows we only need to output the orange bars.
         // The output sequence should look like this | | | | | | | |.
         // Because the sequence is halved, to map the output index back
         // to the previous sequence to find the corresponding value,
         // we need to double the index. When we double the index,
         // we basically interpolate a position, so 2i looks like
         // | _ | _ | _ | _ | _ | _ | _. We move the | to the first k position
         // of each 2k positions by - elemIdx % k. E.g. for output at
         // index 4,5,6,7, we want to get the corresponding element at
         // original index 8,9,10,11, for output at index 8,9,10,11,
         // we want to get the corresponding element at original index
         // 16,17,18,19, so on and so forth.

         int i = elemIdx < k ? elemIdx : (elemIdx * 2 - imod(elemIdx, k));
         int i0 = ${firstPass ? 'i' : 'int(getIndices(batch, i))'};
         int i1 = ${firstPass ? `i + k` : `int(getIndices(batch, i + k))`};

         float x0 = getX(batch, i0);
         float x1 = i1 < n ? getX(batch, i1) : x0;

         setOutput(x0 >= x1 ? float(i0) : float(i1));
       }
     `;
  }

  getCustomSetupFunc(n: number, k: number) {
    return (gpgpu: GPGPUContext, webGLProgram: WebGLProgram) => {
      if (this.k == null) {
        this.k = gpgpu.getUniformLocation(webGLProgram, 'k');
      }
      if (this.n == null) {
        this.n = gpgpu.getUniformLocation(webGLProgram, 'n');
      }
      gpgpu.gl.uniform1i(this.n, n);
      gpgpu.gl.uniform1i(this.k, k);
    };
  }
}
