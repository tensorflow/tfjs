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

import {GPGPUProgram} from './gpgpu_math';

export const COMPLEX_FFT = {
  REAL: 'return real * expR - imag * expI;',
  IMAG: 'return real * expI + imag * expR;'
};

export class FFTProgram implements GPGPUProgram {
  variableNames = ['real', 'imag'];
  outputShape: number[];
  userCode: string;

  constructor(op: string, inputShape: number[]) {
    const size = inputShape[0];
    this.outputShape = [size];

    this.userCode = `
      float unaryOpComplex(float real, float expR, float imag, float expI) {
        ${op}
      }

      float mulMatDFT(int row) {
        // TODO: Gather constants in one place?
        const float PI = 3.1415926535897932384626433832795;
        float result = 0.0;

        for (int i = 0; i < ${size}; i++) {
          float x = -2.0 * PI * float(row * i) / float(${size});
          float expR = cos(x);
          float expI = sin(x);
          float real = getReal(i);
          float imag = getImag(i);

          result += unaryOpComplex(real, expR, imag, expI);
        }

        return result;
      }

      void main() {
        int row = getOutputCoords();
        setOutput(mulMatDFT(row));
      }
    `;
  }
}
