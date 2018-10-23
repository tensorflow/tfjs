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

  constructor(op: string, inputShape: [number, number]) {
    const innerDim = inputShape[1];
    this.outputShape = inputShape;

    this.userCode = `
      const float negativeTwoPi = -2.0 * 3.1415926535897932384626433832795;

      float unaryOpComplex(float real, float expR, float imag, float expI) {
        ${op}
      }

      float mulMatDFT(int batch, int index) {
        float indexRatio = float(index) / float(${innerDim});
        float negativeTwoPiTimesIndexRatio = negativeTwoPi * indexRatio;

        float result = 0.0;

        for (int i = 0; i < ${innerDim}; i++) {
          // x = (-2 * PI / N) * index * i;
          float x = negativeTwoPiTimesIndexRatio * float(i);
          float expR = cos(x);
          float expI = sin(x);
          float real = getReal(batch, i);
          float imag = getImag(batch, i);

          result += unaryOpComplex(real, expR, imag, expI);
        }

        return result;
      }

      void main() {
        ivec2 coords = getOutputCoords();
        setOutput(mulMatDFT(coords[0], coords[1]));
      }
    `;
  }
}
