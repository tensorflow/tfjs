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

import { GPGPUProgram } from "./gpgpu_math";

export const COMPLEX_FFT = {
  REAL: "return resultReal;",
  IMAG: "return resultImag;",
};

export class FFT2DProgram implements GPGPUProgram {
  variableNames = ["real", "imag"];
  outputShape: number[];
  userCode: string;

  constructor(op: string, inputShape: [number, number, number]) {
    this.outputShape = inputShape;

    const [, height, width] = inputShape;
    const mulHeight = `${Math.atan(Math.tan((2 * Math.PI) / height))}`;
    const mulWidth = `${Math.atan(Math.tan((2 * Math.PI) / width))}`;

    this.userCode = `
      float mulWidth = ${mulWidth};
      float mulHeight = ${mulHeight};
      float mulMatDFT(int batch, int y, int x) {
        float resultReal = 0.0;
        float resultImag = 0.0;
        for (int r = 0; r < ${height}; r++) {
          float rowReal = 0.0;
          float rowImag = 0.0;
          for (int c = 0; c < ${width}; c++) {
            float real1 = getReal(batch, r, c);
            float imag1 = getImag(batch, r, c);
            float theta = -float(x) * float(c) * mulWidth;
            float real2 = cos(theta);
            float imag2 = sin(theta);

            rowReal += real1 * real2 - imag1 * imag2;
            rowImag += real1 * imag2 + imag1 * real2;
          }
          float theta = -float(y) * float(r) * mulHeight;
          float real2 = cos(theta);
          float imag2 = sin(theta);
          resultReal += rowReal * real2 - rowImag * imag2;
          resultImag += rowReal * imag2 + rowImag * real2;
        }
        ${op}
      }

      void main() {
        ivec3 coords = getOutputCoords();
        setOutput(mulMatDFT(coords[0], coords[1], coords[2]));
      }
    `;
  }
}
