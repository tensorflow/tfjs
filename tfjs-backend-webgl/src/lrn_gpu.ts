/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
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

export class LRNProgram implements GPGPUProgram {
  variableNames = ['x'];
  outputShape: number[] = [];
  userCode: string;

  constructor(
      xShape: number[], radius: number, bias: number, alpha: number,
      beta: number) {
    const rad = radius;
    const maxD = xShape[3] - 1;
    this.outputShape = xShape;

    // optimize pow(bias + alpha * sum, -beta)
    // src: https://github.com/tensorflow/tensorflow/..
    // blob/26033a1644a9c4a5fbe3170ab2e864b6a4ccd4ca/..
    // tensorflow/core/kernels/mkl_lrn_op.cc#L320
    let powOperator;
    const basis = `float(${bias}) + float(${alpha}) * sum`;
    if (beta === 0.5) {
      powOperator = `inversesqrt(${basis})`;
    } else if (beta === 1.0) {
      powOperator = `1.0/(${basis})`;
    } else {
      powOperator = `exp(log(${basis}) * float(-${beta}));`;
    }

    this.userCode = `
      void main() {
        ivec4 coords = getOutputCoords();
        int b = coords[0];
        int r = coords[1];
        int c = coords[2];
        int d = coords[3];
        float x = getX(b, r, c, d);
        float sum = 0.0;
        for (int j = -${rad}; j <= ${rad}; j++) {
          int idx = d + j;
          if (idx >= 0 && idx <=  ${maxD}) {
            float z = getX(b, r, c, idx);
            sum += z * z;
          }
        }
        float val = x * ${powOperator};
        setOutput(val);
      }
    `;
  }
}
