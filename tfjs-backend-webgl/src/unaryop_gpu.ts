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

import {backend_util} from '@tensorflow/tfjs-core';
import {GPGPUProgram} from './gpgpu_math';

export class UnaryOpProgram implements GPGPUProgram {
  variableNames = ['A'];
  userCode: string;
  outputShape: number[];

  constructor(aShape: number[], opSnippet: string) {
    this.outputShape = aShape;
    this.userCode = `
      float unaryOperation(float x) {
        ${opSnippet}
      }

      void main() {
        float x = getAAtOutCoords();
        float y = unaryOperation(x);

        setOutput(y);
      }
    `;
  }
}

export const CHECK_NAN_SNIPPET = `if (isnan(x)) return x;`;

export const LINEAR = `return x;`;

export const ABS = `return abs(x);`;

export const RELU = CHECK_NAN_SNIPPET + `
  return (x < 0.0) ? 0.0 : x;
`;

export const RELU6 = CHECK_NAN_SNIPPET + `
  return (x < 0.0) ? 0.0 : min(6.0, x);
`;

export const ELU = `return (x >= 0.0) ? x : (exp(x) - 1.0);`;

export const SELU = `
  // Stable and Attracting Fixed Point (0, 1) for Normalized Weights.
  // see: https://arxiv.org/abs/1706.02515
  float scaleAlpha = ${backend_util.SELU_SCALEALPHA};
  float scale = ${backend_util.SELU_SCALE};
  return (x >= 0.0) ? scale * x : scaleAlpha * (exp(x) - 1.0);
`;

export function STEP(alpha = 0.0) {
  return CHECK_NAN_SNIPPET + `
    return x > 0.0 ? 1.0 : float(${alpha});
  `;
}

export const NEG = `return -x;`;

export const CEIL = `return ceil(x);`;

export const IS_NAN = `return float(isnan(x));`;

export const IS_INF = `return float(isinf(x));`;

export const IS_FINITE = `return float(!isnan(x) && !isinf(x));`;

export const ROUND = `
  // OpenGL ES does not support round function.
  // The algorithm is based on banker's rounding.
  float base = floor(x);
  if ((x - base) < 0.5) {
    return floor(x);
  } else if ((x - base) > 0.5) {
    return ceil(x);
  } else {
    if (mod(base, 2.0) == 0.0) {
      return base;
    } else {
      return base + 1.0;
    }
  }
`;

export const EXPM1 = `return exp(x) - 1.0;`;

export const LOG = `if (x < 0.0) return NAN;
  return log(x);`;

export const LOG1P = `return log(1.0 + x);`;

export const SQRT = `return sqrt(x);`;

export const RSQRT = `return inversesqrt(x);`;

export const RECIPROCAL = `return 1.0 / x;`;

export const LOGICAL_NOT = `return float(!(x >= 1.0));`;

export const CLONE = 'return x;';
