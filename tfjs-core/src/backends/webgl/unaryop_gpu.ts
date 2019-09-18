/**
 * @license
 * Copyright 2017 Google Inc. All Rights Reserved.
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

import * as erf_util from '../../ops/erf_util';
import * as selu_util from '../../ops/selu_util';

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

const CHECK_NAN_SNIPPET = `if (isnan(x)) return x;`;

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
  float scaleAlpha = ${selu_util.SELU_SCALEALPHA};
  float scale = ${selu_util.SELU_SCALE};
  return (x >= 0.0) ? scale * x : scaleAlpha * (exp(x) - 1.0);
`;

export function STEP(alpha = 0.0) {
  return CHECK_NAN_SNIPPET + `
    return x > 0.0 ? 1.0 : float(${alpha});
  `;
}

export const NEG = `return -x;`;

export const CEIL = `return ceil(x);`;

export const FLOOR = `return floor(x);`;

export const SIGN = `
  if (isnan(x)) { return 0.0; }
  return sign(x);
`;

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

export const EXP = `return exp(x);`;

export const EXPM1 = `return exp(x) - 1.0;`;

export const LOG = `if (x < 0.0) return NAN;
  return log(x);`;

export const LOG1P = `return log(1.0 + x);`;

export const SQRT = `return sqrt(x);`;

export const RSQRT = `return inversesqrt(x);`;

export const SIGMOID = `return 1.0 / (1.0 + exp(-1.0 * x));`;

/**
 * mirrors the implementation of tf.nn.softplus: https://goo.gl/vkcvwX
 *
 * epsilon is the difference between 1.0 and the next representable
 * float. For a single precision 32 bit float this should be 2^-23, see:
 * https://math.byu.edu/~schow/work/IEEEFloatingPoint.htm
 *
 * too_large = (x > -threshold) is value above which exp(x) may overflow
 * but softplus(x) == x is within machine epsilon
 *
 * too_small = (x < threshold) is value below which exp(x) may underflow,
 * but softplus(x) == exp(x) is within machine epsilon.
 */
export const SOFTPLUS = `
  float epsilon = 1.1920928955078125e-7;
  float threshold = log(epsilon) + 2.0;

  bool too_large = x > -threshold;
  bool too_small = x < threshold;

  float result;
  float exp_x = exp(x);

  if (too_large){
    result = x;
  }
  else if (too_small){
    result = exp_x;
  }
  else{
    result = log(exp_x + 1.0);
  }
  return result;
`;

export const SIN = CHECK_NAN_SNIPPET + `
  return sin(x);
`;

export const COS = CHECK_NAN_SNIPPET + `
  return cos(x);
`;

export const TAN = `return tan(x);`;

export const ASIN = CHECK_NAN_SNIPPET + `
  if (abs(x) > 1.) {
    return NAN;
  }
  return asin(x);
`;

export const ACOS = CHECK_NAN_SNIPPET + `
  if (abs(x) > 1.) {
    return NAN;
  }
  return acos(x);
`;

export const ATAN = CHECK_NAN_SNIPPET + `
  return atan(x);
`;

export const SINH = `
  float e2x = exp(x);
  return (e2x - 1.0 / e2x) / 2.0;
`;

export const COSH = `
  float e2x = exp(-x);
  return (e2x + 1.0 / e2x) / 2.0;
`;

export const TANH = `
  float e2x = exp(-2.0 * abs(x));
  return sign(x) * (1.0 - e2x) / (1.0 + e2x);
`;

export const ASINH = CHECK_NAN_SNIPPET + `return log(x + sqrt(x * x + 1.0));`;

export const ACOSH = CHECK_NAN_SNIPPET + `
  if (x < 1.0) return NAN;
  return log(x + sqrt(x * x - 1.0));`;

export const ATANH = CHECK_NAN_SNIPPET + `
  if ((x < -1.0) || (x > 1.0)) return NAN;
  return (log(1.0 + x) - log(1.0 - x)) / 2.0;`;

export const ERF = `
  // Error function is calculated approximately with elementary function.
  // See "Handbook of Mathematical Functions with Formulas,
  // Graphs, and Mathematical Tables", Abramowitz and Stegun.
  float p = ${erf_util.ERF_P};
  float a1 = ${erf_util.ERF_A1};
  float a2 = ${erf_util.ERF_A2};
  float a3 = ${erf_util.ERF_A3};
  float a4 = ${erf_util.ERF_A4};
  float a5 = ${erf_util.ERF_A5};

  float sign = sign(x);
  x = abs(x);
  float t = 1.0 / (1.0 + p * x);
  return sign * (1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*exp(-x*x));
`;

export const SQUARE = `return x * x;`;

export const RECIPROCAL = `return 1.0 / x;`;

export const LOGICAL_NOT = `return float(!(x >= 1.0));`;

export const TO_INT = `return float(int(x));`;

export const CLONE = 'return x;';
