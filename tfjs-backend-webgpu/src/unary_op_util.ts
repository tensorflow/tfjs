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

import {backend_util} from '@tensorflow/tfjs-core';

export enum UnaryOpType {
  ABS,
  ACOS,
  ACOSH,
  ASIN,
  ASINH,
  ATAN,
  ATANH,
  CEIL,
  COS,
  COSH,
  ELU,
  ERF,
  EXP,
  EXPM1,
  FLOOR,
  IS_FINITE,
  IS_INF,
  IS_NAN,
  LINEAR,
  LOG,
  LOG1P,
  LOGICAL_NOT,
  NEG,
  RELU,
  RELU6,
  LEAKYRELU,
  RECIPROCAL,
  RSQRT,
  SIN,
  SINH,
  SIGMOID,
  SQRT,
  SQUARE,
  TAN,
  TANH,
  TO_INT
}

const ABS = `return abs(a);`;
const ACOS = `
  if (abs(a) > 1.) {
    return uniforms.NAN;
  }
  return acos(a);
`;
const ACOSH = `
  if (a < 1.) {
    return uniforms.NAN;
  }
  return acosh(a);
`;
const ASIN = `
  if (abs(a) > 1.) {
    return uniforms.NAN;
  }
  return asin(a);
`;
const ASINH = `return asinh(a);`;
const ATAN = `
  if (isnan(a)) {
    return uniforms.NAN;
  }
  return atan(a);
`;
const ATANH = `
  if (abs(a) > 1.) {
    return uniforms.NAN;
  }
  if (a == 1.) {
    return uniforms.INFINITY;
  }
  if (a == -1.) {
    return -uniforms.INFINITY;
  }
  return atanh(a);
`;
const CEIL = `return ceil(a);`;
const COS = `return cos(a);`;
const COSH = `
  let e2x = exp(-a);
  return (e2x + 1.0 / e2x) / 2.0;
`;
const EXPM1 = `return exp(a) - 1.0;`;
const ELU = `if (a >= 0.0) { return a; }  return (exp(a) - 1.0);`;
const ELU_VEC4 = `
  var resFloat = exp(a) - vec4<f32>(1.0);
  if (a.r >= 0.0) {
    resFloat.r = a.r;
  }
  if (a.g >= 0.0) {
    resFloat.g = a.g;
  }
  if (a.b >= 0.0) {
    resFloat.b = a.b;
  }
  if (a.a >= 0.0) {
    resFloat.a = a.a;
  }
  return resFloat;
`;
const ERF = `
  // Error function is calculated approximately with elementary function.
  // See "Handbook of Mathematical Functions with Formulas,
  // Graphs, and Mathematical Tables", Abramowitz and Stegun.
  let p = ${backend_util.ERF_P};
  let a1 = ${backend_util.ERF_A1};
  let a2 = ${backend_util.ERF_A2};
  let a3 = ${backend_util.ERF_A3};
  let a4 = ${backend_util.ERF_A4};
  let a5 = ${backend_util.ERF_A5};

  let sign = sign(a);
  let absA = abs(a);
  let t = 1.0 / (1.0 + p * absA);
  return sign * (1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp(-absA * absA));
`;
const EXP = `return exp(a);`;
const FLOOR = `return floor(a);`;
const IS_FINITE = `return f32(!isnan(a) && !isinf(a));`;
const IS_INF = `return f32(isinf(a));`;
const IS_NAN = `return f32(isnan(a));`;
const LINEAR = `return a;`;
const LOG = `if (a < 0.0) { return uniforms.NAN; }
  return log(a);`;
const LOG1P = `
  if (isnan(a)) { return a; }
  return log(1.0 + a);
`;
const LOGICAL_NOT = `return f32(!(a >= 1.0));`;
const NEG = `return -a;`;
const LEAKYRELU = `if (a < 0.0) { return uniforms.alpha * a; } return a;`;
const LEAKYRELU_VEC4 = `
  let aLessThanZero = vec4<f32>(a < vec4<f32>(0.0));
  return (aLessThanZero * (uniforms.alpha * a)) + ((vec4<f32>(1.0) - aLessThanZero) * a);
`;
const RECIPROCAL = `return 1.0 / a;`;
const RELU = `return select(a, 0.0, a < 0.0);`;
const RELU6 = 'return clamp(a, 0.0, 6.0);';
const RELU6_VEC4 =
    'return clamp(a, vec4<f32>(0.0, 0.0, 0.0, 0.0), vec4<f32>(6.0, 6.0, 6.0, 6.0));';
const RELU_VEC4 = `
  return select(a, vec4<f32>(0.0), a < vec4<f32>(0.0));
`;
const RSQRT = `return inverseSqrt(a);`;
const SIGMOID = `return 1.0 / (1.0 + exp(-1.0 * a));`;
const SIN = `return sin(a);`;
const SINH = `
  let e2x = exp(a);
  return (e2x - 1.0 / e2x) / 2.0;
`;
const SQRT = `return sqrt(a);`;
const SQUARE = `return a * a;`;
const TAN = `return tan(a);`;
const TANH = `
  let e2x = exp(-2.0 * abs(a));
  return sign(a) * (1.0 - e2x) / (1.0 + e2x);
`;
const TO_INT = `return f32(i32((a)));`;

export function getUnaryOpString(type: UnaryOpType, useVec4?: boolean): string {
  switch (type) {
    case UnaryOpType.ABS:
      return ABS;
    case UnaryOpType.ACOS:
      return ACOS;
    case UnaryOpType.ACOSH:
      return ACOSH;
    case UnaryOpType.ASIN:
      return ASIN;
    case UnaryOpType.ASINH:
      return ASINH;
    case UnaryOpType.ATAN:
      return ATAN;
    case UnaryOpType.ATANH:
      return ATANH;
    case UnaryOpType.COS:
      return COS;
    case UnaryOpType.COSH:
      return COSH;
    case UnaryOpType.CEIL:
      return CEIL;
    case UnaryOpType.ELU:
      return useVec4 ? ELU_VEC4 : ELU;
    case UnaryOpType.ERF:
      return ERF;
    case UnaryOpType.EXP:
      return EXP;
    case UnaryOpType.EXPM1:
      return EXPM1;
    case UnaryOpType.FLOOR:
      return FLOOR;
    case UnaryOpType.IS_FINITE:
      return IS_FINITE;
    case UnaryOpType.IS_INF:
      return IS_INF;
    case UnaryOpType.IS_NAN:
      return IS_NAN;
    case UnaryOpType.LINEAR:
      return LINEAR;
    case UnaryOpType.LOG:
      return LOG;
    case UnaryOpType.LOG1P:
      return LOG1P;
    case UnaryOpType.LOGICAL_NOT:
      return LOGICAL_NOT;
    case UnaryOpType.NEG:
      return NEG;
    case UnaryOpType.LEAKYRELU:
      return useVec4 ? LEAKYRELU_VEC4 : LEAKYRELU;
    case UnaryOpType.RECIPROCAL:
      return RECIPROCAL;
    case UnaryOpType.RELU:
      return useVec4 ? RELU_VEC4 : RELU;
    case UnaryOpType.RELU6:
      return useVec4 ? RELU6_VEC4 : RELU6;
    case UnaryOpType.RSQRT:
      return RSQRT;
    case UnaryOpType.SIGMOID:
      return SIGMOID;
    case UnaryOpType.SIN:
      return SIN;
    case UnaryOpType.SINH:
      return SINH;
    case UnaryOpType.SQRT:
      return SQRT;
    case UnaryOpType.SQUARE:
      return SQUARE;
    case UnaryOpType.TAN:
      return TAN;
    case UnaryOpType.TANH:
      return TANH;
    case UnaryOpType.TO_INT:
      return TO_INT;

    default:
      throw new Error(`BinaryType ${type} is not implemented!`);
  }
}
