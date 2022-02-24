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

export enum UnaryOpType {
  ABS,
  CEIL,
  COS,
  COSH,
  ELU,
  EXP,
  EXPM1,
  FLOOR,
  LINEAR,
  LOG,
  LOGICAL_NOT,
  NEG,
  PRELU,
  RELU,
  RELU6,
  LEAKYRELU,
  RSQRT,
  SIN,
  SINH,
  SIGMOID,
  SQRT,
  SQUARE,
  TANH,
  TO_INT
}

const ABS = `return abs(a);`;
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
const EXP = `return exp(a);`;
const FLOOR = `return floor(a);`;
const LINEAR = `return a;`;
const LOG = `if (a < 0.0) { return 1.0/0.0; }
  return log(a);`;
const LOGICAL_NOT = `return f32(!(a >= 1.0));`;
const NEG = `return -a;`;
const PRELU = `return (a < 0.0) ? b * a : a;`;
const LEAKYRELU = `if (a < 0.0) { return uniforms.alpha * a; } return a;`;
const RELU = `if(a < 0.0) { return 0.0; } return a;`;
const RELU6 = 'return clamp(a, 0.0, 6.0);';
const RELU6_VEC4 =
    'return clamp(a, vec4<f32>(0.0, 0.0, 0.0, 0.0), vec4<f32>(6.0, 6.0, 6.0, 6.0));';
const RELU_VEC4 = `
  var resFloat = a * vec4<f32>(a >= vec4<f32>(0.0));
  let isNaN = isnanVec4(a);

  if (isNaN.r) {
    resFloat.r = a.r;
  }
  if (isNaN.g) {
    resFloat.g = a.g;
  }
  if (isNaN.b) {
    resFloat.b = a.b;
  }
  if (isNaN.a) {
    resFloat.a = a.a;
  }
  return resFloat;
`;
const RSQRT = `return 1.0/sqrt(a);`;
const SIGMOID = `return 1.0 / (1.0 + exp(-1.0 * a));`;
const SIN = `return sin(a);`;
const SINH = `
  let e2x = exp(a);
  return (e2x - 1.0 / e2x) / 2.0;
`;
const SQRT = `return sqrt(a);`;
const SQUARE = `return a * a;`;
const TANH = `
  let e2x = exp(-2.0 * abs(a));
  return sign(a) * (1.0 - e2x) / (1.0 + e2x);
`;
const TO_INT = `return f32(i32((a)));`;

export function getUnaryOpString(type: UnaryOpType, useVec4?: boolean): string {
  switch (type) {
    case UnaryOpType.ABS:
      return ABS;
    case UnaryOpType.COS:
      return COS;
    case UnaryOpType.COSH:
      return COSH;
    case UnaryOpType.CEIL:
      return CEIL;
    case UnaryOpType.ELU:
      return useVec4 ? ELU_VEC4 : ELU;
    case UnaryOpType.EXP:
      return EXP;
    case UnaryOpType.EXPM1:
      return EXPM1;
    case UnaryOpType.FLOOR:
      return FLOOR;
    case UnaryOpType.LINEAR:
      return LINEAR;
    case UnaryOpType.LOG:
      return LOG;
    case UnaryOpType.LOGICAL_NOT:
      return LOGICAL_NOT;
    case UnaryOpType.NEG:
      return NEG;
    case UnaryOpType.PRELU:
      return PRELU;
    case UnaryOpType.LEAKYRELU:
      return LEAKYRELU;
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
    case UnaryOpType.TANH:
      return TANH;
    case UnaryOpType.TO_INT:
      return TO_INT;

    default:
      throw new Error(`BinaryType ${type} is not implemented!`);
  }
}
