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
  ELU,
  EXP,
  EXPM1,
  FLOOR,
  LINEAR,
  LOG,
  NEG,
  PRELU,
  RELU,
  RELU6,
  RSQRT,
  SIGMOID,
  SQRT,
  SQUARE,
  TANH,
  TO_INT
}

const ABS = `return abs(a);`;
const CEIL = `return ceil(a);`;
const EXPM1 = `return exp(a) - 1.0;`;
const ELU = `return (a >= 0.0) ? a : (exp(a) - 1.0);`;
const ELU_VEC4 = `
  vec4 result;

  result.r = (a.r >= 0.0) ? a.r : (exp(a.r) - 1.0);
  result.g = (a.g >= 0.0) ? a.g : (exp(a.g) - 1.0);
  result.b = (a.b >= 0.0) ? a.b : (exp(a.b) - 1.0);
  result.a = (a.a >= 0.0) ? a.a : (exp(a.a) - 1.0);

  return result;
`;
const EXP = `return exp(a);`;
const FLOOR = `return floor(a);`;
const LINEAR = `return a;`;
const LOG = `if (a < 0.0) return 1.0/0.0;
  return log(a);`;
const NEG = `return -a;`;
const PRELU = `return (a < 0.) ? b * a : a;`;
const RELU = 'return max(a, 0.0);';
const RELU6 = 'return clamp(a, 0.0, 6.0);';
const RELU_VEC4 = `
  vec4 result = a * vec4(greaterThanEqual(a, vec4(0.0)));
  bvec4 isNaN = isnan(a);

  result.r = isNaN.r ? a.r : result.r;
  result.g = isNaN.g ? a.g : result.g;
  result.b = isNaN.b ? a.b : result.b;
  result.a = isNaN.a ? a.a : result.a;

  return result;
`;
const RSQRT = `return inversesqrt(a);`;
const SIGMOID = `return 1.0 / (1.0 + exp(-1.0 * a));`;
const SQRT = `return sqrt(a);`;
const SQUARE = `return a * a;`;
const TANH = `
  float e2x = exp(-2.0 * abs(a));
  return sign(a) * (1.0 - e2x) / (1.0 + e2x);
`;
const TO_INT = `return float(int(a));`;

export function getUnaryOpString(type: UnaryOpType, useVec4?: boolean): string {
  switch (type) {
    case UnaryOpType.ABS:
      return ABS;
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
    case UnaryOpType.NEG:
      return NEG;
    case UnaryOpType.PRELU:
      return PRELU;
    case UnaryOpType.RELU:
      return useVec4 ? RELU_VEC4 : RELU;
    case UnaryOpType.RELU6:
      return RELU6;
    case UnaryOpType.RSQRT:
      return RSQRT;
    case UnaryOpType.SIGMOID:
      return SIGMOID;
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
