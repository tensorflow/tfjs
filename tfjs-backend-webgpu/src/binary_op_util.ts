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

export enum BinaryOpType {
  ADD,
  ATAN2,
  COMPLEX_MULTIPLY_IMAG,
  COMPLEX_MULTIPLY_REAL,
  DIV,
  EQUAL,
  GREATER,
  GREATER_EQUAL,
  INT_DIV,
  LESS,
  LESS_EQUAL,
  LOGICAL_AND,
  LOGICAL_OR,
  MAX,
  MIN,
  MOD,
  MUL,
  NOT_EQUAL,
  POW,
  PRELU,
  SQUARED_DIFFERENCE,
  SUB
}

const CHECK_NAN_SNIPPET = `
  if (isnan(a)) { return a; }
  if (isnan(b)) { return b; }
  `;

const CHECK_NAN_SNIPPET_VEC4_INNER = `
  if (isNaN.r) {
    resultTemp.r = valueForNaN;
  }
  if (isNaN.g) {
    resultTemp.g = valueForNaN;
  }
  if (isNaN.b) {
    resultTemp.b = valueForNaN;
  }
  if (isNaN.a) {
    resultTemp.a = valueForNaN;
  }
  `;

const CHECK_NAN_SNIPPET_VEC4 = `
  let isNaN = isnanVec4(a) | isnanVec4(b);
  ${CHECK_NAN_SNIPPET_VEC4_INNER}
  `;

const ADD = 'return a + b;';
// (Ar + Ai)(Br + Bi) =
// ArBr + ArBi + AiBr + AiBi = ArBr - AB + ArBi + AiBr
// Yr = ArBr - AB
// Yi = ArBi + AiBr
const COMPLEX_MULTIPLY_REAL = 'return areal * breal - aimag * bimag;';
const COMPLEX_MULTIPLY_IMAG = 'return areal * bimag + aimag * breal;';
const DIV = 'return a / b;';
const EQUAL = 'return f32(a == b);';
const EQUAL_VEC4 = 'return vec4<f32>(a == b);';
const GREATER = 'return f32(a > b);';
const GREATER_VEC4 = 'return vec4<f32>(a > b);';
const GREATER_EQUAL = 'return f32(a >= b);';
const GREATER_EQUAL_VEC4 = 'return vec4<f32>(a >= b);';

const INT_DIV = `
  let s = sign(a) * sign(b);
  let ia = i32(round(a));
  let ib = i32(round(b));
  return f32(idiv(ia, ib, s));
`;
const INT_DIV_VEC4 = `
  let ia = vec4<i32>(round(a));
  let ib = vec4<i32>(round(b));
  let cond = ib != vec4<i32>(0);
  var resultTemp = vec4<i32>(0);
  let s = sign(a) * sign(b);

  // Windows (D3D) wants guaranteed non-zero int division at compile-time.
  if (cond[0]) {
    resultTemp[0] = idiv(ia[0], ib[0], s[0]);
  }
  if (cond[1]) {
    resultTemp[1] = idiv(ia[1], ib[1], s[1]);
  }
  if (cond[2]) {
    resultTemp[2] = idiv(ia[2], ib[2], s[2]);
  }
  if (cond[3]) {
    resultTemp[3] = idiv(ia[3], ib[3], s[3]);
  }
  return vec4<f32>(resultTemp);
`;

const LESS = 'return f32(a < b);';
const LESS_VEC4 = 'return vec4<f32>(a < b);';
const LESS_EQUAL = 'return f32(a <= b);';
const LESS_EQUAL_VEC4 = 'return vec4<f32>(a <= b);';
const LOGICAL_AND = 'return f32(a >= 1.0 && b >= 1.0);';
const LOGICAL_AND_VEC4 = `return (vec4<f32>(a >= vec4<f32>(1.0)) *
  vec4<f32>(b >= vec4<f32>(1.0)));`;
const LOGICAL_OR = 'return f32(a >= 1.0 || b >= 1.0);';
const LOGICAL_OR_VEC4 = `return min(vec4<f32>(a >= vec4<f32>(1.0)) +
  vec4<f32>(b >= vec4<f32>(1.0)), vec4<f32>(1.0));`;
const MOD = `
  ${CHECK_NAN_SNIPPET}
  if (b == 0.) {
    return uniforms.NAN;
  }
  var resultTemp = a % b;
  if ((a < 0. && b < 0.) || (a >= 0. && b > 0.)) {
    return resultTemp;
  } else {
    return (resultTemp + b) % b;
  }
`;
const MOD_VEC4 = `
  let valueForNaN = uniforms.NAN;
  var resultTemp = vec4<f32>(a % b);
  ${CHECK_NAN_SNIPPET_VEC4}

  if (b[0] == 0.) {
    resultTemp[0] = uniforms.NAN;
  }
  if (b[1] == 0.) {
    resultTemp[1] = uniforms.NAN;
  }
  if (b[2] == 0.) {
    resultTemp[2] = uniforms.NAN;
  }
  if (b[3] == 0.) {
    resultTemp[3] = uniforms.NAN;
  }

  if (!((a[0] < 0. && b[0] < 0.) || (a[0] >= 0. && b[0] > 0.))) {
    resultTemp[0] = (resultTemp[0] + b[0]) % b[0];
  }
  if (!((a[1] < 0. && b[1] < 0.) || (a[1] >= 0. && b[1] > 0.))) {
    resultTemp[1] = (resultTemp[1] + b[1]) % b[1];
  }
  if (!((a[2] < 0. && b[2] < 0.) || (a[2] >= 0. && b[2] > 0.))) {
    resultTemp[2] = (resultTemp[2] + b[2]) % b[2];
  }
  if (!((a[3] < 0. && b[3] < 0.) || (a[3] >= 0. && b[3] > 0.))) {
    resultTemp[3] = (resultTemp[3] + b[3]) % b[3];
  }

  return resultTemp;
`;
const MUL = 'return a * b;';
const NOT_EQUAL = `
  if (isnan(a) || isnan(b)) {
    return 1.0;
  }
  return f32(a != b);
`;
const NOT_EQUAL_VEC4 = `
  var resultTemp = vec4<f32>(a != b);
  let valueForNaN = 1.0;
  ${CHECK_NAN_SNIPPET_VEC4}

  return resultTemp;
`;

const POW = `
  if(a < 0.0 && floor(b) < b) {
    return uniforms.NAN;
  }
  if (b == 0.0) {
    return 1.0;
  }
  if (round(abs(b) % 2.0) != 1.0) {
    return pow(abs(a), b);
  }
  return sign(a) * pow(abs(a), b);
`;
const POW_VEC4 = `
  let isModRound1Bool = vec4<i32>(round(abs(b) % vec4<f32>(2.0))) == vec4<i32>(1);
  let isModRound1 = vec4<f32>(isModRound1Bool);
  let multiplier = sign(a) * isModRound1 + (vec4<f32>(1.0) - isModRound1);
  var resultTemp = multiplier * pow(abs(a), b);

  // Ensure that a^0 = 1, including 0^0 = 1 as this correspond to TF and JS
  let isExpZero = b == vec4<f32>(0.0);
  if (isExpZero.r) {
    resultTemp.r = 1.0;
  }
  if (isExpZero.g) {
    resultTemp.g = 1.0;
  }
  if (isExpZero.b) {
    resultTemp.b = 1.0;
  }
  if (isExpZero.a) {
    resultTemp.a = 1.0;
  }
  let isNaN = (a < vec4<f32>(0.0)) & (floor(b) < b);
  let valueForNaN = uniforms.NAN;
  ${CHECK_NAN_SNIPPET_VEC4_INNER}
  return resultTemp;
`;

const PRELU = `if (a < 0.0) { return b * a; }  return a;`;
const PRELU_VEC4 = `
  let aLessThanZero = vec4<f32>(a < vec4<f32>(0.0));
  return (aLessThanZero * (b * a)) + ((vec4<f32>(1.0) - aLessThanZero) * a);
`;
const SQUARED_DIFFERENCE = 'return (a - b) * (a - b);';
const SUB = 'return a - b;';

function getBinaryWithNanString(
    op: string, useVec4: boolean, valueForNaN = 'uniforms.NAN') {
  const checkNanSnippet = useVec4 ? CHECK_NAN_SNIPPET_VEC4 : CHECK_NAN_SNIPPET;
  return useVec4 ? `
    let valueForNaN = ${valueForNaN};
    var resultTemp = vec4<f32>(${op}(a, b));
    ` + checkNanSnippet +
          `
    return resultTemp;
  ` :
                   checkNanSnippet + `
    return ${op}(a, b);
  `;
}

export function getBinaryOpString(
    type: BinaryOpType, useVec4?: boolean): string {
  switch (type) {
    case BinaryOpType.ADD:
      return ADD;
    case BinaryOpType.ATAN2:
      return getBinaryWithNanString('atan2', useVec4);
    case BinaryOpType.COMPLEX_MULTIPLY_IMAG:
      return COMPLEX_MULTIPLY_IMAG;
    case BinaryOpType.COMPLEX_MULTIPLY_REAL:
      return COMPLEX_MULTIPLY_REAL;
    case BinaryOpType.DIV:
      return DIV;
    case BinaryOpType.EQUAL:
      return useVec4 ? EQUAL_VEC4 : EQUAL;
    case BinaryOpType.GREATER:
      return useVec4 ? GREATER_VEC4 : GREATER;
    case BinaryOpType.GREATER_EQUAL:
      return useVec4 ? GREATER_EQUAL_VEC4 : GREATER_EQUAL;
    case BinaryOpType.INT_DIV:
      return useVec4 ? INT_DIV_VEC4 : INT_DIV;
    case BinaryOpType.LESS:
      return useVec4 ? LESS_VEC4 : LESS;
    case BinaryOpType.LESS_EQUAL:
      return useVec4 ? LESS_EQUAL_VEC4 : LESS_EQUAL;
    case BinaryOpType.LOGICAL_AND:
      return useVec4 ? LOGICAL_AND_VEC4 : LOGICAL_AND;
    case BinaryOpType.LOGICAL_OR:
      return useVec4 ? LOGICAL_OR_VEC4 : LOGICAL_OR;
    case BinaryOpType.MAX:
      return getBinaryWithNanString('max', useVec4);
    case BinaryOpType.MIN:
      return getBinaryWithNanString('min', useVec4);
    case BinaryOpType.MOD:
      return useVec4 ? MOD_VEC4 : MOD;
    case BinaryOpType.MUL:
      return MUL;
    case BinaryOpType.NOT_EQUAL:
      return useVec4 ? NOT_EQUAL_VEC4 : NOT_EQUAL;
    case BinaryOpType.POW:
      return useVec4 ? POW_VEC4 : POW;
    case BinaryOpType.PRELU:
      return useVec4 ? PRELU_VEC4 : PRELU;
    case BinaryOpType.SQUARED_DIFFERENCE:
      return SQUARED_DIFFERENCE;
    case BinaryOpType.SUB:
      return SUB;
    default:
      throw new Error(`BinaryType ${type} is not implemented!`);
  }
}
