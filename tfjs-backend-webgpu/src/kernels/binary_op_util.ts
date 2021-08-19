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
  MUL,
  ADD,
  SUB,
  DIV,
  EQUAL,
  GREATER,
  GREATER_EQUAL,
  LESS,
  LESS_EQUAL,
  LOGICAL_AND,
  NOT_EQUAL,
  SQUARED_DIFFERENCE,
  INT_DIV,
  POW,
  PRELU,
  MAX,
  MIN,
  COMPLEX_MULTIPLY_REAL,
  COMPLEX_MULTIPLY_IMAG
}

// GLSL shader.
const CHECK_NAN_SNIPPET = `
  if (isnan(a)) return a;
  if (isnan(b)) return b;
  `;
const CHECK_NAN_SNIPPET_VEC4 = `
  result.r = isNaN.r > 0. ? NAN : result.r;
  result.g = isNaN.g > 0. ? NAN : result.g;
  result.b = isNaN.b > 0. ? NAN : result.b;
  result.a = isNaN.a > 0. ? NAN : result.a;
  `;

const ADD = 'return a + b;';
// (Ar + Ai)(Br + Bi) =
// ArBr + ArBi + AiBr + AiBi = ArBr - AB + ArBi + AiBr
// Yr = ArBr - AB
// Yi = ArBi + AiBr
const COMPLEX_MULTIPLY_REAL = 'return areal * breal - aimag * bimag;';
const COMPLEX_MULTIPLY_IMAG = 'return areal * bimag + aimag * breal;';
const DIV = 'return a / b;';
const EQUAL = 'return float(a == b);';
const EQUAL_VEC4 = 'return vec4(equal(a, b));';
const GREATER = 'return float(a > b);';
const GREATER_VEC4 = 'return vec4(greaterThan(a, b));';
const GREATER_EQUAL = 'return float(a >= b);';
const GREATER_EQUAL_VEC4 = 'return vec4(greaterThanEqual(a, b));';
const INT_DIV = `
  float s = sign(a) * sign(b);
  int ia = int(round(a));
  int ib = int(round(b));
  return float(idiv(ia, ib, s));
  `;
const INT_DIV_VEC4 = `
  ivec4 ia = ivec4(round(a));
  ivec4 ib = ivec4(round(b));
  bvec4 cond = notEqual(ib, ivec4(0));
  ivec4 result = ivec4(0);
  vec4 s = sign(a) * sign(b);

  // Windows (D3D) wants guaranteed non-zero int division at compile-time.
  if (cond[0]) {
    result[0] = idiv(ia[0], ib[0], s[0]);
  }
  if (cond[1]) {
    result[1] = idiv(ia[1], ib[1], s[1]);
  }
  if (cond[2]) {
    result[2] = idiv(ia[2], ib[2], s[2]);
  }
  if (cond[3]) {
    result[3] = idiv(ia[3], ib[3], s[3]);
  }
  return vec4(result);
  `;
const LESS = 'return float(a < b);';
const LESS_VEC4 = 'return vec4(lessThan(a, b));';
const LESS_EQUAL = 'return float(a <= b);';
const LESS_EQUAL_VEC4 = 'return vec4(lessThanEqual(a, b));';
const LOGICAL_AND = 'return float(float(a) >= 1.0 && float(b) >= 1.0);';
const LOGICAL_AND_VEC4 = `return vec4(
  vec4(greaterThanEqual(a, vec4(1.0))) *
  vec4(greaterThanEqual(b, vec4(1.0))));`;
const MUL = 'return a * b;';
const NOT_EQUAL = 'return float(a != b);';
const NOT_EQUAL_VEC4 = 'return vec4(notEqual(a, b));';
const POW = `
  if(a < 0.0 && floor(b) < b) {
    return NAN;
  }
  if (b == 0.0) {
    return 1.0;
  }
  return (round(mod(b, 2.0)) != 1) ?
      pow(abs(a), b) : sign(a) * pow(abs(a), b);
  `;
const POW_VEC4 = `
  // isModRound1 has 1 for components with round(mod(b, 2.0)) == 1, 0 otherwise.
  vec4 isModRound1 = vec4(equal(round(mod(b, 2.0)), ivec4(1)));
  vec4 multiplier = sign(a) * isModRound1 + (vec4(1.0) - isModRound1);
  vec4 result = multiplier * pow(abs(a), b);

  // Ensure that a^0 = 1, including 0^0 = 1 as this correspond to TF and JS
  bvec4 isExpZero = equal(b, vec4(0.0));
  result.r = isExpZero.r ? 1.0 : result.r;
  result.g = isExpZero.g ? 1.0 : result.g;
  result.b = isExpZero.b ? 1.0 : result.b;
  result.a = isExpZero.a ? 1.0 : result.a;

  vec4 isNaN = vec4(lessThan(a, vec4(0.0))) * vec4(lessThan(floor(b), b));
  ${CHECK_NAN_SNIPPET_VEC4}
  return result;
  `;
const PRELU = 'return (a < 0.) ? b * a : a;';
const PRELU_VEC4 = `
  vec4 aLessThanZero = vec4(lessThan(a, vec4(0.)));
  return (aLessThanZero * (b * a)) + ((vec4(1.0) - aLessThanZero) * a);
  `;
const SQUARED_DIFFERENCE = 'return (a - b) * (a - b);';
const SUB = 'return a - b;';

// WGSL shader.
const EQUAL_WGSL = 'return f32(a == b);';
const EQUAL_VEC4_WGSL = 'return vec4<f32>(a == b);';
const GREATER_WGSL = 'return f32(a > b);';
const GREATER_VEC4_WGSL = 'return vec4<f32>(a > b);';
const GREATER_EQUAL_WGSL = 'return f32(a >= b);';
const GREATER_EQUAL_VEC4_WGSL = 'return vec4<f32>(a >= b);';
const LESS_WGSL = 'return f32(a < b);';
const LESS_VEC4_WGSL = 'return vec4<f32>(a < b);';
const LESS_EQUAL_WGSL = 'return f32(a <= b);';
const LESS_EQUAL_VEC4_WGSL = 'return vec4<f32>(a <= b);';
const LOGICAL_AND_WGSL = 'return f32(f32(a) >= 1.0 && f32(b) >= 1.0);';
const LOGICAL_AND_VEC4_WGSL = `return (vec4<f32>(a >= vec4<f32>(1.0)) *
  vec4<f32>(b >= vec4<f32>(1.0)));`;
const CHECK_NAN_SNIPPET_WGSL = `
  if (isNanCustom(a)) { return a; }
  if (isNanCustom(b)) { return b; }
  `;
const CHECK_NAN_SNIPPET_VEC4_WGSL = `
  if (isNaN.r > 0.) {
    resultTemp.r = uniforms.NAN;
  }
  if (isNaN.g > 0.) {
    resultTemp.g = uniforms.NAN;
  }
  if (isNaN.b > 0.) {
    resultTemp.b = uniforms.NAN;
  }
  if (isNaN.a > 0.) {
    resultTemp.a = uniforms.NAN;
  }
  `;
const INT_DIV_WGSL = `
  let s = sign(a) * sign(b);
  let ia = i32(round(a));
  let ib = i32(round(b));
  return f32(idiv(ia, ib, s));
  `;
const INT_DIV_VEC4_WGSL = `
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

const NOT_EQUAL_WGSL = 'return f32(a != b);';
const NOT_EQUAL_VEC4_WGSL = 'return vec4<f32>(a != b);';

const POW_WGSL = `
  if(a < 0.0 && floor(b) < b) {
    return f32(uniforms.NAN);
  }
  if (b == 0.0) {
    return 1.0;
  }
  if (i32(round(b % 2.0)) != 1) {
    return pow(abs(a), b);
  }
  return sign(a) * pow(abs(a), b);
  `;

const POW_VEC4_WGSL = `
  let isModRound1Bool = vec4<i32>(round(b % vec4<f32>(2.0))) == vec4<i32>(1);
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
  let isNaN = vec4<f32>(a < vec4<f32>(0.0)) * vec4<f32>(floor(b) < b);
  ${CHECK_NAN_SNIPPET_VEC4_WGSL}
  return resultTemp;
  `;

const PRELU_WGSL = `if (a < 0.0) { return b * a; }  return a;`;
const PRELU_VEC4_WGSL = `
  let aLessThanZero : vec4<bool> = vec4<bool>(a < vec4<f32>(0.0));
  let aLessThanZeroF32 = vec4<f32>(aLessThanZero);
  return (vec4<f32>(aLessThanZeroF32) * (b * a)) + ((vec4<f32>(1.0) - vec4<f32>(aLessThanZeroF32)) * a);
  `;

function getMinMaxString(op: string, useVec4: boolean, useWGSL = false) {
  if (useWGSL) {
    const checkNanSnippetWgsl =
        useVec4 ? CHECK_NAN_SNIPPET_VEC4_WGSL : CHECK_NAN_SNIPPET_WGSL;
    return useVec4 ? `
    var resultTemp = vec4<f32>(${op}(a, b));
    let isNaN = min(vec4<f32>(isNanCustomVec4F32(a)) + vec4<f32>(isNanCustomVec4F32(b)), vec4<f32>(1.0));
    ` + checkNanSnippetWgsl +
            `
    return resultTemp;
  ` :
                     checkNanSnippetWgsl + `
    return ${op}(a, b);
  `;
  }
  const checkNanSnippet = useVec4 ? CHECK_NAN_SNIPPET_VEC4 : CHECK_NAN_SNIPPET;
  return useVec4 ? `
    vec4 result = vec4(${op}(a, b));
    vec4 isNaN = min(vec4(isnan(a)) + vec4(isnan(b)), vec4(1.0));
    ` + checkNanSnippet +
          `
    return result;
  ` :
                   checkNanSnippet + `
    return ${op}(a, b);
  `;
}

export function getBinaryOpString(
    type: BinaryOpType, useVec4?: boolean, useWgsl?: boolean): string {
  switch (type) {
    case BinaryOpType.MUL:
      return MUL;
    case BinaryOpType.ADD:
      return ADD;
    case BinaryOpType.SUB:
      return SUB;
    case BinaryOpType.DIV:
      return DIV;
    case BinaryOpType.EQUAL:
      if (useWgsl) {
        return useVec4 ? EQUAL_VEC4_WGSL : EQUAL_WGSL;
      } else {
        return useVec4 ? EQUAL_VEC4 : EQUAL;
      }
    case BinaryOpType.GREATER:
      if (useWgsl) {
        return useVec4 ? GREATER_VEC4_WGSL : GREATER_WGSL;
      } else {
        return useVec4 ? GREATER_VEC4 : GREATER;
      }
    case BinaryOpType.GREATER_EQUAL:
      if (useWgsl) {
        return useVec4 ? GREATER_EQUAL_VEC4_WGSL : GREATER_EQUAL_WGSL;
      } else {
        return useVec4 ? GREATER_EQUAL_VEC4 : GREATER_EQUAL;
      }
    case BinaryOpType.LESS:
      if (useWgsl) {
        return useVec4 ? LESS_VEC4_WGSL : LESS_WGSL;
      } else {
        return useVec4 ? LESS_VEC4 : LESS;
      }
    case BinaryOpType.LESS_EQUAL:
      if (useWgsl) {
        return useVec4 ? LESS_EQUAL_VEC4_WGSL : LESS_EQUAL_WGSL;
      } else {
        return useVec4 ? LESS_EQUAL_VEC4 : LESS_EQUAL;
      }
    case BinaryOpType.LOGICAL_AND:
      if (useWgsl) {
        return useVec4 ? LOGICAL_AND_VEC4_WGSL : LOGICAL_AND_WGSL;
      } else {
        return useVec4 ? LOGICAL_AND_VEC4 : LOGICAL_AND;
      }
    case BinaryOpType.NOT_EQUAL:
      if (useWgsl) {
        return useVec4 ? NOT_EQUAL_VEC4_WGSL : NOT_EQUAL_WGSL;
      } else {
        return useVec4 ? NOT_EQUAL_VEC4 : NOT_EQUAL;
      }
    case BinaryOpType.SQUARED_DIFFERENCE:
      return SQUARED_DIFFERENCE;
    case BinaryOpType.INT_DIV:
      if (useWgsl) {
        return useVec4 ? INT_DIV_VEC4_WGSL : INT_DIV_WGSL;
      } else {
        return useVec4 ? INT_DIV_VEC4 : INT_DIV;
      }
    case BinaryOpType.PRELU:
      if (useWgsl) {
        return useVec4 ? PRELU_VEC4_WGSL : PRELU_WGSL;
      } else {
        return useVec4 ? PRELU_VEC4 : PRELU;
      }
    case BinaryOpType.MAX:
      return getMinMaxString('max', useVec4, useWgsl);
    case BinaryOpType.MIN:
      return getMinMaxString('min', useVec4, useWgsl);
    case BinaryOpType.POW:
      if (useWgsl) {
        return useVec4 ? POW_VEC4_WGSL : POW_WGSL;
      } else {
        return useVec4 ? POW_VEC4 : POW;
      }
    case BinaryOpType.COMPLEX_MULTIPLY_REAL:
      return COMPLEX_MULTIPLY_REAL;
    case BinaryOpType.COMPLEX_MULTIPLY_IMAG:
      return COMPLEX_MULTIPLY_IMAG;
    default:
      throw new Error(`BinaryType ${type} is not implemented!`);
  }
}
