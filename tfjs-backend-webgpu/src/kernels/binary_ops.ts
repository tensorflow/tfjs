/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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

import {util} from '@tensorflow/tfjs-core';
import {BinaryOpSharedProgram} from './binary_op_shared_webgpu';
import {BinaryOpVec4Program} from './binary_op_vec4_webgpu';
import {BinaryOpProgram} from './binary_op_webgpu';

export enum BinaryOpType {
  MUL,
  ADD,
  SUB,
  DIV,
  GREATER,
  GREATER_EQUAL,
  LESS,
  LESS_EQUAL,
  NOT_EQUAL,
  SQUARED_DIFFERENCE,
  INT_DIV,
  PRELU,
  MAX,
  MIN
}

export function getBinaryOpString(
    type: BinaryOpType, useVec4?: boolean): string {
  switch (type) {
    case BinaryOpType.MUL:
      return 'return a * b;';
    case BinaryOpType.ADD:
      return 'return a + b;';
    case BinaryOpType.SUB:
      return 'return a - b;';
    case BinaryOpType.DIV:
      return 'return a / b;';
    case BinaryOpType.GREATER:
      return useVec4 ? 'return vec4(greaterThan(a, b));' :
                       'return float(a > b);';
    case BinaryOpType.GREATER_EQUAL:
      return useVec4 ? 'return vec4(greaterThanEqual(a, b));' :
                       'return float(a >= b);';
    case BinaryOpType.LESS:
      return useVec4 ? 'return vec4(lessThan(a, b));' : 'return float(a < b);';
    case BinaryOpType.LESS_EQUAL:
      return useVec4 ? 'return vec4(lessThanEqual(a, b));' :
                       'return float(a <= b);';
    case BinaryOpType.NOT_EQUAL:
      return 'return a != b;';
    case BinaryOpType.SQUARED_DIFFERENCE:
      return 'return (a - b) * (a - b);';
    case BinaryOpType.INT_DIV:
      return useVec4 ? `
      ivec4 ia = round(a);
      ivec4 ib = round(b);
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
    ` :
                       `
    float s = sign(a) * sign(b);
    int ia = int(round(a));
    int ib = int(round(b));
    return float(idiv(ia, ib, s));
  `;
    case BinaryOpType.PRELU:
      return useVec4 ? `
      vec4 aLessThanZero = vec4(lessThan(a, vec4(0.)));
      return (aLessThanZero * (b * a)) + ((vec4(1.0) - aLessThanZero) * a);
    ` :
                       'return (a < 0.) ? b * a : a;';
    case BinaryOpType.MAX:
      // TODO (xing.xu@intel.com): Currently, NaN is not supported in WebGPU
      // backend.
      // https://github.com/tensorflow/tfjs/issues/4734
      return `return max(a, b);`;
    case BinaryOpType.MIN:
      // TODO (xing.xu@intel.com): Currently, NaN is not supported in WebGPU
      // backend.
      // https://github.com/tensorflow/tfjs/issues/4734
      return `return min(a, b);`;
    default:
      throw new Error(`BinaryType ${type} is not implemented!`);
  }
}

export function getBinaryProgram(
    op: BinaryOpType, aShape: number[], bShape: number[]) {
  const useVec4 =
      util.arraysEqual(aShape, bShape) && util.sizeFromShape(aShape) % 4 === 0;
  const opStr = getBinaryOpString(op, useVec4);
  if (useVec4) {
    return new BinaryOpVec4Program(opStr, aShape, bShape);
  }
  const useSharedMemoryWithA =
      aShape.length === 1 && bShape.length > 1 && aShape[0] < 2048;
  const useSharedMemoryWithB =
      bShape.length === 1 && aShape.length > 1 && bShape[0] < 2048;
  if (useSharedMemoryWithA || useSharedMemoryWithB) {
    return new BinaryOpSharedProgram(
        opStr, aShape, bShape, useSharedMemoryWithB);
  } else {
    return new BinaryOpProgram(opStr, aShape, bShape);
  }
}
