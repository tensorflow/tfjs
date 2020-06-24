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

import {BinaryOpSharedProgram} from './binary_op_shared_webgpu';
import {BinaryOpUniformProgram} from './binary_op_uniform_webgpu';
import {BinaryOpProgram} from './binary_op_webgpu';

export const MUL = 'return a * b;';
export const ADD = 'return a + b;';
export const SUB = 'return a - b;';
export const DIV = 'return a / b;';
export const GREATER = 'return float(a > b);';
export const GREATER_EQUAL = 'return float(a >= b);';
export const LESS = `return float(a < b);`;
export const LESS_EQUAL = `return float(a <= b);`;
export const SQUARED_DIFFERENCE = 'return (a - b) * (a - b);';

export const INT_DIV = `
  float s = sign(a) * sign(b);
  int ia = int(round(a));
  int ib = int(round(b));
  return float(idiv(ia, ib, s));
`;

export const PRELU = `return (a < 0.) ? b * a : a;`;
const CHECK_NAN_SNIPPET = `
  if (isnan(a)) return a;
  if (isnan(b)) return b;
`;
export const MAX = CHECK_NAN_SNIPPET + `
  return max(a, b);
`;
export function getBinaryProgram(
    op: string, aShape: number[], bShape: number[]) {
  const optimizeA = aShape.length === 1 && bShape.length > 1;
  const optimizeB = bShape.length === 1 && aShape.length > 1;
  const useUniformWithA = optimizeA && (aShape[0] <= 4 || aShape[0] === 64);
  const useUniformWithB = optimizeB && (bShape[0] <= 4 || bShape[0] === 64);
  const useSharedMemoryWithA = optimizeA && aShape[0] < 2048;
  const useSharedMemoryWithB = optimizeB && bShape[0] < 2048;
  if (useUniformWithA || useUniformWithB) {
    return new BinaryOpUniformProgram(op, aShape, bShape, useUniformWithB);
  } else if (useSharedMemoryWithA || useSharedMemoryWithB) {
    return new BinaryOpSharedProgram(op, aShape, bShape, useSharedMemoryWithB);
  } else {
    return new BinaryOpProgram(op, aShape, bShape);
  }
}
