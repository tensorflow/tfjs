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

import {BinaryInputs, DataType, engine, KernelFunc, TypedArray, UnaryInputs, upcastType, util} from '@tensorflow/tfjs-core';
import {WebGPUBackend} from '../backend_webgpu';
import {BinaryOpVec4Program} from '../kernels/binary_op_vec4_webgpu';
import {BinaryOpSharedProgram} from '../kernels/binary_op_shared_webgpu';
import {BinaryOpProgram} from '../kernels/binary_op_webgpu';
import {UnaryOpProgram} from '../kernels/unary_op_webgpu';
import {SimpleBinaryKernelImplCPU, SimpleUnaryKernelImplCPU} from './shared';

type UnaryKernelFuncConfig = {
  opSnippet: string,
  cpuKernelImpl?: SimpleUnaryKernelImplCPU,
  dtype?: DataType
};

/**
 * Template that creates a `KernelFunc` for unary ops.
 * @param opSnippet Op snippet to create `UnaryOpProgram`.
 * @param packedOpSnippet Op snippet to create `UnaryOpPackedProgram`.
 * @param dtype Optional. If set, the result has this dtype. Otherwise, the
 *     result has the same dtype as the first input. This is mainly used in
 *     comparison kernels, such as Equal, Less, Greater, etc.
 */
export function unaryKernelFunc(
    {opSnippet, cpuKernelImpl, dtype}: UnaryKernelFuncConfig):
    KernelFunc {
  return ({inputs, backend}) => {
    const {x} = inputs as UnaryInputs;
    const webgpuBackend = backend as WebGPUBackend;

    const $dtype = dtype || x.dtype;
    if (webgpuBackend.shouldExecuteOnCPU([x]) &&
        cpuKernelImpl != null) {
      const xData = webgpuBackend.tensorMap.get(x.dataId);
      const outValues = cpuKernelImpl(xData.values as TypedArray, $dtype);
      return webgpuBackend.makeTensorInfo(x.shape, $dtype, outValues);
    }

    const program: UnaryOpProgram = new UnaryOpProgram(x.shape, opSnippet);
    return webgpuBackend.runWebGPUProgram(program, [x]);
  };
}

type BinaryKernelFuncConfig = {
  opSnippet: number,
  boolType?: boolean,
  cpuKernelImpl?: SimpleBinaryKernelImplCPU,
  dtype?: DataType
};

/**
 * Template that creates a `KernelFunc` for binary ops.
 * @param opSnippet Op snippet to create `BinaryOpProgram`.
 * @param packedOpSnippet Op snippet to create `BinaryOpPackedProgram`.
 * @param checkOutOfBoundsForPackedProgram Whether to set checkOutOfBounds=true
 *     when creating BinaryOpPackedProgram.
 * @param dtype Optional. If set, the result has this dtype. Otherwise, the
 *     result has the same dtype as the first input. This is mainly used in
 *     comparison kernels, such as Equal, Less, Greater, etc.
 */
export function binaryKernelFunc({
  opSnippet,
  boolType = false,
  cpuKernelImpl,
  dtype
}: BinaryKernelFuncConfig): KernelFunc {
  return ({inputs, backend}) => {
    const {a, b} = inputs as BinaryInputs;
    const webgpuBackend = backend as WebGPUBackend;
    const $dtype = dtype || upcastType(a.dtype, b.dtype);
    if (webgpuBackend.shouldExecuteOnCPU([a, b]) &&
        cpuKernelImpl != null) {
      const aData = webgpuBackend.tensorMap.get(a.dataId);
      const bData = webgpuBackend.tensorMap.get(b.dataId);
      const [outValues, outShape] = cpuKernelImpl(
          a.shape, b.shape, aData.values as TypedArray,
          bData.values as TypedArray, $dtype);

      return webgpuBackend.makeTensorInfo(outShape, $dtype, outValues);
    }
    const program = getBinaryProgram(opSnippet, a.shape, b.shape);
    if (boolType) {
      const dataId = webgpuBackend.write(
          null /*values*/, program.outputShape, 'bool');
      const output = engine().makeTensorFromDataId(
          dataId, program.outputShape, 'bool', webgpuBackend);

      return webgpuBackend.runWebGPUProgram(program, [a, b], output);
    }
    const dataId = webgpuBackend.write(
        null /*values*/, program.outputShape, $dtype);
    const output =
        engine().makeTensorFromDataId(
            dataId, program.outputShape, $dtype, webgpuBackend);

    return webgpuBackend.runWebGPUProgram(program, [a, b], output);
  };
}

export enum BinaryOpType {
  MUL,
  ADD,
  SUB,
  DIV,
  GREATER,
  GREATER_EQUAL,
  LESS,
  LESS_EQUAL,
  SQUARED_DIFFERENCE,
  INT_DIV,
  PRELU,
  MAX
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
      const CHECK_NAN_SNIPPET = useVec4 ? `
        result.r = isNaN.r > 0. ? NAN : result.r;
        result.g = isNaN.g > 0. ? NAN : result.g;
        result.b = isNaN.b > 0. ? NAN : result.b;
        result.a = isNaN.a > 0. ? NAN : result.a;
      ` :
                                          `
        if (isnan(a)) return a;
        if (isnan(b)) return b;
      `;
      return useVec4 ? `
      vec4 result = vec4(max(a, b));
      vec4 isNaN = min(vec4(isnan(a)) + vec4(isnan(b)), vec4(1.0));
      ` + CHECK_NAN_SNIPPET +
              `
      return result;
    ` :
                       CHECK_NAN_SNIPPET + `
      return max(a, b);
    `;
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
