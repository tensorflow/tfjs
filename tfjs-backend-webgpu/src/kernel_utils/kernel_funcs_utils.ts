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

import {BinaryInputs, DataType, KernelFunc, TypedArray, UnaryInputs, upcastType} from '@tensorflow/tfjs-core';
import {WebGPUBackend} from '../backend_webgpu';
import {getBinaryProgram} from '../kernels/binary_ops';
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
 * @param cpuKernelImpl Optional. Shared functionality from tfjs-backend-cpu, it
 *     will be involved when necessary.
 * @param dtype Optional. If set, the result has this dtype. Otherwise, the
 *     result has the same dtype as the first input. This is mainly used in
 *     comparison kernels, such as Equal, Less, Greater, etc.
 */
export function unaryKernelFunc(
    {opSnippet, cpuKernelImpl, dtype}: UnaryKernelFuncConfig): KernelFunc {
  return ({inputs, backend}) => {
    const {x} = inputs as UnaryInputs;
    const webgpuBackend = backend as WebGPUBackend;

    const $dtype = dtype || x.dtype;
    if (webgpuBackend.shouldExecuteOnCPU([x]) && cpuKernelImpl != null) {
      const xData = webgpuBackend.tensorMap.get(x.dataId);
      const outValues = cpuKernelImpl(xData.values as TypedArray, $dtype);
      return webgpuBackend.makeTensorInfo(x.shape, $dtype, outValues);
    }

    const program: UnaryOpProgram = new UnaryOpProgram(x.shape, opSnippet);
    return webgpuBackend.runWebGPUProgram(program, [x], $dtype);
  };
}

type BinaryKernelFuncConfig = {
  opSnippet: number,
  cpuKernelImpl?: SimpleBinaryKernelImplCPU,
  dtype?: DataType
};

/**
 * Template that creates a `KernelFunc` for binary ops.
 * @param opSnippet Op snippet to create `BinaryOpProgram`.
 * @param cpuKernelImpl Optional. Shared functionality from tfjs-backend-cpu, it
 *     will be involved when necessary.
 * @param dtype Optional. If set, the result has this dtype. Otherwise, the
 *     result has the same dtype as the first input. This is mainly used in
 *     comparison kernels, such as Equal, Less, Greater, etc.
 */
export function binaryKernelFunc(
    {opSnippet, cpuKernelImpl, dtype}: BinaryKernelFuncConfig): KernelFunc {
  return ({inputs, backend}) => {
    const {a, b} = inputs as BinaryInputs;
    const webgpuBackend = backend as WebGPUBackend;
    const $dtype = dtype || upcastType(a.dtype, b.dtype);
    if (webgpuBackend.shouldExecuteOnCPU([a, b]) && cpuKernelImpl != null) {
      const aData = webgpuBackend.tensorMap.get(a.dataId);
      const bData = webgpuBackend.tensorMap.get(b.dataId);
      const [outValues, outShape] = cpuKernelImpl(
          a.shape, b.shape, aData.values as TypedArray,
          bData.values as TypedArray, $dtype);

      return webgpuBackend.makeTensorInfo(outShape, $dtype, outValues);
    }
    const program = getBinaryProgram(opSnippet, a.shape, b.shape);
    return webgpuBackend.runWebGPUProgram(program, [a, b], $dtype);
  };
}
