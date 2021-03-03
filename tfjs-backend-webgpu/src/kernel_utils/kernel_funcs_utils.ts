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

import {BinaryInputs, DataType, KernelFunc, TensorInfo, TypedArray, UnaryInputs, upcastType} from '@tensorflow/tfjs-core';

import {WebGPUBackend} from '../backend_webgpu';
import {BinaryOpComplexProgram, COMPLEX_MULTIPLY} from '../kernels/binary_op_complex_webgpu';
import {BinaryOpType, getBinaryProgram} from '../kernels/binary_ops';
import {complex} from '../kernels/Complex';
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
  supportsComplex?: boolean,
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
    {opSnippet, cpuKernelImpl, supportsComplex = false, dtype}:
        BinaryKernelFuncConfig): KernelFunc {
  return ({inputs, backend}) => {
    const {a, b} = inputs as BinaryInputs;
    const webgpuBackend = backend as WebGPUBackend;

    if (supportsComplex && a.dtype === 'complex64') {
      const aData = webgpuBackend.tensorMap.get(a.dataId);
      const bData = webgpuBackend.tensorMap.get(b.dataId);
      let real: TensorInfo, imag: TensorInfo;
      if (opSnippet !== BinaryOpType.MUL) {
        [real, imag] = [
          [aData.complexTensorInfos.real, bData.complexTensorInfos.real],
          [aData.complexTensorInfos.imag, bData.complexTensorInfos.imag]
        ].map(complexParts => {
          const [aPart, bPart] = complexParts;

          const aHandle = {
            dataId: aPart.dataId,
            dtype: aPart.dtype,
            shape: a.shape
          };
          const bHandle = {
            dataId: bPart.dataId,
            dtype: bPart.dtype,
            shape: b.shape
          };

          const program = getBinaryProgram(opSnippet, a.shape, b.shape);
          return webgpuBackend.runWebGPUProgram(
              program, [aHandle, bHandle],
              upcastType(aPart.dtype, bPart.dtype));
        });
      } else {
        const realProgram =
            new BinaryOpComplexProgram(COMPLEX_MULTIPLY.REAL, a.shape, b.shape);
        const imagProgram =
            new BinaryOpComplexProgram(COMPLEX_MULTIPLY.IMAG, a.shape, b.shape);

        const inputs = [
          {
            dataId: aData.complexTensorInfos.real.dataId,
            dtype: aData.complexTensorInfos.real.dtype,
            shape: a.shape
          },
          {
            dataId: aData.complexTensorInfos.imag.dataId,
            dtype: aData.complexTensorInfos.imag.dtype,
            shape: a.shape
          },
          {
            dataId: bData.complexTensorInfos.real.dataId,
            dtype: bData.complexTensorInfos.real.dtype,
            shape: b.shape
          },
          {
            dataId: bData.complexTensorInfos.imag.dataId,
            dtype: bData.complexTensorInfos.imag.dtype,
            shape: b.shape
          }
        ];

        real = webgpuBackend.runWebGPUProgram(realProgram, inputs, 'float32');
        imag = webgpuBackend.runWebGPUProgram(imagProgram, inputs, 'float32');
      }

      const complexOutput =
          complex({inputs: {real, imag}, backend: webgpuBackend});

      webgpuBackend.disposeData(real.dataId);
      webgpuBackend.disposeData(imag.dataId);

      // TODO: Implement CPU forwarding for complex inputs.

      return complexOutput;
    }

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
