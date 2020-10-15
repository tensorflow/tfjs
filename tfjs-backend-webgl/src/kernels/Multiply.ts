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

import {backend_util, BinaryInputs, env, KernelConfig, Multiply, TensorInfo, TypedArray} from '@tensorflow/tfjs-core';

import {MathBackendWebGL} from '../backend_webgl';
import * as binaryop_complex_gpu from '../binaryop_complex_gpu';
import {BinaryOpComplexProgram} from '../binaryop_complex_gpu';
import {BinaryOpProgram} from '../binaryop_gpu';
import {BinaryOpPackedProgram} from '../binaryop_packed_gpu';
import {multiplyImplCPU as cpuMultiply} from '../kernel_utils/shared';

import {complex} from './Complex';

const MUL = 'return a * b;';

export function multiply(
    args: {inputs: BinaryInputs, backend: MathBackendWebGL}): TensorInfo {
  const {inputs, backend} = args;
  const {a, b} = inputs;
  const dtype = backend_util.upcastType(a.dtype, b.dtype);

  if (a.dtype === 'complex64') {
    const aData = backend.texData.get(a.dataId);
    const bData = backend.texData.get(b.dataId);

    const realProgram = new BinaryOpComplexProgram(
        binaryop_complex_gpu.COMPLEX_MULTIPLY.REAL, a.shape, b.shape);
    const imagProgram = new BinaryOpComplexProgram(
        binaryop_complex_gpu.COMPLEX_MULTIPLY.IMAG, a.shape, b.shape);

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

    const realPart = backend.runWebGLProgram(realProgram, inputs, 'float32');
    const imagPart = backend.runWebGLProgram(imagProgram, inputs, 'float32');

    const complexOutput =
        complex({inputs: {real: realPart, imag: imagPart}, backend});

    backend.disposeIntermediateTensorInfo(realPart);
    backend.disposeIntermediateTensorInfo(imagPart);

    // TODO(annxingyuan): CPU forwarding for complex inputs.
    return complexOutput;
  }

  if (backend.shouldExecuteOnCPU([a, b])) {
    const aData = backend.texData.get(a.dataId);
    const bData = backend.texData.get(b.dataId);
    const [outValues, outShape] = cpuMultiply(
        a.shape, b.shape, aData.values as TypedArray,
        bData.values as TypedArray, dtype);

    const out = backend.makeTensorInfo(outShape, dtype);
    const outData = backend.texData.get(out.dataId);
    outData.values = outValues;
    return out;
  }

  let program: BinaryOpProgram|BinaryOpPackedProgram;
  if (env().getBool('WEBGL_PACK_BINARY_OPERATIONS')) {
    program = new BinaryOpPackedProgram(MUL, a.shape, b.shape);
  } else {
    program = new BinaryOpProgram(MUL, a.shape, b.shape);
  }

  return backend.runWebGLProgram(program, [a, b], dtype);
}

export const multiplyConfig: KernelConfig = {
  kernelName: Multiply,
  backendName: 'webgl',
  kernelFunc: multiply
};
