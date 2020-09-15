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

import {env, Tensor} from '@tensorflow/tfjs-core';
import {TensorInfo} from '@tensorflow/tfjs-core';

import {MathBackendWebGL} from '../backend_webgl';
import * as binaryop_complex_gpu from '../binaryop_complex_gpu';
import {BinaryOpComplexProgram} from '../binaryop_complex_gpu';
import * as binaryop_gpu from '../binaryop_gpu';
import {BinaryOpProgram} from '../binaryop_gpu';
import {BinaryOpPackedProgram} from '../binaryop_packed_gpu';

export function multiplyImpl(
    a: TensorInfo, b: TensorInfo, backend: MathBackendWebGL): TensorInfo {
  if (a.dtype === 'complex64') {
    const aData = backend.texData.get(a.dataId);
    const bData = backend.texData.get(b.dataId);

    const realProgram = new BinaryOpComplexProgram(
        binaryop_complex_gpu.COMPLEX_MULTIPLY.REAL, a.shape, b.shape);
    const imagProgram = new BinaryOpComplexProgram(
        binaryop_complex_gpu.COMPLEX_MULTIPLY.IMAG, a.shape, b.shape);

    const inputs = [
      backend.makeComplexComponentTensorInfo(a, aData.complexTensors.real),
      backend.makeComplexComponentTensorInfo(a, aData.complexTensors.imag),
      backend.makeComplexComponentTensorInfo(b, bData.complexTensors.real),
      backend.makeComplexComponentTensorInfo(b, bData.complexTensors.imag)
    ];
    const real = backend.compileAndRun<Tensor>(realProgram, inputs);
    const imag = backend.compileAndRun<Tensor>(imagProgram, inputs);

    const complex = backend.complex(real, imag);
    real.dispose();
    imag.dispose();
    return complex;
  }

  if (backend.shouldExecuteOnCPU([a, b]) && a instanceof Tensor &&
      b instanceof Tensor) {
    return backend.getCPUBackend().multiply(a, b);
  }
  const program = env().getBool('WEBGL_PACK_BINARY_OPERATIONS') ?
      new BinaryOpPackedProgram(binaryop_gpu.MUL, a.shape, b.shape) :
      new BinaryOpProgram(binaryop_gpu.MUL, a.shape, b.shape);
  return backend.runWebGLProgram(program, [a, b], a.dtype);
}
