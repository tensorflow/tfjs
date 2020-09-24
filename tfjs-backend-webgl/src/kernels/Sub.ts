
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

import {BinaryInputs, env, KernelConfig, Sub, TensorInfo, TypedArray, upcastType} from '@tensorflow/tfjs-core';

import {MathBackendWebGL} from '../backend_webgl';
import {BinaryOpProgram} from '../binaryop_gpu';
import {BinaryOpPackedProgram} from '../binaryop_packed_gpu';
import {subImplCPU as cpuSub} from '../kernel_utils/shared';

import {complex} from './Complex';

const SUB = 'return a - b;';

export function sub(args: {inputs: BinaryInputs, backend: MathBackendWebGL}):
    TensorInfo {
  const {inputs, backend} = args;
  const {a, b} = inputs;

  if (a.dtype === 'complex64') {
    const aData = backend.texData.get(a.dataId);
    const bData = backend.texData.get(b.dataId);

    const [real, imag] = [
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

      const program = new BinaryOpProgram(SUB, a.shape, b.shape);
      return backend.runWebGLProgram(
          program, [aHandle, bHandle], upcastType(aPart.dtype, bPart.dtype));
    });

    const complexOutput = complex({inputs: {real, imag}, backend});

    backend.disposeIntermediateTensorInfo(real);
    backend.disposeIntermediateTensorInfo(imag);

    // TODO(annxingyuan): CPU forwarding for complex inputs.

    return complexOutput;
  }

  if (backend.shouldExecuteOnCPU([a, b])) {
    const aData = backend.texData.get(a.dataId);
    const bData = backend.texData.get(b.dataId);
    const [outValues, outShape] = cpuSub(
        a.shape, b.shape, aData.values as TypedArray,
        bData.values as TypedArray, 'float32');

    const out = backend.makeTensorInfo(outShape, 'float32');
    const outData = backend.texData.get(out.dataId);
    outData.values = outValues;
    return out;
  }

  const dtype = upcastType(a.dtype, b.dtype);
  let program: BinaryOpProgram|BinaryOpPackedProgram;
  if (env().getBool('WEBGL_PACK_BINARY_OPERATIONS')) {
    program = new BinaryOpPackedProgram(SUB, a.shape, b.shape);
  } else {
    program = new BinaryOpProgram(SUB, a.shape, b.shape);
  }

  return backend.runWebGLProgram(program, [a, b], dtype);
}

export const subConfig: KernelConfig = {
  kernelName: Sub,
  backendName: 'webgl',
  kernelFunc: sub
};
