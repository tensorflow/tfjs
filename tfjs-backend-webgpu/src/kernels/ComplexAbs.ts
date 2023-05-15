/**
 * @license
 * Copyright 2023 Google LLC.
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

import {ComplexAbs, ComplexAbsInputs, KernelConfig, KernelFunc, TensorInfo} from '@tensorflow/tfjs-core';

import {WebGPUBackend} from '../backend_webgpu';
import {ComplexAbsProgram} from '../complex_abs_webgpu';

// Returns a TensorInfo with the complex shape and the dataId of the
// underlying part. We need to do this because a reshaped complex tensor is
// not reflected in its parts.
function makeComplexComponentTensorInfo(
    complexTensor: TensorInfo, complexPart: TensorInfo): TensorInfo {
  return {
    dataId: complexPart.dataId,
    dtype: complexPart.dtype,
    shape: complexTensor.shape
  };
}

export function complexAbs(
    args: {inputs: ComplexAbsInputs, backend: WebGPUBackend}): TensorInfo {
  const {inputs, backend} = args;
  const {x} = inputs;

  const xData = backend.tensorMap.get(x.dataId);

  const program = new ComplexAbsProgram(x.shape);
  const programInputs = [
    makeComplexComponentTensorInfo(x, xData.complexTensorInfos.real),
    makeComplexComponentTensorInfo(x, xData.complexTensorInfos.imag),
  ];

  return backend.runWebGPUProgram(
      program, programInputs, programInputs[0].dtype);
}

export const complexAbsConfig: KernelConfig = {
  kernelName: ComplexAbs,
  backendName: 'webgpu',
  kernelFunc: complexAbs as unknown as KernelFunc
};
