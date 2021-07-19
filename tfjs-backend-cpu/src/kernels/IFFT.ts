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

import {IFFT, IFFTInputs, KernelConfig, KernelFunc, TensorInfo, util} from '@tensorflow/tfjs-core';

import {MathBackendCPU} from '../backend_cpu';
import {fftBatch} from '../utils/fft_utils';
import {reshape} from './Reshape';

export function ifft(args: {inputs: IFFTInputs, backend: MathBackendCPU}):
    TensorInfo {
  const {inputs, backend} = args;
  const {input} = inputs;

  const inputSize = util.sizeFromShape(input.shape);

  // Collapse all outer dimensions to a single batch dimension.
  const innerDimensionSize = input.shape[input.shape.length - 1];
  const batch = inputSize / innerDimensionSize;

  const input2D = reshape({
    inputs: {x: input},
    backend,
    attrs: {shape: [batch, innerDimensionSize]}
  });

  const result = fftBatch(input2D, true, backend);

  const resultReshaped =
      reshape({inputs: {x: result}, backend, attrs: {shape: input.shape}});

  backend.disposeIntermediateTensorInfo(input2D);
  backend.disposeIntermediateTensorInfo(result);

  return resultReshaped;
}

export const ifftConfig: KernelConfig = {
  kernelName: IFFT,
  backendName: 'cpu',
  kernelFunc: ifft as {} as KernelFunc
};
