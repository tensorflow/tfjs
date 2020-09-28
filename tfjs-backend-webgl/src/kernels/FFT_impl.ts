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

import {TensorInfo, util} from '@tensorflow/tfjs-core';

import {MathBackendWebGL} from '../backend_webgl';
import {FFTProgram} from '../fft_gpu';

import {complex} from './Complex';
import {reshape} from './Reshape';

export function fftImpl(
    x: TensorInfo, inverse: boolean, backend: MathBackendWebGL): TensorInfo {
  const xData = backend.texData.get(x.dataId);

  const inputSize = util.sizeFromShape(x.shape);
  // Collapse all outer dimensions to a single batch dimension.
  const innerDimensionSize = x.shape[x.shape.length - 1];
  const batch = inputSize / innerDimensionSize;

  const input2D = reshape(
      {inputs: {x}, backend, attrs: {shape: [batch, innerDimensionSize]}});

  const xShape = input2D.shape as [number, number];
  const realProgram = new FFTProgram('real', xShape, inverse);
  const imagProgram = new FFTProgram('imag', xShape, inverse);

  const inputs = [
    {
      dataId: xData.complexTensorInfos.real.dataId,
      dtype: xData.complexTensorInfos.real.dtype,
      shape: xShape
    },
    {
      dataId: xData.complexTensorInfos.imag.dataId,
      dtype: xData.complexTensorInfos.imag.dtype,
      shape: xShape
    }
  ];

  const realPart = backend.runWebGLProgram(realProgram, inputs, 'float32');
  const imagPart = backend.runWebGLProgram(imagProgram, inputs, 'float32');

  const complexOutput =
      complex({inputs: {real: realPart, imag: imagPart}, backend});

  backend.disposeIntermediateTensorInfo(realPart);
  backend.disposeIntermediateTensorInfo(imagPart);

  const complexOutputReshaped =
      reshape({inputs: {x: complexOutput}, backend, attrs: {shape: x.shape}});

  backend.disposeIntermediateTensorInfo(complexOutputReshaped);

  return complexOutputReshaped;
}
