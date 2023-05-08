/**
 * @license
 * Copyright 2022 Google LLC.
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

import {WebGPUBackend} from '../backend_webgpu';
import {FFTProgram} from '../fft_webgpu';

import {complex} from './Complex';
import {reshape} from './Reshape';

export function fftImpl(
    x: TensorInfo, inverse: boolean, backend: WebGPUBackend): TensorInfo {
  const xData = backend.tensorMap.get(x.dataId);

  const inputSize = util.sizeFromShape(x.shape);
  // Collapse all outer dimensions to a single batch dimension.
  const innerDimensionSize = x.shape[x.shape.length - 1];
  const batch = inputSize / innerDimensionSize;

  const toDispose = [];
  const input2D = reshape(
      {inputs: {x}, backend, attrs: {shape: [batch, innerDimensionSize]}});
  toDispose.push(input2D);

  const xShape = input2D.shape as [number, number];
  const realProgram = new FFTProgram('real', xShape);
  const imagProgram = new FFTProgram('imag', xShape);

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

  const exponentMultiplier = inverse ? 2.0 * Math.PI : -2.0 * Math.PI;
  const denominator = inverse ? xShape[1] : 1.0;
  const uniformData = [
    {type: 'float32', data: [exponentMultiplier]},
    {type: 'float32', data: [denominator]}
  ];

  const realPart =
      backend.runWebGPUProgram(realProgram, inputs, 'float32', uniformData);
  toDispose.push(realPart);
  const imagPart =
      backend.runWebGPUProgram(imagProgram, inputs, 'float32', uniformData);
  toDispose.push(imagPart);

  const complexOutput =
      complex({inputs: {real: realPart, imag: imagPart}, backend});
  toDispose.push(complexOutput);

  const complexOutputReshaped =
      reshape({inputs: {x: complexOutput}, backend, attrs: {shape: x.shape}});

  toDispose.forEach(t => backend.disposeData(t.dataId));

  return complexOutputReshaped;
}
