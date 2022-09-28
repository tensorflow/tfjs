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

import {backend_util, Conv2D, Conv2DAttrs, Conv2DInputs, KernelConfig, KernelFunc, TensorInfo} from '@tensorflow/tfjs-core';

import {MathBackendWebGL} from '../backend_webgl';
import {Conv2DDensePackedProgram} from '../conv_gpu_dense';

import {reshape} from './Reshape';

export function conv2d(
    args:
        {inputs: Conv2DInputs, attrs: Conv2DAttrs, backend: MathBackendWebGL}) {
  const {inputs, backend, attrs} = args;
  const {x, filter} = inputs;
  const {strides, pad, dataFormat, dilations, dimRoundingMode} = attrs;
  let out: TensorInfo;

  const $dataFormat = backend_util.convertConv2DDataFormat(dataFormat);
  const convInfo = backend_util.computeConv2DInfo(
      x.shape as [number, number, number, number],
      filter.shape as [number, number, number, number], strides, dilations, pad,
      dimRoundingMode, false /* depthwise */, $dataFormat);
  const conv2dProgram = new Conv2DDensePackedProgram(convInfo);
  out = backend.runWebGLProgram(conv2dProgram, [x, filter], 'float32');

  // This reshaping is just to align with the original implementation of conv2d.
  const outReshaped =
      reshape({inputs: {x: out}, backend, attrs: {shape: convInfo.outShape}});
  backend.disposeIntermediateTensorInfo(out);

  return outReshaped;
}

export const conv2DConfig: KernelConfig = {
  kernelName: Conv2D,
  backendName: 'webgl',
  kernelFunc: conv2d as {} as KernelFunc,
};
