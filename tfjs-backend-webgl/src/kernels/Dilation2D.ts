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

import {backend_util, Dilation2D, Dilation2DAttrs, Dilation2DInputs, KernelConfig, KernelFunc, TensorInfo} from '@tensorflow/tfjs-core';

import {MathBackendWebGL} from '../backend_webgl';
import {Dilation2DProgram} from '../dilation_gpu';
import {reshape} from './Reshape';

export function dilation2D(args: {
  inputs: Dilation2DInputs,
  attrs: Dilation2DAttrs,
  backend: MathBackendWebGL
}) {
  const {inputs, backend, attrs} = args;
  const {x, filter} = inputs;
  const {strides, pad, dilations} = attrs;

  const convInfo = backend_util.computeDilation2DInfo(
      x.shape as [number, number, number, number],
      filter.shape as [number, number, number], strides, pad,
      'NHWC' /* dataFormat */, dilations);
  let out: TensorInfo;

  const program = new Dilation2DProgram(convInfo);
  out = backend.runWebGLProgram(program, [x, filter], 'float32');

  const outReshaped =
      reshape({inputs: {x: out}, backend, attrs: {shape: convInfo.outShape}});
  backend.disposeIntermediateTensorInfo(out);

  return outReshaped;
}

export const dilation2DConfig: KernelConfig = {
  kernelName: Dilation2D,
  backendName: 'webgl',
  kernelFunc: dilation2D as {} as KernelFunc,
};
