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

import {backend_util, KernelConfig, KernelFunc, Softmax, SoftmaxAttrs, SoftmaxInputs, TensorInfo, util} from '@tensorflow/tfjs-core';

import {WebGPUBackend} from '../backend_webgpu';
import {SoftmaxProgram} from '../softmax_webgpu';

import {max} from './Max';
import {realDiv} from './RealDiv';
import {reshape} from './Reshape';
import {sum} from './Sum';

export function softmax(
    args: {inputs: SoftmaxInputs, backend: WebGPUBackend, attrs: SoftmaxAttrs}):
    TensorInfo {
  const {inputs, backend, attrs} = args;
  const {logits} = inputs;
  const {dim} = attrs;

  const axes = util.parseAxisParam([dim], logits.shape);

  const maxLogit = max({
    inputs: {x: logits},
    backend,
    attrs: {reductionIndices: axes, keepDims: false}
  });

  const expandedShape = backend_util.expandShapeToKeepDim(maxLogit.shape, axes);

  const maxLogitsReshaped =
      reshape({inputs: {x: maxLogit}, backend, attrs: {shape: expandedShape}});
  const program = new SoftmaxProgram(logits.shape);
  const b = backend.runWebGPUProgram(
      program, [logits, maxLogitsReshaped], logits.dtype);
  const sumExp =
      sum({inputs: {x: b}, backend, attrs: {axis: axes, keepDims: false}});
  const sumExpReshaped =
      reshape({inputs: {x: sumExp}, backend, attrs: {shape: expandedShape}});
  const res =
      realDiv({inputs: {a: b, b: sumExpReshaped}, backend}) as TensorInfo;

  backend.disposeData(maxLogit.dataId);
  backend.disposeData(maxLogitsReshaped.dataId);
  backend.disposeData(b.dataId);
  backend.disposeData(sumExp.dataId);
  backend.disposeData(sumExpReshaped.dataId);

  return res;
}

export const softmaxConfig: KernelConfig = {
  kernelName: Softmax,
  backendName: 'webgpu',
  kernelFunc: softmax as unknown as KernelFunc
};
