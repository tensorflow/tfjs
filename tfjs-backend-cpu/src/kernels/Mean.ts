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

import {backend_util, KernelConfig, KernelFunc, Mean, MeanAttrs, MeanInputs, TensorInfo, util} from '@tensorflow/tfjs-core';

import {MathBackendCPU} from '../backend_cpu';
import {cast} from './Cast';
import {div} from './RealDiv';
import {sum} from './Sum';

export function mean(
    args: {inputs: MeanInputs, backend: MathBackendCPU, attrs: MeanAttrs}):
    TensorInfo {
  const {inputs, backend, attrs} = args;
  const {x} = inputs;
  const {axis, keepDims} = attrs;

  const axes = util.parseAxisParam(axis, x.shape);
  const shapes = backend_util.computeOutAndReduceShapes(x.shape, axes);
  const reduceShape = shapes[1];
  const reduceSize = util.sizeFromShape(reduceShape);
  const toDispose = [];
  const reduceSizeScalar =
      backend.makeTensorInfo([], 'float32', new Float32Array([reduceSize]));
  toDispose.push(reduceSizeScalar);

  const $x = cast({inputs: {x}, backend, attrs: {dtype: 'float32'}});
  toDispose.push($x);

  const res =
      div({inputs: {a: $x, b: reduceSizeScalar}, backend}) as TensorInfo;
  toDispose.push(res);

  const result = sum({inputs: {x: res}, backend, attrs: {axis, keepDims}});

  toDispose.forEach(t => backend.disposeIntermediateTensorInfo(t));

  return result;
}

export const meanConfig: KernelConfig = {
  kernelName: Mean,
  backendName: 'cpu',
  kernelFunc: mean as {} as KernelFunc
};
