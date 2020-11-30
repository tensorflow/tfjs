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

import {KernelConfig, KernelFunc, Pack, PackAttrs, PackInputs, TensorInfo, util} from '@tensorflow/tfjs-core';
import {BackendWasm} from '../backend_wasm';

import {concat} from './Concat';
import {expandDims} from './ExpandDims';

export function pack(
    args: {inputs: PackInputs, backend: BackendWasm, attrs: PackAttrs}):
    TensorInfo {
  const {inputs, backend, attrs} = args;
  const {axis} = attrs;

  if (inputs.length === 1) {
    return expandDims(
        {inputs: {input: inputs[0]}, backend, attrs: {dim: axis}});
  }

  const shape = inputs[0].shape;
  const dtype = inputs[0].dtype;

  inputs.forEach(t => {
    util.assertShapesMatch(
        shape, t.shape,
        'All tensors passed to stack must have matching shapes');
    util.assert(
        dtype === t.dtype,
        () => 'All tensors passed to stack must have matching dtypes');
  });

  const expandedTensors = inputs.map(
      t => expandDims({inputs: {input: t}, backend, attrs: {dim: axis}}));

  return concat({inputs: expandedTensors, backend, attrs: {axis}});
}

export const packConfig: KernelConfig = {
  kernelName: Pack,
  backendName: 'wasm',
  kernelFunc: pack as {} as KernelFunc
};
