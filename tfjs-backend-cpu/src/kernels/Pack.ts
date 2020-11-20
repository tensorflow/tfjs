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

import {expandDims, KernelConfig, KernelFunc, Pack, PackAttrs, PackInputs, TensorInfo, util} from '@tensorflow/tfjs-core';

import {MathBackendCPU} from '../backend_cpu';

export function pack(
    args: {inputs: PackInputs, backend: MathBackendCPU, attrs: PackAttrs}):
    TensorInfo {
  const {inputs, backend, attrs} = args;
  const {axis} = attrs;

  if (inputs.length === 1) {
    return expandDims($tensors[0], axis);
    return expandDims({})
  }

  const rank = $tensors[0].rank;
  const shape = $tensors[0].shape;
  const dtype = $tensors[0].dtype;

  util.assert(axis <= rank, () => 'Axis must be <= rank of the tensor');

  $tensors.forEach(t => {
    util.assertShapesMatch(
        shape, t.shape,
        'All tensors passed to stack must have matching shapes');
    util.assert(
        dtype === t.dtype,
        () => 'All tensors passed to stack must have matching dtypes');
  });

  const expandedTensors = $tensors.map(t => expandDims(t, axis));
  // Stack exists in the TensorFlow C++ API
  // (https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/stack) but not
  // in
  // https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/core/ops/ops.pbtxt.
  // Therefore we are treating it like a high-level op rather than
  // creating a dedicated stack kernel.
  return concat(expandedTensors, axis);
}

export const packConfig: KernelConfig = {
  kernelName: Pack,
  backendName: 'cpu',
  kernelFunc: pack as {} as KernelFunc
};
