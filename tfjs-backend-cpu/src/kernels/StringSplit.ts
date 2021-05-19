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

import {KernelConfig, KernelFunc, StringSplit, StringSplitAttrs, StringSplitInputs, TensorInfo} from '@tensorflow/tfjs-core';

import {MathBackendCPU} from '../backend_cpu';

import {stringSplitImpl} from './StringSplit_impl';

export function stringSplit(args: {
  inputs: StringSplitInputs,
  backend: MathBackendCPU,
  attrs: StringSplitAttrs
}): [TensorInfo, TensorInfo, TensorInfo] {
  const {inputs, backend, attrs} = args;
  const {skipEmpty} = attrs;
  const {input, delimiter} = inputs;

  if (input.dtype !== 'string') {
    throw new Error('Input must be of datatype string');
  }
  if (input.shape.length !== 1) {
    throw new Error(`Input must be a vector, got shape: ${input.shape}`);
  }
  if (delimiter.shape.length !== 0) {
    throw new Error(
        `Delimiter must be a scalar, got shape: ${delimiter.shape}`);
  }

  const $input = backend.data.get(input.dataId).values as Uint8Array[];
  const $delimiter = backend.data.get(delimiter.dataId).values[0] as Uint8Array;

  const [indices, values, shape] =
      stringSplitImpl($input, $delimiter, skipEmpty);
  const outputSize = values.length;
  return [
    backend.makeTensorInfo([outputSize, 2], 'int32', indices),
    backend.makeTensorInfo([outputSize], 'string', values),
    backend.makeTensorInfo([2], 'int32', new Int32Array(shape))
  ];
}

export const stringSplitConfig: KernelConfig = {
  kernelName: StringSplit,
  backendName: 'cpu',
  kernelFunc: stringSplit as {} as KernelFunc,
};
