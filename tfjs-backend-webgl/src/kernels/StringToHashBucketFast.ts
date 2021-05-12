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

import {KernelConfig, KernelFunc, StringToHashBucketFast, StringToHashBucketFastAttrs, StringToHashBucketFastInputs, TensorInfo} from '@tensorflow/tfjs-core';

import {MathBackendWebGL} from '../backend_webgl';
import {stringToHashBucketFastImplCPU} from '../kernel_utils/shared';

export function stringToHashBucketFast(args: {
  inputs: StringToHashBucketFastInputs,
  backend: MathBackendWebGL,
  attrs: StringToHashBucketFastAttrs
}): TensorInfo {
  const {inputs, backend, attrs} = args;
  const {numBuckets} = attrs;
  const {input} = inputs;

  if (input.dtype !== 'string') {
    throw new Error('Input must be of datatype string');
  }
  if (numBuckets <= 0) {
    throw new Error(`Number of buckets must be at least 1`);
  }

  const $input = backend.readSync(input.dataId) as Uint8Array[];

  const output = stringToHashBucketFastImplCPU($input, numBuckets);
  return backend.makeTensorInfo(input.shape, 'int32', output);
}

export const stringToHashBucketFastConfig: KernelConfig = {
  kernelName: StringToHashBucketFast,
  backendName: 'webgl',
  kernelFunc: stringToHashBucketFast as {} as KernelFunc,
};
