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

import {KernelConfig, scalar, TopK, TopKAttrs, TopKInputs} from '@tensorflow/tfjs';
import {isNullOrUndefined} from 'util';

import {createTensorsTypeOpAttr, NodeJSKernelBackend} from '../nodejs_kernel_backend';

export const topKConfig: KernelConfig = {
  kernelName: TopK,
  backendName: 'tensorflow',
  kernelFunc: (args) => {
    const {x} = args.inputs as TopKInputs;
    const backend = args.backend as NodeJSKernelBackend;
    const {k, sorted} = args.attrs as unknown as TopKAttrs;

    const kCount = isNullOrUndefined(k) ? 1 : k;
    const isSorted = isNullOrUndefined(sorted) ? true : sorted;
    const opAttrs = [
      {name: 'sorted', type: backend.binding.TF_ATTR_BOOL, value: isSorted},
      createTensorsTypeOpAttr('T', x.dtype),
    ];
    const kTensor = scalar(kCount, 'int32');

    // 'TopKV2' has two-hard coded output attributes:
    // TODO(yassogba) consider renamine constant in kernel names;
    const res =
        backend.executeMultipleOutputs('TopKV2', opAttrs, [x, kTensor], 2);
    kTensor.dispose();
    return res;
  }
};
