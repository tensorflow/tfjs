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

import {backend_util, KernelConfig, Slice, SliceAttrs, SliceInputs, tensor1d} from '@tensorflow/tfjs';

import {createTensorsTypeOpAttr, NodeJSKernelBackend} from '../nodejs_kernel_backend';

export const sliceConfig: KernelConfig = {
  kernelName: Slice,
  backendName: 'tensorflow',
  kernelFunc: (args) => {
    const {x} = args.inputs as SliceInputs;
    const backend = args.backend as NodeJSKernelBackend;
    const {begin, size} = args.attrs as unknown as SliceAttrs;

    const opAttrs = [
      createTensorsTypeOpAttr('T', x.dtype),
      createTensorsTypeOpAttr('Index', 'int32')
    ];

    // Bind tensor values
    const [begin_, size_] =
        backend_util.slice_util.parseSliceParams(x, begin, size);
    const beginTensor = tensor1d(begin_, 'int32');
    const sizeTensor = tensor1d(size_, 'int32');

    const res = backend.executeSingleOutput(
        Slice, opAttrs, [x, beginTensor, sizeTensor]);
    beginTensor.dispose();
    sizeTensor.dispose();

    return res;
  }
};
