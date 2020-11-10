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

import {Fill, FillAttrs, KernelConfig, scalar, tensor1d} from '@tensorflow/tfjs';

import {NodeJSKernelBackend} from '../nodejs_kernel_backend';

export const fillConfig: KernelConfig = {
  kernelName: Fill,
  backendName: 'tensorflow',
  kernelFunc: (args) => {
    const backend = args.backend as NodeJSKernelBackend;
    const {shape, value} = args.attrs as {} as FillAttrs;
    let {dtype} = args.attrs as {} as FillAttrs;

    // TODO(cais, nkreeger): Investigate whether backend can be made into
    // a dtype helper method. The underlying op kernel doesn't accept undefined
    // or null dtype.
    if (dtype == null) {
      if (typeof value === 'number') {
        dtype = 'float32';
      } else {
        dtype = 'string';
      }
    }
    const shapeTensor = tensor1d(shape, 'int32');
    const valueTensor = scalar(value, dtype);
    const opAttrs = [
      {
        name: 'T',
        type: backend.binding.TF_ATTR_TYPE,
        value: backend.getDTypeInteger(dtype)
      },
      {
        name: 'index_type',
        type: backend.binding.TF_ATTR_TYPE,
        value: backend.binding.TF_INT32
      }
    ];
    const res =
        backend.executeSingleOutput(Fill, opAttrs, [shapeTensor, valueTensor]);
    shapeTensor.dispose();
    valueTensor.dispose();
    return res;
  }
};
