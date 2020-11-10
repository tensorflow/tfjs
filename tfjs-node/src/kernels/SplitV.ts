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

import {backend_util, KernelConfig, scalar, SplitV, SplitVAttrs, SplitVInputs, Tensor, tensor1d, tidy, util} from '@tensorflow/tfjs';

import {createTensorsTypeOpAttr, NodeJSKernelBackend} from '../nodejs_kernel_backend';

export const splitVConfig: KernelConfig = {
  kernelName: SplitV,
  backendName: 'tensorflow',
  kernelFunc: (args) => {
    const {x} = args.inputs as SplitVInputs;
    const backend = args.backend as NodeJSKernelBackend;
    const {numOrSizeSplits, axis} = args.attrs as {} as SplitVAttrs;

    const $axis = util.parseAxisParam(axis, x.shape)[0];
    const splitSizes = backend_util.prepareSplitSize(x, numOrSizeSplits, $axis);

    const opAttrs = [
      {
        name: 'num_split',
        type: backend.binding.TF_ATTR_INT,
        value: splitSizes.length
      },
      createTensorsTypeOpAttr('T', x as Tensor), {
        name: 'Tlen',
        type: backend.binding.TF_ATTR_TYPE,
        value: backend.binding.TF_INT32
      }
    ];
    const inputs = [x];
    return tidy(() => {
      inputs.push(tensor1d(splitSizes, 'int32'));
      inputs.push(scalar($axis, 'int32'));
      return backend.executeMultipleOutputs(
          SplitV, opAttrs, inputs, splitSizes.length);
    });
  }
};
