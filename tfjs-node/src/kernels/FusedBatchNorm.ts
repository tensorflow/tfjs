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

import {add, fill, FusedBatchNorm, FusedBatchNormAttrs, FusedBatchNormInputs, KernelConfig, mul, Rank, rsqrt, scalar, sub, Tensor, Tensor4D, tidy} from '@tensorflow/tfjs';

import {createTensorsTypeOpAttr, NodeJSKernelBackend} from '../nodejs_kernel_backend';

export const fusedBatchNormConfig: KernelConfig = {
  kernelName: FusedBatchNorm,
  backendName: 'tensorflow',
  kernelFunc: (args) => {
    const {x, mean, variance} = args.inputs as FusedBatchNormInputs;
    let {scale, offset} = args.inputs as FusedBatchNormInputs;
    const backend = args.backend as NodeJSKernelBackend;
    const {varianceEpsilon} = args.attrs as {} as FusedBatchNormAttrs;

    return tidy(() => {
      if ((mean as Tensor).rank > 1) {
        // Fused batch norm doesn't work with high-dim mean/var/scale/offset.
        let inv = rsqrt(add(variance as Tensor, scalar(varianceEpsilon)));
        if (scale != null) {
          inv = mul(inv, scale as Tensor);
        }
        const xNorm: Tensor4D = mul(sub(x as Tensor, mean as Tensor), inv);
        return offset != null ? add(xNorm, offset as Tensor) : xNorm;
      }
      const dataFormat = 'NHWC';
      const depth = x.shape[3];
      const opAttrs = [
        createTensorsTypeOpAttr('T', x.dtype),
        {
          name: 'epsilon',
          type: backend.binding.TF_ATTR_FLOAT,
          value: varianceEpsilon
        },
        {
          name: 'data_format',
          type: backend.binding.TF_ATTR_STRING,
          value: dataFormat
        },
        {name: 'is_training', type: backend.binding.TF_ATTR_BOOL, value: false},
      ];
      const numOutputs = 5;

      if (scale == null) {
        scale = fill<Rank.R1>([depth], 1);
      }
      if (offset == null) {
        offset = fill<Rank.R1>([depth], 0);
      }
      return backend.executeMultipleOutputs(
          FusedBatchNorm, opAttrs, [x, scale, offset, mean, variance],
          numOutputs)[0];
    });
  }
};
