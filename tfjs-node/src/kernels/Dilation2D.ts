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

import {backend_util, Dilation2D, Dilation2DAttrs, Dilation2DInputs, KernelConfig} from '@tensorflow/tfjs';

import {createTensorsTypeOpAttr, NodeJSKernelBackend} from '../nodejs_kernel_backend';

export const dilation2dConfig: KernelConfig = {
  kernelName: Dilation2D,
  backendName: 'tensorflow',
  kernelFunc: ({inputs, backend, attrs}) => {
    const {x, filter} = inputs as Dilation2DInputs;
    const {strides, pad, dilations} = attrs as {} as Dilation2DAttrs;
    const {dilationHeight, dilationWidth, padInfo, strideHeight, strideWidth} =
        backend_util.computeDilation2DInfo(
            x.shape as [number, number, number, number],
            filter.shape as [number, number, number], strides, pad,
            'NHWC' /* dataFormat */, dilations);
    const $strides = [1, strideHeight, strideWidth, 1];
    const $dilations = [1, dilationHeight, dilationWidth, 1];

    const nodeBackend = backend as NodeJSKernelBackend;

    const opAttrs = [
      createTensorsTypeOpAttr('T', x.dtype),
      {name: 'strides', type: nodeBackend.binding.TF_ATTR_INT, value: $strides},
      {name: 'rates', type: nodeBackend.binding.TF_ATTR_INT, value: $dilations},
      {
        name: 'padding',
        type: nodeBackend.binding.TF_ATTR_STRING,
        value: padInfo.type
      }
    ];

    return nodeBackend.executeSingleOutput(Dilation2D, opAttrs, [x, filter]);
  }
};
