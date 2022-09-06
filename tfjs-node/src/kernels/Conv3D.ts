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

import {backend_util, Conv3D, Conv3DAttrs, Conv3DInputs, KernelConfig} from '@tensorflow/tfjs';

import {createTensorsTypeOpAttr, NodeJSKernelBackend} from '../nodejs_kernel_backend';

export const conv3DConfig: KernelConfig = {
  kernelName: Conv3D,
  backendName: 'tensorflow',
  kernelFunc: (args) => {
    const {x, filter} = args.inputs as Conv3DInputs;
    const backend = args.backend as NodeJSKernelBackend;
    const {strides, pad, dilations} = args.attrs as {} as Conv3DAttrs;

    const convInfo = backend_util.computeConv3DInfo(
        x.shape as [number, number, number, number, number],
        filter.shape as [number, number, number, number, number], strides,
        dilations, pad);

    const $strides = [
      1, convInfo.strideDepth, convInfo.strideHeight, convInfo.strideWidth, 1
    ];
    const padding = convInfo.padInfo.type;
    const $dataFormat =
        convInfo.dataFormat === 'channelsLast' ? 'NDHWC' : 'NCDHW';

    if (!backend.isGPUPackage && convInfo.dilationDepth > 1) {
      throw new Error('CPU Dilation depth must be 1');
    }
    const $dilations = [
      1, convInfo.dilationDepth, convInfo.dilationHeight,
      convInfo.dilationWidth, 1
    ];

    const opAttrs = [
      createTensorsTypeOpAttr('T', x.dtype),
      {name: 'strides', type: backend.binding.TF_ATTR_INT, value: $strides},
      {name: 'padding', type: backend.binding.TF_ATTR_STRING, value: padding}, {
        name: 'data_format',
        type: backend.binding.TF_ATTR_STRING,
        value: $dataFormat
      },
      {name: 'dilations', type: backend.binding.TF_ATTR_INT, value: $dilations}
    ];
    return backend.executeSingleOutput(Conv3D, opAttrs, [x, filter]);
  }
};
