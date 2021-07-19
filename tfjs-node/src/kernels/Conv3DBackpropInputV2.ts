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

import {backend_util, Conv3DBackpropInputV2, Conv3DBackpropInputV2Attrs, Conv3DBackpropInputV2Inputs, KernelConfig, tensor1d} from '@tensorflow/tfjs';

import {createTensorsTypeOpAttr, NodeJSKernelBackend} from '../nodejs_kernel_backend';

export const conv3DBackpropInputV2Config: KernelConfig = {
  kernelName: Conv3DBackpropInputV2,
  backendName: 'tensorflow',
  kernelFunc: (args) => {
    const {dy, filter} = args.inputs as Conv3DBackpropInputV2Inputs;
    const backend = args.backend as NodeJSKernelBackend;
    const {strides, pad, inputShape} =
        args.attrs as {} as Conv3DBackpropInputV2Attrs;

    const dilations = 1;

    const convInfo = backend_util.computeConv3DInfo(
        inputShape, filter.shape as [number, number, number, number, number],
        strides, dilations, pad);

    const $strides = [
      1, convInfo.strideDepth, convInfo.strideHeight, convInfo.strideWidth, 1
    ];
    const padding = convInfo.padInfo.type;
    const dataFormat =
        convInfo.dataFormat === 'channelsLast' ? 'NDHWC' : 'NCDHW';
    if (!backend.isGPUPackage && convInfo.dilationDepth > 1) {
      throw new Error('CPU Dilation depth must be 1');
    }
    const $dilations = [
      1, convInfo.dilationDepth, convInfo.dilationHeight,
      convInfo.dilationWidth, 1
    ];
    const opAttrs = [
      createTensorsTypeOpAttr('T', dy.dtype),
      {name: 'strides', type: backend.binding.TF_ATTR_INT, value: $strides},
      {name: 'padding', type: backend.binding.TF_ATTR_STRING, value: padding}, {
        name: 'data_format',
        type: backend.binding.TF_ATTR_STRING,
        value: dataFormat
      },
      {name: 'dilations', type: backend.binding.TF_ATTR_INT, value: $dilations},
      createTensorsTypeOpAttr('Tshape', 'int32')
    ];
    const inputSizes = tensor1d(inputShape, 'int32');
    const res = backend.executeSingleOutput(
        Conv3DBackpropInputV2, opAttrs, [inputSizes, filter, dy]);
    inputSizes.dispose();
    return res;
  }
};
