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

import {backend_util, DepthwiseConv2dNativeBackpropInput, DepthwiseConv2dNativeBackpropInputAttrs, DepthwiseConv2dNativeBackpropInputInputs, KernelConfig, tensor1d, Tensor4D, TensorInfo} from '@tensorflow/tfjs';

import {createTensorsTypeOpAttr, NodeJSKernelBackend} from '../nodejs_kernel_backend';

export const depthwiseConv2dNativeBackpropInputConfig: KernelConfig = {
  kernelName: DepthwiseConv2dNativeBackpropInput,
  backendName: 'tensorflow',
  kernelFunc: (args) => {
    const {dy, filter} =
        args.inputs as DepthwiseConv2dNativeBackpropInputInputs;
    const backend = args.backend as NodeJSKernelBackend;
    const {strides, dilations, pad, dimRoundingMode, inputShape} =
        args.attrs as {} as DepthwiseConv2dNativeBackpropInputAttrs;

    const convInfo = backend_util.computeConv2DInfo(
        inputShape, filter.shape as [number, number, number, number], strides,
        dilations, pad, dimRoundingMode, true /* depthwise */);

    return depthwiseConv2dNativeBackpropInputImpl(
        dy, filter, convInfo, backend);
  }
};

function depthwiseConv2dNativeBackpropInputImpl(
    dy: TensorInfo, filter: TensorInfo, convInfo: backend_util.Conv2DInfo,
    backend: NodeJSKernelBackend): Tensor4D {
  const strides = [1, convInfo.strideHeight, convInfo.strideWidth, 1];
  const padding = convInfo.padInfo.type;
  const dataFormat = convInfo.dataFormat === 'channelsLast' ? 'NHWC' : 'NCHW';
  const dilations = [1, convInfo.dilationHeight, convInfo.dilationWidth, 1];
  const opAttrs = [
    createTensorsTypeOpAttr('T', 'float32'),
    {name: 'strides', type: backend.binding.TF_ATTR_INT, value: strides},
    {name: 'padding', type: backend.binding.TF_ATTR_STRING, value: padding}, {
      name: 'data_format',
      type: backend.binding.TF_ATTR_STRING,
      value: dataFormat
    },
    {name: 'dilations', type: backend.binding.TF_ATTR_INT, value: dilations}
  ];

  const inputSizes = tensor1d(convInfo.inShape, 'int32');
  const res = backend.executeSingleOutput(
                  DepthwiseConv2dNativeBackpropInput, opAttrs,
                  [inputSizes, filter, dy]) as Tensor4D;
  inputSizes.dispose();
  return res;
}
