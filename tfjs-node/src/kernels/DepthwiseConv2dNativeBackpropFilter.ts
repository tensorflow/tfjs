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

import {backend_util, DepthwiseConv2dNativeBackpropFilter, DepthwiseConv2dNativeBackpropFilterAttrs, DepthwiseConv2dNativeBackpropFilterInputs, KernelConfig, tensor1d, Tensor4D, TensorInfo} from '@tensorflow/tfjs';

import {createTensorsTypeOpAttr, NodeJSKernelBackend} from '../nodejs_kernel_backend';

export const depthwiseConv2dNativeBackpropFilterConfig: KernelConfig = {
  kernelName: DepthwiseConv2dNativeBackpropFilter,
  backendName: 'tensorflow',
  kernelFunc: (args) => {
    const {x, dy} = args.inputs as DepthwiseConv2dNativeBackpropFilterInputs;
    const backend = args.backend as NodeJSKernelBackend;
    const {strides, dilations, pad, dimRoundingMode, filterShape} =
        args.attrs as unknown as DepthwiseConv2dNativeBackpropFilterAttrs;

    const convInfo = backend_util.computeConv2DInfo(
        x.shape as [number, number, number, number], filterShape, strides,
        dilations, pad, dimRoundingMode, true /* depthwise */);

    return depthwiseConv2dNativeBackpropFilterImpl(x, dy, convInfo, backend);
  }
};

function depthwiseConv2dNativeBackpropFilterImpl(
    x: TensorInfo, dy: TensorInfo, convInfo: backend_util.Conv2DInfo,
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
  const filterSizes = tensor1d(convInfo.filterShape, 'int32');
  const res = backend.executeSingleOutput(
                  DepthwiseConv2dNativeBackpropFilter, opAttrs,
                  [x, filterSizes, dy]) as Tensor4D;
  filterSizes.dispose();
  return res;
}
