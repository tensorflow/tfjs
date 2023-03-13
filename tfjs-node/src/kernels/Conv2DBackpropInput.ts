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

import {backend_util, Conv2DBackpropInput, Conv2DBackpropInputAttrs, Conv2DBackpropInputInputs, KernelConfig, tensor1d, Tensor4D, TensorInfo} from '@tensorflow/tfjs';

import {createTensorsTypeOpAttr, NodeJSKernelBackend} from '../nodejs_kernel_backend';

export const conv2DBackpropInputConfig: KernelConfig = {
  kernelName: Conv2DBackpropInput,
  backendName: 'tensorflow',
  kernelFunc: (args) => {
    const {dy, filter} = args.inputs as Conv2DBackpropInputInputs;
    const backend = args.backend as NodeJSKernelBackend;
    const {strides, pad, dataFormat, dimRoundingMode, inputShape} =
        args.attrs as unknown as Conv2DBackpropInputAttrs;

    const $dataFormat = backend_util.convertConv2DDataFormat(dataFormat);
    const convInfo = backend_util.computeConv2DInfo(
        inputShape, filter.shape as [number, number, number, number], strides,
        1 /* dilations */, pad, dimRoundingMode, false, $dataFormat);

    return conv2DBackpropInputImpl(dy, filter, convInfo, backend);
  }
};

function conv2DBackpropInputImpl(
    dy: TensorInfo, filter: TensorInfo, convInfo: backend_util.Conv2DInfo,
    backend: NodeJSKernelBackend): Tensor4D {
  if (convInfo.padInfo.type !== 'VALID' && convInfo.padInfo.type !== 'SAME') {
    throw new Error(
        `TF Backend supports only 'valid' and 'same' padding ` +
        `while padding was ${convInfo.padInfo.type}`);
  }
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
    {name: 'use_cudnn_on_gpu', type: backend.binding.TF_ATTR_BOOL, value: true},
    {name: 'dilations', type: backend.binding.TF_ATTR_INT, value: dilations}
  ];
  const inputSizes = tensor1d(convInfo.inShape, 'int32');
  const res =
      backend.executeSingleOutput(
          Conv2DBackpropInput, opAttrs, [inputSizes, filter, dy]) as Tensor4D;
  inputSizes.dispose();
  return res;
}
