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

import {backend_util, Conv2D, Conv2DAttrs, Conv2DInputs, KernelConfig, Tensor4D, TensorInfo} from '@tensorflow/tfjs';

import {createTensorsTypeOpAttr, NodeJSKernelBackend} from '../nodejs_kernel_backend';

export const conv2DConfig: KernelConfig = {
  kernelName: Conv2D,
  backendName: 'tensorflow',
  kernelFunc: (args) => {
    const {x, filter} = args.inputs as Conv2DInputs;
    const backend = args.backend as NodeJSKernelBackend;
    const {strides, pad, dataFormat, dilations, dimRoundingMode} =
        args.attrs as unknown as Conv2DAttrs;

    const $dataFormat = backend_util.convertConv2DDataFormat(dataFormat);
    const convInfo = backend_util.computeConv2DInfo(
        x.shape as [number, number, number, number],
        filter.shape as [number, number, number, number], strides, dilations,
        pad, dimRoundingMode, false /* depthwise */, $dataFormat);

    return conv2dImpl(x, filter, convInfo, backend);
  }
};

export function conv2dImpl(
    x: TensorInfo, filter: TensorInfo, convInfo: backend_util.Conv2DInfo,
    backend: NodeJSKernelBackend): Tensor4D {
  if (convInfo.padInfo.type !== 'VALID' && convInfo.padInfo.type !== 'SAME' &&
      convInfo.padInfo.type !== 'EXPLICIT') {
    throw new Error(
        `TF Backend supports only 'valid' and 'same' padding ` +
        `while padding was ${convInfo.padInfo.type}`);
  }
  const strides = [1, convInfo.strideHeight, convInfo.strideWidth, 1];
  const padding = convInfo.padInfo.type;
  const dataFormat = convInfo.dataFormat === 'channelsLast' ? 'NHWC' : 'NCHW';
  const dilations = [1, convInfo.dilationHeight, convInfo.dilationWidth, 1];
  const opAttrs = [
    createTensorsTypeOpAttr('T', x.dtype),
    {name: 'strides', type: backend.binding.TF_ATTR_INT, value: strides},
    {name: 'padding', type: backend.binding.TF_ATTR_STRING, value: padding},
    {
      name: 'data_format',
      type: backend.binding.TF_ATTR_STRING,
      value: dataFormat
    },
    {name: 'use_cudnn_on_gpu', type: backend.binding.TF_ATTR_BOOL, value: true},
    {name: 'dilations', type: backend.binding.TF_ATTR_INT, value: dilations},
  ];
  if (padding === 'EXPLICIT') {
    const padValue = [
      convInfo.padInfo.top, convInfo.padInfo.bottom, convInfo.padInfo.left,
      convInfo.padInfo.right
    ];
    opAttrs.push({
      name: 'explicit_paddings',
      type: backend.binding.TF_ATTR_INT,
      value: dataFormat === 'NHWC' ? [0, 0, ...padValue, 0, 0] :
                                     [0, 0, 0, 0, ...padValue]
    });
  }
  return backend.executeSingleOutput(Conv2D, opAttrs, [x, filter]) as Tensor4D;
}
