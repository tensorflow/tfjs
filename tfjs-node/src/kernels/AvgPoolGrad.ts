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

import {AvgPoolGrad, AvgPoolGradAttrs, AvgPoolGradInputs, backend_util, KernelConfig, tensor1d} from '@tensorflow/tfjs';

import {createTensorsTypeOpAttr, NodeJSKernelBackend} from '../nodejs_kernel_backend';

export const avgPoolGradConfig: KernelConfig = {
  kernelName: AvgPoolGrad,
  backendName: 'tensorflow',
  kernelFunc: (args) => {
    const {dy, input} = args.inputs as AvgPoolGradInputs;
    const backend = args.backend as NodeJSKernelBackend;
    const {filterSize, strides, pad} =
        args.attrs as unknown as AvgPoolGradAttrs;

    const convInfo = backend_util.computePool2DInfo(
        input.shape as [number, number, number, number], filterSize, strides,
        1 /* dilations */, pad);

    if (convInfo.padInfo.type !== 'VALID' && convInfo.padInfo.type !== 'SAME') {
      throw new Error(
          `TF Backend supports only 'valid' and 'same' padding ` +
          `while padding type was ${convInfo.padInfo.type}`);
    }
    const ksize = [1, convInfo.filterHeight, convInfo.filterWidth, 1];
    const $strides = [1, convInfo.strideHeight, convInfo.strideWidth, 1];
    const padding = convInfo.padInfo.type;
    const dataFormat = convInfo.dataFormat === 'channelsLast' ? 'NHWC' : 'NCHW';
    const opAttrs = [
      createTensorsTypeOpAttr('T', input.dtype),
      {name: 'ksize', type: backend.binding.TF_ATTR_INT, value: ksize},
      {name: 'strides', type: backend.binding.TF_ATTR_INT, value: $strides},
      {name: 'padding', type: backend.binding.TF_ATTR_STRING, value: padding},
      {
        name: 'data_format',
        type: backend.binding.TF_ATTR_STRING,
        value: dataFormat
      },
    ];
    const origInputShape = tensor1d(input.shape, 'int32');
    const ret =
        backend.executeSingleOutput(AvgPoolGrad, opAttrs, [origInputShape, dy]);
    origInputShape.dispose();
    return ret;
  }
};
