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

import {backend_util, Conv2dDerAttrs, Conv2dDerInput, Conv2dDerInputInputs, engine, KernelConfig, Tensor} from '@tensorflow/tfjs-core';

import {WebGPUBackend} from '../backend_webgpu';

import {Conv2DDerInputProgram} from './conv_derinput_webgpu';

export const conv2dDerInputConfig: KernelConfig = {
  kernelName: Conv2dDerInput,
  backendName: 'webgpu',
  kernelFunc: ({inputs, attrs, backend}) => {
    const {dy4D, filter} = inputs as Conv2dDerInputInputs;
    const {
      batchSize,
      inHeight,
      inWidth,
      inChannels,
      outHeight,
      outWidth,
      outChannels,
      dataFormat,
      strideHeight,
      strideWidth,
      dilationHeight,
      dilationWidth,
      filterHeight,
      filterWidth,
      effectiveFilterHeight,
      effectiveFilterWidth,
      padInfo,
      inShape,
      outShape,
      filterShape
    } = attrs as {} as Conv2dDerAttrs;

    const convInfo: backend_util.Conv2DInfo = {
      batchSize,
      inHeight,
      inWidth,
      inChannels,
      outHeight,
      outWidth,
      outChannels,
      dataFormat,
      strideHeight,
      strideWidth,
      dilationHeight,
      dilationWidth,
      filterHeight,
      filterWidth,
      effectiveFilterHeight,
      effectiveFilterWidth,
      padInfo,
      inShape,
      outShape,
      filterShape
    };
    const webGPUBackend = backend as WebGPUBackend;
    convInfo.outShape = convInfo.inShape;
    const dataId =
        webGPUBackend.write(null /*values*/, convInfo.outShape, dy4D.dtype);
    const output = engine().makeTensorFromDataId(
        dataId, convInfo.outShape, dy4D.dtype, webGPUBackend);
    const pad = convInfo.padInfo.type === 'VALID' ?
        [0, 0] :
        convInfo.padInfo.type === 'SAME' ?
        [
          -Math.floor((convInfo.filterShape[0] - 1) / 2),
          -Math.floor((convInfo.filterShape[1] - 1) / 2)
        ] :
        [convInfo.padInfo.top, convInfo.padInfo.left];
    const dimensions = [
      convInfo.filterHeight, convInfo.filterWidth, ...pad,
      convInfo.strideHeight, convInfo.strideWidth, convInfo.dilationHeight,
      convInfo.dilationWidth
    ];

    const program = new Conv2DDerInputProgram(convInfo);
    return webGPUBackend.compileAndRun(
        program, [dy4D as Tensor, filter as Tensor], output, dimensions);
  }
};
