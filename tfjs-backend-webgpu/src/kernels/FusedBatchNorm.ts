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

import {FusedBatchNorm, FusedBatchNormAttrs, FusedBatchNormInputs, KernelConfig, Tensor} from '@tensorflow/tfjs-core';

import {WebGPUBackend} from '../backend_webgpu';

import {BatchNormProgram} from './batchnorm_webgpu';

export const fusedBatchNormConfig: KernelConfig = {
  kernelName: FusedBatchNorm,
  backendName: 'webgpu',
  kernelFunc: ({inputs, attrs, backend}) => {
    const {x, scale, offset, mean, variance} = inputs as FusedBatchNormInputs;
    const {varianceEpsilon} = attrs as unknown as FusedBatchNormAttrs;
    const webGPUBackend = backend as WebGPUBackend;
    const batchNormInputs = [x as Tensor, mean as Tensor, variance as Tensor];
    let offsetShape = null;
    if (offset != null) {
      offsetShape = offset.shape;
      batchNormInputs.push(offset as Tensor);
    }
    let scaleShape = null;
    if (scale != null) {
      scaleShape = scale.shape;
      batchNormInputs.push(scale as Tensor);
    }
    const program = new BatchNormProgram(
        x.shape, mean.shape, variance.shape, offsetShape, scaleShape,
        varianceEpsilon);
    return webGPUBackend.runWebGPUProgram(program, batchNormInputs, x.dtype);
  }
};
