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

import {FusedBatchNorm, FusedBatchNormAttrs, FusedBatchNormInputs, KernelConfig, KernelFunc, TensorInfo} from '@tensorflow/tfjs-core';

import {MathBackendWebGL} from '../backend_webgl';

import {batchNorm} from './BatchNorm_impl';

export const batchNormKernelFunc: (params: {
  inputs: FusedBatchNormInputs,
  backend: MathBackendWebGL,
  attrs: FusedBatchNormAttrs
}) => TensorInfo | TensorInfo[] = ({inputs, backend, attrs}) => {
  const {x, mean, variance, offset, scale} = inputs;
  const {varianceEpsilon} = attrs;
  return batchNorm(x, mean, variance, backend, offset, scale, varianceEpsilon);
};

export const batchNormConfig: KernelConfig = {
  kernelName: FusedBatchNorm,
  backendName: 'webgl',
  kernelFunc: batchNormKernelFunc as {} as KernelFunc,
};
