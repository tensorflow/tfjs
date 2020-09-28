
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

import {env, FusedBatchNorm, FusedBatchNormAttrs, FusedBatchNormInputs, KernelConfig, KernelFunc, TensorInfo, util} from '@tensorflow/tfjs-core';

import {MathBackendWebGL} from '../backend_webgl';
import {BatchNormProgram} from '../batchnorm_gpu';
import {BatchNormPackedProgram} from '../batchnorm_packed_gpu';

export const batchNormKernelFunc: (params: {
  inputs: FusedBatchNormInputs,
  backend: MathBackendWebGL,
  attrs: FusedBatchNormAttrs
}) => TensorInfo = ({inputs, backend, attrs}) => {
  const {x, mean, variance, offset, scale} = inputs;

  util.assert(
      mean.shape.length === variance.shape.length,
      () => 'Batch normalization gradient requires mean and variance to have ' +
          'equal ranks.');
  util.assert(
      offset == null || mean.shape.length === offset.shape.length,
      () => 'Batch normalization gradient requires mean and offset to have ' +
          'equal ranks.');
  util.assert(
      scale == null || mean.shape.length === scale.shape.length,
      () => 'Batch normalization gradient requires mean and scale to have ' +
          'equal ranks.');

  let {varianceEpsilon} = attrs;
  if (varianceEpsilon == null) {
    varianceEpsilon = 0.001;
  }

  const finalInputs = [x, mean, variance];

  let offsetShape = null;
  if (offset != null) {
    offsetShape = offset.shape;
    finalInputs.push(offset);
  }

  let scaleShape = null;
  if (scale != null) {
    scaleShape = scale.shape;
    finalInputs.push(scale);
  }

  const program = env().getBool('WEBGL_PACK_NORMALIZATION') ?
      new BatchNormPackedProgram(
          x.shape, mean.shape, variance.shape, offsetShape, scaleShape,
          varianceEpsilon) :
      new BatchNormProgram(
          x.shape, mean.shape, variance.shape, offsetShape, scaleShape,
          varianceEpsilon);
  const output =
      backend.runWebGLProgram(program, finalInputs, finalInputs[0].dtype);

  return output;
};

export const batchNormConfig: KernelConfig = {
  kernelName: FusedBatchNorm,
  backendName: 'webgl',
  kernelFunc: batchNormKernelFunc as {} as KernelFunc,
};
