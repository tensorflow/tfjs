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

import {FusedBatchNorm, FusedBatchNormAttrs, FusedBatchNormInputs, KernelConfig, KernelFunc, TensorInfo, TypedArray, util} from '@tensorflow/tfjs-core';

import {MathBackendCPU} from '../backend_cpu';
import {assertNotComplex} from '../cpu_util';

export function batchNormKernelFunc(args: {
  inputs: FusedBatchNormInputs,
  backend: MathBackendCPU,
  attrs: FusedBatchNormAttrs
}): TensorInfo {
  const {inputs, backend, attrs} = args;
  const {x, scale, offset, mean, variance} = inputs;

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

  assertNotComplex([x, mean, variance, scale, offset], 'batchNorm');

  let {varianceEpsilon} = attrs;
  if (varianceEpsilon == null) {
    varianceEpsilon = 0.001;
  }

  const xVals = backend.data.get(x.dataId).values as TypedArray;
  const mVals = backend.data.get(mean.dataId).values as TypedArray;
  const varVals = backend.data.get(variance.dataId).values as TypedArray;
  const sVals = scale ? backend.data.get(scale.dataId).values as TypedArray :
                        new Float32Array([1]);
  const offVals = offset ?
      backend.data.get(offset.dataId).values as TypedArray :
      new Float32Array([0]);
  const outVals = new Float32Array(xVals.length);

  const offValsLength = offVals.length;
  const sValsLength = sVals.length;
  const varValsLength = varVals.length;
  const mValsLength = mVals.length;

  let offi = 0;
  let mi = 0;
  let si = 0;
  let vi = 0;
  for (let i = 0; i < xVals.length; ++i) {
    outVals[i] = offVals[offi++] +
        (xVals[i] - mVals[mi++]) * sVals[si++] /
            Math.sqrt(varVals[vi++] + varianceEpsilon);
    if (offi >= offValsLength) {
      offi = 0;
    }
    if (mi >= mValsLength) {
      mi = 0;
    }
    if (si >= sValsLength) {
      si = 0;
    }
    if (vi >= varValsLength) {
      vi = 0;
    }
  }
  return backend.makeTensorInfo(x.shape, x.dtype, outVals);
}

export const batchNormConfig: KernelConfig = {
  kernelName: FusedBatchNorm,
  backendName: 'cpu',
  kernelFunc: batchNormKernelFunc as {} as KernelFunc,
};
