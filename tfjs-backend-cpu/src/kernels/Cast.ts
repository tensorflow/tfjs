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
import * as tf from '@tensorflow/tfjs-core';
import {Cast, CastAttrs, CastInputs, KernelConfig, KernelFunc, Tensor, TensorInfo, util} from '@tensorflow/tfjs-core';

import {MathBackendCPU} from '../backend_cpu';

import {complex} from './Complex';
import {identity} from './Identity';
import {real} from './Real';
import {int} from '../utils/cast_utils';

export function cast(
    args: {inputs: CastInputs, backend: MathBackendCPU, attrs: CastAttrs}):
    TensorInfo {
  const {inputs, backend, attrs} = args;
  const {x} = inputs;
  const {dtype} = attrs;

  // Casting to complex64.
  if (dtype === 'complex64') {
    if (x.dtype === 'complex64') {
      return identity({inputs: {x}, backend});
    }

    // TODO(lina128): Import kernel function once zeros is modularized.
    const zerosTensor = tf.zeros(x.shape);
    const floatX = cast({inputs: {x}, backend, attrs: {dtype: 'float32'}});

    const result =
        complex({inputs: {real: floatX, imag: zerosTensor}, backend});

    zerosTensor.dispose();
    backend.disposeIntermediateTensorInfo(floatX);

    return result;
  }

  // Casting from complex64
  if (x.dtype === 'complex64') {
    const realPart = real({inputs: {input: x}, backend});
    const result = cast({inputs: {x: realPart}, backend, attrs: {dtype}});

    backend.disposeIntermediateTensorInfo(realPart);

    return result;
  }

  if (!util.hasEncodingLoss(x.dtype, dtype)) {
    // We don't change the underlying data, since we cast to higher
    // precision.
    const result = identity({inputs: {x}, backend});
    return {dataId: result.dataId, shape: result.shape, dtype};
  }

  if (dtype === 'int32') {
    return int({inputs: {x}, backend});
  }

  if (dtype === 'bool') {
    // TODO(lina128): Import kernel function and just use 0 once notEqual is
    // modularized.
    const zero = tf.scalar(0, x.dtype);
    const result = tf.notEqual(x as Tensor, zero);

    zero.dispose();

    return result;
  }

  throw new Error(`Error in Cast: failed to cast ${x.dtype} to ${dtype}`);
}

export const castConfig: KernelConfig = {
  kernelName: Cast,
  backendName: 'cpu',
  kernelFunc: cast as {} as KernelFunc
};
