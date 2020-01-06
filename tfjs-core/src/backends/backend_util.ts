/**
 * @license
 * Copyright 2018 Google Inc. All Rights Reserved.
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

import {ENGINE} from '../engine';
import {scalar, tensor1d, zeros} from '../ops/tensor_ops';
import {Tensor} from '../tensor';
import {Rank} from '../types';
import {DataType, ShapeMap} from '../types';
import {hasEncodingLoss, makeZerosTypedArray} from '../util';

import {KernelBackend} from './backend';

// Utilities needed by backend consumers of tf-core.
export * from '../ops/axis_util';
export * from '../ops/broadcast_util';
export * from '../ops/concat_util';
export * from '../ops/conv_util';
export {Activation, FusedConv2DConfig} from '../ops/fused_util';
export * from '../ops/reduce_util';
export {BackendValues, TypedArray, upcastType, PixelData} from '../types';
export {MemoryInfo, TimingInfo} from '../engine';

export function castTensor<T extends Tensor>(
    x: T, dtype: DataType, backend: KernelBackend): T {
  if (dtype === 'complex64') {
    if (x.dtype === 'complex64') {
      return x.clone();
    }
    const zerosTensor = zeros(x.shape);
    const floatX = x.toFloat();
    const result = backend.complex(floatX, zerosTensor);
    zerosTensor.dispose();
    floatX.dispose();
    return result as T;
  }

  if (!hasEncodingLoss(x.dtype, dtype)) {
    // We don't change the underlying data, since we cast to higher
    // precision.
    return ENGINE.makeTensorFromDataId(x.dataId, x.shape, dtype) as T;
  }
  if (x.dtype === 'complex64') {
    const real = backend.real(x);
    const result = real.cast(dtype);
    real.dispose();
    return result;
  }
  if (dtype === 'int32') {
    return backend.int(x);
  } else if (dtype === 'bool') {
    const zero = scalar(0, x.dtype);
    const result = backend.notEqual(x, zero) as T;
    zero.dispose();
    return result;
  } else {
    throw new Error(`Error in Cast: failed to cast ${x.dtype} to ${dtype}`);
  }
}

export function reshapeTensor<T extends Tensor, R extends Rank>(
    x: T, shape: ShapeMap[R]): Tensor<R> {
  return ENGINE.makeTensorFromDataId(x.dataId, shape, x.dtype) as Tensor<R>;
}

export function linspaceImpl(start: number, stop: number, num: number) {
  const step = (stop - start) / (num - 1);

  const values = makeZerosTypedArray(num, 'float32');
  values[0] = start;
  for (let i = 1; i < values.length; i++) {
    values[i] = values[i - 1] + step;
  }

  return tensor1d(values, 'float32');
}
