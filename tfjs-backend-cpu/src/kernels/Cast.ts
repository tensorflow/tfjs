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
import {Cast, CastAttrs, CastInputs, KernelConfig, KernelFunc, TensorInfo, TypedArray, util} from '@tensorflow/tfjs-core';

import {MathBackendCPU} from '../backend_cpu';
import {createSimpleBinaryKernelImpl} from '../utils/binary_impl';
import {zeros} from '../utils/zeros_impl';

import {complex} from './Complex';
import {identity} from './Identity';
import {real} from './Real';

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

    const zerosTensorInfo = zeros(backend, x.shape, x.dtype);
    const floatX = cast({inputs: {x}, backend, attrs: {dtype: 'float32'}});

    const result =
        complex({inputs: {real: floatX, imag: zerosTensorInfo}, backend});

    backend.disposeIntermediateTensorInfo(zerosTensorInfo);
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
    const values = backend.data.get(x.dataId).values as TypedArray;
    const resultValues = Int32Array.from(values);
    return backend.makeTensorInfo(x.shape, 'int32', resultValues);
  }

  if (dtype === 'bool') {
    // This is essentially the result of notEqual(x, 0). We avoid using
    // kernel notEqual to avoid circular dependency, i.e. binary_utils ->
    // cast -> notEqual -> binary_utils.
    const xVals = backend.data.get(x.dataId).values as TypedArray;
    const zero = util.toTypedArray([0], x.dtype);

    const [resultData, resultShape] = createSimpleBinaryKernelImpl(
        (a, b) => (a !== b) ? 1 : 0)(x.shape, [], xVals, zero, 'bool');

    return backend.makeTensorInfo(resultShape, 'bool', resultData);
  }

  throw new Error(`Error in Cast: failed to cast ${x.dtype} to ${dtype}`);
}

export const castConfig: KernelConfig = {
  kernelName: Cast,
  backendName: 'cpu',
  kernelFunc: cast as {} as KernelFunc
};
