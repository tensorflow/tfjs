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

import {KernelConfig, KernelFunc, TensorInfo, Transpose, TransposeAttrs, TransposeInputs, TypedArray} from '@tensorflow/tfjs-core';

import {MathBackendCPU} from '../backend_cpu';
import {assertNotComplex} from '../cpu_util';

import {transposeImpl} from './Transpose_impl';

export function transpose(args: {
  inputs: TransposeInputs,
  attrs: TransposeAttrs,
  backend: MathBackendCPU
}): TensorInfo {
  const {inputs, attrs, backend} = args;
  const {x} = inputs;
  const {perm} = attrs;

  assertNotComplex(x, 'transpose');

  const xRank = x.shape.length;

  const newShape: number[] = new Array(xRank);
  for (let i = 0; i < newShape.length; i++) {
    newShape[i] = x.shape[perm[i]];
  }

  const values = backend.data.get(x.dataId).values as TypedArray;
  const result = transposeImpl(values, x.shape, x.dtype, perm, newShape);

  const dataId = backend.write(result, newShape, x.dtype);
  return {dataId, shape: newShape, dtype: x.dtype};
}

export const transposeConfig: KernelConfig = {
  kernelName: Transpose,
  backendName: 'cpu',
  kernelFunc: transpose as {} as KernelFunc
};
