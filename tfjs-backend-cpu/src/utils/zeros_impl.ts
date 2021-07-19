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

import {DataType, TensorInfo, util} from '@tensorflow/tfjs-core';
import {MathBackendCPU} from '../backend_cpu';
import {complex} from '../kernels/Complex';

/**
 * Generates a tensorInfo with all zeros value.
 * @param backend cpu backend.
 * @param shape Shape for the zeros tensor.
 * @param dtype Optional. If set, the result has this dtype.
 */
export function zeros(
    backend: MathBackendCPU, shape: number[],
    dtype: DataType = 'float32'): TensorInfo {
  if (dtype === 'complex64') {
    const real = zeros(backend, shape, 'float32');
    const imag = zeros(backend, shape, 'float32');

    return complex({inputs: {real, imag}, backend});
  }

  const values = util.makeZerosTypedArray(util.sizeFromShape(shape), dtype);

  return backend.makeTensorInfo(shape, dtype, values);
}
