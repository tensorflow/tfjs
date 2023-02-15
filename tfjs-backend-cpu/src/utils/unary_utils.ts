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

import {backend_util, DataTypeFor, KernelFunc, UnaryInputs} from '@tensorflow/tfjs-core';

import {MathBackendCPU} from '../backend_cpu';
import {assertNotComplex} from '../cpu_util';
import {createSimpleUnaryImpl} from './unary_impl';

import {SimpleUnaryImpl, SimpleUnaryOperation} from './unary_types';

/**
 * Template that creates a `KernelFunc` for unary ops.
 * @param name Kernel name.
 * @param op A `SimpleUnaryOperation` for the kernel.
 * @param dtype Optional. If set, the result has this dtype. Otherwise, the
 *     result has the same dtype as the input. This is mainly used in certain
 *     kernels that return bool type, such as isFinite, isInf, etc.
 */
export function unaryKernelFunc<I extends number | string = number,
  O extends number | string = number>(
  name: string, op: SimpleUnaryOperation<I, O>,
  dtype?: DataTypeFor<O>): KernelFunc {

  const impl = createSimpleUnaryImpl<I, O>(op);

  return unaryKernelFuncFromImpl<I, O>(name, impl, dtype);
}

/**
 * Template that creates a `KernelFunc` for unary ops from the given
 * `SimpleUnaryImpl`..
 * @param name Kernel name.
 * @param unaryImpl A `SimpleUnaryImpl` that implements the op.
 * @param dtype Optional. If set, the result has this dtype. Otherwise, the
 *     result has the same dtype as the input. This is mainly used in certain
 *     kernels that return bool type, such as isFinite, isInf, etc.
 */
export function unaryKernelFuncFromImpl<I extends number | string = number,
  O extends number | string = number>(
  name: string, unaryImpl: SimpleUnaryImpl<I, O>,
  dtype?: DataTypeFor<O>): KernelFunc {

  return ({inputs, attrs, backend}) => {
    const {x} = inputs as UnaryInputs;
    assertNotComplex(x, name);

    const cpuBackend = backend as MathBackendCPU;
    const values = cpuBackend.data.get(x.dataId).values;
    let decoded: ArrayLike<I>;
    if (x.dtype === 'string') {
      if (!Array.isArray(values)) {
        throw new Error('String tensor\'s value was not an instance of Array');
      }
      decoded = backend_util.fromUint8ToStringArray(values) as unknown as
        ArrayLike<I>;
    } else {
      decoded = values as unknown as ArrayLike<I>;
    }

    const $dtype = dtype || x.dtype as DataTypeFor<O>;
    const newValues = unaryImpl(decoded, $dtype, attrs);
    return cpuBackend.makeTensorInfo(x.shape, $dtype, newValues);
  };
}
