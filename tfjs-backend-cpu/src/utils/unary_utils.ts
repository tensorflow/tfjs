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

import {DataType, KernelFunc, TypedArray, UnaryInputs, util} from '@tensorflow/tfjs-core';

import {MathBackendCPU} from '../backend_cpu';
import {assertNotComplex} from '../cpu_util';

import {SimpleUnaryImpl, SimpleUnaryOperation} from './unary_types';

/**
 * Template that creates a `KernelFunc` for unary ops.
 * @param name Kernel name.
 * @param op A `SimpleUnaryOperation` for the kernel.
 * @param dtype Optional. If set, the result has this dtype. Otherwise, the
 *     result has the same dtype as the input. This is mainly used in certain
 *     kernels that return bool type, such as isFinite, isInf, etc.
 */
export function unaryKernelFunc(
    name: string, op: SimpleUnaryOperation, dtype?: DataType): KernelFunc {
  return ({inputs, attrs, backend}) => {
    const {x} = inputs as UnaryInputs;
    assertNotComplex(x, name);
    if (x.dtype === 'string' || dtype === 'string') {
      throw new Error('unaryKernelFunc does not support string input/output');
    }

    const cpuBackend = backend as MathBackendCPU;
    const values = cpuBackend.data.get(x.dataId).values as TypedArray;
    const xSize = util.sizeFromShape(x.shape);
    const $dtype = dtype || x.dtype;
    const newValues = util.getArrayFromDType($dtype, xSize);
    for (let i = 0; i < xSize; ++i) {
      newValues[i] = op(values[i], attrs);
    }
    return cpuBackend.makeTensorInfo(x.shape, $dtype, newValues);
  };
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
export function unaryKernelFuncFromImpl(
    name: string, unaryImpl: SimpleUnaryImpl, dtype?: DataType): KernelFunc {
  return ({inputs, attrs, backend}) => {
    const {x} = inputs as UnaryInputs;
    assertNotComplex(x, name);
    if (x.dtype === 'string' || dtype === 'string') {
      throw new Error('unaryKernelFunc does not support string input/output');
    }

    const cpuBackend = backend as MathBackendCPU;
    const values = cpuBackend.data.get(x.dataId).values as TypedArray;
    const $dtype = dtype || x.dtype;
    const newValues = unaryImpl(values, $dtype, attrs);
    return cpuBackend.makeTensorInfo(x.shape, $dtype, newValues);
  };
}
