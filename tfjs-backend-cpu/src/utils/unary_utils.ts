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

import {DataType, KernelFunc, NamedAttrMap, NumericDataType, TypedArray, UnaryInputs, util} from '@tensorflow/tfjs-core';

import {MathBackendCPU} from '../backend_cpu';
import {assertNotComplex} from '../cpu_util';

import {SimpleUnaryOperation} from './unary_types';

/**
 * Template that creates a `KernelFunc` for unary ops.
 * @param name Kernel name.
 * @param op A `SimpleUnaryOperation` for the kernel.
 * @param dtype Optional. If set, the result has this dtype. Otherwise, the
 *     result has the same dtype as the input. This is mainly used in certain
 *     kernels that return bool type, such as isFinite, isInf, etc.
 * @param opComplex A `ComplexUnaryOperation` for the kernel to handle complex64
 *     inputs.
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
    const $dtype = dtype || x.dtype;
    const implFn = unaryOpImpl(op);
    const newValues = implFn(values, $dtype, attrs);
    return cpuBackend.makeTensorInfo(x.shape, $dtype, newValues);
  };
}

export function unaryOpImpl(op: SimpleUnaryOperation): (
    values: TypedArray, dtype: DataType, attrs?: NamedAttrMap) => TypedArray {
  return (values, dtype, attrs) => {
    const newValues =
        util.getTypedArrayFromDType(dtype as NumericDataType, values.length);
    for (let i = 0; i < values.length; ++i) {
      newValues[i] = op(values[i], attrs);
    }
    return newValues;
  };
}
