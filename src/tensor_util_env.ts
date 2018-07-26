/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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

import {ENV} from './environment';
import {Tensor} from './tensor';
import {ArrayData, DataType, TensorLike} from './types';
import {inferShape, isTypedArray, toTypedArray} from './util';

export function convertToTensor<T extends Tensor>(
    x: T|TensorLike, argName: string, functionName: string,
    dtype: DataType = 'float32'): T {
  dtype = dtype || 'float32';
  if (x instanceof Tensor) {
    return x;
  }
  if (!isTypedArray(x) && !Array.isArray(x) && typeof x !== 'number' &&
      typeof x !== 'boolean') {
    throw new Error(
        `Argument '${argName}' passed to '${functionName}' must be a ` +
        `Tensor or TensorLike, but got ${x.constructor.name}`);
  }
  const inferredShape = inferShape(x);
  if (!isTypedArray(x) && !Array.isArray(x)) {
    x = [x] as number[];
  }
  return Tensor.make(
      inferredShape,
      {values: toTypedArray(x as ArrayData<DataType>, dtype, ENV.get('DEBUG'))},
      dtype);
}

export function convertToTensorArray<T extends Tensor>(
    arg: T[]|TensorLike[], argName: string, functionName: string): T[] {
  if (!Array.isArray(arg)) {
    throw new Error(
        `Argument ${argName} passed to ${functionName} must be a ` +
        '`Tensor[]` or `TensorLike[]`');
  }
  const tensors = arg as T[];
  return tensors.map(
      (t, i) => convertToTensor(t, `${argName}[${i}]`, functionName));
}
