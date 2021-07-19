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

import {Tensor} from './tensor';
import {TensorContainer, TensorContainerArray} from './tensor_types';
import {upcastType} from './types';
import {assert} from './util';

export function makeTypesMatch<T extends Tensor>(a: T, b: T): [T, T] {
  if (a.dtype === b.dtype) {
    return [a, b];
  }
  const dtype = upcastType(a.dtype, b.dtype);
  return [a.cast(dtype), b.cast(dtype)];
}

export function assertTypesMatch(a: Tensor, b: Tensor): void {
  assert(
      a.dtype === b.dtype,
      () => `The dtypes of the first(${a.dtype}) and` +
          ` second(${b.dtype}) input must match`);
}

export function isTensorInList(tensor: Tensor, tensorList: Tensor[]): boolean {
  return tensorList.some(x => x.id === tensor.id);
}

/**
 * Extracts any `Tensor`s found within the provided object.
 *
 * @param container an object that may be a `Tensor` or may directly contain
 *   `Tensor`s, such as a `Tensor[]` or `{key: Tensor, ...}`. In general it
 *   is safe to pass any object here, except that `Promise`s are not
 *   supported.
 * @returns An array of `Tensors` found within the passed object. If the
 *   argument is simply a `Tensor', a list containing that `Tensor` is
 *   returned. If the object is not a `Tensor` or does not
 *   contain `Tensors`, an empty list is returned.
 */
export function getTensorsInContainer(result: TensorContainer): Tensor[] {
  const list: Tensor[] = [];
  const seen = new Set<{}|void>();
  walkTensorContainer(result, list, seen);
  return list;
}

function walkTensorContainer(
    container: TensorContainer, list: Tensor[], seen: Set<{}|void>): void {
  if (container == null) {
    return;
  }
  if (container instanceof Tensor) {
    list.push(container);
    return;
  }
  if (!isIterable(container)) {
    return;
  }
  // Iteration over keys works also for arrays.
  const iterable = container as TensorContainerArray;
  for (const k in iterable) {
    const val = iterable[k];
    if (!seen.has(val)) {
      seen.add(val);
      walkTensorContainer(val, list, seen);
    }
  }
}

// tslint:disable-next-line:no-any
function isIterable(obj: any): boolean {
  return Array.isArray(obj) || typeof obj === 'object';
}
