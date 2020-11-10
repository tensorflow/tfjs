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
 *
 * =============================================================================
 */

import * as tf from '@tensorflow/tfjs-core';

// tslint:disable:no-any

/**
 * A return value for a mapping function that can be applied via deepMap.
 *
 * If recurse is true, the value should be empty, and iteration will continue
 * into the object or array.
 */
export type DeepMapResult = {
  value: any,
  recurse: boolean
};

/**
 * Apply a mapping function to a nested structure in a recursive manner.
 *
 * The result of the mapping is an object with the same nested structure (i.e.,
 * of arrays and dicts) as the input, except that some subtrees are replaced,
 * according to the results of the mapping function.
 *
 * Mappings are memoized.  Thus, if the nested structure contains the same
 * object in multiple positions, the output will contain the same mapped object
 * in those positions.  Cycles are not supported, however.
 *
 * @param input: The object to which to apply the mapping function.
 * @param mapFn: A function that expects a single node of the object tree, and
 *   returns a `DeepMapResult`.  The `DeepMapResult` either provides a
 *   replacement value for that node (i.e., replacing the subtree), or indicates
 *   that the node should be processed recursively.
 */
export function deepMap(input: any, mapFn: (x: any) => DeepMapResult): any|
    any[] {
  return deepMapInternal(input, mapFn);
}

/**
 * @param seen: A Map of known object mappings (i.e., memoized results of
 *   `mapFn()`)
 * @param containedIn: An set containing objects on the reference path currently
 *   being processed (used to detect cycles).
 */
function deepMapInternal(
    input: any, mapFn: (x: any) => DeepMapResult,
    seen: Map<any, any> = new Map(), containedIn: Set<{}> = new Set()): any|
    any[] {
  if (input == null) {
    return null;
  }
  if (containedIn.has(input)) {
    throw new Error('Circular references are not supported.');
  }
  if (seen.has(input)) {
    return seen.get(input);
  }
  const result = mapFn(input);

  if (result.recurse && result.value !== null) {
    throw new Error(
        'A deep map function may not return both a value and recurse=true.');
  }

  if (!result.recurse) {
    seen.set(input, result.value);
    return result.value;
  } else if (isIterable(input)) {
    // tslint:disable-next-line:no-any
    const mappedIterable: any|any[] = Array.isArray(input) ? [] : {};
    containedIn.add(input);
    for (const k in input) {
      const child = input[k];
      const childResult = deepMapInternal(child, mapFn, seen, containedIn);
      mappedIterable[k] = childResult;
    }
    containedIn.delete(input);
    return mappedIterable;
  } else {
    throw new Error(`Can't recurse into non-iterable type: ${input}`);
  }
}

// TODO(soergel, kangyizhang) Reconsider naming of deepZip() to avoid confusion
// with zip()

/**
 * Zip nested structures together in a recursive manner.
 *
 * This has the effect of transposing or pivoting data, e.g. converting it from
 * a row-major representation to a column-major representation.
 *
 * For example, `deepZip([{a: 1, b: 2}, {a: 3, b: 4}])` returns
 * `{a: [1, 3], b: [2, 4]}`.
 *
 * The inputs should all have the same nested structure (i.e., of arrays and
 * dicts).  The result is a single object with the same nested structure, where
 * the leaves are arrays collecting the values of the inputs at that location
 * (or, optionally, the result of a custom function applied to those arrays).
 *
 * @param inputs: An array of the objects to zip together.
 * @param zipFn: (optional) A function that expects an array of elements at a
 *   single node of the object tree, and returns a `DeepMapResult`.  The
 *   `DeepMapResult` either provides a result value for that node (i.e.,
 *   representing the subtree), or indicates that the node should be processed
 *   recursively.  The default zipFn recurses as far as possible and places
 *   arrays at the leaves.
 */
export function deepZip(
    inputs: any[], zipFn: (xs: any[]) => DeepMapResult = zipToList): any|any[] {
  return deepZipInternal(inputs, zipFn);
}

/**
 * @param containedIn: An set containing objects on the reference path currently
 *   being processed (used to detect cycles).
 */
function deepZipInternal(
    inputs: any[], zipFn: (xs: any[]) => DeepMapResult,
    containedIn: Set<{}> = new Set()): any|any[] {
  // The recursion follows the structure of input 0; it's assumed that all the
  // other inputs have the same structure.
  const input = inputs[0];
  if (containedIn.has(input)) {
    throw new Error('Circular references are not supported.');
  }
  const result = zipFn(inputs);

  if (result.recurse && result.value !== null) {
    throw new Error(
        'A deep zip function may not return both a value and recurse=true.');
  }

  if (!result.recurse) {
    return result.value;
  } else if (isIterable(input)) {
    // tslint:disable-next-line:no-any
    const mappedIterable: any|any[] = Array.isArray(input) ? [] : {};
    containedIn.add(input);
    for (const k in input) {
      const children = inputs.map(x => x[k]);
      const childResult = deepZipInternal(children, zipFn, containedIn);
      mappedIterable[k] = childResult;
    }
    containedIn.delete(input);
    return mappedIterable;
  } else {
    throw new Error(`Can't recurse into non-iterable type: ${input}`);
  }
}

// tslint:disable-next-line:no-any
export function zipToList(x: any[]): DeepMapResult {
  if (x === null) {
    return null;
  }
  // TODO(soergel): validate array type?

  if (isIterable(x[0])) {
    return {value: null, recurse: true};
  } else {
    return {value: x, recurse: false};
  }
}

/**
 * A return value for an async map function for use with deepMapAndAwaitAll.
 *
 * If recurse is true, the value should be empty, and iteration will continue
 * into the object or array.
 */
export type DeepMapAsyncResult = {
  value: Promise<any>,
  recurse: boolean
};

/**
 * Apply an async mapping function to a nested structure in a recursive manner.
 *
 * This first creates a nested structure of Promises, and then awaits all of
 * those, resulting in a single Promise for a resolved nested structure.
 *
 * The result of the mapping is an object with the same nested structure (i.e.,
 * of arrays and dicts) as the input, except that some subtrees are replaced,
 * according to the results of the mapping function.
 *
 * Mappings are memoized.  Thus, if the nested structure contains the same
 * object in multiple positions, the output will contain the same mapped object
 * in those positions.  Cycles are not supported, however.
 *
 * @param input: The object to which to apply the mapping function.
 * @param mapFn: A function that expects a single node of the object tree, and
 *   returns a `DeepMapAsyncResult`.  The `DeepMapAsyncResult` either provides
 *   a `Promise` for a replacement value for that node (i.e., replacing the
 *   subtree), or indicates that the node should be processed recursively.  Note
 *   that the decision whether or not to recurse must be made immediately; only
 *   the mapped value may be promised.
 */
export async function deepMapAndAwaitAll(
    input: any, mapFn: (x: any) => DeepMapAsyncResult): Promise<any|any[]> {
  const seen: Map<any, Promise<any>> = new Map();

  // First do a normal deepMap, collecting Promises in 'seen' as a side effect.
  deepMapInternal(input, mapFn, seen);

  // Replace the Promises in 'seen' in place.
  // Note TypeScript provides no async map iteration, and regular map iteration
  // is broken too, so sadly we have to do Array.from() to make it work.
  // (There's no advantage to Promise.all(), and that would be tricky anyway.)
  for (const key of Array.from(seen.keys())) {
    const value = seen.get(key);
    if (tf.util.isPromise(value)) {
      const mappedValue = await value;
      seen.set(key, mappedValue);
    }
  }

  // Normal deepMap again, this time filling in the resolved values.
  // It's unfortunate that we have to do two passes.
  // TODO(soergel): test performance and think harder about a fast solution.
  const result = deepMapInternal(input, mapFn, seen);
  return result;
}

/**
 * Determine whether the argument is iterable.
 *
 * @returns true if the argument is an array or any non-Tensor object.
 */
// tslint:disable-next-line:no-any
export function isIterable(obj: any): boolean {
  return obj != null && (!ArrayBuffer.isView(obj)) &&
      (Array.isArray(obj) ||
       (typeof obj === 'object' && !(obj instanceof tf.Tensor)));
}

/**
 * Determine whether the argument can be converted to Tensor.
 *
 * Tensors, primitives, arrays, and TypedArrays all qualify; anything else does
 * not.
 *
 * @returns true if the argument can be converted to Tensor.
 */
// tslint:disable-next-line:no-any
export function canTensorify(obj: any): boolean {
  return obj == null || isPrimitive(obj) || Array.isArray(obj) ||
      (typeof obj === 'object' && (obj instanceof tf.Tensor)) ||
      tf.util.isTypedArray(obj);
}

/**
 * Returns true if the given `value` is a primitive type. Otherwise returns
 * false. This is equivalant to node util.isPrimitive
 */
function isPrimitive(value: any): boolean {
  return (
      value === null ||
      (typeof value !== 'object' && typeof value !== 'function'));
}
