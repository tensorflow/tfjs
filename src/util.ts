/**
 * @license
 * Copyright 2017 Google Inc. All Rights Reserved.
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
// tslint:disable-next-line:max-line-length
import {DataType, DataTypeMap, FlatVector, NamedTensorMap, RecursiveArray, RegularArray, TypedArray} from './types';
/** Shuffles the array using Fisher-Yates algorithm. */
// tslint:disable-next-line:no-any
export function shuffle(array: any[]|Uint32Array|Int32Array|
                        Float32Array): void {
  let counter = array.length;
  let temp = 0;
  let index = 0;
  // While there are elements in the array
  while (counter > 0) {
    // Pick a random index
    index = (Math.random() * counter) | 0;
    // Decrease counter by 1
    counter--;
    // And swap the last element with it
    temp = array[counter];
    array[counter] = array[index];
    array[index] = temp;
  }
}

/** Clamps a value to a specified range. */
export function clamp(min: number, x: number, max: number): number {
  return Math.max(min, Math.min(x, max));
}

/** Returns a sample from a uniform [a, b] distribution. */
export function randUniform(a: number, b: number) {
  return Math.random() * (b - a) + a;
}

/** Returns squared eucledian distance between two vectors. */
export function distSquared(a: FlatVector, b: FlatVector): number {
  let result = 0;
  for (let i = 0; i < a.length; i++) {
    const diff = Number(a[i]) - Number(b[i]);
    result += diff * diff;
  }
  return result;
}

export function assert(expr: boolean, msg: string) {
  if (!expr) {
    throw new Error(msg);
  }
}

export function assertShapesMatch(
    shapeA: number[], shapeB: number[], errorMessagePrefix = ''): void {
  assert(
      arraysEqual(shapeA, shapeB),
      errorMessagePrefix + `Shapes ${shapeA} and ${shapeB} must match`);
}

export function assertTypesMatch(a: Tensor, b: Tensor): void {
  assert(
      a.dtype === b.dtype,
      `The dtypes of the first (${a.dtype}) and ` +
          `second (${b.dtype}) input must match`);
}

// NOTE: We explicitly type out what T extends instead of any so that
// util.flatten on a nested array of number doesn't try to infer T as a
// number[][], causing us to explicitly type util.flatten<number>().
export function flatten<T extends number|boolean|Tensor|Promise<number>>(
    arr: T|RecursiveArray<T>, ret: T[] = []): T[] {
  if (Array.isArray(arr)) {
    for (let i = 0; i < arr.length; ++i) {
      flatten(arr[i], ret);
    }
  } else {
    ret.push(arr as T);
  }
  return ret;
}

export function inferShape(val: TypedArray|number|boolean|RegularArray<number>|
                           RegularArray<boolean>): number[] {
  if (isTypedArray(val)) {
    return [(val as TypedArray).length];
  }
  if (!Array.isArray(val)) {
    return [];  // Scalar.
  }
  const shape: number[] = [];
  while (val instanceof Array) {
    shape.push(val.length);
    val = val[0];
  }
  return shape;
}

export function sizeFromShape(shape: number[]): number {
  if (shape.length === 0) {
    // Scalar.
    return 1;
  }
  let size = shape[0];
  for (let i = 1; i < shape.length; i++) {
    size *= shape[i];
  }
  return size;
}

export function isScalarShape(shape: number[]): boolean {
  return shape.length === 0;
}

export function arraysEqual(n1: FlatVector, n2: FlatVector) {
  if (n1.length !== n2.length) {
    return false;
  }
  for (let i = 0; i < n1.length; i++) {
    if (n1[i] !== n2[i]) {
      return false;
    }
  }
  return true;
}

export function isInt(a: number): boolean {
  return a % 1 === 0;
}

export function tanh(x: number): number {
  // tslint:disable-next-line:no-any
  if ((Math as any).tanh != null) {
    // tslint:disable-next-line:no-any
    return (Math as any).tanh(x);
  }
  if (x === Infinity) {
    return 1;
  } else if (x === -Infinity) {
    return -1;
  } else {
    const e2x = Math.exp(2 * x);
    return (e2x - 1) / (e2x + 1);
  }
}

export function sizeToSquarishShape(size: number): [number, number] {
  for (let a = Math.floor(Math.sqrt(size)); a > 1; --a) {
    if (size % a === 0) {
      return [a, size / a];
    }
  }
  return [1, size];
}

export function createShuffledIndices(n: number): Uint32Array {
  const shuffledIndices = new Uint32Array(n);
  for (let i = 0; i < n; ++i) {
    shuffledIndices[i] = i;
  }
  shuffle(shuffledIndices);
  return shuffledIndices;
}

export function rightPad(a: string, size: number): string {
  if (size <= a.length) {
    return a;
  }
  return a + ' '.repeat(size - a.length);
}

export function repeatedTry(
    checkFn: () => boolean, delayFn = (counter: number) => 0,
    maxCounter?: number): Promise<void> {
  return new Promise<void>((resolve, reject) => {
    let tryCount = 0;

    const tryFn = () => {
      if (checkFn()) {
        resolve();
        return;
      }

      tryCount++;

      const nextBackoff = delayFn(tryCount);

      if (maxCounter != null && tryCount >= maxCounter) {
        reject();
        return;
      }
      setTimeout(tryFn, nextBackoff);
    };

    setTimeout(tryFn, 0);
  });
}

export function getQueryParams(queryString: string): {[key: string]: string} {
  const params = {};
  queryString.replace(/[?&]([^=?&]+)(?:=([^&]*))?/g, (s, ...t) => {
    decodeParam(params, t[0], t[1]);
    return t.join('=');
  });
  return params;
}

function decodeParam(
    params: {[key: string]: string}, name: string, value?: string) {
  params[decodeURIComponent(name)] = decodeURIComponent(value || '');
}

/**
 * Given the full size of the array and a shape that may contain -1 as the
 * implicit dimension, returns the inferred shape where -1 is replaced.
 * E.g. For shape=[2, -1, 3] and size=24, it will return [2, 4, 3].
 *
 * @param shape The shape, which may contain -1 in some dimension.
 * @param size The full size (number of elements) of the array.
 * @return The inferred shape where -1 is replaced with the inferred size.
 */
export function inferFromImplicitShape(
    shape: number[], size: number): number[] {
  let shapeProd = 1;
  let implicitIdx = -1;

  for (let i = 0; i < shape.length; ++i) {
    if (shape[i] > 0) {
      shapeProd *= shape[i];
    } else if (shape[i] === -1) {
      if (implicitIdx !== -1) {
        throw Error(
            `Shapes can only have 1 implicit size. ` +
            `Found -1 at dim ${implicitIdx} and dim ${i}`);
      }
      implicitIdx = i;
    } else if (shape[i] <= 0) {
      throw Error(`Shapes can not be <= 0. Found ${shape[i]} at dim ${i}`);
    }
  }

  if (implicitIdx === -1) {
    if (size > 0 && size !== shapeProd) {
      throw Error(`Size (${size}) must match the product of shape ${shape}`);
    }
    return shape;
  }

  if (size % shapeProd !== 0) {
    throw Error(
        `The implicit shape can't be a fractional number. ` +
        `Got ${size} / ${shapeProd}`);
  }

  const newShape = shape.slice();
  newShape[implicitIdx] = size / shapeProd;
  return newShape;
}

export const NAN_INT32 = 1 << 31;
export const NAN_BOOL = 255;
export const NAN_FLOAT32 = NaN;

export function getNaN(dtype: DataType): number {
  if (dtype === 'float32') {
    return NAN_FLOAT32;
  } else if (dtype === 'int32') {
    return NAN_INT32;
  } else if (dtype === 'bool') {
    return NAN_BOOL;
  } else {
    throw new Error(`Unknown dtype ${dtype}`);
  }
}

export function isValNaN(val: number, dtype: DataType): boolean {
  if (isNaN(val)) {
    return true;
  }
  if (dtype === 'float32') {
    return false;
  } else if (dtype === 'int32') {
    return val === NAN_INT32;
  } else if (dtype === 'bool') {
    return val === NAN_BOOL;
  } else {
    throw new Error(`Unknown dtype ${dtype}`);
  }
}

/** Reduces the shape by removing all dimensions of shape 1. */
export function squeezeShape(shape: number[], axis?: number[]):
    {newShape: number[], keptDims: number[]} {
  const newShape: number[] = [];
  const keptDims: number[] = [];
  let j = 0;
  for (let i = 0; i < shape.length; ++i) {
    if (axis !== undefined) {
      if (axis[j] === i && shape[i] > 1) {
        throw new Error(`axis ${i} is not 1`);
      }
      if ((axis[j] === undefined || axis[j] > i) && shape[i] === 1) {
        newShape.push(shape[i]);
        keptDims.push(i);
      }
      if (axis[j] <= i) j++;
    }
    if (shape[i] > 1) {
      newShape.push(shape[i]);
      keptDims.push(i);
    }
  }
  return {newShape, keptDims};
}

export function getTypedArrayFromDType<D extends DataType>(
    dtype: D, size: number): DataTypeMap[D] {
  let values = null;
  if (dtype == null || dtype === 'float32') {
    values = new Float32Array(size);
  } else if (dtype === 'int32') {
    values = new Int32Array(size);
  } else if (dtype === 'bool') {
    values = new Uint8Array(size);
  } else {
    throw new Error(`Unknown data type ${dtype}`);
  }
  return values;
}

export function isTensorInList(tensor: Tensor, tensorList: Tensor[]): boolean {
  for (let i = 0; i < tensorList.length; i++) {
    if (tensorList[i].id === tensor.id) {
      return true;
    }
  }
  return false;
}

export function checkForNaN<D extends DataType>(
    vals: DataTypeMap[D], dtype: D, name: string): void {
  for (let i = 0; i < vals.length; i++) {
    if (isValNaN(vals[i], dtype)) {
      throw Error(`The result of the '${name}' has NaNs.`);
    }
  }
}

export function flattenNameArrayMap(
    nameArrayMap: Tensor|NamedTensorMap, keys?: string[]): Tensor[] {
  const xs: Tensor[] = [];
  if (nameArrayMap instanceof Tensor) {
    xs.push(nameArrayMap);
  } else {
    const xMap = nameArrayMap as {[xName: string]: Tensor};
    for (let i = 0; i < keys.length; i++) {
      xs.push(xMap[keys[i]]);
    }
  }
  return xs;
}

export function unflattenToNameArrayMap(
    keys: string[], flatArrays: Tensor[]): NamedTensorMap {
  if (keys.length !== flatArrays.length) {
    throw new Error(
        `Cannot unflatten Tensor[], keys and arrays are not of same length.`);
  }
  const result: NamedTensorMap = {};
  for (let i = 0; i < keys.length; i++) {
    result[keys[i]] = flatArrays[i];
  }
  return result;
}

/**
 * Returns true if the new type can't encode the old type without loss of
 * precision.
 */
export function hasEncodingLoss(oldType: DataType, newType: DataType): boolean {
  if (newType === 'float32') {
    return false;
  }
  if (newType === 'int32' && oldType !== 'float32') {
    return false;
  }
  if (newType === 'bool' && oldType === 'bool') {
    return false;
  }
  return true;
}

/**
 * Returns a promise that resolve when a requestAnimationFrame has completed.
 * This is simply a sugar method so that users can do the following:
 * `await dl.nextFrame();`
 */
export function nextFrame(): Promise<void> {
  return new Promise<void>(resolve => requestAnimationFrame(() => resolve()));
}

export function copyTypedArray<D extends DataType>(
    array: DataTypeMap[D]|number[]|boolean[], dtype: D): DataTypeMap[D] {
  if (dtype == null || dtype === 'float32') {
    return new Float32Array(array as number[]);
  } else if (dtype === 'int32') {
    const vals = new Int32Array(array.length);
    for (let i = 0; i < vals.length; ++i) {
      const val = array[i] as number;
      if (isValNaN(val, 'int32')) {
        vals[i] = getNaN('int32');
      } else {
        vals[i] = val;
      }
    }
    return vals;
  } else if (dtype === 'bool') {
    const bool = new Uint8Array(array.length);
    for (let i = 0; i < bool.length; ++i) {
      const val = array[i] as number;
      if (isValNaN(val as number, 'bool')) {
        bool[i] = getNaN('bool');
      } else if (Math.round(val) !== 0) {
        bool[i] = 1;
      }
    }
    return bool;
  } else {
    throw new Error(`Unknown data type ${dtype}`);
  }
}

export function isTypedArray(a: TypedArray|number|boolean|RegularArray<number>|
                             RegularArray<boolean>): boolean {
  return a instanceof Float32Array || a instanceof Int32Array ||
      a instanceof Uint8Array;
}

export function bytesPerElement(dtype: DataType): number {
  if (dtype === 'float32' || dtype === 'int32') {
    return 4;
  } else if (dtype === 'bool') {
    return 1;
  } else {
    throw new Error(`Unknown dtype ${dtype}`);
  }
}
