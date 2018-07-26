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

import {ArrayData, DataType, DataTypeMap, FlatVector, RecursiveArray, RegularArray, TensorLike, TypedArray} from './types';

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

export function assert(expr: boolean, msg: string|(() => string)) {
  if (!expr) {
    throw new Error(typeof msg === 'string' ? msg : msg());
  }
}

export function assertShapesMatch(
    shapeA: number[], shapeB: number[], errorMessagePrefix = ''): void {
  assert(
      arraysEqual(shapeA, shapeB),
      errorMessagePrefix + ` Shapes ${shapeA} and ${shapeB} must match`);
}

export function assertNonNull(a: TensorLike): void {
  assert(
      a != null,
      `The input to the tensor constructor must be a non-null value.`);
}

// NOTE: We explicitly type out what T extends instead of any so that
// util.flatten on a nested array of number doesn't try to infer T as a
// number[][], causing us to explicitly type util.flatten<number>().
export function flatten<T extends number|boolean|Promise<number>>(
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
  let firstElem: typeof val = val;

  if (isTypedArray(val)) {
    return [(val as TypedArray).length];
  }
  if (!Array.isArray(val)) {
    return [];  // Scalar.
  }
  const shape: number[] = [];

  while (firstElem instanceof Array) {
    shape.push(firstElem.length);
    firstElem = firstElem[0];
  }
  if (val instanceof Array) {
    deepAssertShapeConsistency(val, shape, []);
  }
  return shape;
}

function deepAssertShapeConsistency(
    val: number|boolean|RegularArray<number>|RegularArray<boolean>,
    shape: number[], indices?: number[]) {
  indices = indices || [];
  if (!(val instanceof Array)) {
    assert(
        shape.length === 0,
        () => `Element arr[${indices.join('][')}] is a primitive, ` +
            `but should be an array of ${shape[0]} elements`);
    return;
  }
  assert(
      shape.length > 0,
      () => `Element arr[${indices.join('][')}] should be a primitive, ` +
          `but is an array of ${val.length} elements`);
  assert(
      val.length === shape[0],
      () => `Element arr[${indices.join('][')}] should have ${shape[0]} ` +
          `elements, but has ${val.length} elements`);
  const subShape = shape.slice(1);
  for (let i = 0; i < val.length; ++i) {
    deepAssertShapeConsistency(val[i], subShape, indices.concat(i));
  }
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

    tryFn();
  });
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
            `Found - 1 at dim ${implicitIdx} and dim ${i}`);
      }
      implicitIdx = i;
    } else if (shape[i] <= 0) {
      throw Error(`Shapes can not be <= 0. Found ${shape[i]} at dim ${i}`);
    }
  }

  if (implicitIdx === -1) {
    if (size > 0 && size !== shapeProd) {
      throw Error(`Size(${size}) must match the product of shape ${shape}`);
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

/** Reduces the shape by removing all dimensions of shape 1. */
export function squeezeShape(shape: number[], axis?: number[]):
    {newShape: number[], keptDims: number[]} {
  const newShape: number[] = [];
  const keptDims: number[] = [];
  let j = 0;
  for (let i = 0; i < shape.length; ++i) {
    if (axis != null) {
      if (axis[j] === i && shape[i] > 1) {
        throw new Error(
            `Can't squeeze axis ${i} since its dim '${shape[i]}' is not 1`);
      }
      if ((axis[j] == null || axis[j] > i) && shape[i] === 1) {
        newShape.push(shape[i]);
        keptDims.push(i);
      }
      if (axis[j] <= i) {
        j++;
      }
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

export function checkComputationForNaN<D extends DataType>(
    vals: DataTypeMap[D], dtype: D, name: string): void {
  if (dtype !== 'float32') {
    // Only floating point computations will generate NaN values
    return;
  }
  for (let i = 0; i < vals.length; i++) {
    if (isNaN(vals[i])) {
      throw Error(`The result of the '${name}' has NaNs.`);
    }
  }
}

export function checkConversionForNaN<D extends DataType>(
    vals: DataTypeMap[D]|number[], dtype: D): void {
  if (dtype === 'float32') {
    // NaN is valid for floating point conversions
    return;
  }

  for (let i = 0; i < vals.length; i++) {
    if (isNaN(vals[i])) {
      throw Error(`NaN is not a valid value for dtype: '${dtype}'.`);
    }
  }
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

function copyTypedArray<D extends DataType>(
    array: DataTypeMap[D]|number[]|boolean[], dtype: D,
    debugMode: boolean): DataTypeMap[D] {
  if (dtype == null || dtype === 'float32') {
    return new Float32Array(array as number[]);
  } else if (dtype === 'int32') {
    if (debugMode) {
      checkConversionForNaN(array as number[], dtype);
    }
    return new Int32Array(array as number[]);
  } else if (dtype === 'bool') {
    const bool = new Uint8Array(array.length);
    for (let i = 0; i < bool.length; ++i) {
      if (Math.round(array[i] as number) !== 0) {
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

export function isFunction(f: Function) {
  return !!(f && f.constructor && f.call && f.apply);
}

export function nearestDivisor(size: number, start: number): number {
  for (let i = start; i < size; ++i) {
    if (size % i === 0) {
      return i;
    }
  }
  return size;
}

export function computeStrides(shape: number[]): number[] {
  const rank = shape.length;
  if (rank < 2) {
    return [];
  }

  // Last dimension has implicit stride of 1, thus having D-1 (instead of D)
  // strides.
  const strides = new Array(rank - 1);
  strides[rank - 2] = shape[rank - 1];
  for (let i = rank - 3; i >= 0; --i) {
    strides[i] = strides[i + 1] * shape[i + 1];
  }
  return strides;
}

export function toTypedArray<D extends DataType>(
    a: ArrayData<D>, dtype: D, debugMode: boolean): DataTypeMap[D] {
  if (noConversionNeeded(a, dtype)) {
    return a as DataTypeMap[D];
  }
  if (Array.isArray(a)) {
    a = flatten(a as number[]);
  }
  return copyTypedArray(a, dtype, debugMode);
}

function noConversionNeeded<D extends DataType>(
    a: ArrayData<D>, dtype: D): boolean {
  return (a instanceof Float32Array && dtype === 'float32') ||
      (a instanceof Int32Array && dtype === 'int32') ||
      (a instanceof Uint8Array && dtype === 'bool');
}

export function makeOnesTypedArray<D extends DataType>(
    size: number, dtype: D): DataTypeMap[D] {
  const array = makeZerosTypedArray(size, dtype);
  for (let i = 0; i < array.length; i++) {
    array[i] = 1;
  }
  return array;
}

export function makeZerosTypedArray<D extends DataType>(
    size: number, dtype: D): DataTypeMap[D] {
  if (dtype == null || dtype === 'float32') {
    return new Float32Array(size);
  } else if (dtype === 'int32') {
    return new Int32Array(size);
  } else if (dtype === 'bool') {
    return new Uint8Array(size);
  } else {
    throw new Error(`Unknown data type ${dtype}`);
  }
}

/**
 * Returns the current high-resolution real time in milliseconds. It is
 * relative to an arbitrary time in the past.
 */
export function now(): number {
  if (typeof performance !== 'undefined') {
    return performance.now();
  } else if (typeof process !== 'undefined') {
    const time = process.hrtime();
    return time[0] * 1000 + time[1] / 1000000;
  } else {
    throw new Error(
        'Can not measure time in this environment. You should run tf.js ' +
        'in the browser or in Node.js');
  }
}
