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

import {ENGINE} from './engine';
import {inferShape} from './tensor_util_env';
import {RecursiveArray, TensorLike, TypedArray} from './types';
import {arraysEqual, flatten, isString, isTypedArray} from './util';

const TEST_EPSILON_FLOAT32 = 1e-3;
export const TEST_EPSILON_FLOAT16 = 1e-1;

export function expectArraysClose(
    actual: TypedArray|number|RecursiveArray<number>,
    expected: TypedArray|number|RecursiveArray<number>, epsilon?: number) {
  if (epsilon == null) {
    epsilon = testEpsilon();
  }
  return expectArraysPredicate(
      actual, expected, (a, b) => areClose(a as number, b as number, epsilon));
}

export function testEpsilon() {
  return ENGINE.backend.floatPrecision() === 32 ? TEST_EPSILON_FLOAT32 :
                                                  TEST_EPSILON_FLOAT16;
}

function expectArraysPredicate(
    actual: TensorLike, expected: TensorLike,
    predicate: (a: {}, b: {}) => boolean) {
  let checkClassType = true;
  if (isTypedArray(actual) || isTypedArray(expected)) {
    checkClassType = false;
  }
  if (isTypedArray(actual) && isTypedArray(expected)) {
    checkClassType = true;
  }
  if (checkClassType) {
    const aType = actual.constructor.name;
    const bType = expected.constructor.name;

    if (aType !== bType) {
      throw new Error(
          `Arrays are of different type. Actual: ${aType}. ` +
          `Expected: ${bType}`);
    }
  }

  if (Array.isArray(actual) && Array.isArray(expected)) {
    const actualShape = inferShape(actual);
    const expectedShape = inferShape(expected);
    if (!arraysEqual(actualShape, expectedShape)) {
      throw new Error(
          `Arrays have different shapes. ` +
          `Actual: [${actualShape}]. Expected: [${expectedShape}]`);
    }
  }

  const actualFlat =
      isTypedArray(actual) ? actual : flatten(actual as RecursiveArray<number>);
  const expectedFlat = isTypedArray(expected) ?
      expected :
      flatten(expected as RecursiveArray<number>);

  if (actualFlat.length !== expectedFlat.length) {
    throw new Error(
        `Arrays have different lengths actual: ${actualFlat.length} vs ` +
        `expected: ${expectedFlat.length}.\n` +
        `Actual:   ${actualFlat}.\n` +
        `Expected: ${expectedFlat}.`);
  }
  for (let i = 0; i < expectedFlat.length; ++i) {
    const a = actualFlat[i];
    const e = expectedFlat[i];

    if (!predicate(a, e)) {
      throw new Error(
          `Arrays differ: actual[${i}] = ${a}, expected[${i}] = ${e}.\n` +
          `Actual:   ${actualFlat}.\n` +
          `Expected: ${expectedFlat}.`);
    }
  }
}

export interface DoneFn {
  (): void;
  fail: (message?: Error|string) => void;
}

export function expectPromiseToFail(fn: () => Promise<{}>, done: DoneFn): void {
  fn().then(() => done.fail(), () => done());
}

export function expectArraysEqual(actual: TensorLike, expected: TensorLike) {
  const exp = typeof expected === 'string' || typeof expected === 'number' ||
          typeof expected === 'boolean' ?
      [expected] as number[] :
      expected as number[];
  if (isString(actual) || isString((actual as string[])[0]) ||
      isString(expected) || isString((expected as string[])[0])) {
    // tslint:disable-next-line: triple-equals
    return expectArraysPredicate(actual, exp, (a, b) => a == b);
  }
  return expectArraysPredicate(
      actual, expected, (a, b) => areClose(a as number, b as number, 0));
}

export function expectNumbersClose(a: number, e: number, epsilon?: number) {
  if (epsilon == null) {
    epsilon = testEpsilon();
  }
  if (!areClose(a, e, epsilon)) {
    throw new Error(`Numbers differ: actual === ${a}, expected === ${e}`);
  }
}

function areClose(a: number, e: number, epsilon: number): boolean {
  if (!isFinite(a) && !isFinite(e)) {
    return true;
  }
  if (isNaN(a) || isNaN(e) || Math.abs(a - e) > epsilon) {
    return false;
  }
  return true;
}

export function expectValuesInRange(
    actual: TypedArray|number[], low: number, high: number) {
  for (let i = 0; i < actual.length; i++) {
    if (actual[i] < low || actual[i] > high) {
      throw new Error(
          `Value out of range:${actual[i]} low: ${low}, high: ${high}`);
    }
  }
}

export function expectArrayBuffersEqual(
    actual: ArrayBuffer, expected: ArrayBuffer) {
  // Safari & Jasmine don't like comparing ArrayBuffers directly. Wrapping in
  // a Float32Array solves this issue.
  expect(new Float32Array(actual)).toEqual(new Float32Array(expected));
}
