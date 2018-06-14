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

import {ENV, Features} from './environment';
import {Tensor} from './tensor';
import {TypedArray} from './types';
import * as util from './util';

export const WEBGL_ENVS: Features[] = [
  {
    'BACKEND': 'test-webgl',
    'WEBGL_RENDER_FLOAT32_ENABLED': true,
    'WEBGL_DOWNLOAD_FLOAT_ENABLED': true,
    'WEBGL_VERSION': 1
  },
  {
    'BACKEND': 'test-webgl',
    'WEBGL_RENDER_FLOAT32_ENABLED': true,
    'WEBGL_DOWNLOAD_FLOAT_ENABLED': true,
    'WEBGL_VERSION': 2
  }
];
export const CPU_ENVS: Features[] = [{'BACKEND': 'test-cpu'}];
export const CHROME_CPU_ENVS: Features[] =
    [{'BACKEND': 'test-cpu', 'IS_CHROME': true}];

// Emulates the current device.
export const NATIVE_ENV: Features = {};
export const BROWSER_ENVS: Features[] = WEBGL_ENVS.concat(CPU_ENVS);
export const ALL_ENVS: Features[] = [NATIVE_ENV].concat(BROWSER_ENVS);

export function expectArraysClose(
    actual: Tensor|TypedArray|number[],
    expected: Tensor|TypedArray|number[]|boolean[], epsilon?: number) {
  if (epsilon == null) {
    epsilon = ENV.get('TEST_EPSILON');
  }
  if (!(actual instanceof Tensor) && !(expected instanceof Tensor)) {
    const aType = actual.constructor.name;
    const bType = expected.constructor.name;

    if (aType !== bType) {
      throw new Error(
          `Arrays are of different type actual: ${aType} ` +
          `vs expected: ${bType}`);
    }
  } else if (actual instanceof Tensor && expected instanceof Tensor) {
    if (actual.dtype !== expected.dtype) {
      throw new Error(
          `Arrays are of different type actual: ${actual.dtype} ` +
          `vs expected: ${expected.dtype}.`);
    }
    if (!util.arraysEqual(actual.shape, expected.shape)) {
      throw new Error(
          `Arrays are of different shape actual: ${actual.shape} ` +
          `vs expected: ${expected.shape}.`);
    }
  }

  let actualValues: TypedArray|number[];
  let expectedValues: TypedArray|number[]|boolean[];
  if (actual instanceof Tensor) {
    actualValues = actual.dataSync();
  } else {
    actualValues = actual;
  }
  if (expected instanceof Tensor) {
    expectedValues = expected.dataSync();
  } else {
    expectedValues = expected;
  }

  if (actualValues.length !== expectedValues.length) {
    throw new Error(
        `Arrays have different lengths actual: ${actualValues.length} vs ` +
        `expected: ${expectedValues.length}.\n` +
        `Actual:   ${actualValues}.\n` +
        `Expected: ${expectedValues}.`);
  }
  for (let i = 0; i < expectedValues.length; ++i) {
    const a = actualValues[i];
    const e = expectedValues[i];

    if (!areClose(a, Number(e), epsilon)) {
      throw new Error(
          `Arrays differ: actual[${i}] = ${a}, expected[${i}] = ${e}.\n` +
          `Actual:   ${actualValues}.\n` +
          `Expected: ${expectedValues}.`);
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

export function expectArraysEqual(
    actual: Tensor|TypedArray|number[],
    expected: Tensor|TypedArray|number[]|boolean[]) {
  return expectArraysClose(actual, expected, 0);
}

export function expectNumbersClose(a: number, e: number, epsilon?: number) {
  if (epsilon == null) {
    epsilon = ENV.get('TEST_EPSILON');
  }
  if (!areClose(a, e, epsilon)) {
    throw new Error(`Numbers differ: actual === ${a}, expected === ${e}`);
  }
}

function areClose(a: number, e: number, epsilon: number): boolean {
  if (isNaN(a) && isNaN(e)) {
    return true;
  }
  if (isNaN(a) || isNaN(e) || Math.abs(a - e) > epsilon) {
    return false;
  }
  return true;
}

export function expectValuesInRange(
    actual: Tensor|TypedArray|number[], low: number, high: number) {
  let actualVals: TypedArray|number[];
  if (actual instanceof Tensor) {
    actualVals = actual.dataSync();
  } else {
    actualVals = actual;
  }
  for (let i = 0; i < actualVals.length; i++) {
    if (actualVals[i] < low || actualVals[i] > high) {
      throw new Error(
          `Value out of range:${actualVals[i]} low: ${low}, high: ${high}`);
    }
  }
}

export function expectArrayBuffersEqual(
    actual: ArrayBuffer, expected: ArrayBuffer) {
  // Safari & Jasmine dont like comparing ArrayBuffers directly. Wrapping in
  // a Float32Array solves this issue.
  expect(new Float32Array(actual)).toEqual(new Float32Array(expected));
}
