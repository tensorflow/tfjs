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

import * as environment from './environment';
import {ENV, Environment, Features} from './environment';
import {MathBackendCPU} from './math/backends/backend_cpu';
import {MathBackendWebGL} from './math/backends/backend_webgl';
import {NDArrayMath} from './math/math';
import * as util from './util';
import {DType, TypedArray} from './util';

/** Accuracy for tests. */
// TODO(nsthorat || smilkov): Fix this low precision for byte-backed textures.
export const TEST_EPSILON = 1e-2;

export function mean(values: TypedArray|number[]) {
  let sum = 0;
  for (let i = 0; i < values.length; i++) {
    sum += values[i];
  }
  return sum / values.length;
}

export function standardDeviation(values: TypedArray|number[], mean: number) {
  let squareDiffSum = 0;
  for (let i = 0; i < values.length; i++) {
    const diff = values[i] - mean;
    squareDiffSum += diff * diff;
  }
  return Math.sqrt(squareDiffSum / values.length);
}

export function kurtosis(values: TypedArray|number[]) {
  // https://en.wikipedia.org/wiki/Kurtosis
  const valuesMean = mean(values);
  const n = values.length;
  let sum2 = 0;
  let sum4 = 0;
  for (let i = 0; i < n; i++) {
    const v = values[i] - valuesMean;
    sum2 += Math.pow(v, 2);
    sum4 += Math.pow(v, 4);
  }
  return (1 / n) * sum4 / Math.pow((1 / n) * sum2, 2);
}

export function skewness(values: TypedArray|number[]) {
  // https://en.wikipedia.org/wiki/Skewness
  const valuesMean = mean(values);
  const n = values.length;
  let sum2 = 0;
  let sum3 = 0;
  for (let i = 0; i < n; i++) {
    const v = values[i] - valuesMean;
    sum2 += Math.pow(v, 2);
    sum3 += Math.pow(v, 3);
  }
  return (1 / n) * sum3 / Math.pow((1 / (n - 1)) * sum2, 3 / 2);
}

export function jarqueBeraNormalityTest(values: TypedArray|number[]) {
  // https://en.wikipedia.org/wiki/Jarque%E2%80%93Bera_test
  const n = values.length;
  const s = skewness(values);
  const k = kurtosis(values);
  const jb = n / 6 * (Math.pow(s, 2) + 0.25 * Math.pow(k - 3, 2));
  // JB test requires 2-degress of freedom from Chi-Square @ 0.95:
  // http://www.itl.nist.gov/div898/handbook/eda/section3/eda3674.htm
  const CHI_SQUARE_2DEG = 5.991;
  if (jb > CHI_SQUARE_2DEG) {
    throw new Error(`Invalid p-value for JB: ${jb}`);
  }
}

export function expectArrayInMeanStdRange(
    actual: TypedArray|number[], expectedMean: number, expectedStdDev: number,
    epsilon = TEST_EPSILON) {
  const actualMean = mean(actual);
  expectNumbersClose(actualMean, expectedMean, epsilon);
  expectNumbersClose(
      standardDeviation(actual, actualMean), expectedStdDev, epsilon);
}

export function expectArraysClose(
    actual: TypedArray|number[], expected: TypedArray|number[],
    epsilon = TEST_EPSILON) {
  const aType = actual.constructor.name;
  const bType = expected.constructor.name;

  if (aType !== bType) {
    throw new Error(`Arrays are of different type ${aType} vs ${bType}`);
  }
  if (actual.length !== expected.length) {
    throw new Error(
        `Matrices have different lengths (${actual.length} vs ` +
        `${expected.length}).`);
  }
  for (let i = 0; i < expected.length; ++i) {
    const a = actual[i];
    const e = expected[i];

    if (!areClose(a, e, epsilon)) {
      const actualStr = `actual[${i}] === ${a}`;
      const expectedStr = `expected[${i}] === ${e}`;
      throw new Error('Arrays differ: ' + actualStr + ', ' + expectedStr);
    }
  }
}

export function expectNumbersClose(
    a: number, e: number, epsilon = TEST_EPSILON) {
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
    actual: TypedArray|number[], low: number, high: number) {
  for (let i = 0; i < actual.length; i++) {
    if (actual[i] < low || actual[i] > high) {
      throw new Error(
          `Value out of range:${actual[i]} low: ${low}, high: ${high}`);
    }
  }
}

export function randomArrayInRange(
    n: number, minValue: number, maxValue: number): Float32Array {
  const v = new Float32Array(n);
  const range = maxValue - minValue;
  for (let i = 0; i < n; ++i) {
    v[i] = (Math.random() * range) + minValue;
  }
  return v;
}

export function makeIdentity(n: number): Float32Array {
  const i = new Float32Array(n * n);
  for (let j = 0; j < n; ++j) {
    i[(j * n) + j] = 1;
  }
  return i;
}

export function cpuMultiplyMatrix(
    a: Float32Array, aRow: number, aCol: number, b: Float32Array, bRow: number,
    bCol: number) {
  const result = new Float32Array(aRow * bCol);
  for (let r = 0; r < aRow; ++r) {
    const aOffset = (r * aCol);
    const cOffset = (r * bCol);
    for (let c = 0; c < bCol; ++c) {
      let d = 0;
      for (let k = 0; k < aCol; ++k) {
        d += a[aOffset + k] * b[(k * bCol) + c];
      }
      result[cOffset + c] = d;
    }
  }
  return result;
}

export function cpuDotProduct(a: Float32Array, b: Float32Array): number {
  if (a.length !== b.length) {
    throw new Error('cpuDotProduct: incompatible vectors.');
  }
  let d = 0;
  for (let i = 0; i < a.length; ++i) {
    d += a[i] * b[i];
  }
  return d;
}

export type MathTests =
    (it: (name: string, testFn: (math: NDArrayMath) => void) => void) => void;
export type Tests = (it: (name: string, testFn: () => void) => void) => void;

export function describeMathCPU(
    name: string, tests: MathTests[], featuresList?: Features[]) {
  const testNameBase = 'CPU: math.' + name;
  describeWithFeaturesAndExecutor(
      testNameBase, tests as Tests[],
      (testName, tests, features) => executeMathTests(testName, tests, () => {
        const safeMode = true;
        return new NDArrayMath(new MathBackendCPU(), safeMode);
      }, features), featuresList);
}

export function describeMathGPU(
    name: string, tests: MathTests[], featuresList?: Features[]) {
  const testNameBase = 'WebGL: math.' + name;
  describeWithFeaturesAndExecutor(
      testNameBase, tests as Tests[],
      (testName, tests, features) => executeMathTests(testName, tests, () => {
        const safeMode = true;
        return new NDArrayMath(new MathBackendWebGL(), safeMode);
      }, features), featuresList);
}

export function describeCustom(
    name: string, tests: Tests, featuresList?: Features[],
    customBeforeEach?: () => void, customAfterEach?: () => void) {
  describeWithFeaturesAndExecutor(
      name, [tests],
      (testName, tests, features) => executeTests(
          testName, tests, features, customBeforeEach, customAfterEach),
      featuresList);
}

type TestExecutor = (testName: string, tests: Tests[], features?: Features) =>
    void;
function describeWithFeaturesAndExecutor(
    testNameBase: string, tests: Tests[], executor: TestExecutor,
    featuresList?: Features[]) {
  if (featuresList != null) {
    featuresList.forEach(features => {
      const testName = testNameBase + ' ' + JSON.stringify(features);
      executor(testName, tests, features);
    });
  } else {
    executor(testNameBase, tests);
  }
}

// A wrapper around it() that calls done automatically if the function returns
// a Promise, aka if it's an async/await function.
const PROMISE_IT = (name: string, testFunc: () => void|Promise<void>) => {
  it(name, (done: DoneFn) => {
    const result = testFunc();
    if (result instanceof Promise) {
      result.then(done, e => {
        fail(e);
        done();
      });
    } else {
      done();
    }
  });
};

export function executeMathTests(
    testName: string, tests: MathTests[], mathFactory: () => NDArrayMath,
    features?: Features) {
  let math: NDArrayMath;
  let oldMath: NDArrayMath;

  const customBeforeEach = () => {
    oldMath = ENV.math;
    math = mathFactory();
    math.startScope();
  };
  const customAfterEach = () => {
    math.endScope(null);
    math.dispose();
    ENV.setGlobalMath(oldMath);
  };
  const customIt =
      (name: string, testFunc: (math: NDArrayMath) => void|Promise<void>) => {
        PROMISE_IT(name, () => testFunc(math));
      };

  executeTests(
      testName, tests as Tests[], features, customBeforeEach, customAfterEach,
      customIt);
}

export function executeTests(
    testName: string, tests: Tests[], features?: Features,
    customBeforeEach?: () => void, customAfterEach?: () => void,
    customIt: (expectation: string, testFunc: () => void|Promise<void>) =>
        void = PROMISE_IT) {
  describe(testName, () => {
    beforeEach(() => {
      if (features != null) {
        environment.setEnvironment(new Environment(features));
      }

      if (customBeforeEach != null) {
        customBeforeEach();
      }
    });

    afterEach(() => {
      if (customAfterEach != null) {
        customAfterEach();
      }

      if (features != null) {
        environment.setEnvironment(new Environment());
      }
    });

    tests.forEach(test => test(customIt));
  });
}

export function assertIsNan(val: number, dtype: DType) {
  if (!util.isValNaN(val, dtype)) {
    throw new Error(`Value ${val} does not represent NaN for dtype ${dtype}`);
  }
}
