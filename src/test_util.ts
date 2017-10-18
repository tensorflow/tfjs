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
import {Environment, Features} from './environment';
import {NDArrayMath} from './math/math';
import {NDArrayMathCPU} from './math/math_cpu';
import {NDArrayMathGPU} from './math/math_gpu';
import {TypedArray} from './util';

/** Accuracy for tests. */
// TODO(nsthorat || smilkov): Fix this low precision for byte-backed textures.
export const TEST_EPSILON = 1e-2;

export function expectArraysClose(
    actual: TypedArray, expected: TypedArray, epsilon = TEST_EPSILON) {
  const aType = actual.constructor.name;
  const bType = expected.constructor.name;

  if (aType !== bType) {
    throw new Error(`Arrays are of different type ${aType} vs ${bType}`);
  }
  if (actual.length !== expected.length) {
    throw new Error(
        'Matrices have different lengths (' + actual.length + ' vs ' +
        expected.length + ').');
  }
  for (let i = 0; i < expected.length; ++i) {
    const a = actual[i];
    const e = expected[i];

    if (!areClose(a, e, epsilon)) {
      const actualStr = 'actual[' + i + '] === ' + a;
      const expectedStr = 'expected[' + i + '] === ' + e;
      throw new Error('Arrays differ: ' + actualStr + ', ' + expectedStr);
    }
  }
}

export function expectNumbersClose(
    a: number, e: number, epsilon = TEST_EPSILON) {
  if (!areClose(a, e, epsilon)) {
    throw new Error('Numbers differ: actual === ' + a + ', expected === ' + e);
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

export function setValue(
    m: Float32Array, mNumRows: number, mNumCols: number, v: number, row: number,
    column: number) {
  if (row >= mNumRows) {
    throw new Error('row (' + row + ') must be in [0 ' + mNumRows + '].');
  }
  if (column >= mNumCols) {
    throw new Error('column (' + column + ') must be in [0 ' + mNumCols + '].');
  }
  m[(row * mNumCols) + column] = v;
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
  const testNameBase = 'math_cpu.' + name;
  describeWithFeaturesAndExecutor(
      testNameBase, tests as Tests[],
      (testName, tests, features) => executeMathTests(
          testName, tests, () => new NDArrayMathCPU(), features),
      featuresList);
}

export function describeMathGPU(
    name: string, tests: MathTests[], featuresList?: Features[]) {
  const testNameBase = 'math_gpu.' + name;
  describeWithFeaturesAndExecutor(
      testNameBase, tests as Tests[],
      (testName, tests, features) => executeMathTests(
          testName, tests, () => new NDArrayMathGPU(), features),
      featuresList);
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
  const customBeforeEach = () => {
    math = mathFactory();
    math.startScope();
  };
  const customAfterEach = () => {
    math.endScope(null);
    math.dispose();
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
