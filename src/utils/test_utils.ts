/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

/**
 * Testing utilities.
 */

// tslint:disable:max-line-length
import {Tensor, test_util} from '@tensorflow/tfjs-core';
import * as jasmine_util from '@tensorflow/tfjs-core/dist/jasmine_util';
import {disposeScalarCache} from '../backend/tfjs_backend';
import {ValueError} from '../errors';

// tslint:enable:max-line-length

/**
 * Expect values are close between an Tensor or ConcreteTensor.
 * @param actual
 * @param expected
 */
export function expectTensorsClose(
    actual: Tensor|number[], expected: Tensor|number[], epsilon?: number) {
  if (actual == null) {
    throw new ValueError(
        'First argument to expectTensorsClose() is not defined.');
  }
  if (expected == null) {
    throw new ValueError(
        'Second argument to expectTensorsClose() is not defined.');
  }
  test_util.expectArraysClose(actual, expected, epsilon);
}

/**
 * Expect values in array are within a specified range, boundaries inclusive.
 * @param actual
 * @param expected
 */
export function expectTensorsValuesInRange(
    actual: Tensor, low: number, high: number) {
  if (actual == null) {
    throw new ValueError(
        'First argument to expectTensorsClose() is not defined.');
  }
  test_util.expectValuesInRange(actual.dataSync(), low, high);
}


/**
 * Describe tests to be run on CPU and GPU.
 * @param testName
 * @param tests
 */
export function describeMathCPUAndGPU(testName: string, tests: () => void) {
  describeMathCPU(testName, tests);
  describeMathGPU(testName, tests);
}

/**
 * Describe tests to be run on CPU only.
 * @param testName
 * @param tests
 */
export function describeMathCPU(testName: string, tests: () => void) {
  jasmine_util.describeWithFlags(testName, test_util.CPU_ENVS, () => {
    beforeEach(() => {
      disposeScalarCache();
    });
    tests();
  });
}

/**
 * Describe tests to be run on GPU only.
 * @param testName
 * @param tests
 */
export function describeMathGPU(testName: string, tests: () => void) {
  jasmine_util.describeWithFlags(testName, test_util.WEBGL_ENVS, () => {
    beforeEach(() => {
      disposeScalarCache();
    });
    tests();
  });
}
