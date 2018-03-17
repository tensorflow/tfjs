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
import {environment, Tensor, test_util} from '@tensorflow/tfjs-core';

import {setBackend} from '../backend/deeplearnjs_backend';
import {ValueError} from '../errors';

// tslint:enable:max-line-length

const webgl2Features: environment.Features[] = [{
  'BACKEND': 'webgl',
  'WEBGL_FLOAT_TEXTURE_ENABLED': true,
  'WEBGL_VERSION': 2
}];


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
  test_util.describeWithFlags(testName, test_util.CPU_ENVS, () => {
    beforeEach(() => {
      setBackend('cpu');
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
  test_util.describeWithFlags(testName, webgl2Features, () => {
    beforeEach(() => {
      setBackend('webgl');
    });
    tests();
  });
}
