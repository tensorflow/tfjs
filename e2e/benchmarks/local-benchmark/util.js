/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
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

async function convertTensorToData(tensor) {
  const data = await tensor.data();
  tensor.dispose();
  return data;
}

async function getPredictionData(output) {
  if (output instanceof Promise) {
    output = await output;
  }

  if (output instanceof tf.Tensor) {
    output = await convertTensorToData(output);
  } else if (Array.isArray(output)) {
    for (let i = 0; i < output.length; i++) {
      if (output[i] instanceof tf.Tensor) {
        output[i] = await convertTensorToData(output[i]);
      }
    }
  } else if (output != null && typeof output === 'object') {
    for (const property in output) {
      if (output[property] instanceof tf.Tensor) {
        output[property] = await convertTensorToData(output[property]);
      }
    }
  }
  return output;
}

function printTime(elapsed) {
  return elapsed.toFixed(1) + ' ms';
}

function printMemory(bytes) {
  if (bytes < 1024) {
    return bytes + ' B';
  } else if (bytes < 1024 * 1024) {
    return (bytes / 1024).toFixed(2) + ' KB';
  } else {
    return (bytes / (1024 * 1024)).toFixed(2) + ' MB';
  }
}

function sleep(timeMs) {
  return new Promise(resolve => setTimeout(resolve, timeMs));
}

function queryTimerIsEnabled() {
  return _tfengine.ENV.getNumber(
             'WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION') > 0;
}

function areClose(a, e, epsilon) {
  if (!isFinite(a) && !isFinite(e)) {
    return true;
  }
  if (isNaN(a) || isNaN(e) || Math.abs(a - e) > epsilon) {
    return false;
  }
  return true;
}

function expectObjectsPredicate(actual, expected, epsilon, predicate) {
  let actualKeys = Object.getOwnPropertyNames(actual);
  let expectedKeys = Object.getOwnPropertyNames(expected);
  if (actualKeys.length != expectedKeys.length) {
    throw new Error(`Actual length ${
        actualKeys.length} not equal Expected length ${expectedKeys.length}`);
  }
  for (let i = 0; i < actualKeys.length; i++) {
    let key = actualKeys[i];
    let isObject = typeof (actual[key]) === 'object' &&
        typeof (expected[key]) === 'object';
    let isArray = tf.util.isTypedArray(actual[key]) &&
        tf.util.isTypedArray(expected[key]);
    if (isArray) {
      expectArraysClose(actual[key], expected[key], epsilon, key);
    } else if (isObject) {
      expectObjectsPredicate(actual[key], expected[key], epsilon, predicate);
    } else {
      if (!predicate(actual[key], expected[key])) {
        throw new Error(`Objects differ: actual[${key}] = ${
            JSON.stringify(actual[key])}, expected[${key}] = ${
            JSON.stringify(expected[key])}!`);
      }
    }
  }
  return true;
}

function expectObjectsClose(actual, expected, epsilon) {
  if (epsilon == null) {
    epsilon = tf.test_util.testEpsilon();
  }
  expectObjectsPredicate(
      actual, expected, epsilon, (a, b) => areClose(a, b, epsilon));
}

function expectArraysPredicateFuzzy(actual, expected, predicate, errorRate) {
  if (tf.util.isTypedArray(actual) == false ||
      tf.util.isTypedArray(expected) == false) {
    throw new Error(`Actual and Expected are not arrays.`);
  }

  if (actual.length !== expected.length) {
    throw new Error(
        `Arrays have different lengths actual: ${actual.length} vs ` +
        `expected: ${expected.length}.\n` +
        `Actual:   ${actual}.\n` +
        `Expected: ${expected}.`);
  }
  let mismatchCount = 0;
  for (let i = 0; i < expected.length; ++i) {
    const a = actual[i];
    const e = expected[i];
    if (!predicate(a, e)) {
      mismatchCount++;
      const maxMismatch = Math.floor(errorRate * expected.length);
      if (mismatchCount > maxMismatch) {
        throw new Error(
            `Arrays data has more than ${maxMismatch} differs from ${
                expected.length}: actual[${i}] = ${a}, expected[${i}] = ${
                e}.\n` +
            `Actual:   ${actual}.\n` +
            `Expected: ${expected}.`);
      }
    }
  }
}

function expectArraysClose(actual, expected, epsilon, key) {
  if (epsilon == null) {
    epsilon = tf.test_util.testEpsilon();
  }

  if (key == 'data') {
    // For bodypix, the value in data memeber means "1 for the pixels that are
    // part of the person, and 0 otherwise".
    // So for these models, we don't expect all data is exactly match. Default
    // use error rate 0.001 (1/1000).
    const ERROR_RATE = 0.001;
    return expectArraysPredicateFuzzy(
        actual, expected, (a, b) => areClose(a, b, epsilon), ERROR_RATE);
  } else {
    return tf.test_util.expectArraysClose(actual, expected, epsilon);
  }
}
