/**
 * @license
 * Copyright 2018 Google Inc. All Rights Reserved.
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

import {ENV} from '../environment';
import {Tensor} from '../tensor';
import {expectNumbersClose} from '../test_util';
import {TypedArray} from '../types';

export function jarqueBeraNormalityTest(a: Tensor|TypedArray|number[]) {
  let values: TypedArray|number[];
  if (a instanceof Tensor) {
    values = a.dataSync();
  } else {
    values = a;
  }
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
    actual: Tensor|TypedArray|number[], expectedMean: number,
    expectedStdDev: number, epsilon?: number) {
  if (epsilon == null) {
    epsilon = ENV.get('TEST_EPSILON');
  }
  let actualValues: TypedArray|number[];
  if (actual instanceof Tensor) {
    actualValues = actual.dataSync();
  } else {
    actualValues = actual;
  }
  const actualMean = mean(actualValues);
  expectNumbersClose(actualMean, expectedMean, epsilon);
  expectNumbersClose(
      standardDeviation(actualValues, actualMean), expectedStdDev, epsilon);
}

function mean(values: TypedArray|number[]) {
  let sum = 0;
  for (let i = 0; i < values.length; i++) {
    sum += values[i];
  }
  return sum / values.length;
}

function standardDeviation(values: TypedArray|number[], mean: number) {
  let squareDiffSum = 0;
  for (let i = 0; i < values.length; i++) {
    const diff = values[i] - mean;
    squareDiffSum += diff * diff;
  }
  return Math.sqrt(squareDiffSum / values.length);
}

function kurtosis(values: TypedArray|number[]) {
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

function skewness(values: TypedArray|number[]) {
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
