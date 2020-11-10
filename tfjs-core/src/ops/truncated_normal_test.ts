/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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

import * as tf from '../index';
import {ALL_ENVS, describeWithFlags} from '../jasmine_util';
import {TypedArray} from '../types';
import {expectArrayInMeanStdRange} from './rand_util';

describeWithFlags('truncatedNormal', ALL_ENVS, () => {
  // Expect slightly higher variances for truncated values.
  const EPSILON = 0.60;
  const SEED = 2002;

  function assertTruncatedValues(
      values: TypedArray, mean: number, stdv: number) {
    const bounds = mean + stdv * 2;
    for (let i = 0; i < values.length; i++) {
      expect(Math.abs(values[i])).toBeLessThanOrEqual(bounds);
    }
  }

  it('should return a random 1D float32 array', async () => {
    const shape: [number] = [1000];

    // Ensure defaults to float32 w/o type:
    let result = tf.truncatedNormal(shape, 0, 3.5, null, SEED);
    expect(result.dtype).toBe('float32');
    assertTruncatedValues(await result.data(), 0, 3.5);
    expectArrayInMeanStdRange(await result.data(), 0, 3.5, EPSILON);

    result = tf.truncatedNormal(shape, 0, 4.5, 'float32', SEED);
    expect(result.dtype).toBe('float32');
    assertTruncatedValues(await result.data(), 0, 4.5);
    expectArrayInMeanStdRange(await result.data(), 0, 4.5, EPSILON);
  });

  it('should return a randon 1D int32 array', async () => {
    const shape: [number] = [1000];
    const result = tf.truncatedNormal(shape, 0, 5, 'int32', SEED);
    expect(result.dtype).toBe('int32');
    assertTruncatedValues(await result.data(), 0, 5);
    expectArrayInMeanStdRange(await result.data(), 0, 5, EPSILON);
  });

  it('should return a 2D float32 array', async () => {
    const shape: [number, number] = [50, 50];

    // Ensure defaults to float32 w/o type:
    let result = tf.truncatedNormal(shape, 0, 3.5, null, SEED);
    expect(result.dtype).toBe('float32');
    assertTruncatedValues(await result.data(), 0, 3.5);
    expectArrayInMeanStdRange(await result.data(), 0, 3.5, EPSILON);

    result = tf.truncatedNormal(shape, 0, 4.5, 'float32', SEED);
    expect(result.dtype).toBe('float32');
    assertTruncatedValues(await result.data(), 0, 4.5);
    expectArrayInMeanStdRange(await result.data(), 0, 4.5, EPSILON);
  });

  it('should return a 2D int32 array', async () => {
    const shape: [number, number] = [50, 50];
    const result = tf.truncatedNormal(shape, 0, 5, 'int32', SEED);
    expect(result.dtype).toBe('int32');
    assertTruncatedValues(await result.data(), 0, 5);
    expectArrayInMeanStdRange(await result.data(), 0, 5, EPSILON);
  });

  it('should return a 3D float32 array', async () => {
    const shape: [number, number, number] = [10, 10, 10];

    // Ensure defaults to float32 w/o type:
    let result = tf.truncatedNormal(shape, 0, 3.5, null, SEED);
    expect(result.dtype).toBe('float32');
    assertTruncatedValues(await result.data(), 0, 3.5);
    expectArrayInMeanStdRange(await result.data(), 0, 3.5, EPSILON);

    result = tf.truncatedNormal(shape, 0, 4.5, 'float32', SEED);
    expect(result.dtype).toBe('float32');
    assertTruncatedValues(await result.data(), 0, 4.5);
    expectArrayInMeanStdRange(await result.data(), 0, 4.5, EPSILON);
  });

  it('should return a 3D int32 array', async () => {
    const shape: [number, number, number] = [10, 10, 10];
    const result = tf.truncatedNormal(shape, 0, 5, 'int32', SEED);
    expect(result.dtype).toBe('int32');
    assertTruncatedValues(await result.data(), 0, 5);
    expectArrayInMeanStdRange(await result.data(), 0, 5, EPSILON);
  });

  it('should return a 4D float32 array', async () => {
    const shape: [number, number, number, number] = [5, 5, 5, 5];

    // Ensure defaults to float32 w/o type:
    let result = tf.truncatedNormal(shape, 0, 3.5, null, SEED);
    expect(result.dtype).toBe('float32');
    assertTruncatedValues(await result.data(), 0, 3.5);
    expectArrayInMeanStdRange(await result.data(), 0, 3.5, EPSILON);

    result = tf.truncatedNormal(shape, 0, 4.5, 'float32', SEED);
    expect(result.dtype).toBe('float32');
    assertTruncatedValues(await result.data(), 0, 4.5);
    expectArrayInMeanStdRange(await result.data(), 0, 4.5, EPSILON);
  });

  it('should return a 4D int32 array', async () => {
    const shape: [number, number, number, number] = [5, 5, 5, 5];
    const result = tf.truncatedNormal(shape, 0, 5, 'int32', SEED);
    expect(result.dtype).toBe('int32');
    assertTruncatedValues(await result.data(), 0, 5);
    expectArrayInMeanStdRange(await result.data(), 0, 5, EPSILON);
  });
});
