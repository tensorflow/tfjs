
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
import {expectArrayInMeanStdRange, jarqueBeraNormalityTest} from './rand_util';

describeWithFlags('randomNormal', ALL_ENVS, () => {
  const SEED = 2002;
  const EPSILON = 0.05;

  it('should return a float32 1D of random normal values', async () => {
    const SAMPLES = 10000;

    // Ensure defaults to float32.
    let result = tf.randomNormal([SAMPLES], 0, 0.5, null, SEED);
    expect(result.dtype).toBe('float32');
    expect(result.shape).toEqual([SAMPLES]);
    jarqueBeraNormalityTest(await result.data());
    expectArrayInMeanStdRange(await result.data(), 0, 0.5, EPSILON);

    result = tf.randomNormal([SAMPLES], 0, 1.5, 'float32', SEED);
    expect(result.dtype).toBe('float32');
    expect(result.shape).toEqual([SAMPLES]);
    jarqueBeraNormalityTest(await result.data());
    expectArrayInMeanStdRange(await result.data(), 0, 1.5, EPSILON);
  });

  it('should return a int32 1D of random normal values', async () => {
    const SAMPLES = 10000;
    const result = tf.randomNormal([SAMPLES], 0, 2, 'int32', SEED);
    expect(result.dtype).toBe('int32');
    expect(result.shape).toEqual([SAMPLES]);
    jarqueBeraNormalityTest(await result.data());
    expectArrayInMeanStdRange(await result.data(), 0, 2, EPSILON);
  });

  it('should return a float32 2D of random normal values', async () => {
    const SAMPLES = 100;

    // Ensure defaults to float32.
    let result = tf.randomNormal([SAMPLES, SAMPLES], 0, 2.5, null, SEED);
    expect(result.dtype).toBe('float32');
    expect(result.shape).toEqual([SAMPLES, SAMPLES]);
    jarqueBeraNormalityTest(await result.data());
    expectArrayInMeanStdRange(await result.data(), 0, 2.5, EPSILON);

    result = tf.randomNormal([SAMPLES, SAMPLES], 0, 3.5, 'float32', SEED);
    expect(result.dtype).toBe('float32');
    expect(result.shape).toEqual([SAMPLES, SAMPLES]);
    jarqueBeraNormalityTest(await result.data());
    expectArrayInMeanStdRange(await result.data(), 0, 3.5, EPSILON);
  });

  it('should return a int32 2D of random normal values', async () => {
    const SAMPLES = 100;
    const result = tf.randomNormal([SAMPLES, SAMPLES], 0, 2, 'int32', SEED);
    expect(result.dtype).toBe('int32');
    expect(result.shape).toEqual([SAMPLES, SAMPLES]);
    jarqueBeraNormalityTest(await result.data());
    expectArrayInMeanStdRange(await result.data(), 0, 2, EPSILON);
  });

  it('should return a float32 3D of random normal values', async () => {
    const SAMPLES_SHAPE = [20, 20, 20];

    // Ensure defaults to float32.
    let result = tf.randomNormal(SAMPLES_SHAPE, 0, 0.5, null, SEED);
    expect(result.dtype).toBe('float32');
    expect(result.shape).toEqual(SAMPLES_SHAPE);
    jarqueBeraNormalityTest(await result.data());
    expectArrayInMeanStdRange(await result.data(), 0, 0.5, EPSILON);

    result = tf.randomNormal(SAMPLES_SHAPE, 0, 1.5, 'float32', SEED);
    expect(result.dtype).toBe('float32');
    expect(result.shape).toEqual(SAMPLES_SHAPE);
    jarqueBeraNormalityTest(await result.data());
    expectArrayInMeanStdRange(await result.data(), 0, 1.5, EPSILON);
  });

  it('should return a int32 3D of random normal values', async () => {
    const SAMPLES_SHAPE = [20, 20, 20];
    const result = tf.randomNormal(SAMPLES_SHAPE, 0, 2, 'int32', SEED);
    expect(result.dtype).toBe('int32');
    expect(result.shape).toEqual(SAMPLES_SHAPE);
    jarqueBeraNormalityTest(await result.data());
    expectArrayInMeanStdRange(await result.data(), 0, 2, EPSILON);
  });

  it('should return a float32 4D of random normal values', async () => {
    const SAMPLES_SHAPE = [10, 10, 10, 10];

    // Ensure defaults to float32.
    let result = tf.randomNormal(SAMPLES_SHAPE, 0, 0.5, null, SEED);
    expect(result.dtype).toBe('float32');
    expect(result.shape).toEqual(SAMPLES_SHAPE);
    jarqueBeraNormalityTest(await result.data());
    expectArrayInMeanStdRange(await result.data(), 0, 0.5, EPSILON);

    result = tf.randomNormal(SAMPLES_SHAPE, 0, 1.5, 'float32', SEED);
    expect(result.dtype).toBe('float32');
    expect(result.shape).toEqual(SAMPLES_SHAPE);
    jarqueBeraNormalityTest(await result.data());
    expectArrayInMeanStdRange(await result.data(), 0, 1.5, EPSILON);
  });

  it('should return a int32 4D of random normal values', async () => {
    const SAMPLES_SHAPE = [10, 10, 10, 10];

    const result = tf.randomNormal(SAMPLES_SHAPE, 0, 2, 'int32', SEED);
    expect(result.dtype).toBe('int32');
    expect(result.shape).toEqual(SAMPLES_SHAPE);
    jarqueBeraNormalityTest(await result.data());
    expectArrayInMeanStdRange(await result.data(), 0, 2, EPSILON);
  });

  it('should return a int32 5D of random normal values', async () => {
    const SAMPLES_SHAPE = [10, 10, 10, 10, 10];

    const result = tf.randomNormal(SAMPLES_SHAPE, 0, 2, 'int32', SEED);
    expect(result.dtype).toBe('int32');
    expect(result.shape).toEqual(SAMPLES_SHAPE);
    jarqueBeraNormalityTest(await result.data());
    expectArrayInMeanStdRange(await result.data(), 0, 2, EPSILON);
  });
});
