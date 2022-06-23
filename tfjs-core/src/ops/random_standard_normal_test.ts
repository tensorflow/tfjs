
/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
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

describeWithFlags('randomStandardNormal', ALL_ENVS, () => {
  const SEED = 42;
  const EPSILON = 0.05;

  it('should return a float32 1D of random standard normal values',
     async () => {
       const SAMPLES = 10000;

       // Ensure defaults to float32.
       let result = tf.randomStandardNormal([SAMPLES], null, SEED);
       expect(result.dtype).toBe('float32');
       expect(result.shape).toEqual([SAMPLES]);
       jarqueBeraNormalityTest(await result.data());
       expectArrayInMeanStdRange(await result.data(), 0, 1, EPSILON);

       result = tf.randomStandardNormal([SAMPLES], 'float32', SEED);
       expect(result.dtype).toBe('float32');
       expect(result.shape).toEqual([SAMPLES]);
       jarqueBeraNormalityTest(await result.data());
       expectArrayInMeanStdRange(await result.data(), 0, 1, EPSILON);
     });

  it('should return a int32 1D of random standard normal values', async () => {
    const SAMPLES = 10000;
    const result = tf.randomStandardNormal([SAMPLES], 'int32', SEED);
    expect(result.dtype).toBe('int32');
    expect(result.shape).toEqual([SAMPLES]);
    jarqueBeraNormalityTest(await result.data());
    expectArrayInMeanStdRange(await result.data(), 0, 1, EPSILON);
  });

  it('should return a float32 2D of random standard normal values',
     async () => {
       const SAMPLES = 100;

       // Ensure defaults to float32.
       let result = tf.randomStandardNormal([SAMPLES, SAMPLES], null, SEED);
       expect(result.dtype).toBe('float32');
       expect(result.shape).toEqual([SAMPLES, SAMPLES]);
       jarqueBeraNormalityTest(await result.data());
       expectArrayInMeanStdRange(await result.data(), 0, 1, EPSILON);

       result = tf.randomStandardNormal([SAMPLES, SAMPLES], 'float32', SEED);
       expect(result.dtype).toBe('float32');
       expect(result.shape).toEqual([SAMPLES, SAMPLES]);
       jarqueBeraNormalityTest(await result.data());
       expectArrayInMeanStdRange(await result.data(), 0, 1, EPSILON);
     });

  it('should return a int32 2D of random standard normal values', async () => {
    const SAMPLES = 100;
    const result = tf.randomStandardNormal([SAMPLES, SAMPLES], 'int32', SEED);
    expect(result.dtype).toBe('int32');
    expect(result.shape).toEqual([SAMPLES, SAMPLES]);
    jarqueBeraNormalityTest(await result.data());
    expectArrayInMeanStdRange(await result.data(), 0, 1, EPSILON);
  });

  it('should return a float32 3D of random standard normal values',
     async () => {
       const SAMPLES_SHAPE = [20, 20, 20];

       // Ensure defaults to float32.
       let result = tf.randomStandardNormal(SAMPLES_SHAPE, null, SEED);
       expect(result.dtype).toBe('float32');
       expect(result.shape).toEqual(SAMPLES_SHAPE);
       jarqueBeraNormalityTest(await result.data());
       expectArrayInMeanStdRange(await result.data(), 0, 1, EPSILON);

       result = tf.randomStandardNormal(SAMPLES_SHAPE, 'float32', SEED);
       expect(result.dtype).toBe('float32');
       expect(result.shape).toEqual(SAMPLES_SHAPE);
       jarqueBeraNormalityTest(await result.data());
       expectArrayInMeanStdRange(await result.data(), 0, 1, EPSILON);
     });

  it('should return a int32 3D of random standard normal values', async () => {
    const SAMPLES_SHAPE = [20, 20, 20];
    const result = tf.randomStandardNormal(SAMPLES_SHAPE, 'int32', SEED);
    expect(result.dtype).toBe('int32');
    expect(result.shape).toEqual(SAMPLES_SHAPE);
    jarqueBeraNormalityTest(await result.data());
    expectArrayInMeanStdRange(await result.data(), 0, 1, EPSILON);
  });

  it('should return a float32 4D of random standard normal values',
     async () => {
       const SAMPLES_SHAPE = [10, 10, 10, 10];

       // Ensure defaults to float32.
       let result = tf.randomStandardNormal(SAMPLES_SHAPE, null, SEED);
       expect(result.dtype).toBe('float32');
       expect(result.shape).toEqual(SAMPLES_SHAPE);
       jarqueBeraNormalityTest(await result.data());
       expectArrayInMeanStdRange(await result.data(), 0, 1, EPSILON);

       result = tf.randomStandardNormal(SAMPLES_SHAPE, 'float32', SEED);
       expect(result.dtype).toBe('float32');
       expect(result.shape).toEqual(SAMPLES_SHAPE);
       jarqueBeraNormalityTest(await result.data());
       expectArrayInMeanStdRange(await result.data(), 0, 1, EPSILON);
     });

  it('should return a int32 4D of random standard normal values', async () => {
    const SAMPLES_SHAPE = [10, 10, 10, 10];

    const result = tf.randomStandardNormal(SAMPLES_SHAPE, 'int32', SEED);
    expect(result.dtype).toBe('int32');
    expect(result.shape).toEqual(SAMPLES_SHAPE);
    jarqueBeraNormalityTest(await result.data());
    expectArrayInMeanStdRange(await result.data(), 0, 1, EPSILON);
  });

  it('should return a int32 5D of random standard normal values', async () => {
    const SAMPLES_SHAPE = [10, 10, 10, 10, 10];

    const result = tf.randomStandardNormal(SAMPLES_SHAPE, 'int32', SEED);
    expect(result.dtype).toBe('int32');
    expect(result.shape).toEqual(SAMPLES_SHAPE);
    jarqueBeraNormalityTest(await result.data());
    expectArrayInMeanStdRange(await result.data(), 0, 1, EPSILON);
  });
});
