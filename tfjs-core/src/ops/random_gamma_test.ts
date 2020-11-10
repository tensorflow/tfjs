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
import {expectValuesInRange} from '../test_util';

const GAMMA_MIN = 0;
const GAMMA_MAX = 40;

describeWithFlags('randomGamma', ALL_ENVS, () => {
  it('should return a random 1D float32 array', async () => {
    const shape: [number] = [10];

    // Ensure defaults to float32 w/o type:
    let result = tf.randomGamma(shape, 2, 2);
    expect(result.dtype).toBe('float32');
    expectValuesInRange(await result.data(), GAMMA_MIN, GAMMA_MAX);

    result = tf.randomGamma(shape, 2, 2, 'float32');
    expect(result.dtype).toBe('float32');
    expectValuesInRange(await result.data(), GAMMA_MIN, GAMMA_MAX);
  });

  it('should return a random 1D int32 array', async () => {
    const shape: [number] = [10];
    const result = tf.randomGamma(shape, 2, 2, 'int32');
    expect(result.dtype).toBe('int32');
    expectValuesInRange(await result.data(), GAMMA_MIN, GAMMA_MAX);
  });

  it('should return a random 2D float32 array', async () => {
    const shape: [number, number] = [3, 4];

    // Ensure defaults to float32 w/o type:
    let result = tf.randomGamma(shape, 2, 2);
    expect(result.dtype).toBe('float32');
    expectValuesInRange(await result.data(), GAMMA_MIN, GAMMA_MAX);

    result = tf.randomGamma(shape, 2, 2, 'float32');
    expect(result.dtype).toBe('float32');
    expectValuesInRange(await result.data(), GAMMA_MIN, GAMMA_MAX);
  });

  it('should return a random 2D int32 array', async () => {
    const shape: [number, number] = [3, 4];
    const result = tf.randomGamma(shape, 2, 2, 'int32');
    expect(result.dtype).toBe('int32');
    expectValuesInRange(await result.data(), GAMMA_MIN, GAMMA_MAX);
  });

  it('should return a random 3D float32 array', async () => {
    const shape: [number, number, number] = [3, 4, 5];

    // Ensure defaults to float32 w/o type:
    let result = tf.randomGamma(shape, 2, 2);
    expect(result.dtype).toBe('float32');
    expectValuesInRange(await result.data(), GAMMA_MIN, GAMMA_MAX);

    result = tf.randomGamma(shape, 2, 2, 'float32');
    expect(result.dtype).toBe('float32');
    expectValuesInRange(await result.data(), GAMMA_MIN, GAMMA_MAX);
  });

  it('should return a random 3D int32 array', async () => {
    const shape: [number, number, number] = [3, 4, 5];
    const result = tf.randomGamma(shape, 2, 2, 'int32');
    expect(result.dtype).toBe('int32');
    expectValuesInRange(await result.data(), GAMMA_MIN, GAMMA_MAX);
  });

  it('should return a random 4D float32 array', async () => {
    const shape: [number, number, number, number] = [3, 4, 5, 6];

    // Ensure defaults to float32 w/o type:
    let result = tf.randomGamma(shape, 2, 2);
    expect(result.dtype).toBe('float32');
    expectValuesInRange(await result.data(), GAMMA_MIN, GAMMA_MAX);

    result = tf.randomGamma(shape, 2, 2, 'float32');
    expect(result.dtype).toBe('float32');
    expectValuesInRange(await result.data(), GAMMA_MIN, GAMMA_MAX);
  });

  it('should return a random 4D int32 array', async () => {
    const shape: [number, number, number, number] = [3, 4, 5, 6];
    const result = tf.randomGamma(shape, 2, 2, 'int32');
    expect(result.dtype).toBe('int32');
    expectValuesInRange(await result.data(), GAMMA_MIN, GAMMA_MAX);
  });

  it('should return a random 5D float32 array', async () => {
    const shape: [number, number, number, number, number] = [2, 3, 4, 5, 6];

    // Ensure defaults to float32 w/o type:
    let result = tf.randomGamma(shape, 2, 2);
    expect(result.dtype).toBe('float32');
    expectValuesInRange(await result.data(), GAMMA_MIN, GAMMA_MAX);

    result = tf.randomGamma(shape, 2, 2, 'float32');
    expect(result.dtype).toBe('float32');
    expectValuesInRange(await result.data(), GAMMA_MIN, GAMMA_MAX);
  });

  it('should return a random 5D int32 array', async () => {
    const shape: [number, number, number, number, number] = [2, 3, 4, 5, 6];
    const result = tf.randomGamma(shape, 2, 2, 'int32');
    expect(result.dtype).toBe('int32');
    expectValuesInRange(await result.data(), GAMMA_MIN, GAMMA_MAX);
  });
});
