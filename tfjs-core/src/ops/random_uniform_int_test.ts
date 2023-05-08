/**
 * @license
 * Copyright 2023 Google LLC.
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
import { expectArraysEqual, expectValuesInRange} from '../test_util';

describeWithFlags('randomUniformInt', ALL_ENVS, () => {
  it('should return a random 1D int32 array', async () => {
    const shape: [number] = [10];
    const result = tf.randomUniformInt(shape, 0, 2);
    expect(result.dtype).toBe('int32');
    expectValuesInRange(await result.data(), 0, 2);
  });

  it('should return a random 2D int32 array', async () => {
    const shape: [number, number] = [3, 4];
    const result = tf.randomUniformInt(shape, 0, 2);
    expect(result.dtype).toBe('int32');
    expectValuesInRange(await result.data(), 0, 2);
  });

  it('should return a random 3D int32 array', async () => {
    const shape: [number, number, number] = [3, 4, 5];
    const result = tf.randomUniformInt(shape, 0, 2);
    expect(result.dtype).toBe('int32');
    expectValuesInRange(await result.data(), 0, 2);
  });

  it('should return a random 4D int32 array', async () => {
    const shape: [number, number, number, number] = [3, 4, 5, 6];
    const result = tf.randomUniformInt(shape, 0, 2);
    expect(result.dtype).toBe('int32');
    expectValuesInRange(await result.data(), 0, 2);
  });

  it('should return a random 5D int32 array', async () => {
    const shape: [number, number, number, number, number] = [2, 3, 4, 5, 6];
    const result = tf.randomUniformInt(shape, 0, 2);
    expect(result.dtype).toBe('int32');
    expectValuesInRange(await result.data(), 0, 2);
  });

  it('should throw error when shape is not integer', () => {
    expect(() => tf.randomUniformInt([2, 2.22, 3.33], 0, 1)).toThrow();
  });

  it('should return the same result when seed is specified', async () => {
    const seed = 123;
    const shape = [4, 4];
    const result1 = tf.randomUniformInt(shape, 0, 1000, seed);
    const result2 = tf.randomUniformInt(shape, 0, 1000, seed);
    expectArraysEqual(await result1.data(), await result2.data());
  });
});
