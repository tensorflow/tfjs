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
import {expectArraysEqual} from '../test_util';

async function expectTensor(
    tensor: tf.Tensor, shape: number[], value: string|number) {
  const length = shape.length === 0 ? 1 : shape.reduce((a, b) => a * b);
  expect(tensor.shape).toEqual(shape);
  expectArraysEqual(await tensor.data(), new Array(length).fill(value));
}

describeWithFlags('squeeze', ALL_ENVS, () => {
  it('default', async () => {
    const assertType = async (dtype: 'string'|'float32') => {
      const value = dtype === 'string' ? 'test' : 0.0;
      // Nothing to squeeze.
      await expectTensor(tf.squeeze(tf.fill([2], value)), [2], value);

      // Squeeze the middle element away.
      await expectTensor(tf.squeeze(tf.fill([2, 1, 2], value)), [2, 2], value);

      // Squeeze on both ends.
      await expectTensor(
          tf.squeeze(tf.fill([1, 2, 1, 3, 1], value)), [2, 3], value);
    };

    await assertType('string');
    await assertType('float32');
  });

  it('specific dimension', async () => {
    const assertType = async (dtype: 'string'|'float32') => {
      const value = dtype === 'string' ? 'test' : 0.0;
      const shape = [1, 2, 1, 3, 1];
      // Positive squeeze dim index.
      await expectTensor(
          tf.squeeze(tf.fill(shape, value), [0]), [2, 1, 3, 1], value);
      await expectTensor(
          tf.squeeze(tf.fill(shape, value), [2, 4]), [1, 2, 3], value);
      await expectTensor(
          tf.squeeze(tf.fill(shape, value), [0, 4, 2]), [2, 3], value);

      // Negative squeeze dim index.
      await expectTensor(
          tf.squeeze(tf.fill(shape, value), [-1]), [1, 2, 1, 3], value);
      await expectTensor(
          tf.squeeze(tf.fill(shape, value), [-3, -5]), [2, 3, 1], value);
      await expectTensor(
          tf.squeeze(tf.fill(shape, value), [-3, -5, -1]), [2, 3], value);
    };

    await assertType('string');
    await assertType('float32');
  });

  it('all ones', async () => {
    const assertType = async (dtype: 'string'|'float32') => {
      const value = dtype === 'string' ? 'test' : 0.0;
      await expectTensor(tf.squeeze(tf.fill([1, 1, 1], value)), [], value);
    };

    await assertType('string');
    await assertType('float32');
  });

  it('squeeze only ones', async () => {
    const assertType = async (dtype: 'string'|'float32') => {
      const value = dtype === 'string' ? 'test' : 0.0;
      const shape = [1, 1, 3];
      await expectTensor(tf.squeeze(tf.fill(shape, value)), [3], value);
      await expectTensor(tf.squeeze(tf.fill(shape, value), [0]), [1, 3], value);
      await expectTensor(tf.squeeze(tf.fill(shape, value), [1]), [1, 3], value);
      expect(() => tf.squeeze(tf.fill(shape, value), [2])).toThrowError();
    };

    await assertType('string');
    await assertType('float32');
  });

  it('squeeze errors', async () => {
    const assertType = async (dtype: 'string'|'float32') => {
      const value = dtype === 'string' ? 'test' : 0.0;
      const shape = [1, 2, 1];
      expect(() => tf.squeeze(tf.fill(shape, value), [-4])).toThrowError();
      expect(() => tf.squeeze(tf.fill(shape, value), [0, -4])).toThrowError();
      expect(() => tf.squeeze(tf.fill(shape, value), [3])).toThrowError();
      expect(() => tf.squeeze(tf.fill(shape, value), [2, 3])).toThrowError();
    };

    await assertType('string');
    await assertType('float32');
  });
});
