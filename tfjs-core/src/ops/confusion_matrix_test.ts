/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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

/**
 * Unit tests for confusionMatrix().
 */

describeWithFlags('confusionMatrix', ALL_ENVS, () => {
  // Reference (Python) TensorFlow code:
  //
  // ```py
  // import tensorflow as tf
  //
  // tf.enable_eager_execution()
  //
  // labels = tf.constant([0, 1, 2, 1, 0])
  // predictions = tf.constant([0, 2, 2, 1, 0])
  // out = tf.confusion_matrix(labels, predictions, 3)
  //
  // print(out)
  // ```
  it('3x3 all cases present in both labels and predictions', async () => {
    const labels = tf.tensor1d([0, 1, 2, 1, 0], 'int32');
    const predictions = tf.tensor1d([0, 2, 2, 1, 0], 'int32');
    const numClasses = 3;
    const out = tf.math.confusionMatrix(labels, predictions, numClasses);
    expectArraysEqual(await out.data(), [2, 0, 0, 0, 1, 1, 0, 0, 1]);
    expect(out.dtype).toBe('int32');
    expect(out.shape).toEqual([3, 3]);
  });

  it('float32 arguments are accepted', async () => {
    const labels = tf.tensor1d([0, 1, 2, 1, 0], 'float32');
    const predictions = tf.tensor1d([0, 2, 2, 1, 0], 'float32');
    const numClasses = 3;
    const out = tf.math.confusionMatrix(labels, predictions, numClasses);
    expectArraysEqual(await out.data(), [2, 0, 0, 0, 1, 1, 0, 0, 1]);
    expect(out.dtype).toBe('int32');
    expect(out.shape).toEqual([3, 3]);
  });

  // Reference (Python) TensorFlow code:
  //
  // ```py
  // import tensorflow as tf
  //
  // tf.enable_eager_execution()
  //
  // labels = tf.constant([3, 3, 2, 2, 1, 1, 0, 0])
  // predictions = tf.constant([2, 2, 2, 2, 0, 0, 0, 0])
  // out = tf.confusion_matrix(labels, predictions, 4)
  //
  // print(out)
  // ```
  it('4x4 all cases present in labels, but not predictions', async () => {
    const labels = tf.tensor1d([3, 3, 2, 2, 1, 1, 0, 0], 'int32');
    const predictions = tf.tensor1d([2, 2, 2, 2, 0, 0, 0, 0], 'int32');
    const numClasses = 4;
    const out = tf.math.confusionMatrix(labels, predictions, numClasses);
    expectArraysEqual(
        await out.data(), [2, 0, 0, 0, 2, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0]);
    expect(out.dtype).toBe('int32');
    expect(out.shape).toEqual([4, 4]);
  });

  it('4x4 all cases present in predictions, but not labels', async () => {
    const labels = tf.tensor1d([2, 2, 2, 2, 0, 0, 0, 0], 'int32');
    const predictions = tf.tensor1d([3, 3, 2, 2, 1, 1, 0, 0], 'int32');
    const numClasses = 4;
    const out = tf.math.confusionMatrix(labels, predictions, numClasses);
    expectArraysEqual(
        await out.data(), [2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0]);
    expect(out.dtype).toBe('int32');
    expect(out.shape).toEqual([4, 4]);
  });

  it('Plain arrays as inputs', async () => {
    const labels: number[] = [3, 3, 2, 2, 1, 1, 0, 0];
    const predictions: number[] = [2, 2, 2, 2, 0, 0, 0, 0];
    const numClasses = 4;
    const out = tf.math.confusionMatrix(labels, predictions, numClasses);
    expectArraysEqual(
        await out.data(), [2, 0, 0, 0, 2, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0]);
    expect(out.dtype).toBe('int32');
    expect(out.shape).toEqual([4, 4]);
  });

  it('Int32Arrays as inputs', async () => {
    const labels = new Int32Array([3, 3, 2, 2, 1, 1, 0, 0]);
    const predictions = new Int32Array([2, 2, 2, 2, 0, 0, 0, 0]);
    const numClasses = 4;
    const out = tf.math.confusionMatrix(labels, predictions, numClasses);
    expectArraysEqual(
        await out.data(), [2, 0, 0, 0, 2, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0]);
    expect(out.dtype).toBe('int32');
    expect(out.shape).toEqual([4, 4]);
  });

  // Reference (Python) TensorFlow code:
  //
  // ```py
  // import tensorflow as tf
  //
  // tf.enable_eager_execution()
  //
  // labels = tf.constant([0, 4])
  // predictions = tf.constant([4, 0])
  // out = tf.confusion_matrix(labels, predictions, 5)
  //
  // print(out)
  // ```
  it('5x5 predictions and labels both missing some cases', async () => {
    const labels = tf.tensor1d([0, 4], 'int32');
    const predictions = tf.tensor1d([4, 0], 'int32');
    const numClasses = 5;
    const out = tf.math.confusionMatrix(labels, predictions, numClasses);
    expectArraysEqual(await out.data(), [
      0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0
    ]);
    expect(out.dtype).toBe('int32');
    expect(out.shape).toEqual([5, 5]);
  });

  it('Invalid numClasses leads to Error', () => {
    expect(
        () => tf.math.confusionMatrix(
            tf.tensor1d([0, 1]), tf.tensor1d([1, 0]), 2.5))
        .toThrowError(/numClasses .* positive integer.* got 2\.5/);
  });

  it('Incorrect tensor rank leads to Error', () => {
    expect(
        () => tf.math.confusionMatrix(
            // tslint:disable-next-line:no-any
            tf.scalar(0) as any, tf.scalar(0) as any, 1))
        .toThrowError(/rank .* 1.*got 0/);
    expect(
        () =>
            // tslint:disable-next-line:no-any
        tf.math.confusionMatrix(tf.zeros([3, 3]) as any, tf.zeros([9]), 2))
        .toThrowError(/rank .* 1.*got 2/);
    expect(
        () =>
            // tslint:disable-next-line:no-any
        tf.math.confusionMatrix(tf.zeros([9]), tf.zeros([3, 3]) as any, 2))
        .toThrowError(/rank .* 1.*got 2/);
  });

  it('Mismatch in lengths leads to Error', () => {
    expect(
        // tslint:disable-next-line:no-any
        () => tf.math.confusionMatrix(tf.zeros([3]) as any, tf.zeros([9]), 2))
        .toThrowError(/Mismatch .* 3 vs.* 9/);
  });
});
