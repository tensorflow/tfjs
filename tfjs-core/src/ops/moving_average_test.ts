/**
 * @license
 * Copyright 2017 Google Inc. All Rights Reserved.
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
import {expectArraysClose} from '../test_util';

describeWithFlags('movingAverage', ALL_ENVS, () => {
  // Use the following tensorflow to generate reference values for
  // `zeroDebias` = `true`;
  //
  // ```python
  // import tensorflow as tf
  // from tensorflow.python.training.moving_averages import
  // assign_moving_average
  //
  // with tf.Session() as sess:
  //   v = tf.get_variable("v1", shape=[2, 2], dtype=tf.float32,
  //                       initializer=tf.zeros_initializer)
  //   x = tf.Variable([[1.0, 2.0], [3.0, 4.0]])
  //   inc_x = x.assign_add([[10.0, 10.0], [10.0, 10.0]])
  //   update = assign_moving_average(v, x, 0.6)
  //
  //   sess.run(tf.global_variables_initializer())
  //
  //   sess.run(update)
  //   print(sess.run(v))
  //
  //   sess.run(inc_x)
  //   sess.run(update)
  //   print(sess.run(v))
  // ```

  it('zeroDebias=true, decay and step are numbers', async () => {
    const v0 = tf.tensor2d([[0, 0], [0, 0]], [2, 2]);
    const x = tf.tensor2d([[1, 2], [3, 4]], [2, 2]);
    const decay = 0.6;

    const v1 = tf.movingAverage(v0, x, decay, 1);
    expectArraysClose(await v1.array(), [[1, 2], [3, 4]]);

    const y = tf.tensor2d([[11, 12], [13, 14]], [2, 2]);
    const v2 = tf.movingAverage(v1, y, decay, 2);
    expectArraysClose(await v2.array(), [[7.25, 8.25], [9.25, 10.25]]);
  });

  it('zeroDebias=true, decay and step are scalars', async () => {
    const v0 = tf.tensor2d([[0, 0], [0, 0]], [2, 2]);
    const x = tf.tensor2d([[1, 2], [3, 4]], [2, 2]);
    const decay = tf.scalar(0.6);

    const v1 = tf.movingAverage(v0, x, decay, tf.scalar(1));
    expectArraysClose(await v1.array(), [[1, 2], [3, 4]]);

    const y = tf.tensor2d([[11, 12], [13, 14]], [2, 2]);
    const v2 = tf.movingAverage(v1, y, decay, tf.scalar(2));
    expectArraysClose(await v2.array(), [[7.25, 8.25], [9.25, 10.25]]);
  });

  // Use the following tensorflow to generate reference values for
  // `zeroDebias` = `false`;
  //
  // ```python
  // import tensorflow as tf
  // from tensorflow.python.training.moving_averages import
  // assign_moving_average
  //
  // with tf.Session() as sess:
  //   v = tf.get_variable("v1", shape=[2, 2], dtype=tf.float32,
  //                       initializer=tf.zeros_initializer)
  //   x = tf.Variable([[1.0, 2.0], [3.0, 4.0]])
  //   inc_x = x.assign_add([[10.0, 10.0], [10.0, 10.0]])
  //   update = assign_moving_average(v, x, 0.6, zero_debias=False)
  //
  //   sess.run(tf.global_variables_initializer())
  //
  //   sess.run(update)
  //   print(sess.run(v))
  //
  //   sess.run(inc_x)
  //   sess.run(update)
  //   print(sess.run(v))
  // ```

  it('zeroDebias=false, decay and step are numbers', async () => {
    const v0 = tf.tensor2d([[0, 0], [0, 0]], [2, 2]);
    const x = tf.tensor2d([[1, 2], [3, 4]], [2, 2]);
    const decay = 0.6;

    const v1 = tf.movingAverage(v0, x, decay, null, false);
    expectArraysClose(await v1.array(), [[0.4, 0.8], [1.2, 1.6]]);

    const y = tf.tensor2d([[11, 12], [13, 14]], [2, 2]);
    const v2 = tf.movingAverage(v1, y, decay, null, false);
    expectArraysClose(await v2.array(), [[4.64, 5.28], [5.92, 6.56]]);
  });

  it('zeroDebias=false, decay is scalar', async () => {
    const v0 = tf.tensor2d([[0, 0], [0, 0]], [2, 2]);
    const x = tf.tensor2d([[1, 2], [3, 4]], [2, 2]);
    const decay = tf.scalar(0.6);

    const v1 = tf.movingAverage(v0, x, decay, null, false);
    expectArraysClose(await v1.array(), [[0.4, 0.8], [1.2, 1.6]]);

    const y = tf.tensor2d([[11, 12], [13, 14]], [2, 2]);
    const v2 = tf.movingAverage(v1, y, decay, null, false);
    expectArraysClose(await v2.array(), [[4.64, 5.28], [5.92, 6.56]]);
  });

  it('zeroDebias=true, no step throws error', () => {
    const v0 = tf.tensor2d([[0, 0], [0, 0]], [2, 2]);
    const x = tf.tensor2d([[1, 2], [3, 4]], [2, 2]);
    const decay = tf.scalar(0.6);

    expect(() => tf.movingAverage(v0, x, decay, null)).toThrowError();
  });

  it('shape mismatch in v and x throws error', () => {
    const v0 = tf.tensor2d([[0, 0], [0, 0]], [2, 2]);
    const x = tf.tensor2d([[1, 2]], [1, 2]);
    const decay = tf.scalar(0.6);

    expect(() => tf.movingAverage(v0, x, decay, null)).toThrowError();
  });

  it('throws when passed v as a non-tensor', () => {
    const x = tf.tensor2d([[1, 2], [3, 4]], [2, 2]);

    expect(() => tf.movingAverage({} as tf.Tensor, x, 1))
        .toThrowError(
            /Argument 'v' passed to 'movingAverage' must be a Tensor/);
  });
  it('throws when passed v as a non-tensor', () => {
    const v = tf.tensor2d([[0, 0], [0, 0]], [2, 2]);

    expect(() => tf.movingAverage(v, {} as tf.Tensor, 1))
        .toThrowError(
            /Argument 'x' passed to 'movingAverage' must be a Tensor/);
  });

  it('accepts a tensor-like object', async () => {
    const v0 = [[0, 0], [0, 0]];  // 2x2
    const x = [[1, 2], [3, 4]];   // 2x2
    const decay = 0.6;

    const v1 = tf.movingAverage(v0, x, decay, 1);
    expectArraysClose(await v1.array(), [[1, 2], [3, 4]]);
  });
});
