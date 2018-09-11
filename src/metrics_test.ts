/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

/**
 * Unit tests for metrics.ts.
 */

import {scalar, tensor1d, tensor2d} from '@tensorflow/tfjs-core';

import * as tfl from './index';
import {binaryAccuracy, categoricalAccuracy, get} from './metrics';
import {describeMathCPUAndGPU, expectTensorsClose} from './utils/test_utils';

describeMathCPUAndGPU('binaryAccuracy', () => {
  it('1D exact', () => {
    const x = tensor1d([1, 1, 1, 1, 0, 0, 0, 0]);
    const y = tensor1d([1, 0, 1, 0, 0, 1, 0, 1]);
    const accuracy = tfl.metrics.binaryAccuracy(x, y);
    expectTensorsClose(accuracy, scalar(0.5));
  });
  it('2D thresholded', () => {
    const x = tensor1d([1, 1, 1, 1, 0, 0, 0, 0]);
    const y = tensor1d([0.2, 0.4, 0.6, 0.8, 0.2, 0.3, 0.4, 0.7]);
    const accuracy = tfl.metrics.binaryAccuracy(x, y);
    expectTensorsClose(accuracy, scalar(5 / 8));
  });
  it('2D exact', () => {
    const x = tensor2d([[1, 1, 1, 1], [0, 0, 0, 0]], [2, 4]);
    const y = tensor2d([[1, 0, 1, 0], [0, 0, 0, 1]], [2, 4]);
    const accuracy = tfl.metrics.binaryAccuracy(x, y);
    expectTensorsClose(accuracy, tensor1d([0.5, 0.75]));
  });
  it('2D thresholded', () => {
    const x = tensor2d([[1, 1], [1, 1], [0, 0], [0, 0]], [4, 2]);
    const y =
        tensor2d([[0.2, 0.4], [0.6, 0.8], [0.2, 0.3], [0.4, 0.7]], [4, 2]);
    const accuracy = tfl.metrics.binaryAccuracy(x, y);
    expectTensorsClose(accuracy, tensor1d([0, 1, 1, 0.5]));
  });
});

describeMathCPUAndGPU('binaryCrossentropy', () => {
  it('2D single-value yTrue', () => {
    // Use the following Python code to generate the reference values:
    // ```python
    // import keras
    // import numpy as np
    // import tensorflow as tf
    //
    // with tf.Session() as sess:
    //   x = tf.Variable(np.array(
    //       [[0], [0], [0], [1], [1], [1]],
    //       dtype=np.float32))
    //   y = tf.Variable(np.array(
    //       [[0], [0.5], [1], [0], [0.5], [1]],
    //       dtype=np.float32))
    //   z = keras.metrics.binary_crossentropy(x, y)
    //
    //   sess.run(tf.global_variables_initializer())
    //   print(sess.run(z))
    // ```
    const x = tensor2d([[0], [0], [0], [1], [1], [1]]);
    const y = tensor2d([[0], [0.5], [1], [0], [0.5], [1]]);
    const accuracy = tfl.metrics.binaryCrossentropy(x, y);
    expectTensorsClose(accuracy, tensor1d([
                         1.00000015e-07, 6.93147182e-01, 1.59423847e+01,
                         1.61180954e+01, 6.93147182e-01, 1.19209332e-07
                       ]));
  });
  it('2D one-hot binary yTrue', () => {
    // Use the following Python code to generate the reference values:
    // ```python
    // import keras
    // import numpy as np
    // import tensorflow as tf
    //
    // with tf.Session() as sess:
    //   x = tf.Variable(np.array(
    //       [[1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [0, 1]],
    //       dtype=np.float32))
    //   y = tf.Variable(np.array(
    //       [[1, 0], [0.5, 0.5], [0, 1], [1, 0], [0.5, 0.5], [0, 1]],
    //       dtype=np.float32))
    //   z = keras.metrics.binary_crossentropy(x, y)
    //
    //   sess.run(tf.global_variables_initializer())
    //   print(sess.run(z))
    // ```
    const x = tensor2d([[1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [0, 1]]);
    const y =
        tensor2d([[1, 0], [0.5, 0.5], [0, 1], [1, 0], [0.5, 0.5], [0, 1]]);
    const accuracy = tfl.metrics.binaryCrossentropy(x, y);
    expectTensorsClose(accuracy, tensor1d([
                         1.0960467e-07, 6.9314718e-01, 1.6030239e+01,
                         1.6030239e+01, 6.9314718e-01, 1.0960467e-07
                       ]));
  });
});

describeMathCPUAndGPU('categoricalAccuracy', () => {
  it('1D', () => {
    const x = tensor1d([0, 0, 0, 1]);
    const y = tensor1d([0.1, 0.8, 0.05, 0.05]);
    const accuracy = tfl.metrics.categoricalAccuracy(x, y);
    expect(accuracy.dtype).toEqual('float32');
    expect(accuracy.shape).toEqual([]);
    expect(Array.from(accuracy.dataSync())).toEqual([0]);
  });
  it('2D', () => {
    const x = tensor2d([[0, 0, 0, 1], [0, 0, 0, 1]], [2, 4]);
    const y =
        tensor2d([[0.1, 0.8, 0.05, 0.05], [0.1, 0.05, 0.05, 0.8]], [2, 4]);
    const accuracy = tfl.metrics.categoricalAccuracy(x, y);
    expect(accuracy.dtype).toEqual('float32');
    expect(accuracy.shape).toEqual([2]);
    expect(Array.from(accuracy.dataSync())).toEqual([0, 1]);
  });
});

describeMathCPUAndGPU('categoricalCrossentropy metric', () => {
  it('1D', () => {
    const x = tensor1d([0, 0, 0, 1]);
    const y = tensor1d([0.1, 0.8, 0.05, 0.05]);
    const accuracy = tfl.metrics.categoricalCrossentropy(x, y);
    expect(accuracy.dtype).toEqual('float32');
    expectTensorsClose(accuracy, scalar(2.995732));
  });
  it('2D', () => {
    const x = tensor2d([[0, 0, 0, 1], [0, 0, 0, 1]]);
    const y = tensor2d([[0.1, 0.8, 0.05, 0.05], [0.1, 0.05, 0.05, 0.8]]);
    const accuracy = tfl.metrics.categoricalCrossentropy(x, y);
    expect(accuracy.dtype).toEqual('float32');
    expectTensorsClose(accuracy, tensor1d([2.995732, 0.22314353]));
  });
});

describeMathCPUAndGPU('precision metric', () => {
  it('1D', () => {
    const x = tensor1d([0, 0, 0, 1]);
    const y = tensor1d([0, 0, 0, 1]);
    const precision = tfl.metrics.precision(x, y);
    expect(precision.dtype).toEqual('float32');
    expectTensorsClose(precision, scalar(1));
  });
  it('2D', () => {
    // Use the following Python code to generate the reference values:
    // ```python
    //
    // import tensorflow as tf
    // import numpy as np
    //
    // with tf.Session() as session:
    //     labels = tf.constant([
    //       [0, 0, 1],
    //       [0, 0, 1],
    //       [1, 0, 0],
    //       [0, 1, 0]
    //     ])
    //     prediction = tf.constant([
    //       [0, 0, 1],
    //       [1, 0, 0],
    //       [1, 0, 0],
    //       [0, 1, 0]
    //     ])
    //
    //     output = tf.metrics.precision(
    //       labels,
    //       prediction
    //     )
    //
    //     init = tf.local_variables_initializer()
    //     session.run(init)
    //
    //     result = session.run(output)
    //     print(result[0])
    // ```
    const x = tensor2d([[0, 0, 1], [0, 0, 1], [1, 0, 0], [0, 1, 0]]);
    const y = tensor2d([[0, 0, 1], [1, 0, 0], [1, 0, 0], [0, 1, 0]]);
    const precision = tfl.metrics.precision(x, y);
    expect(precision.dtype).toEqual('float32');
    expectTensorsClose(precision, scalar(0.75));
  });

  it('2D edge case', () => {
    const x = tensor2d([[0, 0, 1], [0, 0, 1], [1, 0, 0], [0, 1, 0]]);
    const y = tensor2d([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]);
    const precision = tfl.metrics.precision(x, y);
    expect(precision.dtype).toEqual('float32');
    expectTensorsClose(precision, scalar(0));
  });
});

describe('metrics.get', () => {
  it('valid name, not alias', () => {
    expect(get('binaryAccuracy') === get('categoricalAccuracy')).toEqual(false);
  });
  it('valid name, alias', () => {
    expect(get('mse') === get('MSE')).toEqual(true);
  });
  it('invalid name', () => {
    expect(() => get('InvalidMetricName')).toThrowError(/Unknown metric/);
  });
  it('LossOrMetricFn input', () => {
    expect(get(binaryAccuracy)).toEqual(binaryAccuracy);
    expect(get(categoricalAccuracy)).toEqual(categoricalAccuracy);
  });
});
