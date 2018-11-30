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
 * Unit tests for normalization layers.
 */

import {onesLike, scalar, Tensor, tensor1d, tensor2d, tensor3d, tensor4d, train, zeros, zerosLike} from '@tensorflow/tfjs-core';

import * as tfl from '../index';
import {SymbolicTensor} from '../engine/topology';
import {describeMathCPU, describeMathCPUAndGPU, expectTensorsClose} from '../utils/test_utils';
import {batchNormalization, normalizeBatchInTraining} from './normalization';


describeMathCPUAndGPU('normalizeBatchInTraining', () => {
  // The reference values for assertion below can be obtained with Python code
  // as the following:
  // ```python
  // import keras
  // import numpy as np
  // import tensorflow as tf
  //
  // with tf.Session() as sess:
  //   x = tf.Variable(np.array(
  //       [[1, 2, 3, 4], [2, 4, 6, 8], [12, 11, 10, 9]], dtype=np.float32))
  //   gamma = tf.Variable(np.array([1, 1, 1, 1], dtype=np.float32))
  //   beta = tf.Variable(np.array([0, 0, 0, 0], dtype=np.float32))
  //   reduction_axes = [0]
  //   normed, mean, variance = keras.backend.normalize_batch_in_training(
  //       x, gamma, beta, reduction_axes)
  //   print(normed)
  //   print(mean)
  //   print(variance)
  // ```

  it('2D, no broadcasting', () => {
    const x = tensor2d([[1, 2, 3, 4], [2, 4, 6, 8], [12, 11, 10, 9]], [3, 4]);
    const gamma = tensor1d([1, 1, 1, 1]);
    const beta = tensor1d([0, 0, 0, 0]);
    const reductionAxes = [0];
    const [normed, mean, variance] =
        normalizeBatchInTraining(x, gamma, beta, reductionAxes);
    expectTensorsClose(
        normed,
        tensor2d(
            [
              [-0.805371, -0.9502233, -1.1624058, -1.3885813],
              [-0.6040282, -0.4319197, -0.11624074, 0.46286058],
              [1.4093992, 1.3821429, 1.2786462, 0.92572117]
            ],
            [3, 4]));
    expectTensorsClose(mean, tensor1d([5.0, 5.6666665, 6.3333335, 7.0]));
    expectTensorsClose(
        variance, tensor1d([24.666666, 14.888889, 8.222222, 4.6666665]));
  });

  it('3D, no broadcasting', () => {
    const x = tensor3d(
        [[[1, 2], [3, 4]], [[2, 4], [6, 8]], [[12, 11], [10, 9]]], [3, 2, 2]);
    const gamma = tensor1d([1, 1]);
    const beta = tensor1d([0, 0]);
    const reductionAxes = [0, 1];
    const [normed, mean, variance] =
        normalizeBatchInTraining(x, gamma, beta, reductionAxes);
    expectTensorsClose(
        normed,
        tensor3d(
            [
              [[-1.1355163, -1.3552775], [-0.6488664, -0.7297648]],
              [[-0.8921913, -0.7297648], [0.08110833, 0.5212605]],
              [[1.5410578, 1.4595294], [1.0544081, 0.8340168]]
            ],
            [3, 2, 2]));
    expectTensorsClose(mean, tensor1d([5.6666665, 6.3333335]));
    expectTensorsClose(variance, tensor1d([16.88889, 10.222222]));
  });

  it('3D, broadcasting', () => {
    const x = tensor3d(
        [[[1, 2], [3, 4]], [[2, 4], [6, 8]], [[12, 11], [10, 9]]], [3, 2, 2]);
    const gamma = tensor2d([[1, 1], [1, 1]], [2, 2]);
    const beta = tensor2d([[0, 0], [0, 0]], [2, 2]);
    const reductionAxes = [0];
    const [normed, mean, variance] =
        normalizeBatchInTraining(x, gamma, beta, reductionAxes);
    expectTensorsClose(
        normed,
        tensor3d(
            [
              [[-0.805371, -0.9502233], [-1.1624058, -1.3885813]],
              [[-0.6040282, -0.4319197], [-0.11624074, 0.46286058]],
              [[1.4093992, 1.3821429], [1.2786462, 0.92572117]]
            ],
            [3, 2, 2]));
    expectTensorsClose(
        mean, tensor2d([[5, 5.6666665], [6.3333335, 7]], [2, 2]));
    expectTensorsClose(
        variance,
        tensor2d([[24.666666, 14.888889], [8.222222, 4.6666665]], [2, 2]));
  });

  it('4D, broadcasting', () => {
    const x = tensor4d(
        [[[[1, 2], [3, 4]], [[2, 4], [6, 8]], [[12, 11], [10, 9]]]],
        [1, 3, 2, 2]);
    const gamma = tensor2d([[1, 1], [1, 1]], [2, 2]);
    const beta = tensor2d([[0, 0], [0, 0]], [2, 2]);
    const reductionAxes = [0, 1];
    const [normed, mean, variance] =
        normalizeBatchInTraining(x, gamma, beta, reductionAxes);
    expectTensorsClose(
        normed,
        tensor4d(
            [[
              [[-0.805371, -0.9502233], [-1.1624058, -1.3885813]],
              [[-0.6040282, -0.4319197], [-0.11624074, 0.46286058]],
              [[1.4093992, 1.3821429], [1.2786462, 0.92572117]]
            ]],
            [1, 3, 2, 2]));
    expectTensorsClose(
        mean, tensor2d([[5, 5.6666665], [6.3333335, 7]], [2, 2]));
    expectTensorsClose(
        variance,
        tensor2d([[24.666666, 14.888889], [8.222222, 4.6666665]], [2, 2]));
  });
});

describeMathCPUAndGPU('batchNormalization', () => {
  it('2D, no broadcast, no gamma, no beta', () => {
    const x = tensor2d([[10, 20], [30, 40]], [2, 2]);
    const mean = tensor2d([[5, 5], [5, 5]], [2, 2]);
    const variance = tensor2d([[4, 16], [4, 16]], [2, 2]);
    expectTensorsClose(
        batchNormalization(x, mean, variance, null, null, 0),
        tensor2d([[2.5, 3.75], [12.5, 8.75]], [2, 2]));
  });
  it('2D, no broadcast, no gamma, no beta, custom epsilon', () => {
    const x = tensor2d([[30, 30], [60, 60]], [2, 2]);
    const mean = tensor2d([[0, 0], [0, 0]], [2, 2]);
    const variance = tensor2d([[7, 7], [7, 7]], [2, 2]);
    expectTensorsClose(
        batchNormalization(x, mean, variance, null, null, 2),
        tensor2d([[10, 10], [20, 20]], [2, 2]));
  });
  it('2D, no broadcast, gamma, no beta', () => {
    const x = tensor2d([[10, 20], [30, 40]], [2, 2]);
    const mean = tensor2d([[5, 5], [5, 5]], [2, 2]);
    const variance = tensor2d([[4, 16], [4, 16]], [2, 2]);
    const gamma = tensor2d([[1, 2], [3, 4]], [2, 2]);
    expectTensorsClose(
        batchNormalization(x, mean, variance, null, gamma, 0),
        tensor2d([[2.5, 7.5], [37.5, 35]], [2, 2]));
  });
  it('2D, no broadcast, gamma, beta', () => {
    const x = tensor2d([[10, 20], [30, 40]], [2, 2]);
    const mean = tensor2d([[5, 5], [5, 5]], [2, 2]);
    const variance = tensor2d([[4, 16], [4, 16]], [2, 2]);
    const gamma = tensor2d([[1, 2], [3, 4]], [2, 2]);
    const beta = tensor2d([[-1, -1], [-2, -2]], [2, 2]);
    expectTensorsClose(
        batchNormalization(x, mean, variance, beta, gamma, 0),
        tensor2d([[1.5, 6.5], [35.5, 33]], [2, 2]));
  });
  it('2D, broadcast, gamma, beta', () => {
    const x = tensor2d([[10, 20], [30, 40]], [2, 2]);
    const mean = tensor1d([2, 5]);
    const variance = tensor1d([1, 4]);
    const gamma = tensor1d([3, 4]);
    const beta = tensor1d([-1, -2]);
    expectTensorsClose(
        batchNormalization(x, mean, variance, beta, gamma, 0),
        tensor2d([[23, 28], [83, 68]], [2, 2]));
  });
  it('3D, no broadcast, no gamma, no beta', () => {
    const x = tensor3d([[[10, 20], [30, 40]], [[10, 20], [30, 40]]], [2, 2, 2]);
    const mean = tensor3d([[[5, 5], [5, 5]], [[5, 5], [5, 5]]], [2, 2, 2]);
    const variance =
        tensor3d([[[4, 16], [4, 16]], [[16, 25], [16, 25]]], [2, 2, 2]);
    expectTensorsClose(
        batchNormalization(x, mean, variance, null, null, 0),
        tensor3d(
            [[[2.5, 3.75], [12.5, 8.75]], [[1.25, 3], [6.25, 7]]], [2, 2, 2]));
  });
  it('3D, no broadcast, gamma, beta', () => {
    const x = tensor3d([[[10, 20], [30, 40]], [[10, 20], [30, 40]]], [2, 2, 2]);
    const mean = tensor3d([[[5, 5], [5, 5]], [[5, 5], [5, 5]]], [2, 2, 2]);
    const variance =
        tensor3d([[[4, 16], [4, 16]], [[16, 25], [16, 25]]], [2, 2, 2]);
    const gamma = tensor3d([[[2, 2], [2, 2]], [[4, 4], [4, 4]]], [2, 2, 2]);
    const beta =
        tensor3d([[[-1, -1], [-2, -2]], [[-1, -1], [-2, -2]]], [2, 2, 2]);
    expectTensorsClose(
        batchNormalization(x, mean, variance, beta, gamma, 0),
        tensor3d([[[4, 6.5], [23, 15.5]], [[4, 11], [23, 26]]], [2, 2, 2]));
  });
  it('3D, broadcast, gamma, beta', () => {
    const x = tensor3d([[[10, 20], [30, 40]], [[10, 20], [30, 40]]], [2, 2, 2]);
    const mean = tensor1d([5, 5]);
    const variance = tensor1d([4, 16]);
    const gamma = tensor1d([2, 4]);
    const beta = tensor1d([-1, -2]);
    expectTensorsClose(
        batchNormalization(x, mean, variance, beta, gamma, 0),
        tensor3d([[[4, 13], [24, 33]], [[4, 13], [24, 33]]], [2, 2, 2]));
  });
  it('4D, no broadcast, no gamma, no beta', () => {
    const x = tensor4d(
        [
          [[[10, 20], [30, 40]], [[10, 20], [30, 40]]],
          [[[-10, -20], [-30, -40]], [[-10, -20], [-30, -40]]]
        ],
        [2, 2, 2, 2]);
    const mean = tensor4d(
        [
          [[[5, 5], [5, 5]], [[5, 5], [5, 5]]],
          [[[-5, -5], [-5, -5]], [[-5, -5], [-5, -5]]]
        ],
        [2, 2, 2, 2]);
    const variance = tensor4d(
        [
          [[[4, 16], [4, 16]], [[16, 25], [16, 25]]],
          [[[4, 16], [4, 16]], [[16, 25], [16, 25]]]
        ],
        [2, 2, 2, 2]);
    expectTensorsClose(
        batchNormalization(x, mean, variance, null, null, 0),
        tensor4d(
            [
              [[[2.5, 3.75], [12.5, 8.75]], [[1.25, 3], [6.25, 7]]],
              [[[-2.5, -3.75], [-12.5, -8.75]], [[-1.25, -3], [-6.25, -7]]]
            ],
            [2, 2, 2, 2]));
  });
  it('4D, no broadcast, gamma, beta', () => {
    const x = tensor4d(
        [
          [[[10, 20], [30, 40]], [[10, 20], [30, 40]]],
          [[[-10, -20], [-30, -40]], [[-10, -20], [-30, -40]]]
        ],
        [2, 2, 2, 2]);
    const mean = tensor4d(
        [
          [[[5, 5], [5, 5]], [[5, 5], [5, 5]]],
          [[[-5, -5], [-5, -5]], [[-5, -5], [-5, -5]]]
        ],
        [2, 2, 2, 2]);
    const variance = tensor4d(
        [
          [[[4, 16], [4, 16]], [[16, 25], [16, 25]]],
          [[[4, 16], [4, 16]], [[16, 25], [16, 25]]]
        ],
        [2, 2, 2, 2]);
    const gamma = tensor4d(
        [
          [[[2, 2], [2, 2]], [[4, 4], [4, 4]]],
          [[[2, 2], [2, 2]], [[4, 4], [4, 4]]]
        ],
        [2, 2, 2, 2]);
    const beta = tensor4d(
        [
          [[[-1, -1], [-2, -2]], [[-1, -1], [-2, -2]]],
          [[[1, 1], [2, 2]], [[1, 1], [2, 2]]]
        ],
        [2, 2, 2, 2]);
    expectTensorsClose(
        batchNormalization(x, mean, variance, beta, gamma, 0),
        tensor4d(
            [
              [[[4, 6.5], [23, 15.5]], [[4, 11], [23, 26]]],
              [[[-4, -6.5], [-23, -15.5]], [[-4, -11], [-23, -26]]]
            ],
            [2, 2, 2, 2]));
  });
  it('4D, broadcast, gamma, beta', () => {
    const x = tensor4d(
        [[[[10, 20], [30, 40]]], [[[10, 20], [30, 40]]]], [2, 1, 2, 2]);
    const mean = tensor1d([5, 5]);
    const variance = tensor1d([4, 16]);
    const gamma = tensor1d([2, 4]);
    const beta = tensor1d([-1, -2]);
    expectTensorsClose(
        batchNormalization(x, mean, variance, beta, gamma, 0),
        tensor4d([[[[4, 13], [24, 33]]], [[[4, 13], [24, 33]]]], [2, 1, 2, 2]));
  });
});

describeMathCPU('BatchNormalization Layers: Symbolic', () => {
  const validInputShapes = [[4, 6], [2, 3, 4], [2, 3, 4, 5]];
  for (const inputShape of validInputShapes) {
    const testTitle = `shape=${JSON.stringify(inputShape)}`;
    it(testTitle, () => {
      const x = new SymbolicTensor('float32', inputShape, null, [], null);
      const layer = tfl.layers.batchNormalization({});
      const y = layer.apply(x) as SymbolicTensor;
      expect(y.dtype).toEqual(x.dtype);
      expect(y.shape).toEqual(x.shape);
    });
  }

  it('Undetermined dim axis leads to ValueError', () => {
    const x = new SymbolicTensor('float32', [null, 2, 3], null, [], null);
    const layer = tfl.layers.batchNormalization({axis: 0});
    expect(() => layer.apply(x))
        .toThrowError(
            /Axis 0 of input tensor should have a defined dimension.*/);
  });

  it('batchNormalization constructor works without arg', () => {
    const layer = tfl.layers.batchNormalization();
    expect(layer.getConfig().axis).toEqual(-1);
  });
});

describeMathCPUAndGPU('BatchNormalization Layers: Tensor', () => {
  const dimensions = [2, 3, 4];
  const axisValues = [0, -1];

  for (const dim of dimensions) {
    for (const axis of axisValues) {
      const testTitle = `Inference, ${dim}D, axis=${axis}`;
      it(testTitle, () => {
        const layer = tfl.layers.batchNormalization({axis});
        let x: Tensor;
        if (dim === 2) {
          x = tensor2d([[1, 2], [3, 4]], [2, 2]);
        } else if (dim === 3) {
          x = tensor3d([[[1, 2], [3, 4]], [[-1, -2], [-3, -4]]], [2, 2, 2]);
        } else if (dim === 4) {
          x = tensor4d(
              [
                [[[1, 2], [3, 4]], [[-1, -2], [-3, -4]]],
                [[[-1, -2], [-3, -4]], [[1, 2], [3, 4]]]
              ],
              [2, 2, 2, 2]);
        }
        const y = layer.apply(x, {training: false}) as Tensor;
        expectTensorsClose(y, x, 0.01);
      });
    }
  }

  it('no center', () => {
    const layer = tfl.layers.batchNormalization({center: false, axis: 0});
    const x = tensor2d([[1, 2], [3, 4]], [2, 2]);
    expectTensorsClose(layer.apply(x) as Tensor, x, 0.01);
    expect(layer.getWeights().length).toEqual(3);
    // Firt weight is gamma.
    expectTensorsClose(layer.getWeights()[0], onesLike(layer.getWeights()[0]));
    // Second weight is moving mean.
    expectTensorsClose(layer.getWeights()[1], zerosLike(layer.getWeights()[1]));
    // Third weight is moving variance.
    expectTensorsClose(layer.getWeights()[2], onesLike(layer.getWeights()[2]));
  });

  it('no scale', () => {
    const layer = tfl.layers.batchNormalization({scale: false, axis: 0});
    const x = tensor2d([[1, 2], [3, 4]], [2, 2]);
    expectTensorsClose(layer.apply(x) as Tensor, x, 0.01);
    expect(layer.getWeights().length).toEqual(3);
    // Firt weight is beta.
    expectTensorsClose(layer.getWeights()[0], zerosLike(layer.getWeights()[0]));
    // Second weight is moving mean.
    expectTensorsClose(layer.getWeights()[1], zerosLike(layer.getWeights()[1]));
    // Third weight is moving variance.
    expectTensorsClose(layer.getWeights()[2], onesLike(layer.getWeights()[2]));
  });

  it('no center, no scale', () => {
    const layer = tfl.layers.batchNormalization({scale: false, center: false});
    const x = tensor2d([[1, 2], [3, 4]], [2, 2]);
    expectTensorsClose(layer.apply(x) as Tensor, x, 0.01);
    expect(layer.getWeights().length).toEqual(2);
    // First weight is moving mean.
    expectTensorsClose(layer.getWeights()[0], zerosLike(layer.getWeights()[0]));
    // Second weight is moving variance.
    expectTensorsClose(layer.getWeights()[1], onesLike(layer.getWeights()[1]));
  });

  // Use the following Python code to get the reference values for assertion:
  // ```python
  // from tensorflow import keras
  // import numpy as np
  //
  // layer1 = keras.layers.BatchNormalization(input_shape=(4,))
  // model = keras.Sequential([layer1])
  //
  // model.compile(loss='mean_squared_error', optimizer='sgd')
  //
  // xs = np.array(
  //     [[1, 2, 3, 4], [2, 4, 6, 8], [12, 11, 10, 9]], dtype=np.float32)
  // ys = np.zeros([3, 4])
  // print(layer1.get_weights())
  // history = model.fit(xs, ys, epochs=2, batch_size=3)
  // print(history.history)
  // print(layer1.get_weights())
  // ```
  it('Fit: 2D, BatchNorm Layer Only', async () => {
    const layer1 = tfl.layers.batchNormalization({inputShape: [4]});
    const model = tfl.sequential({layers: [layer1]});
    model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});

    const xs1 = tensor2d([[1, 2, 3, 4], [2, 4, 6, 8], [12, 11, 10, 9]], [3, 4]);
    const ys = zeros([3, 4]);
    const history = await model.fit(xs1, ys, {epochs: 2, batchSize: 3});
    expect(history.history['loss'][0]).toBeCloseTo(0.9998891353607178);
    expect(history.history['loss'][1]).toBeCloseTo(0.9899163246154785);
    const gammaValue = layer1.getWeights()[0];
    expectTensorsClose(
        gammaValue, [0.9900254, 0.9900257, 0.9900262, 0.9900271]);
    const betaValue = layer1.getWeights()[1];
    expectTensorsClose(
        betaValue,
        [2.9802322e-10,  1.4901161e-10,  9.1269614e-10, -7.4505802e-10]);
    const movingMeanValue = layer1.getWeights()[2];
    expectTensorsClose(
        movingMeanValue, [0.09949999, 0.11276666, 0.12603334, 0.13929999]);
    const movingVarianceValue = layer1.getWeights()[3];
    expectTensorsClose(
        movingVarianceValue, [1.4709667, 1.2763889, 1.1437222, 1.0729666]);
  });

  // Use the following Python code to get the reference values for
  // assertion:
  // ```python
  // from tensorflow import keras
  // import numpy as np
  //
  // layer1 = keras.layers.Dense(
  //     4, kernel_initializer='ones', use_bias=False, input_shape=(4,))
  // layer2 = keras.layers.BatchNormalization()
  // layer3 = keras.layers.Dense(1, kernel_initializer='ones',
  //                             use_bias=False)
  // model = keras.Sequential([layer1, layer2, layer3])
  //
  // optimizer = keras.optimizers.sgd(lr=0.1)
  // model.compile(loss='mean_squared_error', optimizer=optimizer)
  //
  // xs = np.array([[1, 2, 3, 4], [2, 4, 6, 8], [12, 11, 10, 9]],
  //                dtype=np.float32)
  // ys = np.zeros([3, 1])
  // history = model.fit(xs, ys, epochs=3, batch_size=3)
  //
  // print(history.history)
  // print(layer1.get_weights())
  // print(layer2.get_weights())
  // print(layer3.get_weights())
  // ```
  it('Fit: 2D, BatchNorm Layer between two Dense Layers', async () => {
    const layer1 = tfl.layers.dense(
        {units: 4, kernelInitializer: 'ones', useBias: false, inputShape: [4]});
    const layer2 = tfl.layers.batchNormalization({inputShape: [4]});
    const layer3 =
        tfl.layers.dense({units: 1, kernelInitializer: 'ones', useBias: false});
    const model = tfl.sequential({layers: [layer1, layer2, layer3]});

    const optimizer = train.sgd(0.1);
    model.compile({loss: 'meanSquaredError', optimizer});

    const xs1 = tensor2d([[1, 2, 3, 4], [2, 4, 6, 8], [12, 11, 10, 9]], [3, 4]);
    const ys = zeros([3, 1]);
    const history = await model.fit(xs1, ys, {epochs: 3, batchSize: 3});
    expect(history.history['loss'][0]).toBeCloseTo(15.999907493591309);
    expect(history.history['loss'][1]).toBeCloseTo(0.025602197274565697);
    expect(history.history['loss'][2]).toBeCloseTo(0.022478966042399406);
    const dense1KernelValue = layer1.getWeights()[0];
    expectTensorsClose(
        dense1KernelValue,
        tensor2d(
            [
            [0.99999833, 0.99999833, 0.99999833, 0.99999833],
            [0.9999987, 0.9999987, 0.9999987, 0.9999987],
            [0.999999, 0.999999, 0.999999, 0.999999],
            [0.99999934, 0.99999934, 0.99999934, 0.99999934]
            ],
            [4, 4]));
    const gammaValue = layer2.getWeights()[0];
    expectTensorsClose(
        gammaValue, [0.18779878, 0.18779878, 0.18779878, 0.18779878]);
    const betaValue = layer2.getWeights()[1];
    expectTensorsClose(
        betaValue,
        [5.5367128e-08, 5.5367128e-08, 5.5367128e-08, 5.5367128e-08]);
    const movingMeanValue = layer2.getWeights()[2];
    // TODO(cais): Update this to tf.keras.
    expectTensorsClose(
        movingMeanValue, [0.7128234, 0.7128234, 0.7128234, 0.7128234]);
    const movingVarianceValue = layer2.getWeights()[3];
    expectTensorsClose(
        movingVarianceValue,[6.276868, 6.276868, 6.276868, 6.276868]);
    const dense2KernelValue = layer3.getWeights()[0];
    expectTensorsClose(
        dense2KernelValue,
        tensor2d(
            [[0.18779878], [0.18779878], [0.18779878], [0.18779878]],
            [4, 1]));
  });

  // Python reference code:
  // ```python
  // import numpy as np
  // from tensorflow import keras
  //
  // model = keras.Sequential()
  // model.add(keras.layers.Conv2D(
  //     4,
  //     2,
  //     kernel_initializer='ones',
  //     bias_initializer='zeros',
  //     input_shape=[5, 5, 1]))
  // model.add(keras.layers.BatchNormalization())
  // model.add(keras.layers.Flatten())
  // model.add(keras.layers.Dense(
  //     1,
  //     kernel_initializer='ones',
  //     bias_initializer='zeros'))
  //
  // model.compile(loss='mse', optimizer='sgd')
  //
  // xs = np.arange(2 * 5 * 5 * 1).reshape([2, 5, 5, 1])
  // ys = np.array([[0], [1]])
  // h = model.fit(xs, ys, epochs=3)
  // print(h.history)
  // ```
  it('Fit: Wtih conv2d layer', async () => {
    const model = tfl.sequential();
    model.add(tfl.layers.conv2d({
      filters: 4,
      kernelSize: 2,
      kernelInitializer: 'ones',
      biasInitializer: 'zeros',
      inputShape: [5, 5, 1]
    }));
    model.add(tfl.layers.batchNormalization());
    model.add(tfl.layers.flatten());
    model.add(tfl.layers.dense({
      units: 1,
      kernelInitializer: 'ones',
      biasInitializer: 'zeros'
    }));
    model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});

    const xsData = [];
    for (let i = 0; i < 2 * 5 * 5 * 1; ++i) {
      xsData.push(i);
    }
    const xs = tensor4d(xsData, [2, 5, 5, 1]);
    const ys = tensor2d([0, 1], [2, 1]);

    const h = await model.fit(xs, ys, {epochs: 2});
    expectTensorsClose(
        h.history.loss as number[], [3332.9971, 2122.5361], 0.01);
  });

  // Reference Python code:
  // ```python
  // import numpy as np
  // import tensorflow as tf
  // from tensorflow import keras
  //
  // model = keras.Sequential()
  // model.add(keras.layers.Conv2DTranspose(
  //     4,
  //     2,
  //     kernel_initializer='ones',
  //     bias_initializer='zeros',
  //     input_shape=[5, 5, 1]))
  // model.add(keras.layers.BatchNormalization())
  // model.add(keras.layers.Flatten())
  // model.add(keras.layers.Dense(
  //     1,
  //     kernel_initializer='ones',
  //     bias_initializer='zeros'))
  //
  // model.compile(loss='mse', optimizer='sgd')
  //
  // xs = np.arange(2 * 5 * 5 * 1).reshape([2, 5, 5, 1]).astype(np.float32)
  // xs = (xs - 25.0) / 100.0
  // ys = np.array([[0], [1]])
  //
  // print(model.layers[1].get_weights())
  // h = model.fit(xs, ys, epochs=2)
  // print(h.history)
  // print(model.layers[1].get_weights())
  // ```
  it('Fit: Wtih conv2dTranspose layer', async () => {
    const model = tfl.sequential();
    model.add(tfl.layers.conv2dTranspose({
      filters: 4,
      kernelSize: 2,
      kernelInitializer: 'ones',
      biasInitializer: 'zeros',
      inputShape: [5, 5, 1]
    }));
    model.add(tfl.layers.batchNormalization());
    model.add(tfl.layers.flatten());
    model.add(tfl.layers.dense({
      units: 1,
      kernelInitializer: 'ones',
      biasInitializer: 'zeros'
    }));
    model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});

    const xsData = [];
    for (let i = 0; i < 2 * 5 * 5 * 1; ++i) {
      xsData.push(i);
    }
    const xs =
        (tensor4d(xsData, [2, 5, 5, 1]).sub(scalar(25))).div(scalar(100));
    const ys = tensor2d([0, 1], [2, 1]);

    const h = await model.fit(xs, ys, {epochs: 2});
    expect(h.history.loss[0]).toBeCloseTo(13922.4492);
    expect(h.history.loss[1]).toBeCloseTo(106532048, -3);
    const weights = model.layers[1].getWeights();
    expect(weights.length).toEqual(4);
    expectTensorsClose(
        weights[0],
        tensor1d([7661.0874, 7661.0874, 7661.0874, 7661.0874]), 1e-2);
    expectTensorsClose(
        weights[1],
        tensor1d([-118.35103, -118.35103, -118.35103, -118.35103]), 1e-2);
    expectTensorsClose(
        weights[2],
        tensor1d([-0.00026271, -0.00026271, -0.00026271, -0.00026271]));
    expectTensorsClose(
        weights[3],
        tensor1d([0.98333836, 0.98333836, 0.98333836, 0.98333836]));
  });

  // Use the following Python code to get the reference values for assertion:
  // ```python
  // from tensorflow import keras
  // import numpy as np
  //
  // layer1 = keras.layers.BatchNormalization(input_shape=[2, 2])
  // model = keras.Sequential([layer1])
  //
  // model.compile(loss='mean_squared_error', optimizer='sgd')
  //
  // xs = np.array(
  //     [[[1, 2], [3, 4]], [[2, 4], [6, 8]], [[12, 11], [10, 9]]],
  //     dtype=np.float32)
  // ys = np.zeros([3, 2, 2], dtype=np.float32)
  // print(layer1.get_weights())
  // history = model.fit(xs, ys, epochs=2, batch_size=3)
  // print(history.history)
  // print(layer1.get_weights())
  // ```
  it('Fit: 3D, BatchNorm Layer Only', async () => {
    const layer1 = tfl.layers.batchNormalization({inputShape: [2, 2]});
    const model = tfl.sequential({layers: [layer1]});
    model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});

    const xs1 = tensor3d(
        [[[1, 2], [3, 4]], [[2, 4], [6, 8]], [[12, 11], [10, 9]]], [3, 2, 2]);
    const ys = zeros([3, 2, 2]);
    const history = await model.fit(xs1, ys, {epochs: 2, batchSize: 3});
    expect(history.history['loss'][0]).toBeCloseTo(0.9999215006828308);
    expect(history.history['loss'][1]).toBeCloseTo(0.980024516582489);
    const gammaValue = layer1.getWeights()[0];
    expectTensorsClose(gammaValue, [0.98010117, 0.98010194]);
    const betaValue = layer1.getWeights()[1];
    expectTensorsClose(betaValue, [-1.1175870e-09, 8.1956386e-10]);
    const movingMeanValue = layer1.getWeights()[2];
    expectTensorsClose(movingMeanValue, [0.11276666, 0.12603334]);
    const movingVarianceValue = layer1.getWeights()[3];
    expectTensorsClose(movingVarianceValue, [1.3161889, 1.1835222], 1e-5);
  });
});
