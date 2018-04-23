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

// tslint:disable:max-line-length
import {onesLike, Tensor, tensor2d, tensor3d, tensor4d, train, zeros, zerosLike} from '@tensorflow/tfjs-core';

import * as tfl from '../index';
import {DType} from '../types';
import {SymbolicTensor} from '../types';
import {describeMathCPU, describeMathCPUAndGPU, expectTensorsClose} from '../utils/test_utils';
// tslint:enable:max-line-length

describeMathCPU('BatchNormalization Layers: Symbolic', () => {
  const validInputShapes = [[4, 6], [2, 3, 4], [2, 3, 4, 5]];
  for (const inputShape of validInputShapes) {
    const testTitle = `shape=${JSON.stringify(inputShape)}`;
    it(testTitle, () => {
      const x = new SymbolicTensor(DType.float32, inputShape, null, [], null);
      const layer = tfl.layers.batchNormalization({});
      const y = layer.apply(x) as SymbolicTensor;
      expect(y.dtype).toEqual(x.dtype);
      expect(y.shape).toEqual(x.shape);
    });
  }

  it('Undetermined dim axis leads to ValueError', () => {
    const x = new SymbolicTensor(DType.float32, [null, 2, 3], null, [], null);
    const layer = tfl.layers.batchNormalization({axis: 0});
    expect(() => layer.apply(x))
        .toThrowError(
            /Axis 0 of input tensor should have a defined dimension.*/);
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

  it('Fit: 2D, BatchNorm Layer Only', async done => {
    // Use the following Python code to get the reference values for assertion:
    // ```python
    // import keras
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
    const layer1 = tfl.layers.batchNormalization({inputShape: [4]});
    const model = tfl.sequential({layers: [layer1]});
    model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});

    const xs1 = tensor2d([[1, 2, 3, 4], [2, 4, 6, 8], [12, 11, 10, 9]], [3, 4]);
    const ys = zeros([3, 4]);
    model.fit(xs1, ys, {epochs: 2, batchSize: 3})
        .then(history => {
          expect(history.history['loss'][0]).toBeCloseTo(0.9998891353607178);
          expect(history.history['loss'][1]).toBeCloseTo(0.9899163246154785);
          const gammaValue = layer1.getWeights()[0];
          expectTensorsClose(
              gammaValue, [0.9900254, 0.9900257, 0.9900262, 0.9900271]);
          const betaValue = layer1.getWeights()[1];
          expectTensorsClose(
              betaValue,
              [2.9802322e-10, 1.4901161e-10, 8.9406960e-10, -7.4505802e-10]);
          const movingMeanValue = layer1.getWeights()[2];
          expectTensorsClose(
              movingMeanValue, [5.0000086, 5.6666765, 6.333345, 7.000012]);
          const movingVarianceValue = layer1.getWeights()[3];
          expectTensorsClose(
              movingVarianceValue, [37.018574, 22.344547, 12.339525, 7.003515]);
          done();
        })
        .catch(err => {
          console.error(err.stack);
        });
  });

  it('Fit: 2D, BatchNorm Layer between two Dense Layers', async done => {
    // Use the following Python code to get the reference values for
    // assertion:
    // ```python
    // import keras
    // from keras import backend as keras_backend
    // import numpy as np
    //
    // layer1 = keras.layers.Dense(
    //     4, kernel_initializer='ones', use_bias=False, input_shape=(4,))
    // layer2 = keras.layers.BatchNormalization()
    // layer3 = keras.layers.Dense(1, kernel_initializer='ones',
    // use_bias=False) model = keras.Sequential([layer1, layer2, layer3])
    //
    // optimizer = keras.optimizers.sgd(lr=0.1)
    // model.compile(loss='mean_squared_error', optimizer=optimizer)
    //
    // xs = np.array([[1, 2, 3, 4], [2, 4, 6, 8], [12, 11, 10, 9]],
    // dtype=np.float32) ys = np.zeros([3, 1]) history = model.fit(xs, ys,
    // epochs=3, batch_size=3)
    //
    // print(history.history)
    // print(layer1.get_weights())
    // print(layer2.get_weights())
    // print(layer3.get_weights())
    // ```
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
    model.fit(xs1, ys, {epochs: 3, batchSize: 3})
        .then(history => {
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
          expectTensorsClose(
              movingMeanValue, [23.999907, 23.999907, 23.999907, 23.999907]);
          const movingVarianceValue = layer2.getWeights()[3];
          expectTensorsClose(
              movingVarianceValue,
              [268.13364, 268.13364, 268.13364, 268.13364]);
          const dense2KernelValue = layer3.getWeights()[0];
          expectTensorsClose(
              dense2KernelValue,
              tensor2d(
                  [[0.18779878], [0.18779878], [0.18779878], [0.18779878]],
                  [4, 1]));
          done();
        })
        .catch(err => {
          console.error(err.stack);
        });
  });

  it('Fit: 3D, BatchNorm Layer Only', async done => {
    // Use the following Python code to get the reference values for assertion:
    // ```python
    // import keras
    // from keras import backend as keras_backend
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
    const layer1 = tfl.layers.batchNormalization({inputShape: [2, 2]});
    const model = tfl.sequential({layers: [layer1]});
    model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});

    const xs1 = tensor3d(
        [[[1, 2], [3, 4]], [[2, 4], [6, 8]], [[12, 11], [10, 9]]], [3, 2, 2]);
    const ys = zeros([3, 2, 2]);
    model.fit(xs1, ys, {epochs: 2, batchSize: 3})
        .then(history => {
          expect(history.history['loss'][0]).toBeCloseTo(0.9999215006828308);
          expect(history.history['loss'][1]).toBeCloseTo(0.980024516582489);
          const gammaValue = layer1.getWeights()[0];
          expectTensorsClose(gammaValue, [0.98010117, 0.98010194]);
          const betaValue = layer1.getWeights()[1];
          expectTensorsClose(betaValue, [-1.1175870e-09, 8.1956386e-10]);
          const movingMeanValue = layer1.getWeights()[2];
          expectTensorsClose(movingMeanValue, [5.6666765, 6.333345]);
          const movingVarianceValue = layer1.getWeights()[3];
          expectTensorsClose(movingVarianceValue, [20.270758, 12.269142]);
          done();
        })
        .catch(err => {
          console.error(err.stack);
        });
  });
});
