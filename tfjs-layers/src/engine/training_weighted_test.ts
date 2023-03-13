/**
 * @license
 * Copyright 2019 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

/** Unit tests for training with {sample, class} weights. */

import {memory, ones, Tensor, tensor2d, train, zeros} from '@tensorflow/tfjs-core';

import * as tfl from '../index';

import {describeMathCPUAndWebGL2, expectTensorsClose} from '../utils/test_utils';
import {FakeNumericDataset} from './dataset_fakes';
import {ClassWeight, ClassWeightMap} from './training_utils';

describeMathCPUAndWebGL2('LayersModel.fit() with classWeight', () => {
  // Reference Python code:
  // ```py
  // import numpy as np
  // import tensorflow as tf
  //
  // model = tf.keras.Sequential()
  // model.add(tf.keras.layers.Dense(
  //     units=3,
  //     input_shape=[2],
  //     kernel_initializer='zeros',
  //     activation='softmax'))
  // model.compile(loss='categorical_crossentropy',
  //               metrics=['acc'],
  //               optimizer=tf.keras.optimizers.SGD(1.0))
  // model.summary()
  //
  // xs = np.array([[0, 1], [0, 2], [1, 10], [1, 20], [2, -10], [2, -20]],
  //               dtype=np.float32)
  // ys = np.array([[1, 0, 0],
  //                [1, 0, 0],
  //                [0, 1, 0],
  //                [0, 1, 0],
  //                [0, 0, 1],
  //                [0, 0, 1]], dtype=np.float32)
  //
  // model.fit(xs,
  //           ys,
  //           epochs=2,
  //           class_weight=[{
  //             0: 1,
  //             1: 10,
  //             2: 1
  //           }])
  // print(model.get_weights()[0])
  // ```
  it('One output, multi-class, one-hot encoding', async () => {
    const model = tfl.sequential();
    model.add(tfl.layers.dense({
      units: 3,
      inputShape: [2],
      kernelInitializer: 'zeros',
      activation: 'softmax'
    }));
    model.compile({
      loss: 'categoricalCrossentropy',
      metrics: ['acc'],
      optimizer: train.sgd(1)
    });

    const xs = tensor2d([[0, 1], [0, 2], [1, 10], [1, 20], [2, -10], [2, -20]]);
    const ys = tensor2d(
        [[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]]);
    const numTensors0 = memory().numTensors;
    const history = await model.fit(
        xs, ys, {epochs: 2, classWeight: [{0: 1, 1: 10, 2: 1}]});
    expect(memory().numTensors).toEqual(numTensors0);  // Assert no memory leak.
    expect(history.history.loss.length).toEqual(2);
    // These loss values are different than what the values would be
    // if there is no class weighting.
    expect(history.history.loss[0]).toBeCloseTo(4.3944);
    expect(history.history.loss[1]).toBeCloseTo(5.3727);
    expect(history.history.acc.length).toEqual(2);
    expect(history.history.acc[0]).toBeCloseTo(0.3333);
    expect(history.history.acc[1]).toBeCloseTo(0.6667);
  });

  it('One output, multi-class, one-hot encoding, batchSize = 2', async () => {
    const model = tfl.sequential();
    model.add(tfl.layers.dense({
      units: 3,
      inputShape: [2],
      kernelInitializer: 'zeros',
      activation: 'softmax'
    }));
    model.compile({
      loss: 'categoricalCrossentropy',
      metrics: ['acc'],
      optimizer: train.sgd(1)
    });

    const xs = tensor2d([[0, 1], [0, 2], [1, 10], [1, 20], [2, -10], [2, -20]]);
    const ys = tensor2d(
        [[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]]);
    const numTensors0 = memory().numTensors;
    const history = await model.fit(xs, ys, {
      epochs: 2,
      batchSize: 2,
      shuffle: false,
      classWeight: [{0: 1, 1: 10, 2: 1}]
    });
    expect(memory().numTensors).toEqual(numTensors0);  // Assert no memory leak.
    expect(history.history.loss.length).toEqual(2);
    // These loss values are different than what the values would be
    // if there is no class weighting.
    expect(history.history.loss[0]).toBeCloseTo(59.2691);
    expect(history.history.loss[1]).toBeCloseTo(10.7454);
    expect(history.history.acc.length).toEqual(2);
    expect(history.history.acc[0]).toBeCloseTo(0.3333);
    expect(history.history.acc[1]).toBeCloseTo(0.3333);
  });

  it('One output, multi-class, one-hot encoding, validationData', async () => {
    const model = tfl.sequential();
    model.add(tfl.layers.dense({
      units: 3,
      inputShape: [2],
      kernelInitializer: 'zeros',
      activation: 'softmax'
    }));
    model.compile({
      loss: 'categoricalCrossentropy',
      metrics: ['acc'],
      optimizer: train.sgd(1)
    });

    const xs = tensor2d([[0, 1], [0, 2], [1, 10], [1, 20], [2, -10], [2, -20]]);
    const ys = tensor2d(
        [[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]]);
    const numTensors0 = memory().numTensors;
    const history = await model.fit(xs, ys, {
      epochs: 2,
      classWeight: [{0: 1, 1: 10, 2: 1}],
      validationData: [xs, ys]
    });
    expect(memory().numTensors).toEqual(numTensors0);  // Assert no memory leak.
    expect(history.history.loss.length).toEqual(2);
    // These loss values are different than what the values would be
    // if there is no class weighting.
    expect(history.history.loss[0]).toBeCloseTo(4.3944);
    expect(history.history.loss[1]).toBeCloseTo(5.3727);
    expect(history.history.acc.length).toEqual(2);
    expect(history.history.acc[0]).toBeCloseTo(0.3333);
    expect(history.history.acc[1]).toBeCloseTo(0.6667);

    expect(history.history.val_loss.length).toEqual(2);
    expect(history.history.val_loss[0]).toBeCloseTo(5.3727);
    expect(history.history.val_loss[1]).toBeCloseTo(5.3727);
    expect(history.history.val_acc.length).toEqual(2);
    expect(history.history.val_acc[0]).toBeCloseTo(0.6667);
    expect(history.history.val_acc[1]).toBeCloseTo(0.6667);
  });

  it('One output, multi-class, one-hot encoding, validationSplit', async () => {
    const model = tfl.sequential();
    model.add(tfl.layers.dense({
      units: 3,
      inputShape: [2],
      kernelInitializer: 'zeros',
      activation: 'softmax'
    }));
    model.compile({
      loss: 'categoricalCrossentropy',
      metrics: ['acc'],
      optimizer: train.sgd(1)
    });

    const xs = tensor2d([[0, 1], [0, 2], [1, 10], [1, 20], [2, -10], [2, -20]]);
    const ys = tensor2d(
        [[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]]);
    const numTensors0 = memory().numTensors;
    const history = await model.fit(
        xs, ys,
        {epochs: 2, classWeight: [{0: 1, 1: 10, 2: 1}], validationSplit: 0.5});
    expect(memory().numTensors).toEqual(numTensors0);  // Assert no memory leak.
    expect(history.history.loss.length).toEqual(2);
    // These loss values are different than what the values would be
    // if there is no class weighting.
    expect(history.history.loss[0]).toBeCloseTo(4.3944);
    expect(history.history.loss[1]).toBeCloseTo(10.7454);
    expect(history.history.acc.length).toEqual(2);
    expect(history.history.acc[0]).toBeCloseTo(0.6667);
    expect(history.history.acc[1]).toBeCloseTo(0.3333);

    expect(history.history.val_loss.length).toEqual(2);
    expect(history.history.val_loss[0]).toBeCloseTo(2.9903e-05);
    expect(history.history.val_loss[1]).toBeCloseTo(2.9903e-05);
    expect(history.history.val_acc.length).toEqual(2);
    expect(history.history.val_acc[0]).toBeCloseTo(1);
    expect(history.history.val_acc[1]).toBeCloseTo(1);
  });

  it('One output, multi-class, sparse encoding', async () => {
    const model = tfl.sequential();
    model.add(tfl.layers.dense({
      units: 3,
      inputShape: [2],
      kernelInitializer: 'zeros',
      activation: 'softmax'
    }));
    model.compile(
        {loss: 'sparseCategoricalCrossentropy', optimizer: train.sgd(1)});

    const xs = tensor2d([[0, 1], [0, 2], [1, 10], [1, 20], [2, -10], [2, -20]]);
    const ys = tensor2d([[0], [0], [1], [1], [2], [2]]);
    const numTensors0 = memory().numTensors;
    const history =
        await model.fit(xs, ys, {epochs: 2, classWeight: {0: 1, 1: 10, 2: 1}});
    expect(memory().numTensors).toEqual(numTensors0);  // Assert no memory leak.
    expect(history.history.loss.length).toEqual(2);
    // These loss values are different than what the values would be
    // if there is no class weighting.
    expect(history.history.loss[0]).toBeCloseTo(4.3944);
    expect(history.history.loss[1]).toBeCloseTo(5.3727);
  });

  // Reference Python code.
  // ```py
  // import numpy as np
  // import tensorflow as tf
  //
  // model = tf.keras.Sequential()
  // model.add(tf.keras.layers.Dense(
  //     units=1,
  //     input_shape=[2],
  //     kernel_initializer='zeros',
  //     activation='sigmoid'))
  // model.compile(loss='binary_crossentropy',
  //               optimizer=tf.keras.optimizers.SGD(1.0))
  // model.summary()
  //
  // xs = np.array([[0, 1], [0, 2], [1, 10], [1, 20]],
  //               dtype=np.float32)
  // ys = np.array([[0], [0], [1], [1]], dtype=np.float32)
  //
  // # model.fit(xs, ys, epochs=1)
  // model.fit(xs,
  //           ys,
  //           epochs=3,
  //           class_weight=[{
  //               0: 0.1,
  //               1: 0.9
  //           }])
  // print(model.get_weights()[0])
  // ```
  it('One output, binary classes, sparse encoding', async () => {
    const model = tfl.sequential();
    model.add(tfl.layers.dense({
      units: 1,
      inputShape: [2],
      kernelInitializer: 'zeros',
      activation: 'sigmoid'
    }));
    model.compile({loss: 'binaryCrossentropy', optimizer: train.sgd(1)});

    const xs = tensor2d([[0, 1], [0, 2], [1, 10], [1, 20]]);
    const ys = tensor2d([[0], [0], [1], [1]]);
    const numTensors0 = memory().numTensors;
    const history =
        await model.fit(xs, ys, {epochs: 2, classWeight: [{0: 0.1, 1: 0.9}]});
    expect(memory().numTensors).toEqual(numTensors0);  // Assert no memory leak.
    expect(history.history.loss.length).toEqual(2);
    // These loss values are different than what the values would be
    // if there is no class weighting.
    expect(history.history.loss[0]).toBeCloseTo(0.3466);
    expect(history.history.loss[1]).toBeCloseTo(0.2611);
  });

  // Python Reference Code:
  // ```py
  // import numpy as np
  // import tensorflow as tf
  //
  // inp = tf.keras.Input(shape=[2])
  // y1 = tf.keras.layers.Dense(
  //     units=3,
  //     input_shape=[2],
  //     kernel_initializer='zeros',
  //     activation='softmax')(inp)
  // y2 = tf.keras.layers.Dense(
  //     units=1,
  //     input_shape=[2],
  //     kernel_initializer='zeros',
  //     activation='sigmoid')(inp)
  // model = tf.keras.Model(inp, [y1, y2])
  // model.compile(
  //     loss=['sparse_categorical_crossentropy', 'binary_crossentropy'],
  //     optimizer=tf.keras.optimizers.SGD(1.0))
  // model.summary()
  //
  // xs = np.array(
  //     [[0, 1], [0, 2], [1, 10], [1, 20], [2, 10], [2, 20]],
  //     dtype=np.float32)
  // y1s = np.array(
  //     [[0], [0], [1], [1], [2], [2]], dtype=np.float32)
  // y2s = np.array(
  //     [[0], [0], [1], [1], [1], [1]], dtype=np.float32)
  //
  // # model.fit(xs, ys, epochs=1)
  // model.fit(xs,
  //           [y1s, y2s],
  //           epochs=3,
  //           class_weight=[{
  //               0: 0.1,
  //               1: 0.2,
  //               2: 0.7
  //           }, {
  //               0: 0.1,
  //               1: 0.9
  //           }])
  // ```
  it('Two outputs, classWeight as array', async () => {
    const inp = tfl.input({shape: [2]});
    const y1 = tfl.layers
                   .dense({
                     units: 3,
                     inputShape: [2],
                     kernelInitializer: 'zeros',
                     activation: 'softmax'
                   })
                   .apply(inp) as tfl.SymbolicTensor;
    const y2 = tfl.layers
                   .dense({
                     units: 1,
                     inputShape: [2],
                     kernelInitializer: 'zeros',
                     activation: 'sigmoid'
                   })
                   .apply(inp) as tfl.SymbolicTensor;
    const model = tfl.model({inputs: inp, outputs: [y1, y2]});
    model.compile({
      loss: ['sparseCategoricalCrossentropy', 'binaryCrossentropy'],
      optimizer: train.sgd(1)
    });

    const xs = tensor2d([[0, 1], [0, 2], [1, 10], [1, 20], [2, 10], [2, 20]]);
    const y1s = tensor2d([[0], [0], [1], [1], [2], [2]]);
    const y2s = tensor2d([[0], [0], [1], [1], [1], [1]]);

    const numTensors0 = memory().numTensors;
    const history = await model.fit(
        xs, [y1s, y2s],
        {epochs: 3, classWeight: [{0: 0.1, 1: 0.2, 2: 0.7}, {0: 0.1, 1: 0.9}]});
    expect(memory().numTensors).toEqual(numTensors0);  // Assert no memory leak.
    expect(history.history.loss.length).toEqual(3);
    expect(history.history.loss[0]).toBeCloseTo(0.8052);
    expect(history.history.loss[1]).toBeCloseTo(1.4887);
    expect(history.history.loss[2]).toBeCloseTo(1.4782);
    const lossKey0 = `${model.outputNames[0]}_loss`;
    expect(history.history[lossKey0].length).toEqual(3);
    expect(history.history[lossKey0][0]).toBeCloseTo(0.3662);
    expect(history.history[lossKey0][1]).toBeCloseTo(1.2553);
    expect(history.history[lossKey0][2]).toBeCloseTo(1.2485);
    const lossKey1 = `${model.outputNames[1]}_loss`;
    expect(history.history[lossKey1].length).toEqual(3);
    expect(history.history[lossKey1][0]).toBeCloseTo(0.4390);
    expect(history.history[lossKey1][1]).toBeCloseTo(0.2333);
    expect(history.history[lossKey1][2]).toBeCloseTo(0.2297);
  });

  it('Two outputs, classWeight as array, one being null', async () => {
    const inp = tfl.input({shape: [2]});
    const y1 = tfl.layers
                   .dense({
                     units: 3,
                     inputShape: [2],
                     kernelInitializer: 'zeros',
                     activation: 'softmax'
                   })
                   .apply(inp) as tfl.SymbolicTensor;
    const y2 = tfl.layers
                   .dense({
                     units: 1,
                     inputShape: [2],
                     kernelInitializer: 'zeros',
                     activation: 'sigmoid'
                   })
                   .apply(inp) as tfl.SymbolicTensor;
    const model = tfl.model({inputs: inp, outputs: [y1, y2]});
    model.compile({
      loss: ['sparseCategoricalCrossentropy', 'binaryCrossentropy'],
      optimizer: train.sgd(1)
    });

    const xs = tensor2d([[0, 1], [0, 2], [1, 10], [1, 20], [2, 10], [2, 20]]);
    const y1s = tensor2d([[0], [0], [1], [1], [2], [2]]);
    const y2s = tensor2d([[0], [0], [1], [1], [1], [1]]);

    const numTensors0 = memory().numTensors;
    const history = await model.fit(
        xs, [y1s, y2s],
        {epochs: 3, classWeight: [null, {0: 0.1, 1: 0.9}], shuffle: false});
    expect(memory().numTensors).toEqual(numTensors0);  // Assert no memory leak.
    // Note that the following values don't match results from Python,
    // which is a bug in Python TensorFlow / tf.keras. But the final
    // kernel value does match Python results. (See b/136123157).
    expect(history.history.loss.length).toEqual(3);
    expect(history.history.loss[0]).toBeCloseTo(1.5376);
    expect(history.history.loss[1]).toBeCloseTo(3.1466);
    expect(history.history.loss[2]).toBeCloseTo(7.9620);
    const lossKey0 = `${model.outputNames[0]}_loss`;
    expect(history.history[lossKey0].length).toEqual(3);
    expect(history.history[lossKey0][0]).toBeCloseTo(1.0986);
    expect(history.history[lossKey0][1]).toBeCloseTo(2.9113);
    expect(history.history[lossKey0][2]).toBeCloseTo(7.7323);
    const lossKey1 = `${model.outputNames[1]}_loss`;
    expect(history.history[lossKey1].length).toEqual(3);
    expect(history.history[lossKey1][0]).toBeCloseTo(0.4390);
    expect(history.history[lossKey1][1]).toBeCloseTo(0.2333);
    expect(history.history[lossKey1][2]).toBeCloseTo(0.2298);
    expectTensorsClose(model.getWeights()[0], tensor2d([
                         [-0.3333333, -0.03197281, 0.3653062],
                         [-2.0025878, 1.9823718, 0.02021614]
                       ]));
  });

  it('Two outputs, classWeight as map, one output no weighting', async () => {
    const inp = tfl.input({shape: [2]});
    const y1 = tfl.layers
                   .dense({
                     units: 3,
                     inputShape: [2],
                     kernelInitializer: 'zeros',
                     activation: 'softmax'
                   })
                   .apply(inp) as tfl.SymbolicTensor;
    const y2 = tfl.layers
                   .dense({
                     units: 1,
                     inputShape: [2],
                     kernelInitializer: 'zeros',
                     activation: 'sigmoid'
                   })
                   .apply(inp) as tfl.SymbolicTensor;
    const model = tfl.model({inputs: inp, outputs: [y1, y2]});
    model.compile({
      loss: ['sparseCategoricalCrossentropy', 'binaryCrossentropy'],
      optimizer: train.sgd(1)
    });

    const xs = tensor2d([[0, 1], [0, 2], [1, 10], [1, 20], [2, 10], [2, 20]]);
    const y1s = tensor2d([[0], [0], [1], [1], [2], [2]]);
    const y2s = tensor2d([[0], [0], [1], [1], [1], [1]]);

    const numTensors0 = memory().numTensors;
    const history = await model.fit(xs, [y1s, y2s], {
      epochs: 3,
      classWeight: {
        [model.outputNames[1]]: {0: 0.1, 1: 0.9},
      },
      shuffle: false
    });
    expect(memory().numTensors).toEqual(numTensors0);  // Assert no memory leak.
    // Note that the following values don't match results from Python,
    // which is a bug in Python TensorFlow / tf.keras. But the final
    // kernel value does match Python results.
    expect(history.history.loss.length).toEqual(3);
    expect(history.history.loss[0]).toBeCloseTo(1.5376);
    expect(history.history.loss[1]).toBeCloseTo(3.1466);
    expect(history.history.loss[2]).toBeCloseTo(7.9620);
    const lossKey0 = `${model.outputNames[0]}_loss`;
    expect(history.history[lossKey0].length).toEqual(3);
    expect(history.history[lossKey0][0]).toBeCloseTo(1.0986);
    expect(history.history[lossKey0][1]).toBeCloseTo(2.9113);
    expect(history.history[lossKey0][2]).toBeCloseTo(7.7323);
    const lossKey1 = `${model.outputNames[1]}_loss`;
    expect(history.history[lossKey1].length).toEqual(3);
    expect(history.history[lossKey1][0]).toBeCloseTo(0.4390);
    expect(history.history[lossKey1][1]).toBeCloseTo(0.2333);
    expect(history.history[lossKey1][2]).toBeCloseTo(0.2298);
    expectTensorsClose(model.getWeights()[0], tensor2d([
                         [-0.3333333, -0.03197281, 0.3653062],
                         [-2.0025878, 1.9823718, 0.02021614]
                       ]));
  });

  it('Two outputs, classWeight as map', async () => {
    const inp = tfl.input({shape: [2]});
    const y1 = tfl.layers
                   .dense({
                     units: 3,
                     inputShape: [2],
                     kernelInitializer: 'zeros',
                     activation: 'softmax'
                   })
                   .apply(inp) as tfl.SymbolicTensor;
    const y2 = tfl.layers
                   .dense({
                     units: 1,
                     inputShape: [2],
                     kernelInitializer: 'zeros',
                     activation: 'sigmoid'
                   })
                   .apply(inp) as tfl.SymbolicTensor;
    const model = tfl.model({inputs: inp, outputs: [y1, y2]});
    model.compile({
      loss: ['sparseCategoricalCrossentropy', 'binaryCrossentropy'],
      optimizer: train.sgd(1)
    });

    const xs = tensor2d([[0, 1], [0, 2], [1, 10], [1, 20], [2, 10], [2, 20]]);
    const y1s = tensor2d([[0], [0], [1], [1], [2], [2]]);
    const y2s = tensor2d([[0], [0], [1], [1], [1], [1]]);

    const numTensors0 = memory().numTensors;
    const history = await model.fit(xs, [y1s, y2s], {
      epochs: 3,
      classWeight: {
        [model.outputNames[0]]: {0: 0.1, 1: 0.2, 2: 0.7},
        [model.outputNames[1]]: {0: 0.1, 1: 0.9}
      }
    });
    expect(memory().numTensors).toEqual(numTensors0);  // Assert no memory leak.
    expect(history.history.loss.length).toEqual(3);
    expect(history.history.loss[0]).toBeCloseTo(0.8052);
    expect(history.history.loss[1]).toBeCloseTo(1.4887);
    expect(history.history.loss[2]).toBeCloseTo(1.4782);
    const lossKey0 = `${model.outputNames[0]}_loss`;
    expect(history.history[lossKey0].length).toEqual(3);
    expect(history.history[lossKey0][0]).toBeCloseTo(0.3662);
    expect(history.history[lossKey0][1]).toBeCloseTo(1.2553);
    expect(history.history[lossKey0][2]).toBeCloseTo(1.2485);
    const lossKey1 = `${model.outputNames[1]}_loss`;
    expect(history.history[lossKey1].length).toEqual(3);
    expect(history.history[lossKey1][0]).toBeCloseTo(0.4390);
    expect(history.history[lossKey1][1]).toBeCloseTo(0.2333);
    expect(history.history[lossKey1][2]).toBeCloseTo(0.2297);
  });
});

describeMathCPUAndWebGL2('LayersModel.fitDataset() with classWeight', () => {
  // Reference Python code:
  // ```py
  // import numpy as np
  // import tensorflow as tf
  //
  // batch_size = 8
  // num_batches = 3
  // epochs = 2
  //
  // xs = np.ones([batch_size * num_batches * epochs, 1])
  // ys = np.concatenate([
  //     np.zeros([int(batch_size * num_batches * epochs / 2), 1]),
  //     np.ones([int(batch_size * num_batches * epochs / 2), 1])],
  //     axis=0)
  //
  // dataset = tf.data.Dataset.from_tensor_slices(
  //     (xs, ys)).batch(batch_size)
  //
  // model = tf.keras.Sequential()
  // model.add(tf.keras.layers.Dense(
  //     1,
  //     input_shape=[1],
  //     activation='sigmoid',
  //     kernel_initializer='zeros'))
  // model.compile(loss='binary_crossentropy',
  //               metrics=['acc'],
  //               optimizer='sgd')
  //
  // history = model.fit(
  //     dataset,
  //     steps_per_epoch=num_batches,
  //     epochs=epochs,
  //     class_weight={0: 0.1, 1: 2})
  // print(history.history)
  // ```
  it('One output, binary crossentropy, acc metric', async () => {
    const model = tfl.sequential();
    model.add(tfl.layers.dense({
      units: 1,
      inputShape: [1],
      kernelInitializer: 'zeros',
      activation: 'sigmoid'
    }));
    model.compile(
        {loss: 'binaryCrossentropy', metrics: ['acc'], optimizer: 'sgd'});

    const batchSize = 8;
    const batchesPerEpoch = 3;
    const epochs = 2;
    const xTensorsFunc = () =>
        [ones([batchSize, 1]), ones([batchSize, 1]), ones([batchSize, 1]),
         ones([batchSize, 1]), ones([batchSize, 1]), ones([batchSize, 1])];
    const yTensorsFunc = () =>
        [zeros([batchSize, 1]), zeros([batchSize, 1]), zeros([batchSize, 1]),
         ones([batchSize, 1]), ones([batchSize, 1]), ones([batchSize, 1])];
    const dataset = new FakeNumericDataset({
      xShape: [1],
      yShape: [1],
      batchSize,
      numBatches: batchesPerEpoch * epochs,
      xTensorsFunc,
      yTensorsFunc
    });

    const classWeight: ClassWeight = {0: 0.1, 1: 2};

    // Do a burn-in call to account for initialization of cached tensors (for
    // the memory-leak check below).
    await model.fitDataset(dataset, {batchesPerEpoch, epochs: 1, classWeight});
    model.setWeights([zeros([1, 1]), zeros([1])]);

    const numTensors0 = memory().numTensors;
    const history =
        await model.fitDataset(dataset, {batchesPerEpoch, epochs, classWeight});
    const numTensors1 = memory().numTensors;
    expect(numTensors1).toEqual(numTensors0);

    // The loss and acc values are different if no classWeight is used.
    expect(history.history.loss.length).toEqual(2);
    expect(history.history.loss[0]).toBeCloseTo(0.0693);
    expect(history.history.loss[1]).toBeCloseTo(1.3695);

    expect(history.history.acc.length).toEqual(2);
    expect(history.history.acc[0]).toBeCloseTo(1);
    expect(history.history.acc[1]).toBeCloseTo(0.6667);
  });

  // Reference Python code:
  // ```py
  // import numpy as np
  // import tensorflow as tf
  //
  // batch_size = 8
  // num_batches = 3
  // epochs = 2
  //
  // xs = np.ones([batch_size * num_batches * epochs, 1])
  // ys = np.concatenate([
  //     np.zeros([int(batch_size * num_batches * epochs / 2), 1]),
  //     np.ones([int(batch_size * num_batches * epochs / 2), 1])],
  //     axis=0)
  //
  // dataset = tf.data.Dataset.from_tensor_slices(
  //     (xs, ys)).batch(batch_size)
  //
  // model = tf.keras.Sequential()
  // model.add(tf.keras.layers.Dense(
  //     1,
  //     input_shape=[1],
  //     activation='sigmoid',
  //     kernel_initializer='zeros'))
  // model.compile(loss='binary_crossentropy',
  //               metrics=['acc'],
  //               optimizer='sgd')
  //
  // history = model.fit(
  //     dataset,
  //     steps_per_epoch=num_batches,
  //     epochs=epochs,
  //     class_weight={0: 0.1, 1: 2})
  // print(history.history)
  // ```
  it('One output, binary crossentropy，validation', async () => {
    const model = tfl.sequential();
    model.add(tfl.layers.dense({
      units: 1,
      inputShape: [1],
      kernelInitializer: 'zeros',
      activation: 'sigmoid'
    }));
    model.compile(
        {loss: 'binaryCrossentropy', metrics: ['acc'], optimizer: 'sgd'});

    const batchSize = 8;
    const batchesPerEpoch = 3;
    const epochs = 2;
    const xTensorsFunc = () =>
        [ones([batchSize, 1]), ones([batchSize, 1]), ones([batchSize, 1]),
         ones([batchSize, 1]), ones([batchSize, 1]), ones([batchSize, 1])];
    const yTensorsFunc = () =>
        [zeros([batchSize, 1]), zeros([batchSize, 1]), zeros([batchSize, 1]),
         ones([batchSize, 1]), ones([batchSize, 1]), ones([batchSize, 1])];
    const dataset = new FakeNumericDataset({
      xShape: [1],
      yShape: [1],
      batchSize,
      numBatches: batchesPerEpoch * epochs,
      xTensorsFunc,
      yTensorsFunc
    });

    const classWeight: ClassWeight = {0: 0.1, 1: 2};

    // Do a burn-in call to account for initialization of cached tensors (for
    // the memory-leak check below).
    await model.fitDataset(
        dataset,
        {batchesPerEpoch, epochs: 1, classWeight, validationData: dataset});
    model.setWeights([zeros([1, 1]), zeros([1])]);

    const numTensors0 = memory().numTensors;
    const history = await model.fitDataset(
        dataset,
        {batchesPerEpoch, epochs, classWeight, validationData: dataset});
    const numTensors1 = memory().numTensors;
    expect(numTensors1).toEqual(numTensors0);

    // The loss and acc values are different if no classWeight is used.
    expect(history.history.loss.length).toEqual(2);
    expect(history.history.loss[0]).toBeCloseTo(0.0693);
    expect(history.history.loss[1]).toBeCloseTo(1.3695);

    expect(history.history.acc.length).toEqual(2);
    expect(history.history.acc[0]).toBeCloseTo(1);
    expect(history.history.acc[1]).toBeCloseTo(0.6667);

    expect(history.history.val_loss.length).toEqual(2);
    expect(history.history.val_loss[0]).toBeCloseTo(0.693148);
    expect(history.history.val_loss[1]).toBeCloseTo(0.693546);

    expect(history.history.val_acc.length).toEqual(2);
    expect(history.history.val_acc[0]).toBeCloseTo(0.5);
    expect(history.history.val_acc[1]).toBeCloseTo(0.5);
  });

  // Reference Python code:
  // ```py
  // import numpy as np
  // import tensorflow as tf
  //
  // batch_size = 8
  // num_batches = 3
  // epochs = 2
  //
  // xs = np.ones([batch_size * num_batches * epochs, 1])
  // ys1 = np.concatenate([
  //     np.zeros([int(batch_size * num_batches * epochs / 2), 1]),
  //     np.ones([int(batch_size * num_batches * epochs / 2), 1])],
  //     axis=0)
  // ys2 = np.concatenate([
  //     np.ones([int(batch_size * num_batches * epochs / 2), 1]),
  //     np.zeros([int(batch_size * num_batches * epochs / 2), 1])],
  //     axis=0)
  //
  // dataset = tf.data.Dataset.from_tensor_slices(
  //     (xs, {'out1': ys1, 'out2': ys2})).batch(batch_size)
  //
  // inp = tf.keras.Input(shape=[1])
  // out1 = tf.keras.layers.Dense(
  //     1,
  //     input_shape=[1],
  //     activation='sigmoid',
  //     kernel_initializer='zeros',
  //     name='out1')(inp)
  // out2 = tf.keras.layers.Dense(
  //     1,
  //     input_shape=[1],
  //     activation='sigmoid',
  //     kernel_initializer='zeros',
  //     name='out2')(inp)
  //
  // model = tf.keras.Model(inp, [out1, out2])
  // model.compile(
  //     loss=['binary_crossentropy', 'binary_crossentropy'],
  //     optimizer='sgd')
  //
  // history = model.fit(
  //     dataset,
  //     steps_per_epoch=num_batches,
  //     epochs=epochs,
  //     class_weight=[{0: 0.1, 1: 2}, {0: 5, 1: 0.5}])
  // print(history.history)
  // ```
  it('Two outputs, binary crossentropy, acc metric', async () => {
    const inp = tfl.input({shape: [1]});
    const out1 = tfl.layers
                     .dense({
                       units: 1,
                       inputShape: [1],
                       kernelInitializer: 'zeros',
                       activation: 'sigmoid'
                     })
                     .apply(inp) as tfl.SymbolicTensor;
    const out2 = tfl.layers
                     .dense({
                       units: 1,
                       inputShape: [1],
                       kernelInitializer: 'zeros',
                       activation: 'sigmoid'
                     })
                     .apply(inp) as tfl.SymbolicTensor;
    const model = tfl.model({inputs: inp, outputs: [out1, out2]});
    model.compile(
        {loss: ['binaryCrossentropy', 'binaryCrossentropy'], optimizer: 'sgd'});

    const batchSize = 8;
    const batchesPerEpoch = 3;
    const epochs = 2;
    const xTensorsFunc = () =>
        [ones([batchSize, 1]), ones([batchSize, 1]), ones([batchSize, 1]),
         ones([batchSize, 1]), ones([batchSize, 1]), ones([batchSize, 1])];
    const yTensorsFunc = () => {
      const output: {[name: string]: Tensor[]} = {};
      output[model.outputNames[0]] = [
        zeros([batchSize, 1]), zeros([batchSize, 1]), zeros([batchSize, 1]),
        ones([batchSize, 1]), ones([batchSize, 1]), ones([batchSize, 1])
      ];
      output[model.outputNames[1]] = [
        ones([batchSize, 1]), ones([batchSize, 1]), ones([batchSize, 1]),
        zeros([batchSize, 1]), zeros([batchSize, 1]), zeros([batchSize, 1])
      ];
      return output;
    };
    const dataset = new FakeNumericDataset({
      xShape: [1],
      yShape: {[model.outputNames[0]]: [1], [model.outputNames[1]]: [1]},
      batchSize,
      numBatches: batchesPerEpoch * epochs,
      xTensorsFunc,
      yTensorsFunc
    });

    const classWeight: ClassWeightMap = {
      [model.outputNames[0]]: {0: 0.1, 1: 2},
      [model.outputNames[1]]: {0: 5, 1: 0.5}
    };

    // Do a burn-in call to account for initialization of cached tensors (for
    // the memory-leak check below).
    await model.fitDataset(dataset, {batchesPerEpoch, epochs: 1, classWeight});
    // model.setWeights([zeros([1, 1]), zeros([1])]);

    const numTensors0 = memory().numTensors;
    const history =
        await model.fitDataset(dataset, {batchesPerEpoch, epochs, classWeight});
    const numTensors1 = memory().numTensors;
    expect(numTensors1).toEqual(numTensors0);

    // The loss and acc values are different if no classWeight is used.
    const totalLoss = history.history.loss;
    expect(totalLoss.length).toEqual(2);
    expect(totalLoss[0]).toBeCloseTo(0.4146);
    expect(totalLoss[1]).toBeCloseTo(4.7882);

    const loss0 = history.history[`${model.outputNames[0]}_loss`];
    expect(loss0.length).toEqual(2);
    expect(loss0[0]).toBeCloseTo(0.0693);
    expect(loss0[1]).toBeCloseTo(1.3695);

    const loss1 = history.history[`${model.outputNames[1]}_loss`];
    expect(loss1.length).toEqual(2);
    expect(loss1[0]).toBeCloseTo(0.3453);
    expect(loss1[1]).toBeCloseTo(3.4158);
  });
});
