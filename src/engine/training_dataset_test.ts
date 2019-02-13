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
 * Unit tests for training.ts, focusing on the tf.Model.fitDataset() and
 * tf.Model.evaluateDataset() methods.
 */

import * as tfc from '@tensorflow/tfjs-core';
import {expectArraysClose} from '@tensorflow/tfjs-core/dist/test_util';

import {CustomCallback} from '../base_callbacks';
import * as tfl from '../index';
import {Logs} from '../logs';
import {describeMathCPUAndGPU, expectTensorsClose} from '../utils/test_utils';

import {FakeNumericDataset, FakeNumericDatasetLegacyArrayForm} from './dataset_fakes';

function createDenseModel(): tfl.Model {
  const model = tfl.sequential();
  model.add(tfl.layers.dense(
      {units: 1, inputShape: [1], kernelInitializer: 'zeros'}));
  return model;
}

describeMathCPUAndGPU('Model.fitDataset', () => {
  // Reference Python tf.keras code:
  //
  // ```py
  // import numpy as np
  // import tensorflow as tf
  //
  // batch_size = 8
  // num_batches = 3
  // epochs = 2
  //
  // xs = np.ones([batch_size * num_batches * epochs, 1])
  // ys = np.ones([batch_size * num_batches * epochs, 1])
  // dataset = tf.data.Dataset.from_tensor_slices((xs, ys)).batch(batch_size)
  //
  // model = tf.keras.Sequential()
  // model.add(tf.keras.layers.Dense(
  //     1,
  //     input_shape=[1],
  //     kernel_initializer='zeros'))
  // model.compile(loss='mean_squared_error', optimizer='sgd')
  //
  // history = model.fit(dataset, steps_per_epoch=num_batches, epochs=epochs)
  // print(history.history)
  // print(model.get_weights()[0])
  // print(model.get_weights()[1])
  // ```
  it('1 input, 1 output, no metric, no validation, with batchesPerEpoch',
     async () => {
       const model = createDenseModel();
       model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});

       const batchSize = 8;
       const epochs = 2;
       const batchesPerEpoch = 3;
       const xTensorsFunc = () =>
           [tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]),
            tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]),
            tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1])];
       const yTensorsFunc = () =>
           [tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]),
            tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]),
            tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1])];
       const dataset = new FakeNumericDataset({
         xShape: [1],
         yShape: [1],
         batchSize,
         numBatches: batchesPerEpoch * epochs,
         xTensorsFunc,
         yTensorsFunc
       });

       // Do a burn-in call to account for initialization of cached tensors (for
       // the memory-leak check below).
       await model.fitDataset(dataset, {batchesPerEpoch, epochs: 1});
       model.setWeights([tfc.zeros([1, 1]), tfc.zeros([1])]);

       const numTensors0 = tfc.memory().numTensors;
       const history =
           await model.fitDataset(dataset, {batchesPerEpoch, epochs});
       const numTensors1 = tfc.memory().numTensors;
       expect(numTensors1).toEqual(numTensors0);
       expect(Object.keys(history.history)).toEqual(['loss']);
       expect(history.history.loss.length).toEqual(2);
       expect(history.history.loss[0]).toBeCloseTo(0.923649);
       expect(history.history.loss[1]).toBeCloseTo(0.722993);
       expectArraysClose(model.getWeights()[0], tfc.tensor2d([[0.108621]]));
       expectArraysClose(model.getWeights()[1], tfc.tensor1d([0.108621]));
     });

  it('1 input, 1 output, no metric, no validation, no batchesPerEpoch',
     async () => {
       const model = createDenseModel();
       model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});

       const batchSize = 8;
       const epochs = 2;
       const batchesPerEpoch = 3;
       const xTensorsFunc =
           () => [tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]), tfc.ones([
             batchSize, 1
           ])];
       const yTensorsFunc =
           () => [tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]), tfc.ones([
             batchSize, 1
           ])];
       const dataset = new FakeNumericDataset({
         xShape: [1],
         yShape: [1],
         batchSize,
         numBatches: batchesPerEpoch,
         xTensorsFunc,
         yTensorsFunc
       });

       // Do a burn-in call to account for initialization of cached tensors (for
       // the memory-leak check below).
       await model.fitDataset(dataset, {epochs: 1});
       model.setWeights([tfc.zeros([1, 1]), tfc.zeros([1])]);

       const numTensors0 = tfc.memory().numTensors;
       const history = await model.fitDataset(dataset, {epochs});
       const numTensors1 = tfc.memory().numTensors;
       expect(numTensors1).toEqual(numTensors0);
       expect(Object.keys(history.history)).toEqual(['loss']);
       expect(history.history.loss.length).toEqual(2);
       expect(history.history.loss[0]).toBeCloseTo(0.923649);
       expect(history.history.loss[1]).toBeCloseTo(0.722993);
       expectArraysClose(model.getWeights()[0], tfc.tensor2d([[0.108621]]));
       expectArraysClose(model.getWeights()[1], tfc.tensor1d([0.108621]));
     });

  // Reference Python tf.keras code:
  //
  // ```py
  // import numpy as np
  // import tensorflow as tf
  //
  // tf.enable_eager_execution():
  //
  // batch_size = 8
  // num_batches = 3
  // epochs = 2
  //
  // xs = np.ones([batch_size * num_batches * epochs, 1])
  // ys = np.ones([batch_size * num_batches * epochs, 1])
  // dataset = tf.data.Dataset.from_tensor_slices((xs, ys)).batch(batch_size)
  //
  // model = tf.keras.Sequential()
  // model.add(tf.keras.layers.Dense(
  //     1,
  //     input_shape=[1],
  //     kernel_initializer='zeros'))
  // model.compile(loss='mean_squared_error',
  //               optimizer='sgd',
  //               metrics=['acc'])
  //
  // history = model.fit(dataset, steps_per_epoch=num_batches, epochs=epochs)
  // print(history.history)
  // print(model.get_weights()[0])
  // print(model.get_weights()[1])
  // ```
  it('1 input, 1 output, 1 metric, no validation, with batchesPerEpoch',
     async () => {
       const model = createDenseModel();
       model.compile(
           {loss: 'meanSquaredError', optimizer: 'sgd', metrics: ['accuracy']});

       const batchSize = 8;
       const epochs = 2;
       const batchesPerEpoch = 3;
       const xTensorsFunc = () =>
           [tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]),
            tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]),
            tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1])];
       const yTensorsFunc = () =>
           [tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]),
            tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]),
            tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1])];
       const dataset = new FakeNumericDataset({
         xShape: [1],
         yShape: [1],
         batchSize,
         numBatches: batchesPerEpoch * epochs,
         xTensorsFunc,
         yTensorsFunc
       });

       // Do a burn-in call to account for initialization of cached tensors (for
       // the memory-leak check below).
       await model.fitDataset(dataset, {batchesPerEpoch, epochs: 1});
       model.setWeights([tfc.zeros([1, 1]), tfc.zeros([1])]);

       const numTensors0 = tfc.memory().numTensors;
       const history =
           await model.fitDataset(dataset, {batchesPerEpoch, epochs});
       const numTensors1 = tfc.memory().numTensors;
       expect(numTensors1).toEqual(numTensors0);
       expect(Object.keys(history.history)).toEqual(['loss', 'acc']);
       expect(history.history.loss.length).toEqual(2);
       expect(history.history.loss[0]).toBeCloseTo(0.923649);
       expect(history.history.loss[1]).toBeCloseTo(0.722993);
       expect(history.history.acc.length).toEqual(2);
       expect(history.history.acc[0]).toBeCloseTo(0);
       expect(history.history.acc[1]).toBeCloseTo(0);
       expectArraysClose(model.getWeights()[0], tfc.tensor2d([[0.108621]]));
       expectArraysClose(model.getWeights()[1], tfc.tensor1d([0.108621]));
     });

  it('1 input, 1 output, 1 metric, no validation, no batchesPerEpoch',
     async () => {
       const model = createDenseModel();
       model.compile(
           {loss: 'meanSquaredError', optimizer: 'sgd', metrics: ['accuracy']});

       const batchSize = 8;
       const epochs = 2;
       const batchesPerEpoch = 3;
       const xTensorsFunc =
           () => [tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]), tfc.ones([
             batchSize, 1
           ])];
       const yTensorsFunc =
           () => [tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]), tfc.ones([
             batchSize, 1
           ])];
       const dataset = new FakeNumericDataset({
         xShape: [1],
         yShape: [1],
         batchSize,
         numBatches: batchesPerEpoch,
         xTensorsFunc,
         yTensorsFunc
       });

       // Do a burn-in call to account for initialization of cached tensors (for
       // the memory-leak check below).
       await model.fitDataset(dataset, {epochs: 1});
       model.setWeights([tfc.zeros([1, 1]), tfc.zeros([1])]);

       const numTensors0 = tfc.memory().numTensors;
       const history = await model.fitDataset(dataset, {epochs});
       const numTensors1 = tfc.memory().numTensors;
       expect(numTensors1).toEqual(numTensors0);
       expect(Object.keys(history.history)).toEqual(['loss', 'acc']);
       expect(history.history.loss.length).toEqual(2);
       expect(history.history.loss[0]).toBeCloseTo(0.923649);
       expect(history.history.loss[1]).toBeCloseTo(0.722993);
       expect(history.history.acc.length).toEqual(2);
       expect(history.history.acc[0]).toBeCloseTo(0);
       expect(history.history.acc[1]).toBeCloseTo(0);
       expectArraysClose(model.getWeights()[0], tfc.tensor2d([[0.108621]]));
       expectArraysClose(model.getWeights()[1], tfc.tensor1d([0.108621]));
     });

  // Reference Python tf.keras code.
  //
  // ```py
  // import numpy as np
  // import tensorflow as tf
  //
  // tf.enable_eager_execution():
  //
  // batch_size = 8
  // num_batches = 3
  // epochs = 2
  //
  // xs = np.ones([batch_size * num_batches * epochs, 1])
  // ys = np.ones([batch_size * num_batches * epochs, 1])
  // dataset = tf.data.Dataset.from_tensor_slices((xs, ys)).batch(batch_size)
  // val_xs = np.zeros([batch_size * 2, 1])
  // val_ys = np.zeros([batch_size * 2, 1])
  //
  // model = tf.keras.Sequential()
  // model.add(tf.keras.layers.Dense(
  //     1,
  //     input_shape=[1],
  //     kernel_initializer='zeros'))
  // model.compile(loss='mean_squared_error', optimizer='sgd',
  // metrics=['accuracy'])
  //
  // class CustomCallback(tf.keras.callbacks.Callback):
  //   def on_epoch_end(self, epoch, logs):
  //     print('epoch = %d; logs = %s' % (epoch, logs))
  //
  // history = model.fit(dataset,
  //                     steps_per_epoch=num_batches,
  //                     epochs=epochs,
  //                     validation_steps=2,
  //                     validation_data=(val_xs, val_ys),
  //                     callbacks=[CustomCallback()])
  // print(history.history)
  // print(model.get_weights()[0])
  // print(model.get_weights()[1])
  // ```
  it('1 input, 1 output, 1 metric, tensor validation, callback, ' +
         'with batchesPerEpoch',
     async () => {
       const model = createDenseModel();
       model.compile(
           {loss: 'meanSquaredError', optimizer: 'sgd', metrics: ['accuracy']});

       const batchSize = 8;
       const epochs = 2;
       const batchesPerEpoch = 3;
       const xTensorsFunc = () =>
           [tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]),
            tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]),
            tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1])];
       const yTensorsFunc = () =>
           [tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]),
            tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]),
            tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1])];
       const dataset = new FakeNumericDataset({
         xShape: [1],
         yShape: [1],
         batchSize,
         numBatches: batchesPerEpoch * epochs,
         xTensorsFunc,
         yTensorsFunc
       });
       const valXs = tfc.zeros([batchSize * 2, 1]);
       const valYs = tfc.zeros([batchSize * 2, 1]);

       // Do a burn-in call to account for initialization of cached
       // tensors (for the memory-leak check below).
       await model.fitDataset(
           dataset,
           {batchesPerEpoch, epochs: 1, validationData: [valXs, valYs]});
       model.setWeights([tfc.zeros([1, 1]), tfc.zeros([1])]);

       const numTensors0 = tfc.memory().numTensors;
       const epochEndValLosses: number[] = [];
       const epochEndValAccs: number[] = [];
       const history = await model.fitDataset(dataset, {
         batchesPerEpoch,
         epochs,
         validationData: [valXs, valYs],
         callbacks: {
           onEpochEnd: async (epoch, logs) => {
             epochEndValLosses.push(logs.val_loss);
             epochEndValAccs.push(logs.val_acc);
           }
         }
       });
       const numTensors1 = tfc.memory().numTensors;
       expect(numTensors1).toEqual(numTensors0);
       expect(Object.keys(history.history).sort()).toEqual([
         'loss', 'acc', 'val_loss', 'val_acc'
       ].sort());
       expect(history.history.loss.length).toEqual(2);
       expect(history.history.loss[0]).toBeCloseTo(0.923649);
       expect(history.history.loss[1]).toBeCloseTo(0.722993);
       expect(history.history.acc.length).toEqual(2);
       expect(history.history.acc[0]).toBeCloseTo(0);
       expect(history.history.acc[1]).toBeCloseTo(0);
       expect(history.history.val_loss.length).toEqual(2);
       expect(history.history.val_loss[0]).toBeCloseTo(0.003321);
       expect(history.history.val_loss[1]).toBeCloseTo(0.011799);
       expect(history.history.val_acc.length).toEqual(2);
       expect(history.history.val_acc[0]).toBeCloseTo(1);
       expect(history.history.val_acc[1]).toBeCloseTo(1);
       expectArraysClose(model.getWeights()[0], tfc.tensor2d([[0.108621]]));
       expectArraysClose(model.getWeights()[1], tfc.tensor1d([0.108621]));

       expect(epochEndValLosses.length).toEqual(2);
       expect(epochEndValLosses[0]).toBeCloseTo(0.003321);
       expect(epochEndValLosses[1]).toBeCloseTo(0.011799);
       expect(epochEndValAccs.length).toEqual(2);
       expect(epochEndValAccs[0]).toBeCloseTo(1);
       expect(epochEndValAccs[1]).toBeCloseTo(1);
     });

  it('1 input, 1 output, 1 metric, tensor validation, callback, ' +
         'no batchesPerEpoch',
     async () => {
       const model = createDenseModel();
       model.compile(
           {loss: 'meanSquaredError', optimizer: 'sgd', metrics: ['accuracy']});

       const batchSize = 8;
       const epochs = 2;
       const batchesPerEpoch = 3;
       const xTensorsFunc =
           () => [tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]), tfc.ones([
             batchSize, 1
           ])];
       const yTensorsFunc =
           () => [tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]), tfc.ones([
             batchSize, 1
           ])];
       const dataset = new FakeNumericDataset({
         xShape: [1],
         yShape: [1],
         batchSize,
         numBatches: batchesPerEpoch,
         xTensorsFunc,
         yTensorsFunc
       });
       const valXs = tfc.zeros([batchSize * 2, 1]);
       const valYs = tfc.zeros([batchSize * 2, 1]);

       // Do a burn-in call to account for initialization of cached
       // tensors (for the memory-leak check below).
       await model.fitDataset(
           dataset, {epochs: 1, validationData: [valXs, valYs]});
       model.setWeights([tfc.zeros([1, 1]), tfc.zeros([1])]);

       const numTensors0 = tfc.memory().numTensors;
       const epochEndValLosses: number[] = [];
       const epochEndValAccs: number[] = [];
       const history = await model.fitDataset(dataset, {
         epochs,
         validationData: [valXs, valYs],
         callbacks: {
           onEpochEnd: async (epoch, logs) => {
             epochEndValLosses.push(logs.val_loss);
             epochEndValAccs.push(logs.val_acc);
           }
         }
       });
       const numTensors1 = tfc.memory().numTensors;
       expect(numTensors1).toEqual(numTensors0);
       expect(Object.keys(history.history).sort()).toEqual([
         'loss', 'acc', 'val_loss', 'val_acc'
       ].sort());
       expect(history.history.loss.length).toEqual(2);
       expect(history.history.loss[0]).toBeCloseTo(0.923649);
       expect(history.history.loss[1]).toBeCloseTo(0.722993);
       expect(history.history.acc.length).toEqual(2);
       expect(history.history.acc[0]).toBeCloseTo(0);
       expect(history.history.acc[1]).toBeCloseTo(0);
       expect(history.history.val_loss.length).toEqual(2);
       expect(history.history.val_loss[0]).toBeCloseTo(0.003321);
       expect(history.history.val_loss[1]).toBeCloseTo(0.011799);
       expect(history.history.val_acc.length).toEqual(2);
       expect(history.history.val_acc[0]).toBeCloseTo(1);
       expect(history.history.val_acc[1]).toBeCloseTo(1);
       expectArraysClose(model.getWeights()[0], tfc.tensor2d([[0.108621]]));
       expectArraysClose(model.getWeights()[1], tfc.tensor1d([0.108621]));

       expect(epochEndValLosses.length).toEqual(2);
       expect(epochEndValLosses[0]).toBeCloseTo(0.003321);
       expect(epochEndValLosses[1]).toBeCloseTo(0.011799);
       expect(epochEndValAccs.length).toEqual(2);
       expect(epochEndValAccs[0]).toBeCloseTo(1);
       expect(epochEndValAccs[1]).toBeCloseTo(1);
     });

  it('Earlier logs are not overwritten', async () => {
    const model = createDenseModel();
    model.compile(
        {loss: 'meanSquaredError', optimizer: 'sgd', metrics: ['accuracy']});

    const batchSize = 8;
    const epochs = 2;
    const batchesPerEpoch = 3;
    const xTensorsFunc =
        () => [tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]), tfc.ones([
          batchSize, 1
        ])];
    const yTensorsFunc =
        () => [tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]), tfc.ones([
          batchSize, 1
        ])];
    const dataset = new FakeNumericDataset({
      xShape: [1],
      yShape: [1],
      batchSize,
      numBatches: batchesPerEpoch,
      xTensorsFunc,
      yTensorsFunc
    });

    const trainLogs: Logs[] = [];
    await model.fitDataset(dataset, {
      epochs,
      callbacks: {
        onEpochEnd: async (epoch, logs) => {
          trainLogs.push(logs);
        }
      }
    });
    expect(trainLogs.length).toEqual(2);
    // Assert that the the first log and the second logs do not overwrite each
    // other.
    expect(trainLogs[0].loss).not.toEqual(trainLogs[1].loss);
  });

  it('dataset.size != null feeds stepPerEpoch to callbacks', async () => {
    const batchSize = 8;
    const batchesPerEpoch = 3;
    const xTensorsFunc =
        () => [tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]), tfc.ones([
          batchSize, 1
        ])];
    const yTensorsFunc =
        () => [tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]), tfc.ones([
          batchSize, 1
        ])];
    const dataset = new FakeNumericDataset({
      xShape: [1],
      yShape: [1],
      batchSize,
      numBatches: batchesPerEpoch,
      xTensorsFunc,
      yTensorsFunc
    });

    let recordedSteps: number;
    class TestCallback extends CustomCallback {
      constructor() {
        super({
          onTrainBegin: async (logs?: Logs) => {
            recordedSteps = this.params.steps as number;
          }
        });
      }
    }

    const model = createDenseModel();
    model.compile(
        {loss: 'meanSquaredError', optimizer: 'sgd', metrics: ['accuracy']});
    const epochs = 1;
    await model.fitDataset(dataset, {epochs, callbacks: new TestCallback()});

    expect(dataset.size).toEqual(batchesPerEpoch);
    expect(recordedSteps).toEqual(batchesPerEpoch);
  });

  it('explicit stepPerEpoch overrides dataset.size for callbacks', async () => {
    const batchSize = 8;
    const numBatches = 3;
    const xTensorsFunc =
        () => [tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]), tfc.ones([
          batchSize, 1
        ])];
    const yTensorsFunc =
        () => [tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]), tfc.ones([
          batchSize, 1
        ])];
    const dataset = new FakeNumericDataset({
      xShape: [1],
      yShape: [1],
      batchSize,
      numBatches,
      xTensorsFunc,
      yTensorsFunc
    });

    let recordedSteps: number;
    class TestCallback extends CustomCallback {
      constructor() {
        super({
          onTrainBegin: async (logs?: Logs) => {
            recordedSteps = this.params.steps as number;
          }
        });
      }
    }

    const model = createDenseModel();
    model.compile(
        {loss: 'meanSquaredError', optimizer: 'sgd', metrics: ['accuracy']});
    const epochs = 1;
    await model.fitDataset(dataset, {
      epochs,
      batchesPerEpoch: numBatches - 1,
      callbacks: new TestCallback()
    });

    expect(dataset.size).toEqual(numBatches);
    expect(recordedSteps).toEqual(numBatches - 1);
  });

  // Reference Python tf.keras code:
  //
  // ```py
  // import numpy as np
  // import tensorflow as tf
  //
  // tf.enable_eager_execution()
  //
  // batch_size = 8
  // num_batches = 3
  // epochs = 2
  //
  // xs = np.ones([batch_size * num_batches * epochs, 1])
  // ys = np.ones([batch_size * num_batches * epochs, 1])
  // dataset = tf.data.Dataset.from_tensor_slices((xs, ys)).batch(batch_size)
  // val_xs = np.zeros([batch_size * 2, 1])
  // val_ys = np.zeros([batch_size * 2, 1])
  // val_dataset = tf.data.Dataset.from_tensor_slices(
  //     (val_xs, val_ys)).batch(batch_size)
  //
  // model = tf.keras.Sequential()
  // model.add(tf.keras.layers.Dense(
  //     1,
  //     input_shape=[1],
  //     kernel_initializer='zeros'))
  // model.compile(loss='mean_squared_error',
  //               optimizer=tf.train.GradientDescentOptimizer(0.01),
  //               metrics=['accuracy'])
  //
  // class CustomCallback(tf.keras.callbacks.Callback):
  //   def on_batch_end(self, batch, logs):
  //     print('batch = %d; logs = %s' % (batch, logs))
  //
  //   def on_epoch_end(self, epoch, logs):
  //     print('epoch = %d; logs = %s' % (epoch, logs))
  //
  // history = model.fit(dataset,
  //                     steps_per_epoch=num_batches,
  //                     epochs=epochs,
  //                     batch_size=4,
  //                     validation_steps=2,
  //                     validation_data=val_dataset,
  //                     callbacks=[CustomCallback()])
  // print(history.history)
  // print(model.get_weights()[0])
  // print(model.get_weights()[1])
  // ```
  it('1 input, 1 output, 1 metric, dataset validation, callback, ' +
         'with batchesPerEpoch',
     async () => {
       const model = createDenseModel();
       model.compile(
           {loss: 'meanSquaredError', optimizer: 'sgd', metrics: ['accuracy']});

       const batchSize = 8;
       const epochs = 2;
       const batchesPerEpoch = 3;

       // Training dataset.
       const xTensorsFunc = () =>
           [tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]),
            tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]),
            tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1])];
       const yTensorsFunc = () =>
           [tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]),
            tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]),
            tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1])];
       const dataset = new FakeNumericDataset({
         xShape: [1],
         yShape: [1],
         batchSize,
         numBatches: batchesPerEpoch * epochs,
         xTensorsFunc,
         yTensorsFunc
       });

       // Validation dataset.
       const valXTensorsFunc = () =>
           [tfc.zeros([batchSize, 1]), tfc.zeros([batchSize, 1]),
            tfc.zeros([batchSize, 1]), tfc.zeros([batchSize, 1]),
            tfc.zeros([batchSize, 1]), tfc.zeros([batchSize, 1])];
       const valYTensorsFunc = () =>
           [tfc.zeros([batchSize, 1]), tfc.zeros([batchSize, 1]),
            tfc.zeros([batchSize, 1]), tfc.zeros([batchSize, 1]),
            tfc.zeros([batchSize, 1]), tfc.zeros([batchSize, 1])];
       const valDataset = new FakeNumericDataset({
         xShape: [1],
         yShape: [1],
         batchSize,
         numBatches: batchesPerEpoch * epochs,
         xTensorsFunc: valXTensorsFunc,
         yTensorsFunc: valYTensorsFunc
       });

       // Do a burn-in call to account for initialization of cached
       // tensors (for the memory-leak check below).
       await model.fitDataset(dataset, {
         batchesPerEpoch,
         epochs,
         validationData: valDataset,
         validationBatches: batchesPerEpoch * epochs
       });
       model.setWeights([tfc.zeros([1, 1]), tfc.zeros([1])]);

       const numTensors0 = tfc.memory().numTensors;
       const epochEndValLosses: number[] = [];
       const epochEndValAccs: number[] = [];
       const history = await model.fitDataset(dataset, {
         batchesPerEpoch,
         epochs,
         validationData: valDataset,
         validationBatches: batchesPerEpoch * epochs,
         callbacks: {
           onEpochEnd: async (epoch, logs) => {
             epochEndValLosses.push(logs.val_loss);
             epochEndValAccs.push(logs.val_acc);
           }
         }
       });
       const numTensors1 = tfc.memory().numTensors;
       expect(numTensors1).toEqual(numTensors0);
       expect(Object.keys(history.history).sort()).toEqual([
         'loss', 'acc', 'val_loss', 'val_acc'
       ].sort());
       expect(history.history.loss.length).toEqual(2);
       expect(history.history.loss[0]).toBeCloseTo(0.923649);
       expect(history.history.loss[1]).toBeCloseTo(0.722993);
       expect(history.history.acc.length).toEqual(2);
       expect(history.history.acc[0]).toBeCloseTo(0);
       expect(history.history.acc[1]).toBeCloseTo(0);
       expect(history.history.val_loss.length).toEqual(2);
       expect(history.history.val_loss[0]).toBeCloseTo(0.003321);
       expect(history.history.val_loss[1]).toBeCloseTo(0.011799);
       expect(history.history.val_acc.length).toEqual(2);
       expect(history.history.val_acc[0]).toBeCloseTo(1);
       expect(history.history.val_acc[1]).toBeCloseTo(1);
       expectArraysClose(model.getWeights()[0], tfc.tensor2d([[0.108621]]));
       expectArraysClose(model.getWeights()[1], tfc.tensor1d([0.108621]));

       expect(epochEndValLosses.length).toEqual(2);
       expect(epochEndValLosses[0]).toBeCloseTo(0.003321);
       expect(epochEndValLosses[1]).toBeCloseTo(0.011799);
       expect(epochEndValAccs.length).toEqual(2);
       expect(epochEndValAccs[0]).toBeCloseTo(1);
       expect(epochEndValAccs[1]).toBeCloseTo(1);
     });

  it('1 input, 1 output, 1 metric, dataset validation, callback, ' +
         'no batchesPerEpoch',
     async () => {
       const model = createDenseModel();
       model.compile(
           {loss: 'meanSquaredError', optimizer: 'sgd', metrics: ['accuracy']});

       const batchSize = 8;
       const epochs = 2;
       const batchesPerEpoch = 3;

       // Training dataset.
       const xTensorsFunc =
           () => [tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]), tfc.ones([
             batchSize, 1
           ])];
       const yTensorsFunc =
           () => [tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]), tfc.ones([
             batchSize, 1
           ])];
       const dataset = new FakeNumericDataset({
         xShape: [1],
         yShape: [1],
         batchSize,
         numBatches: batchesPerEpoch,
         xTensorsFunc,
         yTensorsFunc
       });

       // Validation dataset.
       const valXTensorsFunc = () =>
           [tfc.zeros([batchSize, 1]), tfc.zeros([batchSize, 1]),
            tfc.zeros([batchSize, 1])];
       const valYTensorsFunc = () =>
           [tfc.zeros([batchSize, 1]), tfc.zeros([batchSize, 1]),
            tfc.zeros([batchSize, 1])];
       const valDataset = new FakeNumericDataset({
         xShape: [1],
         yShape: [1],
         batchSize,
         numBatches: batchesPerEpoch,
         xTensorsFunc: valXTensorsFunc,
         yTensorsFunc: valYTensorsFunc
       });

       // Do a burn-in call to account for initialization of cached
       // tensors (for the memory-leak check below).
       await model.fitDataset(dataset, {epochs, validationData: valDataset});
       model.setWeights([tfc.zeros([1, 1]), tfc.zeros([1])]);

       const numTensors0 = tfc.memory().numTensors;
       const epochEndValLosses: number[] = [];
       const epochEndValAccs: number[] = [];
       const history = await model.fitDataset(dataset, {
         epochs,
         validationData: valDataset,
         callbacks: {
           onEpochEnd: async (epoch, logs) => {
             epochEndValLosses.push(logs.val_loss);
             epochEndValAccs.push(logs.val_acc);
           }
         }
       });
       const numTensors1 = tfc.memory().numTensors;
       expect(numTensors1).toEqual(numTensors0);
       expect(Object.keys(history.history).sort()).toEqual([
         'loss', 'acc', 'val_loss', 'val_acc'
       ].sort());
       expect(history.history.loss.length).toEqual(2);
       expect(history.history.loss[0]).toBeCloseTo(0.923649);
       expect(history.history.loss[1]).toBeCloseTo(0.722993);
       expect(history.history.acc.length).toEqual(2);
       expect(history.history.acc[0]).toBeCloseTo(0);
       expect(history.history.acc[1]).toBeCloseTo(0);
       expect(history.history.val_loss.length).toEqual(2);
       expect(history.history.val_loss[0]).toBeCloseTo(0.003321);
       expect(history.history.val_loss[1]).toBeCloseTo(0.011799);
       expect(history.history.val_acc.length).toEqual(2);
       expect(history.history.val_acc[0]).toBeCloseTo(1);
       expect(history.history.val_acc[1]).toBeCloseTo(1);
       expectArraysClose(model.getWeights()[0], tfc.tensor2d([[0.108621]]));
       expectArraysClose(model.getWeights()[1], tfc.tensor1d([0.108621]));

       expect(epochEndValLosses.length).toEqual(2);
       expect(epochEndValLosses[0]).toBeCloseTo(0.003321);
       expect(epochEndValLosses[1]).toBeCloseTo(0.011799);
       expect(epochEndValAccs.length).toEqual(2);
       expect(epochEndValAccs[0]).toBeCloseTo(1);
       expect(epochEndValAccs[1]).toBeCloseTo(1);
     });

  it('Memory leak check with metric and validation, with batchesPerEpoch',
     async () => {
       const model = createDenseModel();
       model.compile(
           {loss: 'meanSquaredError', optimizer: 'sgd', metrics: ['accuracy']});

       const batchSize = 8;
       const epochs = 3;
       const batchesPerEpoch = 3;
       const dataset = new FakeNumericDataset({
         xShape: [1],
         yShape: [1],
         batchSize,
         numBatches: batchesPerEpoch * epochs
       });
       const valXs = tfc.zeros([batchSize * 2, 1]);
       const valYs = tfc.zeros([batchSize * 2, 1]);

       // Do a burn-in call to account for initialization of cached
       // tensors (for the memory-leak check below).
       await model.fitDataset(
           dataset,
           {batchesPerEpoch, epochs: 1, validationData: [valXs, valYs]});
       model.setWeights([tfc.zeros([1, 1]), tfc.zeros([1])]);

       const numTensors0 = tfc.memory().numTensors;
       await model.fitDataset(dataset, {
         batchesPerEpoch,
         epochs,
         validationData: [valXs, valYs],
         callbacks: {
           onEpochEnd: async (epoch, logs) => {
             expect(tfc.memory().numTensors).toEqual(numTensors0);
           }
         }
       });
       const numTensors1 = tfc.memory().numTensors;
       expect(numTensors1).toEqual(numTensors0);
     });

  it('Memory leak check with metric and validation, no batchesPerEpoch',
     async () => {
       const model = createDenseModel();
       model.compile(
           {loss: 'meanSquaredError', optimizer: 'sgd', metrics: ['accuracy']});

       const batchSize = 8;
       const epochs = 3;
       const batchesPerEpoch = 3;
       const dataset = new FakeNumericDataset(
           {xShape: [1], yShape: [1], batchSize, numBatches: batchesPerEpoch});
       const valXs = tfc.zeros([batchSize * 2, 1]);
       const valYs = tfc.zeros([batchSize * 2, 1]);

       // Do a burn-in call to account for initialization of cached
       // tensors (for the memory-leak check below).
       await model.fitDataset(
           dataset, {epochs: 1, validationData: [valXs, valYs]});
       model.setWeights([tfc.zeros([1, 1]), tfc.zeros([1])]);

       const numTensors0 = tfc.memory().numTensors;
       await model.fitDataset(dataset, {
         epochs,
         validationData: [valXs, valYs],
         callbacks: {
           onEpochEnd: async (epoch, logs) => {
             expect(tfc.memory().numTensors).toEqual(numTensors0);
           }
         }
       });
       const numTensors1 = tfc.memory().numTensors;
       expect(numTensors1).toEqual(numTensors0);
     });

  // Refence Python tf.keras code:
  //
  // ```py
  // import numpy as np
  // import tensorflow as tf
  //
  // tf.enable_eager_execution():
  //
  // batch_size = 8
  // num_batches = 3
  // epochs = 2
  //
  // xs = np.ones([batch_size * num_batches * epochs, 1])
  // ys = np.ones([batch_size * num_batches * epochs, 1])
  // dataset = tf.data.Dataset.from_tensor_slices((xs, ys)).batch(batch_size)
  //
  // model = tf.keras.Sequential()
  // model.add(tf.keras.layers.Dense(
  //     1,
  //     input_shape=[1],
  //     kernel_initializer='zeros'))
  // model.compile(loss='mean_squared_error', optimizer='sgd',
  // metrics=['accuracy'])
  //
  // class CustomCallback(tf.keras.callbacks.Callback):
  //   def on_batch_end(self, batch, logs):
  //     print('batch = %d; logs = %s' % (batch, logs))
  //
  // history = model.fit(dataset,
  //                     steps_per_epoch=num_batches,
  //                     epochs=epochs,
  //                     callbacks=[CustomCallback()])
  // print(history.history)
  // ```
  it('1 input, 1 output, 1 metric, no validation, callback, ' +
         'with batchesPerEpoch',
     async () => {
       const model = createDenseModel();
       model.compile(
           {loss: 'meanSquaredError', optimizer: 'sgd', metrics: ['accuracy']});

       const batchSize = 8;
       const epochs = 2;
       const batchesPerEpoch = 3;
       const xTensorsFunc = () =>
           [tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]),
            tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]),
            tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1])];
       const yTensorsFunc = () =>
           [tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]),
            tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]),
            tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1])];
       const dataset = new FakeNumericDataset({
         xShape: [1],
         yShape: [1],
         batchSize,
         numBatches: batchesPerEpoch * epochs,
         xTensorsFunc,
         yTensorsFunc
       });

       // Do a burn-in call to account for initialization of cached tensors (for
       // the memory-leak check below).
       await model.fitDataset(dataset, {batchesPerEpoch, epochs: 1});
       model.setWeights([tfc.zeros([1, 1]), tfc.zeros([1])]);

       const numTensors0 = tfc.memory().numTensors;
       let onTrainBeginCalls = 0;
       let onTrainEndCalls = 0;
       const epochBeginEpochs: number[] = [];
       const epochEndEpochs: number[] = [];
       const batchBeginBatches: number[] = [];
       const batchEndBatches: number[] = [];
       const epochEndLosses: number[] = [];
       const epochEndAccs: number[] = [];
       const batchEndLosses: number[] = [];
       const batchEndAccs: number[] = [];
       const history = await model.fitDataset(dataset, {
         batchesPerEpoch,
         epochs,
         callbacks: {
           onTrainBegin: async () => {
             onTrainBeginCalls++;
           },
           onTrainEnd: async () => {
             onTrainEndCalls++;
           },
           onEpochBegin: async (epoch) => {
             epochBeginEpochs.push(epoch);
           },
           onEpochEnd: async (epoch, logs) => {
             epochEndEpochs.push(epoch);
             epochEndLosses.push(logs.loss);
             epochEndAccs.push(logs.acc);
           },
           onBatchBegin: async (batch, logs) => {
             batchBeginBatches.push(batch);
           },
           onBatchEnd: async (batch, logs) => {
             batchEndBatches.push(batch);
             batchEndLosses.push(logs.loss);
             batchEndAccs.push(logs.acc);
           },
         }
       });
       const numTensors1 = tfc.memory().numTensors;
       expect(numTensors1).toEqual(numTensors0);
       expect(Object.keys(history.history)).toEqual(['loss', 'acc']);
       expect(history.history.loss.length).toEqual(2);
       expect(history.history.loss[0]).toBeCloseTo(0.923649);
       expect(history.history.loss[1]).toBeCloseTo(0.722993);
       expect(history.history.acc.length).toEqual(2);
       expect(history.history.acc[0]).toBeCloseTo(0);
       expect(history.history.acc[1]).toBeCloseTo(0);
       expectArraysClose(model.getWeights()[0], tfc.tensor2d([[0.108621]]));
       expectArraysClose(model.getWeights()[1], tfc.tensor1d([0.108621]));

       expect(onTrainBeginCalls).toEqual(1);
       expect(onTrainEndCalls).toEqual(1);
       expect(epochBeginEpochs).toEqual([0, 1]);
       expect(epochEndEpochs).toEqual([0, 1]);
       expect(batchBeginBatches).toEqual([0, 1, 2, 0, 1, 2]);
       expect(batchEndBatches).toEqual([0, 1, 2, 0, 1, 2]);
       expect(epochEndLosses.length).toEqual(2);
       expect(epochEndLosses[0]).toBeCloseTo(0.923649);
       expect(epochEndLosses[1]).toBeCloseTo(0.722993);
       expect(epochEndAccs.length).toEqual(2);
       expect(epochEndAccs[0]).toBeCloseTo(0);
       expect(epochEndAccs[1]).toBeCloseTo(0);
       expectArraysClose(
           batchEndLosses, [1, 0.9216, 0.849347, 0.782758, 0.721390, 0.664832]);
     });

  it('1 input, 1 output, 1 metric, no validation, callback, no batchesPerEpoch',
     async () => {
       const model = createDenseModel();
       model.compile(
           {loss: 'meanSquaredError', optimizer: 'sgd', metrics: ['accuracy']});

       const batchSize = 8;
       const epochs = 2;
       const batchesPerEpoch = 3;
       const xTensorsFunc =
           () => [tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]), tfc.ones([
             batchSize, 1
           ])];
       const yTensorsFunc =
           () => [tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]), tfc.ones([
             batchSize, 1
           ])];
       const dataset = new FakeNumericDataset({
         xShape: [1],
         yShape: [1],
         batchSize,
         numBatches: batchesPerEpoch,
         xTensorsFunc,
         yTensorsFunc
       });

       // Do a burn-in call to account for initialization of cached tensors (for
       // the memory-leak check below).
       await model.fitDataset(dataset, {epochs: 1});
       model.setWeights([tfc.zeros([1, 1]), tfc.zeros([1])]);

       const numTensors0 = tfc.memory().numTensors;
       let onTrainBeginCalls = 0;
       let onTrainEndCalls = 0;
       const epochBeginEpochs: number[] = [];
       const epochEndEpochs: number[] = [];
       const batchBeginBatches: number[] = [];
       const batchEndBatches: number[] = [];
       const epochEndLosses: number[] = [];
       const epochEndAccs: number[] = [];
       const batchEndLosses: number[] = [];
       const batchEndAccs: number[] = [];
       const history = await model.fitDataset(dataset, {
         epochs,
         callbacks: {
           onTrainBegin: async () => {
             onTrainBeginCalls++;
           },
           onTrainEnd: async () => {
             onTrainEndCalls++;
           },
           onEpochBegin: async (epoch) => {
             epochBeginEpochs.push(epoch);
           },
           onEpochEnd: async (epoch, logs) => {
             epochEndEpochs.push(epoch);
             epochEndLosses.push(logs.loss);
             epochEndAccs.push(logs.acc);
           },
           onBatchBegin: async (batch, logs) => {
             batchBeginBatches.push(batch);
           },
           onBatchEnd: async (batch, logs) => {
             batchEndBatches.push(batch);
             batchEndLosses.push(logs.loss);
             batchEndAccs.push(logs.acc);
           },
         }
       });
       const numTensors1 = tfc.memory().numTensors;
       expect(numTensors1).toEqual(numTensors0);
       expect(Object.keys(history.history)).toEqual(['loss', 'acc']);
       expect(history.history.loss.length).toEqual(2);
       expect(history.history.loss[0]).toBeCloseTo(0.923649);
       expect(history.history.loss[1]).toBeCloseTo(0.722993);
       expect(history.history.acc.length).toEqual(2);
       expect(history.history.acc[0]).toBeCloseTo(0);
       expect(history.history.acc[1]).toBeCloseTo(0);
       expectArraysClose(model.getWeights()[0], tfc.tensor2d([[0.108621]]));
       expectArraysClose(model.getWeights()[1], tfc.tensor1d([0.108621]));

       expect(onTrainBeginCalls).toEqual(1);
       expect(onTrainEndCalls).toEqual(1);
       expect(epochBeginEpochs).toEqual([0, 1]);
       expect(epochEndEpochs).toEqual([0, 1]);
       expect(batchBeginBatches).toEqual([0, 1, 2, 0, 1, 2]);
       expect(batchEndBatches).toEqual([0, 1, 2, 0, 1, 2]);
       expect(epochEndLosses.length).toEqual(2);
       expect(epochEndLosses[0]).toBeCloseTo(0.923649);
       expect(epochEndLosses[1]).toBeCloseTo(0.722993);
       expect(epochEndAccs.length).toEqual(2);
       expect(epochEndAccs[0]).toBeCloseTo(0);
       expect(epochEndAccs[1]).toBeCloseTo(0);
       expectArraysClose(
           batchEndLosses, [1, 0.9216, 0.849347, 0.782758, 0.721390, 0.664832]);
     });

  // Reference Python tf.keras code:
  //
  // ```py
  // import numpy as np
  // import tensorflow as tf
  //
  // batch_size = 8
  // num_batches = 3
  // epochs = 2
  //
  // input1 = tf.keras.Input(shape = [1], name = 'x1')
  // input2 = tf.keras.Input(shape = [1], name = 'x2')
  // concat = tf.keras.layers.concatenate([input1, input2])
  // y = tf.keras.layers.Dense(
  //     1, kernel_initializer = 'zeros')(concat)
  // model = tf.keras.Model(inputs = [input1, input2], outputs = y)
  // model.compile(
  //     loss = 'mean_squared_error', optimizer = 'sgd', metrics =
  //     ['accuracy'])
  // model.summary()
  // print(input1.name)
  // print(input2.name)
  //
  // xs1 = np.ones([batch_size * num_batches * epochs, 1])
  // xs2 = np.ones([batch_size * num_batches * epochs, 1])
  // ys = np.ones([batch_size * num_batches * epochs, 1])
  // dataset = tf.data.Dataset.from_tensor_slices(
  //     ({'x1': xs1, 'x2': xs2}, ys)).batch(batch_size)
  //
  // history = model.fit(dataset,
  //                     steps_per_epoch=num_batches,
  //                     epochs=epochs)
  // print(history.history)
  // print(model.get_weights()[0])
  // print(model.get_weights()[1])
  // ```
  it('2 inputs, 1 output, 1 metric, no validation, with batchesPerEpoch',
     async () => {
       // Create a functional model with 2 inputs.
       const input1 = tfl.layers.input({shape: [1]});
       const input2 = tfl.layers.input({shape: [1]});
       const concat = tfl.layers.concatenate().apply([input1, input2]);
       const y = tfl.layers.dense({units: 1, kernelInitializer: 'zeros'})
                     .apply(concat) as tfl.SymbolicTensor;
       const model = tfl.model({inputs: [input1, input2], outputs: y});
       model.compile(
           {loss: 'meanSquaredError', optimizer: 'sgd', metrics: ['accuracy']});

       const batchSize = 8;
       const epochs = 2;
       const batchesPerEpoch = 3;
       const xTensorsFunc = () => {
         const output: {[name: string]: tfc.Tensor[]} = {};
         output[input1.name] = [
           tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]),
           tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]),
           tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1])
         ];
         output[input2.name] = [
           tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]),
           tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]),
           tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1])
         ];
         return output;
       };
       const yTensorsFunc = () =>
           [tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]),
            tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]),
            tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1])];
       const dataset = new FakeNumericDataset({
         xShape: [1],
         yShape: [1],
         batchSize,
         numBatches: batchesPerEpoch * epochs,
         xTensorsFunc,
         yTensorsFunc
       });

       // Do a burn-in call to account for initialization of cached tensors (for
       // the memory-leak check below).
       await model.fitDataset(dataset, {batchesPerEpoch, epochs: 1});
       model.setWeights([tfc.zeros([2, 1]), tfc.zeros([1])]);

       const numTensors0 = tfc.memory().numTensors;
       const history =
           await model.fitDataset(dataset, {batchesPerEpoch, epochs});
       const numTensors1 = tfc.memory().numTensors;
       expect(numTensors1).toEqual(numTensors0);
       expect(Object.keys(history.history)).toEqual(['loss', 'acc']);
       expect(history.history.loss.length).toEqual(2);
       expect(history.history.loss[0]).toBeCloseTo(0.888116);
       expect(history.history.loss[1]).toBeCloseTo(0.612685);
       expect(history.history.acc.length).toEqual(2);
       expect(history.history.acc[0]).toBeCloseTo(0);
       expect(history.history.acc[1]).toBeCloseTo(0);
       expectArraysClose(
           model.getWeights()[0], tfc.tensor2d([[0.103377], [0.103377]]));
       expectArraysClose(model.getWeights()[1], tfc.tensor1d([0.103377]));
     });

  it('2 inputs, 1 output, 1 metric, no validation, no batchesPerEpoch',
     async () => {
       // Create a functional model with 2 inputs.
       const input1 = tfl.layers.input({shape: [1]});
       const input2 = tfl.layers.input({shape: [1]});
       const concat = tfl.layers.concatenate().apply([input1, input2]);
       const y = tfl.layers.dense({units: 1, kernelInitializer: 'zeros'})
                     .apply(concat) as tfl.SymbolicTensor;
       const model = tfl.model({inputs: [input1, input2], outputs: y});
       model.compile(
           {loss: 'meanSquaredError', optimizer: 'sgd', metrics: ['accuracy']});

       const batchSize = 8;
       const epochs = 2;
       const batchesPerEpoch = 3;
       const xTensorsFunc = () => {
         const output: {[name: string]: tfc.Tensor[]} = {};
         output[input1.name] = [
           tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]),
           tfc.ones([batchSize, 1])
         ];
         output[input2.name] = [
           tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]),
           tfc.ones([batchSize, 1])
         ];
         return output;
       };
       const yTensorsFunc =
           () => [tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]), tfc.ones([
             batchSize, 1
           ])];
       const dataset = new FakeNumericDataset({
         xShape: [1],
         yShape: [1],
         batchSize,
         numBatches: batchesPerEpoch,
         xTensorsFunc,
         yTensorsFunc
       });

       // Do a burn-in call to account for initialization of cached tensors (for
       // the memory-leak check below).
       await model.fitDataset(dataset, {epochs: 1});
       model.setWeights([tfc.zeros([2, 1]), tfc.zeros([1])]);

       const numTensors0 = tfc.memory().numTensors;
       const history = await model.fitDataset(dataset, {epochs});
       const numTensors1 = tfc.memory().numTensors;
       expect(numTensors1).toEqual(numTensors0);
       expect(Object.keys(history.history)).toEqual(['loss', 'acc']);
       expect(history.history.loss.length).toEqual(2);
       expect(history.history.loss[0]).toBeCloseTo(0.888116);
       expect(history.history.loss[1]).toBeCloseTo(0.612685);
       expect(history.history.acc.length).toEqual(2);
       expect(history.history.acc[0]).toBeCloseTo(0);
       expect(history.history.acc[1]).toBeCloseTo(0);
       expectArraysClose(
           model.getWeights()[0], tfc.tensor2d([[0.103377], [0.103377]]));
       expectArraysClose(model.getWeights()[1], tfc.tensor1d([0.103377]));
     });

  // Reference Python tf.keras code:
  //
  // ```py
  // import numpy as np
  // import tensorflow as tf
  //
  // tf.enable_eager_execution()
  //
  // batch_size = 8
  // num_batches = 3
  // epochs = 2
  //
  // input1 = tf.keras.Input(shape = [1], name = 'x1')
  // input2 = tf.keras.Input(shape = [1], name = 'x2')
  // concat = tf.keras.layers.concatenate([input1, input2])
  // y = tf.keras.layers.Dense(
  //     1, kernel_initializer = 'zeros')(concat)
  // model = tf.keras.Model(inputs = [input1, input2], outputs = y)
  // model.compile(
  //     loss='mean_squared_error',
  //     optimizer=tf.train.GradientDescentOptimizer(0.01),
  //     metrics=['accuracy'])
  // model.summary()
  // print(input1.name)
  // print(input2.name)
  //
  // xs1 = np.ones([batch_size * num_batches * epochs, 1])
  // xs2 = np.ones([batch_size * num_batches * epochs, 1])
  // ys = np.ones([batch_size * num_batches * epochs, 1])
  // dataset = tf.data.Dataset.from_tensor_slices(
  //     ({'x1': xs1, 'x2': xs2}, ys)).batch(batch_size)
  //
  // val_xs = [np.zeros([batch_size, 1]),
  //           np.zeros([batch_size, 1])]
  // val_ys = np.zeros([batch_size, 1])
  //
  // history = model.fit(dataset,
  //                     steps_per_epoch=num_batches,
  //                     epochs=epochs,
  //                     batch_size=batch_size,
  //                     validation_data=[val_xs, val_ys])
  // print(history.history)
  // print(model.get_weights()[0])
  // print(model.get_weights()[1])
  // ```
  it('2 inputs, 1 output, 1 metric, tensor array validation, ' +
         'with batchesPerEpoch',
     async () => {
       // Create a functional model with 2 inputs.
       const input1 = tfl.layers.input({shape: [1]});
       const input2 = tfl.layers.input({shape: [1]});
       const concat = tfl.layers.concatenate().apply([input1, input2]);
       const y = tfl.layers.dense({units: 1, kernelInitializer: 'zeros'})
                     .apply(concat) as tfl.SymbolicTensor;
       const model = tfl.model({inputs: [input1, input2], outputs: y});
       model.compile(
           {loss: 'meanSquaredError', optimizer: 'sgd', metrics: ['accuracy']});

       const batchSize = 8;
       const epochs = 2;
       const batchesPerEpoch = 3;

       // Training data.
       const xTensorsFunc = () => {
         const output: {[name: string]: tfc.Tensor[]} = {};
         output[input1.name] = [
           tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]),
           tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]),
           tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1])
         ];
         output[input2.name] = [
           tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]),
           tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]),
           tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1])
         ];
         return output;
       };
       const yTensorsFunc = () =>
           [tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]),
            tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]),
            tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1])];
       const dataset = new FakeNumericDataset({
         xShape: [1],
         yShape: [1],
         batchSize,
         numBatches: batchesPerEpoch * epochs,
         xTensorsFunc,
         yTensorsFunc
       });

       // Validation data.
       const valXs: tfc.Tensor[] =
           [tfc.zeros([batchSize, 1]), tfc.zeros([batchSize, 1])];
       const valYs = tfc.zeros([batchSize, 1]);

       // Do a burn-in call to account for initialization of cached tensors
       // (for the memory-leak check below).
       await model.fitDataset(dataset, {
         batchesPerEpoch,
         epochs: 1,
         validationData: [valXs, valYs],
         validationBatchSize: batchSize
       });
       model.setWeights([tfc.zeros([2, 1]), tfc.zeros([1])]);

       const numTensors0 = tfc.memory().numTensors;
       const history = await model.fitDataset(dataset, {
         batchesPerEpoch,
         epochs,
         validationData: [valXs, valYs],
         validationBatchSize: batchSize
       });
       const numTensors1 = tfc.memory().numTensors;
       expect(numTensors1).toEqual(numTensors0);
       expect(Object.keys(history.history).sort()).toEqual([
         'acc', 'loss', 'val_acc', 'val_loss'
       ]);
       expect(history.history.loss.length).toEqual(2);
       expect(history.history.loss[0]).toBeCloseTo(0.888116);
       expect(history.history.loss[1]).toBeCloseTo(0.612685);
       expect(history.history.acc.length).toEqual(2);
       expect(history.history.acc[0]).toBeCloseTo(0);
       expect(history.history.acc[1]).toBeCloseTo(0);
       expect(history.history.val_loss.length).toEqual(2);
       expect(history.history.val_loss[0]).toBeCloseTo(0.003189);
       expect(history.history.val_loss[1]).toBeCloseTo(0.010687);
       expect(history.history.val_acc.length).toEqual(2);
       expect(history.history.val_acc[0]).toBeCloseTo(1.0);
       expect(history.history.val_acc[1]).toBeCloseTo(1.0);
       expectArraysClose(
           model.getWeights()[0], tfc.tensor2d([[0.103377], [0.103377]]));
       expectArraysClose(model.getWeights()[1], tfc.tensor1d([0.103377]));
     });

  it('2 inputs, 1 output, 1 metric, tensor array validation, ' +
         'no batchesPerEpoch',
     async () => {
       // Create a functional model with 2 inputs.
       const input1 = tfl.layers.input({shape: [1]});
       const input2 = tfl.layers.input({shape: [1]});
       const concat = tfl.layers.concatenate().apply([input1, input2]);
       const y = tfl.layers.dense({units: 1, kernelInitializer: 'zeros'})
                     .apply(concat) as tfl.SymbolicTensor;
       const model = tfl.model({inputs: [input1, input2], outputs: y});
       model.compile(
           {loss: 'meanSquaredError', optimizer: 'sgd', metrics: ['accuracy']});

       const batchSize = 8;
       const epochs = 2;
       const batchesPerEpoch = 3;

       // Training data.
       const xTensorsFunc = () => {
         const output: {[name: string]: tfc.Tensor[]} = {};
         output[input1.name] = [
           tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]),
           tfc.ones([batchSize, 1])
         ];
         output[input2.name] = [
           tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]),
           tfc.ones([batchSize, 1])
         ];
         return output;
       };
       const yTensorsFunc =
           () => [tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]), tfc.ones([
             batchSize, 1
           ])];
       const dataset = new FakeNumericDataset({
         xShape: [1],
         yShape: [1],
         batchSize,
         numBatches: batchesPerEpoch,
         xTensorsFunc,
         yTensorsFunc
       });

       // Validation data.
       const valXs: tfc.Tensor[] =
           [tfc.zeros([batchSize, 1]), tfc.zeros([batchSize, 1])];
       const valYs = tfc.zeros([batchSize, 1]);

       // Do a burn-in call to account for initialization of cached tensors (for
       // the memory-leak check below).
       await model.fitDataset(dataset, {
         epochs: 1,
         validationData: [valXs, valYs],
         validationBatchSize: batchSize
       });
       model.setWeights([tfc.zeros([2, 1]), tfc.zeros([1])]);

       const numTensors0 = tfc.memory().numTensors;
       const history = await model.fitDataset(dataset, {
         epochs,
         validationData: [valXs, valYs],
         validationBatchSize: batchSize
       });
       const numTensors1 = tfc.memory().numTensors;
       expect(numTensors1).toEqual(numTensors0);
       expect(Object.keys(history.history).sort()).toEqual([
         'acc', 'loss', 'val_acc', 'val_loss'
       ]);
       expect(history.history.loss.length).toEqual(2);
       expect(history.history.loss[0]).toBeCloseTo(0.888116);
       expect(history.history.loss[1]).toBeCloseTo(0.612685);
       expect(history.history.acc.length).toEqual(2);
       expect(history.history.acc[0]).toBeCloseTo(0);
       expect(history.history.acc[1]).toBeCloseTo(0);
       expect(history.history.val_loss.length).toEqual(2);
       expect(history.history.val_loss[0]).toBeCloseTo(0.003189);
       expect(history.history.val_loss[1]).toBeCloseTo(0.010687);
       expect(history.history.val_acc.length).toEqual(2);
       expect(history.history.val_acc[0]).toBeCloseTo(1.0);
       expect(history.history.val_acc[1]).toBeCloseTo(1.0);
       expectArraysClose(
           model.getWeights()[0], tfc.tensor2d([[0.103377], [0.103377]]));
       expectArraysClose(model.getWeights()[1], tfc.tensor1d([0.103377]));
     });

  // Reference Python tf.keras code:
  //
  // ```py
  // import numpy as np
  // import tensorflow as tf
  //
  // tf.enable_eager_execution()
  //
  // batch_size = 8
  // num_batches = 3
  // epochs = 2
  //
  // input1 = tf.keras.Input(shape = [1], name = 'x1')
  // input2 = tf.keras.Input(shape = [1], name = 'x2')
  // concat = tf.keras.layers.concatenate([input1, input2])
  // y = tf.keras.layers.Dense(
  //     1, kernel_initializer = 'zeros')(concat)
  // model = tf.keras.Model(inputs = [input1, input2], outputs = y)
  // model.compile(
  //     loss='mean_squared_error',
  //     optimizer=tf.train.GradientDescentOptimizer(0.01),
  //     metrics=['accuracy'])
  // model.summary()
  // print(input1.name)
  // print(input2.name)
  //
  // xs1 = np.ones([batch_size * num_batches * epochs, 1])
  // xs2 = np.ones([batch_size * num_batches * epochs, 1])
  // ys = np.ones([batch_size * num_batches * epochs, 1])
  // dataset = tf.data.Dataset.from_tensor_slices(
  //     ({'x1': xs1, 'x2': xs2}, ys)).batch(batch_size)
  //
  // val_xs = {
  //     'x1': np.zeros([batch_size, 1]),
  //     'x2': np.zeros([batch_size, 1])
  // }
  // val_ys = np.zeros([batch_size, 1])
  //
  // history = model.fit(dataset,
  //                     steps_per_epoch=num_batches,
  //                     epochs=epochs,
  //                     batch_size=batch_size,
  //                     validation_data=[val_xs, val_ys])
  // print(history.history)
  // print(model.get_weights()[0])
  // print(model.get_weights()[1])
  // ```
  it('2 input, 1 output, 1 metric, tensor array validation, ' +
         'with batchesPerEpoch',
     async () => {
       // Create a functional model with 2 inputs.
       const input1 = tfl.layers.input({shape: [1]});
       const input2 = tfl.layers.input({shape: [1]});
       const concat = tfl.layers.concatenate().apply([input1, input2]);
       const y = tfl.layers.dense({units: 1, kernelInitializer: 'zeros'})
                     .apply(concat) as tfl.SymbolicTensor;
       const model = tfl.model({inputs: [input1, input2], outputs: y});
       model.compile(
           {loss: 'meanSquaredError', optimizer: 'sgd', metrics: ['accuracy']});

       const batchSize = 8;
       const epochs = 2;
       const batchesPerEpoch = 3;

       // Training data.
       const xTensorsFunc = () => {
         const output: {[name: string]: tfc.Tensor[]} = {};
         output[input1.name] = [
           tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]),
           tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]),
           tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1])
         ];
         output[input2.name] = [
           tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]),
           tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]),
           tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1])
         ];
         return output;
       };
       const yTensorsFunc = () =>
           [tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]),
            tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]),
            tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1])];
       const dataset = new FakeNumericDataset({
         xShape: [1],
         yShape: [1],
         batchSize,
         numBatches: batchesPerEpoch * epochs,
         xTensorsFunc,
         yTensorsFunc
       });

       // Validation data.
       const valXs: tfc.NamedTensorMap = {};
       valXs[input1.name] = tfc.zeros([batchSize, 1]);
       valXs[input2.name] = tfc.zeros([batchSize, 1]);
       const valYs = tfc.zeros([batchSize, 1]);

       // Do a burn-in call to account for initialization of cached tensors (for
       // the memory-leak check below).
       await model.fitDataset(dataset, {
         batchesPerEpoch,
         epochs: 1,
         validationData: [valXs, valYs],
         validationBatchSize: batchSize
       });
       model.setWeights([tfc.zeros([2, 1]), tfc.zeros([1])]);

       const numTensors0 = tfc.memory().numTensors;
       const history = await model.fitDataset(dataset, {
         batchesPerEpoch,
         epochs,
         validationData: [valXs, valYs],
         validationBatchSize: batchSize
       });
       const numTensors1 = tfc.memory().numTensors;
       expect(numTensors1).toEqual(numTensors0);
       expect(Object.keys(history.history).sort()).toEqual([
         'acc', 'loss', 'val_acc', 'val_loss'
       ]);
       expect(history.history.loss.length).toEqual(2);
       expect(history.history.loss[0]).toBeCloseTo(0.888116);
       expect(history.history.loss[1]).toBeCloseTo(0.612685);
       expect(history.history.acc.length).toEqual(2);
       expect(history.history.acc[0]).toBeCloseTo(0);
       expect(history.history.acc[1]).toBeCloseTo(0);
       expect(history.history.val_loss.length).toEqual(2);
       expect(history.history.val_loss[0]).toBeCloseTo(0.003189);
       expect(history.history.val_loss[1]).toBeCloseTo(0.010687);
       expect(history.history.val_acc.length).toEqual(2);
       expect(history.history.val_acc[0]).toBeCloseTo(1.0);
       expect(history.history.val_acc[1]).toBeCloseTo(1.0);
       expectArraysClose(
           model.getWeights()[0], tfc.tensor2d([[0.103377], [0.103377]]));
       expectArraysClose(model.getWeights()[1], tfc.tensor1d([0.103377]));
     });

  it('2 input, 1 output, 1 metric, tensor array validation, ' +
         'no batchesPerEpoch',
     async () => {
       // Create a functional model with 2 inputs.
       const input1 = tfl.layers.input({shape: [1]});
       const input2 = tfl.layers.input({shape: [1]});
       const concat = tfl.layers.concatenate().apply([input1, input2]);
       const y = tfl.layers.dense({units: 1, kernelInitializer: 'zeros'})
                     .apply(concat) as tfl.SymbolicTensor;
       const model = tfl.model({inputs: [input1, input2], outputs: y});
       model.compile(
           {loss: 'meanSquaredError', optimizer: 'sgd', metrics: ['accuracy']});

       const batchSize = 8;
       const epochs = 2;
       const batchesPerEpoch = 3;

       // Training data.
       const xTensorsFunc = () => {
         const output: {[name: string]: tfc.Tensor[]} = {};
         output[input1.name] = [
           tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]),
           tfc.ones([batchSize, 1])
         ];
         output[input2.name] = [
           tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]),
           tfc.ones([batchSize, 1])
         ];
         return output;
       };
       const yTensorsFunc =
           () => [tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]), tfc.ones([
             batchSize, 1
           ])];
       const dataset = new FakeNumericDataset({
         xShape: [1],
         yShape: [1],
         batchSize,
         numBatches: batchesPerEpoch,
         xTensorsFunc,
         yTensorsFunc
       });

       // Validation data.
       const valXs: tfc.NamedTensorMap = {};
       valXs[input1.name] = tfc.zeros([batchSize, 1]);
       valXs[input2.name] = tfc.zeros([batchSize, 1]);
       const valYs = tfc.zeros([batchSize, 1]);

       // Do a burn-in call to account for initialization of cached tensors (for
       // the memory-leak check below).
       await model.fitDataset(dataset, {
         epochs: 1,
         validationData: [valXs, valYs],
         validationBatchSize: batchSize
       });
       model.setWeights([tfc.zeros([2, 1]), tfc.zeros([1])]);

       const numTensors0 = tfc.memory().numTensors;
       const history = await model.fitDataset(dataset, {
         epochs,
         validationData: [valXs, valYs],
         validationBatchSize: batchSize
       });
       const numTensors1 = tfc.memory().numTensors;
       expect(numTensors1).toEqual(numTensors0);
       expect(Object.keys(history.history).sort()).toEqual([
         'acc', 'loss', 'val_acc', 'val_loss'
       ]);
       expect(history.history.loss.length).toEqual(2);
       expect(history.history.loss[0]).toBeCloseTo(0.888116);
       expect(history.history.loss[1]).toBeCloseTo(0.612685);
       expect(history.history.acc.length).toEqual(2);
       expect(history.history.acc[0]).toBeCloseTo(0);
       expect(history.history.acc[1]).toBeCloseTo(0);
       expect(history.history.val_loss.length).toEqual(2);
       expect(history.history.val_loss[0]).toBeCloseTo(0.003189);
       expect(history.history.val_loss[1]).toBeCloseTo(0.010687);
       expect(history.history.val_acc.length).toEqual(2);
       expect(history.history.val_acc[0]).toBeCloseTo(1.0);
       expect(history.history.val_acc[1]).toBeCloseTo(1.0);
       expectArraysClose(
           model.getWeights()[0], tfc.tensor2d([[0.103377], [0.103377]]));
       expectArraysClose(model.getWeights()[1], tfc.tensor1d([0.103377]));
     });

  it('2 input, 1 missing input in dataset, with batchesPerEpoch', async () => {
    // Create a functional model with 2 inputs.
    const input1 = tfl.layers.input({shape: [1]});
    const input2 = tfl.layers.input({shape: [1]});
    const concat = tfl.layers.concatenate().apply([input1, input2]);
    const y = tfl.layers.dense({units: 1, kernelInitializer: 'zeros'})
                  .apply(concat) as tfl.SymbolicTensor;
    const model = tfl.model({inputs: [input1, input2], outputs: y});
    model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});

    const batchSize = 8;
    const epochs = 2;
    const batchesPerEpoch = 3;
    const xTensorsFunc = () => {
      const output: {[name: string]: tfc.Tensor[]} = {};
      output[input1.name] = [
        tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]),
        tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]),
        tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1])
      ];
      // Note: input2 is missing from the data, by intention.
      return output;
    };
    const yTensorsFunc = () =>
        [tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]),
         tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]),
         tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1])];
    const dataset = new FakeNumericDataset({
      xShape: [1],
      yShape: [1],
      batchSize,
      numBatches: batchesPerEpoch * epochs,
      xTensorsFunc,
      yTensorsFunc
    });

    let errorCaught: Error;
    try {
      await model.fitDataset(dataset, {batchesPerEpoch, epochs});
    } catch (err) {
      errorCaught = err;
    }
    expect(errorCaught.message)
        .toEqual(
            'The feature data generated by the dataset lacks the required ' +
            `input key '${input2.name}'.`);
  });

  it('2 input, 1 missing input in dataset, no batchesPerEpoch', async () => {
    // Create a functional model with 2 inputs.
    const input1 = tfl.layers.input({shape: [1]});
    const input2 = tfl.layers.input({shape: [1]});
    const concat = tfl.layers.concatenate().apply([input1, input2]);
    const y = tfl.layers.dense({units: 1, kernelInitializer: 'zeros'})
                  .apply(concat) as tfl.SymbolicTensor;
    const model = tfl.model({inputs: [input1, input2], outputs: y});
    model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});

    const batchSize = 8;
    const epochs = 2;
    const batchesPerEpoch = 3;
    const xTensorsFunc = () => {
      const output: {[name: string]: tfc.Tensor[]} = {};
      output[input1.name] = [
        tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]),
        tfc.ones([batchSize, 1])
      ];
      // Note: input2 is missing from the data, by intention.
      return output;
    };
    const yTensorsFunc =
        () => [tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]), tfc.ones([
          batchSize, 1
        ])];
    const dataset = new FakeNumericDataset({
      xShape: [1],
      yShape: [1],
      batchSize,
      numBatches: batchesPerEpoch,
      xTensorsFunc,
      yTensorsFunc
    });

    let errorCaught: Error;
    try {
      await model.fitDataset(dataset, {epochs});
    } catch (err) {
      errorCaught = err;
    }
    expect(errorCaught.message)
        .toEqual(
            'The feature data generated by the dataset lacks the required ' +
            `input key '${input2.name}'.`);
  });

  it('Legacy array input format works but throws warnings', async () => {
    const model = createDenseModel();
    model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});
    const batchSize = 8;
    const batchesPerEpoch = 3;
    const dataset = new FakeNumericDatasetLegacyArrayForm(
        {xShape: [1], yShape: [1], batchSize, numBatches: batchesPerEpoch});
    // Do a burn-in call to account for initialization of cached tensors (for
    // the memory-leak check below).
    await model.fitDataset(dataset, {batchesPerEpoch, epochs: 1});
    model.setWeights([tfc.zeros([1, 1]), tfc.zeros([1])]);
    const warningMessages: string[] = [];
    spyOn(console, 'warn')
        .and.callFake((msg: string) => warningMessages.push(msg));
    const numTensors0 = tfc.memory().numTensors;
    const epochs = 3;
    const history = await model.fitDataset(dataset, {batchesPerEpoch, epochs});
    const numTensors1 = tfc.memory().numTensors;
    expect(numTensors1).toEqual(numTensors0);
    expect(Object.keys(history.history)).toEqual(['loss']);
    // Only the loss value from the first epoch should be logged.
    // The 2nd and 3rd epochs are cut short because of dataset iterator
    // exhaustion.
    expect(history.history.loss.length).toEqual(1);
    expect(warningMessages.length).toEqual(5);

    const deprecatedWarning =
      'Deprecated argument format: fitDataset() will soon no longer accept ' +
      'Datasets that produce elements in the array form `[xs, ys]`.  Instead ' +
      'it now expects elements of the form `{xs: xVal, ys: yVal}`, where the ' +
      'two values may be `tf.Tensor`, an array of Tensors, or a map of ' +
      'string to Tensor.  You can disable deprecation warnings with ' +
      'tf.disableDeprecationWarnings().';

    expect(warningMessages[0]).toEqual(deprecatedWarning);
    expect(warningMessages[1]).toEqual(deprecatedWarning);
    expect(warningMessages[2]).toEqual(deprecatedWarning);
    expect(warningMessages[3])
        .toMatch(/You provided `batchesPerEpoch` as .* 9 batches/);
    expect(warningMessages[4])
        .toMatch(/You provided `batchesPerEpoch` as .* 9 batches/);
  });

  // Reference Python tf.keras code:
  //
  // ```py
  // import numpy as np
  // import tensorflow as tf
  //
  // batch_size = 8
  // num_batches = 3
  // epochs = 2
  //
  // x = tf.keras.Input(shape=[1], name='x')
  //
  // output1 = tf.keras.layers.Dense(
  //     1, kernel_initializer='zeros')(x)
  // output2 = tf.keras.layers.Dense(
  //     1, kernel_initializer='zeros')(x)
  //
  // model = tf.keras.Model(inputs=x, outputs=[output1, output2])
  // model.compile(
  //     loss = 'mean_squared_error', optimizer = 'sgd', metrics =
  //     ['accuracy'])
  // model.summary()
  // output1_name = model.output_names[0]
  // output2_name = model.output_names[1]
  // print(output1_name)
  // print(output2_name)
  //
  // xs = np.ones([batch_size * num_batches * epochs, 1])
  // ys1 = np.ones([batch_size * num_batches * epochs, 1])
  // ys2 = np.ones([batch_size * num_batches * epochs, 1])
  // dataset = tf.data.Dataset.from_tensor_slices(
  //     (xs, {output1_name: ys1, output2_name: ys2})).batch(batch_size)
  //
  // history = model.fit(dataset,
  //                     steps_per_epoch=num_batches,
  //                     epochs=epochs)
  // print(history.history)
  // print(model.get_weights()[0])
  // print(model.get_weights()[1])
  // print(model.get_weights()[2])
  // print(model.get_weights()[3])
  // ```
  it('1 input, 2 outputs, 1 metric, no validation, with batchesPerEpoch',
     async () => {
       // Create a functional model with 2 outputs.
       const x = tfl.layers.input({shape: [1]});
       const output1 =
           tfl.layers.dense({units: 1, kernelInitializer: 'zeros'}).apply(x) as
           tfl.SymbolicTensor;
       const output2 =
           tfl.layers.dense({units: 1, kernelInitializer: 'zeros'}).apply(x) as
           tfl.SymbolicTensor;
       const model = tfl.model({inputs: x, outputs: [output1, output2]});
       model.compile(
           {loss: 'meanSquaredError', optimizer: 'sgd', metrics: ['accuracy']});

       const batchSize = 8;
       const epochs = 2;
       const batchesPerEpoch = 3;

       const xTensorsFunc = () =>
           [tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]),
            tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]),
            tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1])];
       const yTensorsFunc = () => {
         const output: {[name: string]: tfc.Tensor[]} = {};
         output[model.outputNames[0]] = [
           tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]),
           tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]),
           tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1])
         ];
         output[model.outputNames[1]] = [
           tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]),
           tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]),
           tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1])
         ];
         return output;
       };

       const yShape: {[name: string]: number[]} = {};
       yShape[model.outputNames[0]] = [1];
       yShape[model.outputNames[1]] = [1];
       const dataset = new FakeNumericDataset({
         xShape: [1],
         yShape,
         batchSize,
         numBatches: batchesPerEpoch * epochs,
         xTensorsFunc,
         yTensorsFunc
       });

       // Do a burn-in call to account for initialization of cached tensors (for
       // the memory-leak check below).
       await model.fitDataset(dataset, {batchesPerEpoch, epochs: 1});
       model.setWeights([
         tfc.zeros([1, 1]), tfc.zeros([1]), tfc.zeros([1, 1]), tfc.zeros([1])
       ]);

       const numTensors0 = tfc.memory().numTensors;
       const history =
           await model.fitDataset(dataset, {batchesPerEpoch, epochs});

       const numTensors1 = tfc.memory().numTensors;
       expect(numTensors1).toEqual(numTensors0);

       const output1AccName = model.outputNames[0] + '_acc';
       const output1LossName = model.outputNames[0] + '_loss';
       const output2AccName = model.outputNames[1] + '_acc';
       const output2LossName = model.outputNames[1] + '_loss';

       expect(Object.keys(history.history)).toEqual([
         'loss', output1LossName, output2LossName, output1AccName,
         output2AccName
       ]);
       expect(history.history.loss.length).toEqual(2);
       expect(history.history.loss[0]).toBeCloseTo(1.847297);
       expect(history.history.loss[1]).toBeCloseTo(1.445986);
       expect(history.history[output1LossName].length).toEqual(2);
       expect(history.history[output1LossName][0]).toBeCloseTo(0.923648);
       expect(history.history[output1LossName][1]).toBeCloseTo(0.722993);
       expect(history.history[output2LossName].length).toEqual(2);
       expect(history.history[output2LossName][0]).toBeCloseTo(0.923648);
       expect(history.history[output2LossName][1]).toBeCloseTo(0.722993);
       expect(history.history[output1AccName].length).toEqual(2);
       expect(history.history[output1AccName][0]).toBeCloseTo(0);
       expect(history.history[output1AccName][1]).toBeCloseTo(0);
       expect(history.history[output2AccName].length).toEqual(2);
       expect(history.history[output2AccName][0]).toBeCloseTo(0);
       expect(history.history[output2AccName][1]).toBeCloseTo(0);
       expectArraysClose(model.getWeights()[0], tfc.tensor2d([[0.108621]]));
       expectArraysClose(model.getWeights()[1], tfc.tensor1d([0.108621]));
       expectArraysClose(model.getWeights()[2], tfc.tensor2d([[0.108621]]));
       expectArraysClose(model.getWeights()[3], tfc.tensor1d([0.108621]));
     });

  it('1 input, 2 outputs, 1 metric, no validation, no batchesPerEpoch',
     async () => {
       // Create a functional model with 2 outputs.
       const x = tfl.layers.input({shape: [1]});
       const output1 =
           tfl.layers.dense({units: 1, kernelInitializer: 'zeros'}).apply(x) as
           tfl.SymbolicTensor;
       const output2 =
           tfl.layers.dense({units: 1, kernelInitializer: 'zeros'}).apply(x) as
           tfl.SymbolicTensor;
       const model = tfl.model({inputs: x, outputs: [output1, output2]});
       model.compile(
           {loss: 'meanSquaredError', optimizer: 'sgd', metrics: ['accuracy']});

       const batchSize = 8;
       const epochs = 2;
       const batchesPerEpoch = 3;

       const xTensorsFunc =
           () => [tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]), tfc.ones([
             batchSize, 1
           ])];
       const yTensorsFunc = () => {
         const output: {[name: string]: tfc.Tensor[]} = {};
         output[model.outputNames[0]] = [
           tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]),
           tfc.ones([batchSize, 1])
         ];
         output[model.outputNames[1]] = [
           tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]),
           tfc.ones([batchSize, 1])
         ];
         return output;
       };

       const yShape: {[name: string]: number[]} = {};
       yShape[model.outputNames[0]] = [1];
       yShape[model.outputNames[1]] = [1];
       const dataset = new FakeNumericDataset({
         xShape: [1],
         yShape,
         batchSize,
         numBatches: batchesPerEpoch,
         xTensorsFunc,
         yTensorsFunc
       });

       // Do a burn-in call to account for initialization of cached tensors (for
       // the memory-leak check below).
       await model.fitDataset(dataset, {epochs: 1});
       model.setWeights([
         tfc.zeros([1, 1]), tfc.zeros([1]), tfc.zeros([1, 1]), tfc.zeros([1])
       ]);

       const numTensors0 = tfc.memory().numTensors;
       const history = await model.fitDataset(dataset, {epochs});

       const numTensors1 = tfc.memory().numTensors;
       expect(numTensors1).toEqual(numTensors0);

       const output1AccName = model.outputNames[0] + '_acc';
       const output1LossName = model.outputNames[0] + '_loss';
       const output2AccName = model.outputNames[1] + '_acc';
       const output2LossName = model.outputNames[1] + '_loss';

       expect(Object.keys(history.history)).toEqual([
         'loss', output1LossName, output2LossName, output1AccName,
         output2AccName
       ]);
       expect(history.history.loss.length).toEqual(2);
       expect(history.history.loss[0]).toBeCloseTo(1.847297);
       expect(history.history.loss[1]).toBeCloseTo(1.445986);
       expect(history.history[output1LossName].length).toEqual(2);
       expect(history.history[output1LossName][0]).toBeCloseTo(0.923648);
       expect(history.history[output1LossName][1]).toBeCloseTo(0.722993);
       expect(history.history[output2LossName].length).toEqual(2);
       expect(history.history[output2LossName][0]).toBeCloseTo(0.923648);
       expect(history.history[output2LossName][1]).toBeCloseTo(0.722993);
       expect(history.history[output1AccName].length).toEqual(2);
       expect(history.history[output1AccName][0]).toBeCloseTo(0);
       expect(history.history[output1AccName][1]).toBeCloseTo(0);
       expect(history.history[output2AccName].length).toEqual(2);
       expect(history.history[output2AccName][0]).toBeCloseTo(0);
       expect(history.history[output2AccName][1]).toBeCloseTo(0);
       expectArraysClose(model.getWeights()[0], tfc.tensor2d([[0.108621]]));
       expectArraysClose(model.getWeights()[1], tfc.tensor1d([0.108621]));
       expectArraysClose(model.getWeights()[2], tfc.tensor2d([[0.108621]]));
       expectArraysClose(model.getWeights()[3], tfc.tensor1d([0.108621]));
     });

  // Reference Python tf.keras code:
  //
  // ```py
  // import numpy as np
  // import tensorflow as tf
  //
  // batch_size = 8
  // num_batches = 3
  // epochs = 2
  //
  // x = tf.keras.Input(shape=[1], name='x')
  //
  // output1 = tf.keras.layers.Dense(
  //     1, kernel_initializer='zeros')(x)
  // output2 = tf.keras.layers.Dense(
  //     1, kernel_initializer='zeros')(x)
  //
  // model = tf.keras.Model(inputs=x, outputs=[output1, output2])
  // model.compile(
  //     loss = 'mean_squared_error', optimizer = 'sgd', metrics =
  //     ['accuracy'])
  // model.summary()
  // output1_name = model.output_names[0]
  // output2_name = model.output_names[1]
  // print(output1_name)
  // print(output2_name)
  //
  // xs = np.ones([batch_size * num_batches * epochs, 1])
  // ys1 = np.ones([batch_size * num_batches * epochs, 1])
  // ys2 = np.ones([batch_size * num_batches * epochs, 1])
  // dataset = tf.data.Dataset.from_tensor_slices(
  //     (xs, {output1_name: ys1, output2_name: ys2})).batch(batch_size)
  //
  // val_xs = np.zeros([batch_size, 1])
  // val_ys = {output1_name: np.zeros([batch_size, 1]),
  //           output1_name: np.zeros([batch_size, 1])}
  //
  // history = model.fit(dataset,
  //                     steps_per_epoch=num_batches,
  //                     epochs=epochs,
  //                     validation_data=[val_xs, val_ys])
  // print(history.history)
  // print(model.get_weights()[0])
  // print(model.get_weights()[1])
  // print(model.get_weights()[2])
  // print(model.get_weights()[3])
  // ```
  it('1 input, 2 outputs, 1 metric, tensor array validation, ' +
         'with batchesPerEpoch',
     async () => {
       // Create a functional model with 2 outputs.
       const x = tfl.layers.input({shape: [1]});
       const output1 =
           tfl.layers.dense({units: 1, kernelInitializer: 'zeros'}).apply(x) as
           tfl.SymbolicTensor;
       const output2 =
           tfl.layers.dense({units: 1, kernelInitializer: 'zeros'}).apply(x) as
           tfl.SymbolicTensor;
       const model = tfl.model({inputs: x, outputs: [output1, output2]});
       model.compile(
           {loss: 'meanSquaredError', optimizer: 'sgd', metrics: ['accuracy']});

       const batchSize = 8;
       const epochs = 2;
       const batchesPerEpoch = 3;

       const xTensorsFunc = () =>
           [tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]),
            tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]),
            tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1])];
       const yTensorsFunc = () => {
         const output: {[name: string]: tfc.Tensor[]} = {};
         output[model.outputNames[0]] = [
           tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]),
           tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]),
           tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1])
         ];
         output[model.outputNames[1]] = [
           tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]),
           tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]),
           tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1])
         ];
         return output;
       };

       const yShape: {[name: string]: number[]} = {};
       yShape[model.outputNames[0]] = [1];
       yShape[model.outputNames[1]] = [1];
       const dataset = new FakeNumericDataset({
         xShape: [1],
         yShape,
         batchSize,
         numBatches: batchesPerEpoch * epochs,
         xTensorsFunc,
         yTensorsFunc
       });

       // Validation data.
       const valXs = tfc.zeros([batchSize, 1]);
       const valYs: tfc.NamedTensorMap = {};
       valYs[model.outputNames[0]] = tfc.zeros([batchSize, 1]);
       valYs[model.outputNames[1]] = tfc.zeros([batchSize, 1]);

       // Do a burn-in call to account for initialization of cached tensors (for
       // the memory-leak check below).
       await model.fitDataset(dataset, {
         batchesPerEpoch,
         epochs: 1,
         validationData: [valXs, valYs],
         validationBatchSize: batchSize
       });
       model.setWeights([
         tfc.zeros([1, 1]), tfc.zeros([1]), tfc.zeros([1, 1]), tfc.zeros([1])
       ]);

       const history = await model.fitDataset(dataset, {
         batchesPerEpoch,
         epochs,
         validationData: [valXs, valYs],
         validationBatchSize: batchSize
       });

       const output1AccName = model.outputNames[0] + '_acc';
       const output1LossName = model.outputNames[0] + '_loss';
       const output2AccName = model.outputNames[1] + '_acc';
       const output2LossName = model.outputNames[1] + '_loss';
       const valOutput1AccName = 'val_' + output1AccName;
       const valOutput1LossName = 'val_' + output1LossName;
       const valOutput2AccName = 'val_' + output2AccName;
       const valOutput2LossName = 'val_' + output2LossName;

       expect(Object.keys(history.history)).toEqual([
         'val_loss', valOutput1LossName, valOutput2LossName, valOutput1AccName,
         valOutput2AccName, 'loss', output1LossName, output2LossName,
         output1AccName, output2AccName
       ]);
       expect(history.history.loss.length).toEqual(2);
       expect(history.history.loss[0]).toBeCloseTo(1.847297);
       expect(history.history.loss[1]).toBeCloseTo(1.445986);
       expect(history.history[output1LossName].length).toEqual(2);
       expect(history.history[output1LossName][0]).toBeCloseTo(0.923648);
       expect(history.history[output1LossName][1]).toBeCloseTo(0.722993);
       expect(history.history[output2LossName].length).toEqual(2);
       expect(history.history[output2LossName][0]).toBeCloseTo(0.923648);
       expect(history.history[output2LossName][1]).toBeCloseTo(0.722993);
       expect(history.history[output1AccName].length).toEqual(2);
       expect(history.history[output1AccName][0]).toBeCloseTo(0);
       expect(history.history[output1AccName][1]).toBeCloseTo(0);
       expect(history.history[output2AccName].length).toEqual(2);
       expect(history.history[output2AccName][0]).toBeCloseTo(0);
       expect(history.history[output2AccName][1]).toBeCloseTo(0);
       expect(history.history.val_loss.length).toEqual(2);
       expect(history.history.val_loss[0]).toBeCloseTo(0.003321);
       expect(history.history.val_loss[1]).toBeCloseTo(0.011798);
       expect(history.history[valOutput1LossName].length).toEqual(2);
       expect(history.history[valOutput1LossName][0]).toBeCloseTo(0.006642);
       expect(history.history[valOutput1LossName][1]).toBeCloseTo(0.023597);
       expect(history.history[valOutput2LossName].length).toEqual(2);
       expect(history.history[valOutput2LossName][0]).toBeCloseTo(0.003321);
       expect(history.history[valOutput2LossName][1]).toBeCloseTo(0.011798);
       expect(history.history[valOutput1AccName].length).toEqual(2);
       expect(history.history[valOutput1AccName][0]).toBeCloseTo(0.003321);
       expect(history.history[valOutput1AccName][1]).toBeCloseTo(0.011798);
       expect(history.history[valOutput2AccName].length).toEqual(2);
       expect(history.history[valOutput2AccName][0]).toBeCloseTo(1);
       expect(history.history[valOutput2AccName][1]).toBeCloseTo(1);
       expectArraysClose(model.getWeights()[0], tfc.tensor2d([[0.108621]]));
       expectArraysClose(model.getWeights()[1], tfc.tensor1d([0.108621]));
       expectArraysClose(model.getWeights()[2], tfc.tensor2d([[0.108621]]));
       expectArraysClose(model.getWeights()[3], tfc.tensor1d([0.108621]));
     });

  it('1 input, 2 outputs, 1 metric, tensor array validation, ' +
         'no batchesPerEpoch',
     async () => {
       // Create a functional model with 2 outputs.
       const x = tfl.layers.input({shape: [1]});
       const output1 =
           tfl.layers.dense({units: 1, kernelInitializer: 'zeros'}).apply(x) as
           tfl.SymbolicTensor;
       const output2 =
           tfl.layers.dense({units: 1, kernelInitializer: 'zeros'}).apply(x) as
           tfl.SymbolicTensor;
       const model = tfl.model({inputs: x, outputs: [output1, output2]});
       model.compile(
           {loss: 'meanSquaredError', optimizer: 'sgd', metrics: ['accuracy']});

       const batchSize = 8;
       const epochs = 2;
       const batchesPerEpoch = 3;

       const xTensorsFunc =
           () => [tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]), tfc.ones([
             batchSize, 1
           ])];
       const yTensorsFunc = () => {
         const output: {[name: string]: tfc.Tensor[]} = {};
         output[model.outputNames[0]] = [
           tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]),
           tfc.ones([batchSize, 1])
         ];
         output[model.outputNames[1]] = [
           tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]),
           tfc.ones([batchSize, 1])
         ];
         return output;
       };

       const yShape: {[name: string]: number[]} = {};
       yShape[model.outputNames[0]] = [1];
       yShape[model.outputNames[1]] = [1];
       const dataset = new FakeNumericDataset({
         xShape: [1],
         yShape,
         batchSize,
         numBatches: batchesPerEpoch,
         xTensorsFunc,
         yTensorsFunc
       });

       // Validation data.
       const valXs = tfc.zeros([batchSize, 1]);
       const valYs: tfc.NamedTensorMap = {};
       valYs[model.outputNames[0]] = tfc.zeros([batchSize, 1]);
       valYs[model.outputNames[1]] = tfc.zeros([batchSize, 1]);

       // Do a burn-in call to account for initialization of cached tensors (for
       // the memory-leak check below).
       await model.fitDataset(dataset, {
         epochs: 1,
         validationData: [valXs, valYs],
         validationBatchSize: batchSize
       });
       model.setWeights([
         tfc.zeros([1, 1]), tfc.zeros([1]), tfc.zeros([1, 1]), tfc.zeros([1])
       ]);

       const history = await model.fitDataset(dataset, {
         epochs,
         validationData: [valXs, valYs],
         validationBatchSize: batchSize
       });

       const output1AccName = model.outputNames[0] + '_acc';
       const output1LossName = model.outputNames[0] + '_loss';
       const output2AccName = model.outputNames[1] + '_acc';
       const output2LossName = model.outputNames[1] + '_loss';
       const valOutput1AccName = 'val_' + output1AccName;
       const valOutput1LossName = 'val_' + output1LossName;
       const valOutput2AccName = 'val_' + output2AccName;
       const valOutput2LossName = 'val_' + output2LossName;

       expect(Object.keys(history.history)).toEqual([
         'val_loss', valOutput1LossName, valOutput2LossName, valOutput1AccName,
         valOutput2AccName, 'loss', output1LossName, output2LossName,
         output1AccName, output2AccName
       ]);
       expect(history.history.loss.length).toEqual(2);
       expect(history.history.loss[0]).toBeCloseTo(1.847297);
       expect(history.history.loss[1]).toBeCloseTo(1.445986);
       expect(history.history[output1LossName].length).toEqual(2);
       expect(history.history[output1LossName][0]).toBeCloseTo(0.923648);
       expect(history.history[output1LossName][1]).toBeCloseTo(0.722993);
       expect(history.history[output2LossName].length).toEqual(2);
       expect(history.history[output2LossName][0]).toBeCloseTo(0.923648);
       expect(history.history[output2LossName][1]).toBeCloseTo(0.722993);
       expect(history.history[output1AccName].length).toEqual(2);
       expect(history.history[output1AccName][0]).toBeCloseTo(0);
       expect(history.history[output1AccName][1]).toBeCloseTo(0);
       expect(history.history[output2AccName].length).toEqual(2);
       expect(history.history[output2AccName][0]).toBeCloseTo(0);
       expect(history.history[output2AccName][1]).toBeCloseTo(0);
       expect(history.history.val_loss.length).toEqual(2);
       expect(history.history.val_loss[0]).toBeCloseTo(0.003321);
       expect(history.history.val_loss[1]).toBeCloseTo(0.011798);
       expect(history.history[valOutput1LossName].length).toEqual(2);
       expect(history.history[valOutput1LossName][0]).toBeCloseTo(0.006642);
       expect(history.history[valOutput1LossName][1]).toBeCloseTo(0.023597);
       expect(history.history[valOutput2LossName].length).toEqual(2);
       expect(history.history[valOutput2LossName][0]).toBeCloseTo(0.003321);
       expect(history.history[valOutput2LossName][1]).toBeCloseTo(0.011798);
       expect(history.history[valOutput1AccName].length).toEqual(2);
       expect(history.history[valOutput1AccName][0]).toBeCloseTo(0.003321);
       expect(history.history[valOutput1AccName][1]).toBeCloseTo(0.011798);
       expect(history.history[valOutput2AccName].length).toEqual(2);
       expect(history.history[valOutput2AccName][0]).toBeCloseTo(1);
       expect(history.history[valOutput2AccName][1]).toBeCloseTo(1);
       expectArraysClose(model.getWeights()[0], tfc.tensor2d([[0.108621]]));
       expectArraysClose(model.getWeights()[1], tfc.tensor1d([0.108621]));
       expectArraysClose(model.getWeights()[2], tfc.tensor2d([[0.108621]]));
       expectArraysClose(model.getWeights()[3], tfc.tensor1d([0.108621]));
     });

  it('2 outputs, 1 missing output in dataset, with batchesPerEpoch',
     async () => {
       // Create a functional model with 2 outputs.
       const x = tfl.layers.input({shape: [1]});
       const output1 =
           tfl.layers.dense({units: 1, kernelInitializer: 'zeros'}).apply(x) as
           tfl.SymbolicTensor;
       const output2 =
           tfl.layers.dense({units: 1, kernelInitializer: 'zeros'}).apply(x) as
           tfl.SymbolicTensor;
       const model = tfl.model({inputs: x, outputs: [output1, output2]});
       model.compile(
           {loss: 'meanSquaredError', optimizer: 'sgd', metrics: ['accuracy']});

       const batchSize = 8;
       const epochs = 2;
       const batchesPerEpoch = 3;

       const xTensorsFunc = () =>
           [tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]),
            tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]),
            tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1])];
       const yTensorsFunc = () => {
         const output: {[name: string]: tfc.Tensor[]} = {};
         output[model.outputNames[0]] = [
           tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]),
           tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]),
           tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1])
         ];
         return output;
       };

       const yShape: {[name: string]: number[]} = {};
       yShape[model.outputNames[0]] = [1];
       yShape[model.outputNames[1]] = [1];
       const dataset = new FakeNumericDataset({
         xShape: [1],
         yShape,
         batchSize,
         numBatches: batchesPerEpoch * epochs,
         xTensorsFunc,
         yTensorsFunc
       });

       let errorCaught: Error;
       try {
         await model.fitDataset(dataset, {batchesPerEpoch, epochs});
       } catch (err) {
         errorCaught = err;
       }
       expect(errorCaught.message)
           .toEqual(
               'The feature data generated by the dataset lacks the required ' +
               `output key '${model.outputNames[1]}'.`);
     });

  it('2 outputs, 1 missing output in dataset, no batchesPerEpoch', async () => {
    // Create a functional model with 2 outputs.
    const x = tfl.layers.input({shape: [1]});
    const output1 =
        tfl.layers.dense({units: 1, kernelInitializer: 'zeros'}).apply(x) as
        tfl.SymbolicTensor;
    const output2 =
        tfl.layers.dense({units: 1, kernelInitializer: 'zeros'}).apply(x) as
        tfl.SymbolicTensor;
    const model = tfl.model({inputs: x, outputs: [output1, output2]});
    model.compile(
        {loss: 'meanSquaredError', optimizer: 'sgd', metrics: ['accuracy']});

    const batchSize = 8;
    const epochs = 2;
    const batchesPerEpoch = 3;

    const xTensorsFunc = () =>
        [tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]),
         tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]),
         tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1])];
    const yTensorsFunc = () => {
      const output: {[name: string]: tfc.Tensor[]} = {};
      output[model.outputNames[0]] = [
        tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]),
        tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]),
        tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1])
      ];
      return output;
    };

    const yShape: {[name: string]: number[]} = {};
    yShape[model.outputNames[0]] = [1];
    yShape[model.outputNames[1]] = [1];
    const dataset = new FakeNumericDataset({
      xShape: [1],
      yShape,
      batchSize,
      numBatches: batchesPerEpoch,
      xTensorsFunc,
      yTensorsFunc
    });

    let errorCaught: Error;
    try {
      await model.fitDataset(dataset, {epochs});
    } catch (err) {
      errorCaught = err;
    }
    expect(errorCaught.message)
        .toEqual(
            'The feature data generated by the dataset lacks the required ' +
            `output key '${model.outputNames[1]}'.`);
  });

  // Reference Python tf.keras code:
  //
  // ```py
  // import numpy as np
  // import tensorflow as tf
  //
  // batch_size = 8
  // num_batches = 3
  // epochs = 2
  //
  // input1 = tf.keras.Input(shape=[1], name='x1')
  // input2 = tf.keras.Input(shape=[1], name='x2')
  // concat = tf.keras.layers.concatenate([input1, input2])
  //
  // output1 = tf.keras.layers.Dense(
  //     1, kernel_initializer='zeros')(concat)
  // output2 = tf.keras.layers.Dense(
  //     1, kernel_initializer='zeros')(concat)
  //
  // model = tf.keras.Model(
  //      inputs=[input1, input2],
  //      outputs=[output1, output2])
  // model.compile(
  //     loss = 'mean_squared_error', optimizer = 'sgd', metrics =
  //     ['accuracy'])
  // model.summary()
  // output1_name = model.output_names[0]
  // output2_name = model.output_names[1]
  // print(x1)
  // print(x2)
  // print(output1_name)
  // print(output2_name)
  //
  // xs1 = np.ones([batch_size * num_batches * epochs, 1])
  // xs2 = np.ones([batch_size * num_batches * epochs, 1])
  // ys1 = np.ones([batch_size * num_batches * epochs, 1])
  // ys2 = np.ones([batch_size * num_batches * epochs, 1])
  // dataset = tf.data.Dataset.from_tensor_slices(
  //     ({
  //        x1: xs1,
  //        x2: xs2
  //      },{
  //        output1_name: ys1,
  //        output2_name: ys2
  //      })).batch(batch_size)
  //
  // history = model.fit(dataset,
  //                     steps_per_epoch=num_batches,
  //                     epochs=epochs)
  // print(history.history)
  // print(model.get_weights()[0])
  // print(model.get_weights()[1])
  // print(model.get_weights()[2])
  // print(model.get_weights()[3])
  // ```
  it('2 inputs, 2 outputs, 1 metric, no validation, with batchesPerEpoch',
     async () => {
       // Create a functional model with 2 inputs and 2 outputs.
       const input1 = tfl.layers.input({shape: [1]});
       const input2 = tfl.layers.input({shape: [1]});
       const concat = tfl.layers.concatenate().apply([input1, input2]);
       const output1 = tfl.layers.dense({units: 1, kernelInitializer: 'zeros'})
                           .apply(concat) as tfl.SymbolicTensor;
       const output2 = tfl.layers.dense({units: 1, kernelInitializer: 'zeros'})
                           .apply(concat) as tfl.SymbolicTensor;
       const model =
           tfl.model({inputs: [input1, input2], outputs: [output1, output2]});
       model.compile(
           {loss: 'meanSquaredError', optimizer: 'sgd', metrics: ['accuracy']});

       const batchSize = 8;
       const epochs = 2;
       const batchesPerEpoch = 3;

       // Training data.
       const xTensorsFunc = () => {
         const output: {[name: string]: tfc.Tensor[]} = {};
         output[input1.name] = [
           tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]),
           tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]),
           tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1])
         ];
         output[input2.name] = [
           tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]),
           tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]),
           tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1])
         ];
         return output;
       };
       const yTensorsFunc = () => {
         const output: {[name: string]: tfc.Tensor[]} = {};
         output[model.outputNames[0]] = [
           tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]),
           tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]),
           tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1])
         ];
         output[model.outputNames[1]] = [
           tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]),
           tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]),
           tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1])
         ];
         return output;
       };

       const yShape: {[name: string]: number[]} = {};
       yShape[model.outputNames[0]] = [1];
       yShape[model.outputNames[1]] = [1];
       const dataset = new FakeNumericDataset({
         xShape: [1],
         yShape,
         batchSize,
         numBatches: batchesPerEpoch * epochs,
         xTensorsFunc,
         yTensorsFunc
       });

       // Do a burn-in call to account for initialization of cached tensors (for
       // the memory-leak check below).
       await model.fitDataset(dataset, {batchesPerEpoch, epochs: 1});
       model.setWeights([
         tfc.zeros([2, 1]), tfc.zeros([1]), tfc.zeros([2, 1]), tfc.zeros([1])
       ]);

       const numTensors0 = tfc.memory().numTensors;
       const history =
           await model.fitDataset(dataset, {batchesPerEpoch, epochs});

       const numTensors1 = tfc.memory().numTensors;
       expect(numTensors1).toEqual(numTensors0);

       const output1AccName = model.outputNames[0] + '_acc';
       const output1LossName = model.outputNames[0] + '_loss';
       const output2AccName = model.outputNames[1] + '_acc';
       const output2LossName = model.outputNames[1] + '_loss';

       expect(Object.keys(history.history)).toEqual([
         'loss', output1LossName, output2LossName, output1AccName,
         output2AccName
       ]);
       expect(history.history.loss.length).toEqual(2);
       expect(history.history.loss[0]).toBeCloseTo(1.776232);
       expect(history.history.loss[1]).toBeCloseTo(1.225369);
       expect(history.history[output1LossName].length).toEqual(2);
       expect(history.history[output1LossName][0]).toBeCloseTo(0.888116);
       expect(history.history[output1LossName][1]).toBeCloseTo(0.612684);
       expect(history.history[output2LossName].length).toEqual(2);
       expect(history.history[output2LossName][0]).toBeCloseTo(0.888116);
       expect(history.history[output2LossName][1]).toBeCloseTo(0.612684);
       expect(history.history[output1AccName].length).toEqual(2);
       expect(history.history[output1AccName][0]).toBeCloseTo(0);
       expect(history.history[output1AccName][1]).toBeCloseTo(0);
       expect(history.history[output2AccName].length).toEqual(2);
       expect(history.history[output2AccName][0]).toBeCloseTo(0);
       expect(history.history[output2AccName][1]).toBeCloseTo(0);
       expectArraysClose(
           model.getWeights()[0], tfc.tensor2d([[0.103376], [0.103376]]));
       expectArraysClose(model.getWeights()[1], tfc.tensor1d([0.103376]));
       expectArraysClose(
           model.getWeights()[2], tfc.tensor2d([[0.103376], [0.103376]]));
       expectArraysClose(model.getWeights()[3], tfc.tensor1d([0.103376]));
     });

  it('2 inputs, 2 outputs, 1 metric, no validation, no batchesPerEpoch',
     async () => {
       // Create a functional model with 2 inputs and 2 outputs.
       const input1 = tfl.layers.input({shape: [1]});
       const input2 = tfl.layers.input({shape: [1]});
       const concat = tfl.layers.concatenate().apply([input1, input2]);
       const output1 = tfl.layers.dense({units: 1, kernelInitializer: 'zeros'})
                           .apply(concat) as tfl.SymbolicTensor;
       const output2 = tfl.layers.dense({units: 1, kernelInitializer: 'zeros'})
                           .apply(concat) as tfl.SymbolicTensor;
       const model =
           tfl.model({inputs: [input1, input2], outputs: [output1, output2]});
       model.compile(
           {loss: 'meanSquaredError', optimizer: 'sgd', metrics: ['accuracy']});

       const batchSize = 8;
       const epochs = 2;
       const batchesPerEpoch = 3;

       // Training data.
       const xTensorsFunc = () => {
         const output: {[name: string]: tfc.Tensor[]} = {};
         output[input1.name] = [
           tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]),
           tfc.ones([batchSize, 1])
         ];
         output[input2.name] = [
           tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]),
           tfc.ones([batchSize, 1])
         ];
         return output;
       };
       const yTensorsFunc = () => {
         const output: {[name: string]: tfc.Tensor[]} = {};
         output[model.outputNames[0]] = [
           tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]),
           tfc.ones([batchSize, 1])
         ];
         output[model.outputNames[1]] = [
           tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]),
           tfc.ones([batchSize, 1])
         ];
         return output;
       };

       const yShape: {[name: string]: number[]} = {};
       yShape[model.outputNames[0]] = [1];
       yShape[model.outputNames[1]] = [1];
       const dataset = new FakeNumericDataset({
         xShape: [1],
         yShape,
         batchSize,
         numBatches: batchesPerEpoch,
         xTensorsFunc,
         yTensorsFunc
       });

       // Do a burn-in call to account for initialization of cached tensors (for
       // the memory-leak check below).
       await model.fitDataset(dataset, {epochs: 1});
       model.setWeights([
         tfc.zeros([2, 1]), tfc.zeros([1]), tfc.zeros([2, 1]), tfc.zeros([1])
       ]);

       const numTensors0 = tfc.memory().numTensors;
       const history = await model.fitDataset(dataset, {epochs});

       const numTensors1 = tfc.memory().numTensors;
       expect(numTensors1).toEqual(numTensors0);

       const output1AccName = model.outputNames[0] + '_acc';
       const output1LossName = model.outputNames[0] + '_loss';
       const output2AccName = model.outputNames[1] + '_acc';
       const output2LossName = model.outputNames[1] + '_loss';

       expect(Object.keys(history.history)).toEqual([
         'loss', output1LossName, output2LossName, output1AccName,
         output2AccName
       ]);
       expect(history.history.loss.length).toEqual(2);
       expect(history.history.loss[0]).toBeCloseTo(1.776232);
       expect(history.history.loss[1]).toBeCloseTo(1.225369);
       expect(history.history[output1LossName].length).toEqual(2);
       expect(history.history[output1LossName][0]).toBeCloseTo(0.888116);
       expect(history.history[output1LossName][1]).toBeCloseTo(0.612684);
       expect(history.history[output2LossName].length).toEqual(2);
       expect(history.history[output2LossName][0]).toBeCloseTo(0.888116);
       expect(history.history[output2LossName][1]).toBeCloseTo(0.612684);
       expect(history.history[output1AccName].length).toEqual(2);
       expect(history.history[output1AccName][0]).toBeCloseTo(0);
       expect(history.history[output1AccName][1]).toBeCloseTo(0);
       expect(history.history[output2AccName].length).toEqual(2);
       expect(history.history[output2AccName][0]).toBeCloseTo(0);
       expect(history.history[output2AccName][1]).toBeCloseTo(0);
       expectArraysClose(
           model.getWeights()[0], tfc.tensor2d([[0.103376], [0.103376]]));
       expectArraysClose(model.getWeights()[1], tfc.tensor1d([0.103376]));
       expectArraysClose(
           model.getWeights()[2], tfc.tensor2d([[0.103376], [0.103376]]));
       expectArraysClose(model.getWeights()[3], tfc.tensor1d([0.103376]));
     });

  it('Exhausting iterator with batchesPerEpoch throws warning', async () => {
    const model = createDenseModel();
    model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});
    const batchSize = 8;
    const batchesPerEpoch = 3;
    const dataset = new FakeNumericDataset(
        {xShape: [1], yShape: [1], batchSize, numBatches: batchesPerEpoch});
    // Do a burn-in call to account for initialization of cached tensors (for
    // the memory-leak check below).
    await model.fitDataset(dataset, {batchesPerEpoch, epochs: 1});
    model.setWeights([tfc.zeros([1, 1]), tfc.zeros([1])]);
    const warningMessages: string[] = [];
    spyOn(console, 'warn')
        .and.callFake((msg: string) => warningMessages.push(msg));
    const numTensors0 = tfc.memory().numTensors;
    const epochs = 3;
    const history = await model.fitDataset(dataset, {batchesPerEpoch, epochs});
    const numTensors1 = tfc.memory().numTensors;
    expect(numTensors1).toEqual(numTensors0);
    expect(Object.keys(history.history)).toEqual(['loss']);
    // Only the loss value from the first epoch should be logged.
    // The 2nd and 3rd epochs are cut short because of dataset iterator
    // exhaustion.
    expect(history.history.loss.length).toEqual(1);
    expect(warningMessages.length).toEqual(2);
    expect(warningMessages[0])
        .toMatch(/You provided `batchesPerEpoch` as .* 9 batches/);
    expect(warningMessages[1])
        .toMatch(/You provided `batchesPerEpoch` as .* 9 batches/);
  });

  it('Calling fitDataset() without calling compile() errors', async () => {
    const model = createDenseModel();

    const batchSize = 8;
    const numBatches = 3;
    const epochs = 2;
    const dataset = new FakeNumericDataset({
      xShape: [1],
      yShape: [1],
      batchSize,
      numBatches,
    });

    let errorCaught: Error;
    try {
      await model.fitDataset(dataset, {epochs});
    } catch (err) {
      errorCaught = err;
    }
    expect(errorCaught.message)
        .toEqual('The model needs to be compiled before being used.');
  });

  it('Wrong validationBatches leads to Error', async () => {
    const model = createDenseModel();
    model.compile(
        {loss: 'meanSquaredError', optimizer: 'sgd', metrics: ['accuracy']});
    const batchSize = 8;
    const epochs = 2;
    const batchesPerEpoch = 3;
    // Training dataset.
    const dataset = new FakeNumericDataset({
      xShape: [1],
      yShape: [1],
      batchSize,
      numBatches: batchesPerEpoch * epochs
    });
    // Validation dataset.
    const valDataset = new FakeNumericDataset({
      xShape: [1],
      yShape: [1],
      batchSize,
      numBatches: batchesPerEpoch * epochs
    });
    // Do a burn-in call to account for initialization of cached
    // tensors (for the memory-leak check below).
    let errorCaught: Error;
    try {
      await model.fitDataset(dataset, {
        batchesPerEpoch,
        epochs,
        validationData: valDataset,
        validationBatches: 0
      });
    } catch (err) {
      errorCaught = err;
    }
    expect(errorCaught.message)
        .toMatch(/fitDataset.*dataset-based validation.*not to be provided.*0/);
  });

  it('Calling fitDataset with validationSplit leads to Error', async () => {
    const model = createDenseModel();
    model.compile(
        {loss: 'meanSquaredError', optimizer: 'sgd', metrics: ['accuracy']});

    const batchSize = 8;
    const epochs = 2;
    const batchesPerEpoch = 3;

    // Training dataset.
    const dataset = new FakeNumericDataset({
      xShape: [1],
      yShape: [1],
      batchSize,
      numBatches: batchesPerEpoch * epochs
    });

    let errorCaught: Error;
    try {
      await model.fitDataset(
          dataset,
          // tslint:disable-next-line:no-any
          {epochs: 1, batchesPerEpoch: 2, validationSplit: 0.25} as any);
    } catch (err) {
      errorCaught = err;
    }
    expect(errorCaught.message)
        .toMatch(/.*validationSplit.*not supported.*validationData/);
  });

  class StopAfterNBatches extends tfl.Callback {
    private readonly batchesToTrain: number;
    constructor(epochsToTrain: number) {
      super();
      this.batchesToTrain = epochsToTrain;
    }

    async onBatchEnd(batch: number, logs?: Logs) {
      if (batch === this.batchesToTrain - 1) {
        this.model.stopTraining = true;
      }
    }
  }

  it('Stop training resets at start of Model.fitDataset()', async () => {
    const model = createDenseModel();
    model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});

    const batchSize = 8;
    const epochs = 2;
    const batchesPerEpoch = 1;
    const xTensorsFunc =
        () => [tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1])];
    const yTensorsFunc =
        () => [tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1])];
    const dataset = new FakeNumericDataset({
      xShape: [1],
      yShape: [1],
      batchSize,
      numBatches: batchesPerEpoch * epochs,
      xTensorsFunc,
      yTensorsFunc
    });
    // Order 2 epochs of training, but the training should stop after only one
    // epochs due to the callback that orders the training to stop after one
    // batches.
    let history = await model.fitDataset(
        dataset,
        {batchesPerEpoch, epochs, callbacks: [new StopAfterNBatches(1)]});
    expect(history.history.loss.length).toEqual(1);

    // Running fitDataset again should now run to completion
    history = await model.fitDataset(dataset, {batchesPerEpoch, epochs});
    expect(history.history.loss.length).toEqual(2);
  });
});

// TODO(cais): The corresponding test for predict() and evaluate().

describeMathCPUAndGPU('Model.evaluateDataset', () => {
  // Reference Python tf.keras code:
  //
  // ```py
  // import numpy as np
  // import tensorflow as tf
  //
  // tf.enable_eager_execution()
  //
  // batch_size = 8
  // num_batches = 3
  //
  // xs = np.ones([batch_size * num_batches, 1])
  // ys = np.ones([batch_size * num_batches, 1])
  // dataset = tf.data.Dataset.from_tensor_slices((xs, ys)).batch(batch_size)
  //
  // model = tf.keras.Sequential()
  // model.add(tf.keras.layers.Dense(
  //     1,
  //     input_shape=[1],
  //     kernel_initializer='zeros'))
  // model.compile(loss='mean_squared_error',
  //               optimizer=tf.train.GradientDescentOptimizer(0.01))
  //
  // out = model.evaluate(dataset, steps=3, verbose=0)
  // print(out)
  // ```
  it('1 input, 1 output, no metric, with batches specified', async () => {
    const model = createDenseModel();
    model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});

    const batchSize = 8;
    const batches = 3;
    const xTensorsFunc =
        () => [tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]), tfc.ones([
          batchSize, 1
        ])];
    const yTensorsFunc =
        () => [tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]), tfc.ones([
          batchSize, 1
        ])];
    const dataset = new FakeNumericDataset({
      xShape: [1],
      yShape: [1],
      batchSize,
      numBatches: batches,
      xTensorsFunc,
      yTensorsFunc
    });

    // Do a burn-in call to account for initialization of cached tensors (for
    // the memory-leak check below).
    tfc.dispose(
        await model.evaluateDataset(dataset, {batches}) as tfc.Scalar[]);

    const numTensors0 = tfc.memory().numTensors;
    const evalOut =
        await model.evaluateDataset(dataset, {batches}) as tfc.Scalar;
    const expectedLoss = tfc.scalar(1.0);
    expectTensorsClose(evalOut, expectedLoss);
    tfc.dispose(evalOut);
    tfc.dispose(expectedLoss);
    const numTensors1 = tfc.memory().numTensors;
    expect(numTensors1).toEqual(numTensors0);
  });

  it('1 input, 1 output, no metric, no batches specified', async () => {
    const model = createDenseModel();
    model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});

    const batchSize = 8;
    const batches = 3;
    const xTensorsFunc =
        () => [tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]), tfc.ones([
          batchSize, 1
        ])];
    const yTensorsFunc =
        () => [tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]), tfc.ones([
          batchSize, 1
        ])];
    const dataset = new FakeNumericDataset({
      xShape: [1],
      yShape: [1],
      batchSize,
      numBatches: batches,
      xTensorsFunc,
      yTensorsFunc
    });

    // Do a burn-in call to account for initialization of cached tensors (for
    // the memory-leak check below).
    tfc.dispose(await model.evaluateDataset(dataset, {}) as tfc.Scalar[]);

    const numTensors0 = tfc.memory().numTensors;
    const evalOut = await model.evaluateDataset(dataset, {}) as tfc.Scalar;
    const expectedLoss = tfc.scalar(1.0);
    expectTensorsClose(evalOut, expectedLoss);
    tfc.dispose(evalOut);
    tfc.dispose(expectedLoss);
    const numTensors1 = tfc.memory().numTensors;
    expect(numTensors1).toEqual(numTensors0);
  });

  // Reference Python tf.keras code:
  //
  // ```py
  // import numpy as np
  // import tensorflow as tf
  //
  // tf.enable_eager_execution()
  //
  // batch_size = 8
  // num_batches = 3
  //
  // xs = np.ones([batch_size * num_batches, 1])
  // ys = np.ones([batch_size * num_batches, 1])
  // dataset = tf.data.Dataset.from_tensor_slices((xs, ys)).batch(batch_size)
  //
  // model = tf.keras.Sequential()
  // model.add(tf.keras.layers.Dense(
  //     1,
  //     input_shape=[1],
  //     kernel_initializer='zeros'))
  // model.compile(loss='mean_squared_error',
  //               optimizer=tf.train.GradientDescentOptimizer(0.01),
  //               metrics=['accuracy'])
  //
  // out = model.evaluate(dataset, steps=3, verbose=0)
  // print(out)
  // ```
  it('1 input, 1 output, 1 metric, with batches specified', async () => {
    const model = createDenseModel();
    model.compile(
        {loss: 'meanSquaredError', optimizer: 'sgd', metrics: ['acc']});

    const batchSize = 8;
    const batches = 3;
    const xTensorsFunc =
        () => [tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]), tfc.ones([
          batchSize, 1
        ])];
    const yTensorsFunc =
        () => [tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]), tfc.ones([
          batchSize, 1
        ])];
    const dataset = new FakeNumericDataset({
      xShape: [1],
      yShape: [1],
      batchSize,
      numBatches: batches,
      xTensorsFunc,
      yTensorsFunc
    });

    // Do a burn-in call to account for initialization of cached tensors (for
    // the memory-leak check below).
    tfc.dispose(
        await model.evaluateDataset(dataset, {batches}) as tfc.Scalar[]);

    const numTensors0 = tfc.memory().numTensors;
    const evalOut =
        await model.evaluateDataset(dataset, {batches}) as tfc.Scalar[];
    expect(evalOut.length).toEqual(2);
    const expectedLoss = tfc.scalar(1.0);
    const expectedAcc = tfc.scalar(0.0);
    expectTensorsClose(evalOut[0], expectedLoss);
    expectTensorsClose(evalOut[1], expectedAcc);
    tfc.dispose(evalOut);
    tfc.dispose([expectedLoss, expectedAcc]);
    const numTensors1 = tfc.memory().numTensors;
    expect(numTensors1).toEqual(numTensors0);
  });

  it('1 input, 1 output, 1 metric, no batches specified', async () => {
    const model = createDenseModel();
    model.compile(
        {loss: 'meanSquaredError', optimizer: 'sgd', metrics: ['acc']});

    const batchSize = 8;
    const batches = 3;
    const xTensorsFunc =
        () => [tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]), tfc.ones([
          batchSize, 1
        ])];
    const yTensorsFunc =
        () => [tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]), tfc.ones([
          batchSize, 1
        ])];
    const dataset = new FakeNumericDataset({
      xShape: [1],
      yShape: [1],
      batchSize,
      numBatches: batches,
      xTensorsFunc,
      yTensorsFunc
    });

    // Do a burn-in call to account for initialization of cached tensors (for
    // the memory-leak check below).
    tfc.dispose(await model.evaluateDataset(dataset, {}) as tfc.Scalar[]);

    const numTensors0 = tfc.memory().numTensors;
    const evalOut = await model.evaluateDataset(dataset, {}) as tfc.Scalar[];
    expect(evalOut.length).toEqual(2);
    const expectedLoss = tfc.scalar(1.0);
    const expectedAcc = tfc.scalar(0.0);
    expectTensorsClose(evalOut[0], expectedLoss);
    expectTensorsClose(evalOut[1], expectedAcc);
    tfc.dispose(evalOut);
    tfc.dispose([expectedLoss, expectedAcc]);
    const numTensors1 = tfc.memory().numTensors;
    expect(numTensors1).toEqual(numTensors0);
  });

  it('1 input, 1 output, 1 metric, no batches, only 1 arg', async () => {
    const model = createDenseModel();
    model.compile(
        {loss: 'meanSquaredError', optimizer: 'sgd', metrics: ['acc']});

    const batchSize = 8;
    const batches = 3;
    const xTensorsFunc =
        () => [tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]), tfc.ones([
          batchSize, 1
        ])];
    const yTensorsFunc =
        () => [tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]), tfc.ones([
          batchSize, 1
        ])];
    const dataset = new FakeNumericDataset({
      xShape: [1],
      yShape: [1],
      batchSize,
      numBatches: batches,
      xTensorsFunc,
      yTensorsFunc
    });

    // Do a burn-in call to account for initialization of cached tensors (for
    // the memory-leak check below). Use 1-arg call.
    tfc.dispose(await model.evaluateDataset(dataset) as tfc.Scalar[]);

    const numTensors0 = tfc.memory().numTensors;
    // Use 1-arg call, omitting the config object.
    const evalOut = await model.evaluateDataset(dataset) as tfc.Scalar[];
    expect(evalOut.length).toEqual(2);
    const expectedLoss = tfc.scalar(1.0);
    const expectedAcc = tfc.scalar(0.0);
    expectTensorsClose(evalOut[0], expectedLoss);
    expectTensorsClose(evalOut[1], expectedAcc);
    tfc.dispose(evalOut);
    tfc.dispose([expectedLoss, expectedAcc]);
    const numTensors1 = tfc.memory().numTensors;
    expect(numTensors1).toEqual(numTensors0);
  });

  it('1 input, 1 output, iterator exhaustion with batches', async () => {
    const model = createDenseModel();
    model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});

    const batchSize = 8;
    const batches = 3;
    const dataset = new FakeNumericDataset(
        {xShape: [1], yShape: [1], batchSize, numBatches: batches});

    // Do a burn-in call to account for initialization of cached tensors (for
    // the memory-leak check below).
    tfc.dispose(
        await model.evaluateDataset(dataset, {batches}) as tfc.Scalar[]);

    const warningMessages: string[] = [];
    spyOn(console, 'warn')
        .and.callFake((msg: string) => warningMessages.push(msg));

    const numTensors0 = tfc.memory().numTensors;
    tfc.dispose(await model.evaluateDataset(dataset, {batches: batches + 2}));
    const numTensors1 = tfc.memory().numTensors;
    expect(numTensors1).toEqual(numTensors0);

    expect(warningMessages.length).toEqual(1);
    expect(warningMessages[0])
        .toMatch(
            /dataset iterator ran out of data during evaluate.* 5 batches/);
  });

  // Reference Python tf.keras code:
  //
  // ```py
  // import numpy as np
  // import tensorflow as tf
  //
  // tf.enable_eager_execution()
  //
  // batch_size = 8
  // num_batches = 3
  //
  // xs1 = np.ones([batch_size * num_batches, 1])
  // xs2 = np.ones([batch_size * num_batches, 1])
  // ys = np.ones([batch_size * num_batches, 1])
  // dataset = tf.data.Dataset.from_tensor_slices(
  //     ({'input1': xs1, 'input2': xs2}, ys)).batch(batch_size)
  //
  // input1 = tf.keras.Input(shape=[1], name='input1')
  // input2 = tf.keras.Input(shape=[1], name='input2')
  // concat = tf.keras.layers.concatenate([input1, input2])
  // output = tf.keras.layers.Dense(
  //     1,
  //     input_shape=[1],
  //     kernel_initializer='zeros').apply(concat)
  // model = tf.keras.Model(inputs=[input1, input2], outputs=output)
  // model.compile(loss='mean_squared_error',
  //               optimizer=tf.train.GradientDescentOptimizer(0.01),
  //               metrics=['accuracy'])
  //
  // out = model.evaluate(dataset, steps=3, verbose=0)
  // print(out)
  // ```
  it('2 inputs, 1 output, 1 metric, no validation, with batches specified',
     async () => {
       // Create a functional model with 2 inputs.
       const input1 = tfl.layers.input({shape: [1]});
       const input2 = tfl.layers.input({shape: [1]});
       const concat = tfl.layers.concatenate().apply([input1, input2]);
       const y = tfl.layers.dense({units: 1, kernelInitializer: 'zeros'})
                     .apply(concat) as tfl.SymbolicTensor;
       const model = tfl.model({inputs: [input1, input2], outputs: y});
       model.compile(
           {loss: 'meanSquaredError', optimizer: 'sgd', metrics: ['accuracy']});

       const batchSize = 8;
       const batches = 3;
       const xTensorsFunc = () => {
         const output: {[name: string]: tfc.Tensor[]} = {};
         output[input1.name] = [
           tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]),
           tfc.ones([batchSize, 1])
         ];
         output[input2.name] = [
           tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]),
           tfc.ones([batchSize, 1])
         ];
         return output;
       };
       const yTensorsFunc =
           () => [tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]), tfc.ones([
             batchSize, 1
           ])];
       const dataset = new FakeNumericDataset({
         xShape: [1],
         yShape: [1],
         batchSize,
         numBatches: batches,
         xTensorsFunc,
         yTensorsFunc
       });

       // Do a burn-in call to account for initialization of cached tensors (for
       // the memory-leak check below).
       await model.evaluateDataset(dataset, {batches});
       model.setWeights([tfc.zeros([2, 1]), tfc.zeros([1])]);

       const numTensors0 = tfc.memory().numTensors;
       const evalOut =
           await model.evaluateDataset(dataset, {batches}) as tfc.Scalar[];
       const expectedLoss = tfc.scalar(1.0);
       const expectedAcc = tfc.scalar(0.0);
       expectTensorsClose(evalOut[0], expectedLoss);
       expectTensorsClose(evalOut[1], expectedAcc);
       tfc.dispose(evalOut);
       tfc.dispose([expectedLoss, expectedAcc]);
       const numTensors1 = tfc.memory().numTensors;
       expect(numTensors1).toEqual(numTensors0);
     });

  it('2 inputs, 1 output, 1 metric, no validation, no batches specified',
     async () => {
       // Create a functional model with 2 inputs.
       const input1 = tfl.layers.input({shape: [1]});
       const input2 = tfl.layers.input({shape: [1]});
       const concat = tfl.layers.concatenate().apply([input1, input2]);
       const y = tfl.layers.dense({units: 1, kernelInitializer: 'zeros'})
                     .apply(concat) as tfl.SymbolicTensor;
       const model = tfl.model({inputs: [input1, input2], outputs: y});
       model.compile(
           {loss: 'meanSquaredError', optimizer: 'sgd', metrics: ['accuracy']});

       const batchSize = 8;
       const batches = 3;
       const xTensorsFunc = () => {
         const output: {[name: string]: tfc.Tensor[]} = {};
         output[input1.name] = [
           tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]),
           tfc.ones([batchSize, 1])
         ];
         output[input2.name] = [
           tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]),
           tfc.ones([batchSize, 1])
         ];
         return output;
       };
       const yTensorsFunc =
           () => [tfc.ones([batchSize, 1]), tfc.ones([batchSize, 1]), tfc.ones([
             batchSize, 1
           ])];
       const dataset = new FakeNumericDataset({
         xShape: [1],
         yShape: [1],
         batchSize,
         numBatches: batches,
         xTensorsFunc,
         yTensorsFunc
       });

       // Do a burn-in call to account for initialization of cached tensors (for
       // the memory-leak check below).
       await model.evaluateDataset(dataset, {});
       model.setWeights([tfc.zeros([2, 1]), tfc.zeros([1])]);

       const numTensors0 = tfc.memory().numTensors;
       const evalOut = await model.evaluateDataset(dataset, {}) as tfc.Scalar[];
       const expectedLoss = tfc.scalar(1.0);
       const expectedAcc = tfc.scalar(0.0);
       expectTensorsClose(evalOut[0], expectedLoss);
       expectTensorsClose(evalOut[1], expectedAcc);
       tfc.dispose(evalOut);
       tfc.dispose([expectedLoss, expectedAcc]);
       const numTensors1 = tfc.memory().numTensors;
       expect(numTensors1).toEqual(numTensors0);
     });
});
