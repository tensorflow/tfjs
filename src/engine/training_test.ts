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
 * Unit tests for training.ts
 */

import * as tfc from '@tensorflow/tfjs-core';
import {abs, mean, memory, mul, NamedTensorMap, ones, Scalar, scalar, SGDOptimizer, Tensor, tensor1d, tensor2d, tensor3d, test_util, util, zeros} from '@tensorflow/tfjs-core';

import * as K from '../backend/tfjs_backend';
import {CustomCallback, CustomCallbackConfig, ModelTrainingYielder} from '../base_callbacks';
import * as tfl from '../index';
import {Logs, UnresolvedLogs} from '../logs';
import {Regularizer} from '../regularizers';
import {Kwargs} from '../types';
import {pyListRepeat, stringsEqual, unique} from '../utils/generic_utils';
import {describeMathCPU, describeMathCPUAndGPU, describeMathGPU, expectTensorsClose} from '../utils/test_utils';

// TODO(bileschi): Use external version of Layer.
import {Layer, SymbolicTensor} from './topology';
import {checkArrayLengths, isDataArray, isDataDict, isDataTensor, makeBatches, sliceArraysByIndices, standardizeInputData} from './training';


describeMathCPU('isDataTensor', () => {
  it('Positive case', () => {
    expect(isDataTensor(scalar(3.14))).toEqual(true);
  });
  it('Negative cases', () => {
    expect(isDataTensor([scalar(3.14), scalar(-3.14)])).toEqual(false);
    expect(isDataTensor({'Pie': scalar(3.14)})).toEqual(false);
    expect(isDataTensor({})).toEqual(false);
  });
});

describeMathCPU('isDataArray', () => {
  it('Positive case', () => {
    expect(isDataArray([scalar(3.14), scalar(-3.14)])).toEqual(true);
    expect(isDataArray([])).toEqual(true);
  });
  it('Negative cases', () => {
    expect(isDataArray(scalar(3.14))).toEqual(false);
    expect(isDataArray({'Pie': scalar(3.14)})).toEqual(false);
    expect(isDataArray({})).toEqual(false);
  });
});

describeMathCPU('isDataDict', () => {
  it('Positive case', () => {
    expect(isDataDict({'Pie': scalar(3.14)})).toEqual(true);
    expect(isDataDict({})).toEqual(true);
  });
  it('Negative cases', () => {
    expect(isDataDict(scalar(3.14))).toEqual(false);
    expect(isDataDict([scalar(3.14), scalar(-3.14)])).toEqual(false);
    expect(isDataDict([])).toEqual(false);
  });
});

describeMathCPU('standardizeInputData', () => {
  it('Singleton Tensor, Array of one name', () => {
    const outputs = standardizeInputData(scalar(42), ['Foo']);
    expect(outputs.length).toEqual(1);
    expectTensorsClose(outputs[0], scalar(42));
  });
  it('Array of one Tensor, Array of one name', () => {
    const outputs = standardizeInputData([scalar(42)], ['Foo']);
    expect(outputs.length).toEqual(1);
    expectTensorsClose(outputs[0], scalar(42));
  });
  it('Array of two Tensors, Array of two names', () => {
    const outputs =
        standardizeInputData([scalar(42), scalar(21)], ['Foo', 'Bar']);
    expect(outputs.length).toEqual(2);
    expectTensorsClose(outputs[0], scalar(42));
    expectTensorsClose(outputs[1], scalar(21));
  });
  it('Dict of two Tensors, Array of two names', () => {
    const outputs = standardizeInputData(
        {'Foo': scalar(42), 'Bar': scalar(21)}, ['Foo', 'Bar']);
    expect(outputs.length).toEqual(2);
    expectTensorsClose(outputs[0], scalar(42));
    expectTensorsClose(outputs[1], scalar(21));
  });
  it('Unexpected data leads to exception: singleton Tensor', () => {
    expect(() => standardizeInputData(scalar(42), []))
        .toThrowError(/expected no data/);
  });
  it('Unexpected data leads to exception: Array of two Tensors', () => {
    expect(() => standardizeInputData([scalar(42), scalar(21)], []))
        .toThrowError(/expected no data/);
  });
  it('Unexpected data leads to exception: Dict', () => {
    expect(() => standardizeInputData({'Pie': scalar(42)}, []))
        .toThrowError(/expected no data/);
  });
  it('Length mismatch: 1 singleton Scalar vs two names', () => {
    expect(() => standardizeInputData(scalar(42), ['Foo', 'Bar']))
        .toThrowError(/expects 2 Tensor.* but only received one/);
  });
  it('Length mismatch: Array of 2 Scalars vs one name', () => {
    expect(() => standardizeInputData([scalar(42), scalar(-42)], ['Foo']))
        .toThrowError(/Expected to see 1 Tensor/);
  });
  it('Length mismatch: Dict of 1 Scalar vs 2 names', () => {
    expect(() => standardizeInputData({'Foo': scalar(42)}, ['Foo', 'Bar']))
        .toThrowError(/No data provided for \"Bar\"/);
  });
});

describeMathCPU('checkArrayLengths', () => {
  it('Batch mismatch in inputs', () => {
    const inputs = [zeros([2, 1]), zeros([3, 1])];
    const targets = [zeros([2, 1]), zeros([2, 1])];
    expect(() => checkArrayLengths(inputs, targets))
        .toThrowError(/All input .* should have the same number of samples/);
  });
  it('Batch mismatch in targets', () => {
    const inputs = [zeros([2, 1]), zeros([2, 1])];
    const targets = [zeros([2, 1]), zeros([3, 1])];
    expect(() => checkArrayLengths(inputs, targets))
        .toThrowError(/All target .* should have the same number of samples/);
  });
  it('Batch mismatch between inputs and targets', () => {
    const inputs = [zeros([2, 1]), zeros([2, 1])];
    const targets = [zeros([3, 1]), zeros([3, 1])];
    expect(() => checkArrayLengths(inputs, targets))
        .toThrowError(
            /Input Tensors should have the same number of samples as target/);
  });
});

describeMathCPUAndGPU('sliceArraysByIndices', () => {
  it('Single 2D', () => {
    const x = tensor2d([[1, 2], [3, 4], [5, 6]], [3, 2]);
    const y = sliceArraysByIndices(x, tensor1d([0, 2])) as Tensor;
    expectTensorsClose(y, tensor2d([[1, 2], [5, 6]], [2, 2]));
  });
  it('Array of two 2Ds', () => {
    const xs = [
      tensor2d([[1, 2], [3, 4], [5, 6]], [3, 2]),
      tensor2d([[10, 20], [30, 40], [50, 60]], [3, 2])
    ];
    const ys = sliceArraysByIndices(xs, tensor1d([0, 2])) as Tensor[];
    expect(ys.length).toEqual(2);
    expectTensorsClose(ys[0], tensor2d([[1, 2], [5, 6]], [2, 2]));
    expectTensorsClose(ys[1], tensor2d([[10, 20], [50, 60]], [2, 2]));
  });
  it('Array of two 3Ds', () => {
    const xs = [
      tensor3d([[[1]], [[2]], [[3]]], [3, 1, 1]),
      tensor3d([[[10]], [[20]], [[30]]], [3, 1, 1]),
    ];
    const ys = sliceArraysByIndices(xs, tensor1d([0, 2])) as Tensor[];
    expect(ys.length).toEqual(2);
    expectTensorsClose(ys[0], tensor3d([[[1]], [[3]]], [2, 1, 1]));
    expectTensorsClose(ys[1], tensor3d([[[10]], [[30]]], [2, 1, 1]));
  });
  it('null array input', () => {
    expect(sliceArraysByIndices(null, tensor1d([0, 2]))).toBeNull();
  });
  it('casts indices automatically', () => {
    const x = tensor2d([[1, 2], [3, 4], [5, 6]], [3, 2]);
    const y =
        sliceArraysByIndices(x, tensor1d([0.1, 2.0], 'float32')) as Tensor;
    expectTensorsClose(y, tensor2d([[1, 2], [5, 6]], [2, 2]));
  });
});

describe('makeBatches', () => {
  it('divisible', () => {
    expect(makeBatches(6, 3)).toEqual([[0, 3], [3, 6]]);
  });

  it('indivisible', () => {
    expect(makeBatches(7, 3)).toEqual([[0, 3], [3, 6], [6, 7]]);
    expect(makeBatches(2, 4)).toEqual([[0, 2]]);
  });

  it('empty size', () => {
    expect(makeBatches(0, 4)).toEqual([]);
  });
});

describeMathCPUAndGPU('Model.predict', () => {
  it('1 input, 1 output', () => {
    const inputTensor = tfl.layers.input(
        {shape: [3, 4], name: 'inputLayer1', dtype: 'float32'});
    const layer = tfl.layers.reshape({targetShape: [2, 6]});
    const output = layer.apply(inputTensor) as tfl.SymbolicTensor;
    const model = new tfl.Model(
        {inputs: [inputTensor], outputs: [output], name: 'model1x1'});
    const xs = ones([10, 3, 4]);
    const ys = model.predict(xs, {batchSize: 4}) as Tensor;
    expectTensorsClose(ys, ones([10, 2, 6]));
  });

  it('1 input, 1 output, tensor as input argument', () => {
    const inputTensor = tfl.layers.input(
        {shape: [3, 4], name: 'inputLayer1', dtype: 'float32'});
    const layer = tfl.layers.reshape({targetShape: [2, 6]});
    const output = layer.apply(inputTensor) as tfl.SymbolicTensor;
    const model = new tfl.Model(
        {inputs: [inputTensor], outputs: [output], name: 'model1x1'});
    const xs = ones([10, 3, 4]);
    const ys = model.predict(xs) as Tensor;
    expectTensorsClose(ys, ones([10, 2, 6]));
  });

  it('1 input as Array, 1 output', () => {
    const inputTensor = tfl.layers.input(
        {shape: [3, 4], name: 'inputLayer1', dtype: 'float32'});
    const layer = tfl.layers.reshape({targetShape: [2, 6]});
    const output = layer.apply(inputTensor) as tfl.SymbolicTensor;
    const model = new tfl.Model(
        {inputs: [inputTensor], outputs: [output], name: 'model1x1'});
    const xs = ones([10, 3, 4]);
    const ys = model.predict([xs], {batchSize: 4}) as Tensor;
    expectTensorsClose(ys, ones([10, 2, 6]));
  });

  it('1 input, 2 outputs', () => {
    const inputTensor = tfl.layers.input(
        {shape: [3, 4], name: 'inputLayer2', dtype: 'float32'});
    const layer1 = tfl.layers.reshape({targetShape: [2, 6]});
    const layer2 = tfl.layers.flatten();
    const output1 = layer1.apply(inputTensor) as tfl.SymbolicTensor;
    const output2 = layer2.apply(output1) as tfl.SymbolicTensor;
    const model = new tfl.Model(
        {inputs: [inputTensor], outputs: [output1, output2], name: 'model1x2'});
    const xs = ones([10, 3, 4]);
    const ys = model.predict(xs, {batchSize: 4}) as Tensor[];
    expect(ys.length).toEqual(2);
    expectTensorsClose(ys[0], ones([10, 2, 6]));
    expectTensorsClose(ys[1], ones([10, 12]));
  });

  it('2 inputs, 2 outputs', () => {
    const inputTensor1 = tfl.layers.input(
        {shape: [3, 4], name: 'inputLayer3', dtype: 'float32'});
    const inputTensor2 = tfl.layers.input(
        {shape: [3, 3], name: 'inputLayer4', dtype: 'float32'});
    const layer1 = tfl.layers.reshape({targetShape: [2, 6]});
    const layer2 = tfl.layers.flatten();
    const output1 = layer1.apply(inputTensor1) as tfl.SymbolicTensor;
    const output2 = layer2.apply(inputTensor2) as tfl.SymbolicTensor;
    const model = new tfl.Model({
      inputs: [inputTensor1, inputTensor2],
      outputs: [output1, output2],
      name: 'model2x2'
    });
    const xs1 = ones([10, 3, 4]);
    const xs2 = ones([10, 3, 3]);
    const ys = model.predict([xs1, xs2], {batchSize: 4}) as Tensor[];
    expect(ys.length).toEqual(2);
    expectTensorsClose(ys[0], ones([10, 2, 6]));
    expectTensorsClose(ys[1], ones([10, 9]));
  });

  it('Incorrect number of inputs leads to exception: 1 vs 2', () => {
    const inputTensor = tfl.layers.input(
        {shape: [3, 4], name: 'inputLayer_inc_1', dtype: 'float32'});
    const layer = tfl.layers.reshape({targetShape: [2, 6]});
    const output = layer.apply(inputTensor) as tfl.SymbolicTensor;
    const model = new tfl.Model(
        {inputs: [inputTensor], outputs: [output], name: 'model_inc_1x1'});
    const xs1 = ones([10, 3, 4]);

    expect(() => model.predict([
      xs1, xs1
    ])).toThrowError(/.*Expected.*1 Tensor.*got 2 Tensor.*/);
  });

  it('Incorrect number of inputs leads to exception: 2 vs 3', () => {
    const inputTensor1 = tfl.layers.input(
        {shape: [3, 4], name: 'inputLayer_inc_3', dtype: 'float32'});
    const inputTensor2 = tfl.layers.input(
        {shape: [3, 3], name: 'inputLayer_inc_4', dtype: 'float32'});
    const layer1 = tfl.layers.reshape({targetShape: [2, 6]});
    const layer2 = tfl.layers.flatten();
    const output1 = layer1.apply(inputTensor1) as tfl.SymbolicTensor;
    const output2 = layer2.apply(inputTensor2) as tfl.SymbolicTensor;
    const model = new tfl.Model({
      inputs: [inputTensor1, inputTensor2],
      outputs: [output1, output2],
      name: 'model_inc_2x2'
    });
    const xs1 = ones([10, 3, 4]);

    expect(() => model.predict([
      xs1, xs1, xs1
    ])).toThrowError(/.*Expected.*2 Tensor.*got 3 Tensor.*/);
  });

  it('Incorrect input shape leads to exception', () => {
    const inputTensor = tfl.layers.input(
        {shape: [3, 4], name: 'inputLayer_inc_1', dtype: 'float32'});
    const layer = tfl.layers.reshape({targetShape: [2, 6]});
    const output = layer.apply(inputTensor) as tfl.SymbolicTensor;
    const model = new tfl.Model(
        {inputs: [inputTensor], outputs: [output], name: 'model_inc_1x1'});
    const xs1 = ones([2, 4, 3]);

    expect(() => model.predict(xs1))
        .toThrowError(/.*expected.* shape \[null,3,4\].*but got.*\[2,4,3\]/);
  });
});

describeMathCPUAndGPU('Model.fit', () => {
  const inputSize = 4;   // Input vector size for model with one input.
  const inputSize1 = 3;  // 1st input vector size for model with two inputs.
  const inputSize2 = 4;  // 2nd input vector size for model with two inputs.
  const numSamples = 5;  // Number of samples in a batch.

  const inputTensor = tfl.layers.input(
      {shape: [inputSize], name: 'inputLayer1', dtype: 'float32'});
  const inputTensor1 = tfl.layers.input(
      {shape: [inputSize1], name: 'inputLayer1of2', dtype: 'float32'});
  const inputTensor2 = tfl.layers.input(
      {shape: [inputSize2], name: 'inputLayer2of2', dtype: 'float32'});

  // For model with one input.
  let model: tfl.Model;
  let inputs: Tensor;
  let targets: Tensor;

  // For model with two inputs (and two outputs).
  let twoOutputModel: tfl.Model;
  let inputs1: Tensor;
  let inputs2: Tensor;
  let targets1: Tensor;
  let targets2: Tensor;

  function createDenseModelAndData(
      useBias = false,
      kernelRegularizer?: string|Regularizer,
      biasRegularizer?: string|Regularizer,
      ): void {
    const layer = tfl.layers.dense(
        {units: 1, useBias, kernelInitializer: 'ones', kernelRegularizer});
    const output = layer.apply(inputTensor) as tfl.SymbolicTensor;
    model = new tfl.Model({inputs: [inputTensor], outputs: [output]});
    inputs = ones([numSamples, inputSize]);
    targets = ones([numSamples, 1]);
  }

  function createDenseCategoricalModelAndData(useBias = false): void {
    const layer =
        tfl.layers.dense({units: 2, useBias, kernelInitializer: 'ones'});
    const output = layer.apply(inputTensor) as tfl.SymbolicTensor;
    model = new tfl.Model({inputs: [inputTensor], outputs: [output]});
    inputs = ones([numSamples, inputSize]);
    targets = K.oneHot(ones([numSamples]), 2);
  }

  function createTwoLayerDenseModelAndData(useBias = false): [Layer, Layer] {
    const layer1 =
        tfl.layers.dense({units: 10, useBias, kernelInitializer: 'ones'});
    const layer2 =
        tfl.layers.dense({units: 1, useBias, kernelInitializer: 'ones'});
    const output =
        layer2.apply(layer1.apply(inputTensor)) as tfl.SymbolicTensor;
    model = new tfl.Model({inputs: [inputTensor], outputs: [output]});
    inputs = ones([numSamples, inputSize]);
    targets = ones([numSamples, 1]);
    return [layer1, layer2];
  }

  function createDenseModelWithTwoOutputsAndData(): void {
    const layer1 =
        tfl.layers.dense({units: 1, useBias: false, kernelInitializer: 'ones'});
    const layer2 =
        tfl.layers.dense({units: 1, useBias: false, kernelInitializer: 'ones'});
    const output1 = layer1.apply(inputTensor1) as tfl.SymbolicTensor;
    const output2 = layer2.apply(inputTensor2) as tfl.SymbolicTensor;
    twoOutputModel = new tfl.Model(
        {inputs: [inputTensor1, inputTensor2], outputs: [output1, output2]});
    inputs1 = ones([numSamples, inputSize1]);
    inputs2 = ones([numSamples, inputSize2]);
    targets1 = ones([numSamples, 1]);
    targets2 = ones([numSamples, 1]);
  }

  it('1 input, 1 output, dense, 1 weight, string optimizer, 1 batch',
     async done => {
       createDenseModelAndData();

       model.compile({optimizer: 'SGD', loss: 'meanSquaredError'});
       // Use batchSize === numSamples to get exactly one batch.
       model.fit(inputs, targets, {batchSize: numSamples, epochs: 1})
           .then(history => {
             expect(history.epoch).toEqual([0]);
             const newWeightsValue = model.trainableWeights[0].read();

             const lr = 0.01;  // This is the default learning rate of SGD.
             const expectedValueArray =
                 pyListRepeat([1.0 - (inputSize - 1) * 2 * lr], inputSize);
             expectTensorsClose(
                 newWeightsValue, tensor2d(expectedValueArray, [inputSize, 1]));
             done();
           })
           .catch(err => {
             done.fail(err.stack);
           });
     });

  it('training with custom loss', async done => {
    // Use the following Python code snippet to get reference values
    // for assertion:
    //
    // ```python
    // import keras
    // import keras.backend as K;
    // import numpy as np
    //
    // def abs_diff_loss(x, y):
    //   return K.mean(K.abs(x - y))
    //
    // input1 = keras.Input(shape=[4])
    // layer = keras.layers.Dense(
    //     units=1, use_bias=False, kernel_initializer='ones')
    // output = layer(input1)
    // model = keras.Model(input1, output)
    // model.compile(optimizer='SGD', loss=abs_diff_loss)
    // inputs = np.ones([5, 4])
    // targets = np.ones([5])
    // history = model.fit(
    //     inputs, targets, batch_size=5, epochs=2,
    //     validation_split=0.2)
    // print(history.history)
    // ```

    createDenseModelAndData();

    const absDiffLoss = (x: Tensor, y: Tensor) => mean(abs(x.sub(y)));

    model.compile({optimizer: 'SGD', loss: absDiffLoss});
    // Use batchSize === numSamples to get exactly one batch.
    model
        .fit(
            inputs, targets,
            {batchSize: numSamples, epochs: 2, validationSplit: 0.2})
        .then(history => {
          test_util.expectArraysClose(
              history.history['loss'] as number[], [3, 2.96]);
          test_util.expectArraysClose(
              history.history['val_loss'] as number[], [2.96, 2.92]);
          done();
        })
        .catch(err => {
          done.fail(err.stack);
        });
  });

  it('Using only x and y input arguments', async done => {
    createDenseModelAndData();

    model.compile({optimizer: 'SGD', loss: 'meanSquaredError'});
    model.fit(inputs, targets, {epochs: 100})
        .then(history => {
          expect(history.epoch.length).toEqual(100);
          // 100 is the default number of epochs.
          for (let i = 0; i < 100; ++i) {
            expect(history.epoch[i]).toEqual(i);
          }
          done();
        })
        .catch(err => {
          done.fail(err.stack);
        });
  });

  it('Default Model.fit epochs is 1', async done => {
    createDenseModelAndData();

    model.compile({optimizer: 'SGD', loss: 'meanSquaredError'});
    model.fit(inputs, targets)
        .then(history => {
          expect(history.epoch.length).toEqual(1);
          expect(history.epoch[0]).toEqual(0);
          done();
        })
        .catch(err => {
          done.fail(err.stack);
        });
  });

  it('1 input, 1 output, dense, 1 weight, string optimizer, 2 epochs',
     async done => {
       createDenseModelAndData();

       model.compile({optimizer: 'SGD', loss: 'meanSquaredError'});
       model.fit(inputs, targets, {batchSize: numSamples, epochs: 2})
           .then(history => {
             expect(history.epoch).toEqual([0, 1]);
             done();
           });
     });

  it('1 input, 1 output, dense, 1 weight, string optimizer, 2 epochs, ' +
         '1 initialEpoch',
     async done => {
       createDenseModelAndData();

       model.compile({optimizer: 'SGD', loss: 'meanSquaredError'});
       model
           .fit(
               inputs, targets,
               {batchSize: numSamples, epochs: 2, initialEpoch: 1})
           .then(history => {
             expect(history.epoch).toEqual([1]);
             expect(history.history.loss.length).toEqual(1);
             done();
           });
     });

  it('Training with Dropout layer', async done => {
    const inputSize = 2;
    const batchSize = 4;
    const input = tfl.layers.input({shape: [inputSize]});
    const dense1 =
        tfl.layers.dense({units: 2, kernelInitializer: 'ones', useBias: false});
    const dropout = tfl.layers.dropout({rate: 0.5});
    const dense2 =
        tfl.layers.dense({units: 1, kernelInitializer: 'ones', useBias: false});
    const output =
        dense2.apply(dropout.apply(dense1.apply(input))) as tfl.SymbolicTensor;
    const model = new tfl.Model({inputs: input, outputs: output});
    model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});
    const x = ones([batchSize, inputSize]);
    const y = ones([batchSize, 1]);
    model.fit(x, y, {batchSize, epochs: 1}).then(history => {
      done();
    });
  });

  const validationSplits = [0.2, 0.01];
  for (const validationSplit of validationSplits) {
    const testTitle =
        '1 input, 1 output, dense, 1 weight, string optimizer, 2 epochs, ' +
        `validationSplit=${validationSplit}`;
    it(testTitle, async done => {
      createDenseModelAndData();
      model.compile({optimizer: 'SGD', loss: 'meanSquaredError'});
      model
          .fit(
              inputs, targets,
              {batchSize: numSamples, epochs: 2, validationSplit})
          .then(history => {
            expect(history.epoch).toEqual([0, 1]);
            const losses = history.history['loss'];
            expect(losses.length).toEqual(2);
            const valLosses = history.history['val_loss'];
            expect(valLosses.length).toEqual(2);
            // Reference values of the losses can be obtained from PyKeras:
            // ```python
            // import keras
            // import numpy as np
            // input1 = keras.Input(shape=[4])
            // layer = keras.layers.Dense(
            //     units=1, use_bias=False, kernel_initializer='ones')
            // output = layer(input1)
            // model = keras.Model(input1, output)
            // model.compile(optimizer='SGD', loss='mean_squared_error')
            // inputs = np.ones([5, 4])
            // targets = np.ones([5])
            // history = model.fit(
            //     inputs, targets, batch_size=5, epochs=2,
            //     validation_split=0.2)
            // print(history.history)
            // ```
            expectTensorsClose(losses as number[], [9, 7.617599964141846]);
            expectTensorsClose(
                valLosses as number[], [7.617599964141846, 6.447536945343018]);
            done();
          });
    });
  }

  it('1 input, 1 output, dense, 1 weight, string optimizer, 2 epochs, ' +
         'use validationData',
     async done => {
       createDenseModelAndData();
       model.compile({optimizer: 'SGD', loss: 'meanSquaredError'});
       model
           .fit(inputs, targets, {
             batchSize: numSamples,
             epochs: 2,
             validationData:
                 [zeros(inputs.shape as [number, number]), targets]
           })
           .then(history => {
             expect(history.epoch).toEqual([0, 1]);
             const losses = history.history['loss'];
             expect(losses.length).toEqual(2);
             const valLosses = history.history['val_loss'];
             expect(valLosses.length).toEqual(2);
             expectTensorsClose(losses as number[], [9, 7.617599964141846]);
             done();
           });
     });

  it('1 input, 1 output, dense, 1 weight, string optimizer, 2 epochs, ' +
         'validationSplit = 0.2, with additional metric',
     async done => {
       createDenseModelAndData();
       model.compile(
           {optimizer: 'SGD', loss: 'meanSquaredError', metrics: ['accuracy']});
       expect(model.metricsNames).toEqual(['loss', 'acc']);
       model
           .fit(inputs, targets, {
             batchSize: numSamples,
             epochs: 2,
             validationSplit: 0.2,
           })
           .then(history => {
             expect(history.epoch).toEqual([0, 1]);
             const losses = history.history['loss'];
             expect(losses.length).toEqual(2);
             const valLosses = history.history['val_loss'];
             expect(valLosses.length).toEqual(2);
             expectTensorsClose(losses as number[], [9, 7.617599964141846]);
             expectTensorsClose(
                 valLosses as number[], [7.617599964141846, 6.447536945343018]);
             done();
           });
     });

  it('Return sequences; Fit with metric', async done => {
    // The golden values for history used in the assertion below can be obtained
    // with the following Python Keras code.
    // Ran with Python Keras verion 2.1.2 and TensorFlow (CPU) version
    // 1.7.0-dev20180226.
    // ```python
    // import keras
    // import numpy as np
    //
    // sequenceLength = 3
    // inputSize = 4
    // dataSize = 16
    // validationSplit = 0.5
    // batchSize = 3
    // outputSize = 2
    //
    // model = keras.Sequential()
    //
    // model.add(keras.layers.SimpleRNN(
    //     outputSize,
    //     kernel_initializer='ones',
    //     recurrent_initializer='ones',
    //     use_bias=False,
    //     return_sequences=True,
    //     input_shape=[sequenceLength, inputSize]))
    // model.add(keras.layers.TimeDistributed(
    //     keras.layers.Dense(
    //         outputSize,
    //         kernel_initializer='ones',
    //         use_bias=False)))
    //
    // model.compile(optimizer='sgd',
    //               loss='categorical_crossentropy',
    //               metrics=['accuracy'])
    // history = model.fit(np.ones([dataSize, sequenceLength, inputSize]),
    //                     np.ones([dataSize, sequenceLength, outputSize]),
    //                     batch_size=batchSize,
    //                     epochs=2,
    //                     validation_split=validationSplit)
    // print(history.history)
    // ```

    const sequenceLength = 3;
    const inputSize = 4;
    const dataSize = 16;
    const validationSplit = 0.5;
    const batchSize = 3;
    // So there are 8 examples for train and validation, respectivly. The actual
    // batches during training and validation will be 3, 3 and 2. This tests the
    // correct averaging of the loss values happens without broadcasting.
    const outputSize = 2;
    const simpleRNN = tfl.layers.simpleRNN({
      units: outputSize,
      kernelInitializer: 'ones',
      recurrentInitializer: 'ones',
      useBias: false,
      returnSequences: true,
    });
    const timeDistributed = tfl.layers.timeDistributed({
      layer: tfl.layers.dense(
          {units: outputSize, kernelInitializer: 'ones', useBias: false})
    });
    const input = tfl.layers.input({shape: [sequenceLength, inputSize]});
    const output =
        timeDistributed.apply(simpleRNN.apply(input)) as tfl.SymbolicTensor;
    const model = new tfl.Model({inputs: input, outputs: output});
    model.compile({
      optimizer: 'sgd',
      loss: 'categoricalCrossentropy',
      metrics: ['accuracy'],
    });
    const history = await model.fit(
        ones([dataSize, sequenceLength, inputSize]),
        ones([dataSize, sequenceLength, outputSize]), {
          batchSize,
          epochs: 1,
          validationSplit,
        });
    expectTensorsClose(
        history.history['loss'] as number[], [1.3862943649291992]);
    expectTensorsClose(
        history.history['val_loss'] as number[], [1.3862943649291992]);
    expectTensorsClose(history.history['acc'] as number[], [1.0]);
    expectTensorsClose(history.history['val_acc'] as number[], [1.0]);
    done();
  });

  // TODO(cais): Test metric as a "dict", for models with >1 outputs.

  const metricsToTest: string[][] = [['acc'], ['accuracy']];
  // TODO(cais): Add 'acc', 'accuracy' and assertion acc_1, acc_2.
  for (const metrics of metricsToTest) {
    const testTitle = `categoricalCrossentropy model, validationSplit = 0.2, ` +
        `${JSON.stringify(metrics)}`;
    it(testTitle, async done => {
      createDenseCategoricalModelAndData();
      model.compile(
          {optimizer: 'SGD', loss: 'categoricalCrossentropy', metrics});
      if (stringsEqual(metrics, ['acc']) ||
          stringsEqual(metrics, ['accuracy'])) {
        expect(model.metricsNames).toEqual(['loss', 'acc']);
      } else if (stringsEqual(metrics, ['acc', 'accuracy'])) {
        expect(model.metricsNames).toEqual(['loss', 'acc', 'acc']);
      }
      model
          .fit(
              inputs, targets,
              {batchSize: numSamples, epochs: 2, validationSplit: 0.2})
          .then(history => {
            const losses = history.history['loss'];
            expectTensorsClose(
                losses as number[], [0.6931471824645996, 0.6918979287147522]);
            const valLosses = history.history['val_loss'];
            expectTensorsClose(
                valLosses as number[],
                [0.6918979287147522, 0.6906517744064331]);
            const acc = history.history['acc'];
            expectTensorsClose(acc as number[], [0, 1]);
            const valAcc = history.history['val_acc'];
            expectTensorsClose(valAcc as number[], [1, 1]);
            done();
          });
    });
  }

  it('Two layers, freeze one layer', async done => {
    // The golden values used below can be obtained with the following PyKeras
    // code.
    // ```python
    // import keras
    // import numpy as np
    //
    // input_size = 4
    // num_samples = 5
    //
    // input_tensor = keras.Input([input_size], name='inputLayer1')
    // layer1 = keras.layers.Dense(
    //     units=10, use_bias=False, kernel_initializer='ones')
    // layer2 = keras.layers.Dense(
    //     units=1, use_bias=False, kernel_initializer='ones')
    // output = layer2(layer1(input_tensor))
    // model = keras.Model(input_tensor, output)
    //
    // inputs = np.ones([num_samples, input_size])
    // targets = np.ones([num_samples, 1])
    //
    // optimizer = keras.optimizers.SGD(lr=1e-2)
    // model.compile(optimizer=optimizer, loss='mean_squared_error')
    // history = model.fit(inputs,
    //                     targets,
    //                     batch_size=num_samples,
    //                     epochs=2,
    //                     validation_split=0.2)
    //
    // print(history.history)
    // print(layer1.get_weights())
    // print(layer2.get_weights())
    //
    // # Freeze layer 1.
    // layer1.trainable = False
    // model.compile(optimizer=optimizer, loss='mean_squared_error')
    // history = model.fit(inputs,
    //                     targets,
    //                     batch_size=num_samples,
    //                     epochs=2,
    //                     validation_split=0.2)
    //
    // print(history.history)
    // print(layer1.get_weights())
    // print(layer2.get_weights())
    // ```
    const layers = createTwoLayerDenseModelAndData();
    const layer1 = layers[0];
    const layer2 = layers[1];
    const optimizer = new SGDOptimizer(1e-2);
    model.compile({optimizer, loss: 'meanSquaredError'});
    let history = await model.fit(
        inputs, targets,
        {batchSize: numSamples, epochs: 2, validationSplit: 0.2});
    let losses = history.history['loss'];
    expectTensorsClose(losses as number[], [1521.0, 386.35842895507812]);
    let valLosses = history.history['val_loss'];
    expectTensorsClose(
        valLosses as number[], [386.35848999023438, 1808.7342529296875]);
    expectTensorsClose(
        layer1.getWeights()[0], mul(scalar(-0.61341441), ones([4, 10])));
    expectTensorsClose(
        layer2.getWeights()[0], mul(scalar(-1.77405429), ones([10, 1])));

    // Freeze the 1st layer and compile the model again.
    layer1.trainable = false;
    model.compile({optimizer, loss: 'meanSquaredError'});

    history = await model.fit(
        inputs, targets,
        {batchSize: numSamples, epochs: 2, validationSplit: 0.2});
    losses = history.history['loss'];
    expectTensorsClose(
        losses as number[], [1808.7342529296875, 75.336509704589844]);
    valLosses = history.history['val_loss'];
    expectTensorsClose(
        valLosses as number[], [75.336524963378906, 3.1378798484802246]);
    // Expect no change in the value of layer1's kernel, due to the freezing.
    expectTensorsClose(
        layer1.getWeights()[0], mul(scalar(-0.61341441), ones([4, 10])));
    // Expect change in the value of layer2's kernel.
    expectTensorsClose(
        layer2.getWeights()[0], mul(scalar(-0.11295), ones([10, 1])));
    done();
  });

  it('Unknown metric', () => {
    createDenseCategoricalModelAndData();
    expect(() => model.compile({
      optimizer: 'SGD',
      loss: 'categoricalCrossentropy',
      metrics: ['foo']
    })).toThrowError(/Unknown metric foo/);
  });

  it('1 input, 1 output, dense, 2 weights, string optimizer, 1 batch',
     async done => {
       createDenseModelAndData(true);

       model.compile({optimizer: 'SGD', loss: 'meanSquaredError'});
       model.fit(inputs, targets, {batchSize: numSamples, epochs: 1})
           .then(history => {
             expect(history.epoch).toEqual([0]);

             expect(model.trainableWeights.length).toEqual(2);
             const lr = 0.01;  // This is the default learning rate of SGD.
             const newKernelValue = model.trainableWeights[0].read();
             const expectedKernelArray =
                 pyListRepeat([1.0 - (inputSize - 1) * 2 * lr], inputSize);
             expectTensorsClose(
                 newKernelValue, tensor2d(expectedKernelArray, [inputSize, 1]));
             const newBiasValue = model.trainableWeights[1].read();
             const expectedBiasArray = [0.0 - (inputSize - 1) * 2 * lr];
             expectTensorsClose(newBiasValue, tensor1d(expectedBiasArray));
             done();
           });
     });

  it('1 input, 1 output, dense, 1 weight, optimizer object, 1 batch',
     async done => {
       createDenseModelAndData();

       // Use a custom learning rate for SGD.
       const lr = 0.025;
       model.compile(
           {optimizer: new SGDOptimizer(lr), loss: 'meanSquaredError'});
       model.fit(inputs, targets, {batchSize: numSamples, epochs: 1})
           .then(history => {
             expect(history.epoch).toEqual([0]);
             const newWeightsValue = model.trainableWeights[0].read();

             const expectedValueArray =
                 pyListRepeat([1.0 - (inputSize - 1) * 2 * lr], inputSize);
             expectTensorsClose(
                 newWeightsValue, tensor2d(expectedValueArray, [inputSize, 1]));
             done();
           });
     });

  it('2 inputs, 2 outputs, dense, optimizer object, 1 batch', async done => {
    createDenseModelWithTwoOutputsAndData();

    const lr = 0.01;
    twoOutputModel.compile({
      optimizer: new SGDOptimizer(lr),
      loss: ['meanSquaredError', 'meanSquaredError']
    });
    const trainableWeights = twoOutputModel.trainableWeights;
    let newWeightsValue1 = trainableWeights[0].read();
    let newWeightsValue2 = trainableWeights[1].read();
    twoOutputModel
        .fit(
            [inputs1, inputs2], [targets1, targets2],
            {batchSize: numSamples, epochs: 1})
        .then(history => {
          expect(history.epoch).toEqual([0]);

          expect(twoOutputModel.trainableWeights.length).toEqual(2);
          newWeightsValue1 = twoOutputModel.trainableWeights[0].read();
          newWeightsValue2 = twoOutputModel.trainableWeights[1].read();

          // Check the weight updates to layer1.
          const expectedValueArray1 =
              pyListRepeat([1.0 - (inputSize1 - 1) * 2 * lr], inputSize1);
          expectTensorsClose(
              newWeightsValue1, tensor2d(expectedValueArray1, [inputSize1, 1]));
          // Check the weight updates to layer2 (different from those to
          // layer1).
          const expectedValueArray2 =
              pyListRepeat([1.0 - (inputSize2 - 1) * 2 * lr], inputSize2);
          expectTensorsClose(
              newWeightsValue2, tensor2d(expectedValueArray2, [inputSize2, 1]));
          done();
        });
  });

  const isCustomCallbackConfig = [false, true];
  const isCustomCallbackArray = [false, true];
  for (const isConfig of isCustomCallbackConfig) {
    for (const isArray of isCustomCallbackArray) {
      const testTitle = `Fit with custom callback object: isConfig=${
          isConfig}, isArray=${isArray}`;
      it(testTitle, async done => {
        createDenseModelAndData();
        const trainBeginLogs: Logs[] = [];
        const trainEndLogs: Logs[] = [];
        const epochBeginEpochs: number[] = [];
        const epochEndEpochs: number[] = [];
        const batchBeginBatches: number[] = [];
        const batchEndBatches: number[] = [];
        const batchEndLosses: number[] = [];
        const epochEndLosses: number[] = [];
        const customCallbackConfig: CustomCallbackConfig = {
          onTrainBegin: async (logs?: Logs) => {
            trainBeginLogs.push(logs);
          },
          onTrainEnd: async (logs?: Logs) => {
            trainEndLogs.push(logs);
          },
          onEpochBegin: async (epoch: number, logs?: Logs) => {
            epochBeginEpochs.push(epoch);
          },
          onEpochEnd: async (epoch: number, logs?: Logs) => {
            epochEndEpochs.push(epoch);
            epochEndLosses.push(logs['loss']);
          },
          onBatchBegin: async (batch: number, logs?: Logs) => {
            batchBeginBatches.push(batch);
          },
          onBatchEnd: async (batch: number, logs?: Logs) => {
            batchEndBatches.push(batch);
            batchEndLosses.push(logs['loss']);
          }
        };
        const customCallback = isConfig ?
            customCallbackConfig :
            new CustomCallback(customCallbackConfig);
        model.compile({optimizer: 'sgd', loss: 'meanSquaredError'});
        await model.fit(inputs, targets, {
          batchSize: 2,
          epochs: 2,
          callbacks: isArray ? [customCallback] : customCallback,
        });
        expect(trainBeginLogs.length).toEqual(1);
        expect(trainEndLogs.length).toEqual(1);
        expect(epochBeginEpochs).toEqual([0, 1]);
        expect(epochEndEpochs).toEqual([0, 1]);
        expect(batchBeginBatches).toEqual([0, 1, 2, 0, 1, 2]);
        expect(batchEndBatches).toEqual([0, 1, 2, 0, 1, 2]);

        // The optimization problem is a convex one (a single Dense layer),
        // the learning rate low (default 0.01 for SGD). So it should be fine to
        // assert monotonic assert monotonic decrease in loss value.
        expect(batchEndLosses.length).toEqual(6);
        for (let i = 1; i < batchEndLosses.length; ++i) {
          expect(batchEndLosses[i]).toBeLessThan(batchEndLosses[i - 1]);
        }
        expect(epochEndLosses.length).toEqual(2);
        expect(epochEndLosses[1]).toBeLessThan(epochEndLosses[0]);
        done();
      });
    }
  }

  it('Using custom regularizer', async done => {
    // The golden values used for assertion can be obtained with PyKeras code:
    //
    // ```python
    // import keras
    // import numpy as np
    //
    // model = keras.Sequential([
    //     keras.layers.Dense(
    //         1, kernel_initializer='ones', use_bias=False, input_shape=[4],
    //         kernel_regularizer=keras.regularizers.l1_l2(1, 1))
    // ]);
    //
    // xs = np.ones([5, 4])
    // ys = np.ones([5, 1])
    //
    // model.compile(optimizer='sgd', loss='mean_squared_error')
    //
    // history = model.fit(xs, ys, epochs=2)
    // print(model.get_weights()[0])
    // print(history.history)
    //
    // ```
    createDenseModelAndData(false, tfl.regularizers.l1l2({l1: 1, l2: 1}));

    model.compile({optimizer: 'SGD', loss: 'meanSquaredError'});
    // Use batchSize === numSamples to get exactly one batch.
    model.fit(inputs, targets, {batchSize: numSamples, epochs: 2})
        .then(history => {
          expectTensorsClose(
              model.layers[1].getWeights()[0],
              tensor2d([0.829, 0.829, 0.829, 0.829], [4, 1]));
          expect(history.history.loss.length).toEqual(2);
          expect(history.history.loss[0]).toBeCloseTo(17);
          expect(history.history.loss[1]).toBeCloseTo(13.92);
          done();
        })
        .catch(err => {
          done.fail(err.stack);
        });
  });

  it('Using string regularizer', async done => {
    // The golden values used for assertion can be obtained with PyKeras code:
    //
    // ```python
    // import keras
    // import numpy as np
    //
    // model = keras.Sequential([
    //     keras.layers.Dense(
    //         1, kernel_initializer='ones', use_bias=False, input_shape=[4],
    //         kernel_regularizer='l1l2')
    // ]);
    //
    // xs = np.ones([5, 4])
    // ys = np.ones([5, 1])
    //
    // model.compile(optimizer='sgd', loss='mean_squared_error')
    //
    // history = model.fit(xs, ys, epochs=2)
    // print(model.get_weights()[0])
    // print(history.history)
    //
    // ```
    createDenseModelAndData(false, 'l1l2');

    model.compile({optimizer: 'SGD', loss: 'meanSquaredError'});
    // Use batchSize === numSamples to get exactly one batch.
    model.fit(inputs, targets, {batchSize: numSamples, epochs: 2})
        .then(history => {
          expectTensorsClose(
              model.layers[1].getWeights()[0],
              tensor2d([0.884, 0.884, 0.884, 0.884], [4, 1]));
          expect(history.history.loss.length).toEqual(2);
          expect(history.history.loss[0]).toBeCloseTo(9.08);
          expect(history.history.loss[1]).toBeCloseTo(7.68);
          done();
        })
        .catch(err => {
          done.fail(err.stack);
        });
  });

  class StopAfterNEpochs extends tfl.Callback {
    private readonly epochsToTrain: number;
    constructor(epochsToTrain: number) {
      super();
      this.epochsToTrain = epochsToTrain;
    }

    async onEpochEnd(epoch: number, logs?: UnresolvedLogs) {
      if (epoch === this.epochsToTrain - 1) {
        this.model.stopTraining = true;
      }
    }
  }

  it('Stop training at the end of an epoch: Functional model', done => {
    createDenseModelAndData(true);
    model.compile({optimizer: 'SGD', loss: 'meanSquaredError'});
    // Order 10 epochs of training, but the training should stop after two
    // epochs due to the callback.
    model
        .fit(inputs, targets, {
          batchSize: numSamples,
          epochs: 10,
          callbacks: [new StopAfterNEpochs(2)]
        })
        .then(history => {
          expect(history.history.loss.length).toEqual(2);
          done();
        })
        .catch(err => done.fail(err.stack));
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

  it('Stop training at the end of a batch: Sequential model', done => {
    const sequentialModel = tfl.sequential();
    sequentialModel.add(tfl.layers.dense(
        {units: 1, kernelInitializer: 'ones', inputShape: [inputSize]}));
    // numSamples is 5.
    inputs = ones([numSamples, inputSize]);
    targets = ones([numSamples, 1]);
    sequentialModel.compile({optimizer: 'SGD', loss: 'meanSquaredError'});
    // Order 10 epochs of training, but the training should stop after only one
    // epochs due to the callback that orders the training to stop after two
    // batches. The first epoch should have five batches  due to a batchSize
    // of 1.
    sequentialModel
        .fit(
            inputs, targets,
            {batchSize: 1, epochs: 10, callbacks: [new StopAfterNBatches(2)]})
        .then(history => {
          expect(history.history.loss.length).toEqual(1);
          done();
        })
        .catch(err => done.fail(err.stack));
  });

  it('Invalid dict loss: nonexistent output name', () => {
    createDenseModelAndData();
    expect(() => model.compile({
      optimizer: 'SGD',
      loss: {'Foo': 'meanSquaredError'}
    })).toThrowError(/Unknown entry in loss dictionary:.*Foo.*/);
  });

  it('Invalid Array loss: missing loss for an output', () => {
    createDenseModelWithTwoOutputsAndData();
    expect(() => twoOutputModel.compile({
      optimizer: 'SGD',
      loss: ['meanSquaredError']
    })).toThrowError(/should have one entry per model output.*has 2 output/);
  });

  it('Calling fit without compile leads to error', () => {
    createDenseModelAndData(true);
    const fitPromise =
        model.fit(inputs, targets, {batchSize: numSamples, epochs: 1});
    fitPromise.catch(error => {
      expect(error.message).toContain('You must compile a model before');
    });
  });
});

describeMathCPUAndGPU('Model.fit with training-sensitive layers', () => {
  it('Correct training arg during fit/evaluate/predict', async done => {
    const inputTensor =
        tfl.layers.input({shape: [1], name: 'inputLayer1', dtype: 'float32'});
    const layer1 = tfl.layers.dense({units: 1});
    const layer2 = tfl.layers.dropout({rate: 0.5});

    // Hook the dropout layer to observe the training arg values during the
    // fit(), evaluate() and predict() calls.
    const dropoutLayerTrainingFlags: boolean[] = [];
    const recordDropoutTrainingArgHook =
        (inputs: Tensor|Tensor[], kwargs: Kwargs) => {
          dropoutLayerTrainingFlags.push(kwargs.training as boolean);
        };
    layer2.setCallHook(recordDropoutTrainingArgHook);

    const output =
        layer2.apply(layer1.apply(inputTensor)) as tfl.SymbolicTensor;
    const model = new tfl.Model({inputs: [inputTensor], outputs: [output]});
    model.compile({optimizer: 'sgd', loss: 'meanSquaredError'});
    const xs = ones([4, 1]);
    const ys = ones([4, 1]);

    // 1. Call fit: Dropout layer should be called twice, with training as
    // true.
    try {
      await model.fit(xs, ys, {epochs: 2, batchSize: 4});
    } catch (err) {
      done.fail(err.stack);
    }
    expect(dropoutLayerTrainingFlags).toEqual([true, true]);

    // 2. Call evaluate, Dropout layer should be called once, without
    // training defined.
    model.evaluate(xs, ys, {batchSize: 4});
    expect(dropoutLayerTrainingFlags).toEqual([true, true, undefined]);

    // 3. Call predict, Dropout layer should be called once, without training
    //   defined.
    model.predict(xs, {batchSize: 4});
    expect(dropoutLayerTrainingFlags).toEqual([
      true, true, undefined, undefined
    ]);

    done();
  });
});

describeMathCPUAndGPU(
    'Model.predict and Model.evaluate: No memory leak', () => {
      const inputSize = 4;  // Input vector size for model with one input.

      const inputTensor = tfl.layers.input(
          {shape: [inputSize], name: 'inputLayer1', dtype: 'float32'});
      let model: tfl.Model;
      let inputs: Tensor;
      let targets: Tensor;

      function createDenseModelAndData(
          numSamples: number,
          kernelRegularizer?: string|Regularizer,
          biasRegularizer?: string|Regularizer,
          ): void {
        const layer = tfl.layers.dense(
            {units: 1, kernelInitializer: 'ones', kernelRegularizer});
        const output = layer.apply(inputTensor) as tfl.SymbolicTensor;
        model = new tfl.Model({inputs: [inputTensor], outputs: [output]});
        inputs = ones([numSamples, inputSize]);
        targets = ones([numSamples, 1]);
      }

      it('predict: Single batch', () => {
        const numExamples = 5;
        const batchSize = 32;  // batchSize >= numExamples ==> a single batch.
        createDenseModelAndData(numExamples);
        // Burn-in call.
        let out = model.predict(inputs, {batchSize}) as Tensor;
        out.dispose();
        const numTensors0 = memory().numTensors;

        // Actual call.
        out = model.predict(inputs, {batchSize}) as Tensor;
        out.dispose();
        const numTensors1 = memory().numTensors;
        expect(numTensors1).toEqual(numTensors0);
      });

      it('predict: Two batches', () => {
        const numExamples = 5;
        const batchSize = 3;  // batchSize < numExamples ==> multiple batches.
        createDenseModelAndData(numExamples);
        // Burn-in call.
        let out = model.predict(inputs, {batchSize}) as Tensor;
        out.dispose();
        const numTensors0 = memory().numTensors;

        // Actual call.
        out = model.predict(inputs, {batchSize}) as Tensor;
        out.dispose();
        const numTensors1 = memory().numTensors;
        expect(numTensors1).toEqual(numTensors0);
      });

      it('evaluate: Single batch, no metric', () => {
        const numExamples = 5;
        createDenseModelAndData(numExamples);
        model.compile({optimizer: 'SGD', loss: 'meanSquaredError'});
        const batchSize = 32;  // batchSize >= numExamples ==> a single batch.
        // Burn-in call.
        let out = model.evaluate(inputs, targets, {batchSize}) as Tensor;
        out.dispose();
        const numTensors0 = memory().numTensors;

        // Actual call.
        out = model.evaluate(inputs, targets, {batchSize}) as Tensor;
        out.dispose();
        const numTensors1 = memory().numTensors;
        expect(numTensors1).toEqual(numTensors0);
      });

      it('evaluate: Two batches, no metric', () => {
        const numExamples = 5;
        createDenseModelAndData(numExamples);
        model.compile({optimizer: 'SGD', loss: 'meanSquaredError'});
        const batchSize = 3;  // batchSize < numExamples ==> multiple batches.
        // Burn-in call.
        let out = model.evaluate(inputs, targets, {batchSize}) as Tensor;
        out.dispose();
        const numTensors0 = memory().numTensors;

        // Actual call.
        out = model.evaluate(inputs, targets, {batchSize}) as Tensor;
        out.dispose();
        const numTensors1 = memory().numTensors;
        expect(numTensors1).toEqual(numTensors0);
      });

      it('evaluate: Two batches, with metric', () => {
        const numExamples = 5;
        createDenseModelAndData(numExamples);
        model.compile(
            {optimizer: 'SGD', loss: 'meanSquaredError', metrics: ['mae']});
        const batchSize = 3;  // batchSize < numExamples ==> multiple batches.
        // Burn-in call.
        let out = model.evaluate(inputs, targets, {batchSize}) as Tensor[];
        out.forEach(tensor => tensor.dispose());
        const numTensors0 = memory().numTensors;

        // Actual call.
        out = model.evaluate(inputs, targets, {batchSize}) as Tensor[];
        out.forEach(tensor => tensor.dispose());
        const numTensors1 = memory().numTensors;
        expect(numTensors1).toEqual(numTensors0);
      });
    });

describeMathCPUAndGPU('Model.fit: No memory leak', () => {
  const inputSize = 4;   // Input vector size for model with one input.
  const numSamples = 5;  // Number of samples in a batch.

  const inputTensor = tfl.layers.input(
      {shape: [inputSize], name: 'inputLayer1', dtype: 'float32'});
  let model: tfl.Model;
  let inputs: Tensor;
  let targets: Tensor;
  let valInputs: Tensor;
  let valTargets: Tensor;

  function createDenseModelAndData(
      useBias = false,
      kernelRegularizer?: string|Regularizer,
      biasRegularizer?: string|Regularizer,
      ): void {
    const layer = tfl.layers.dense(
        {units: 1, useBias, kernelInitializer: 'ones', kernelRegularizer});
    const output = layer.apply(inputTensor) as tfl.SymbolicTensor;
    model = new tfl.Model({inputs: [inputTensor], outputs: [output]});
    inputs = ones([numSamples, inputSize]);
    targets = ones([numSamples, 1]);
    valInputs = zeros([numSamples, inputSize]);
    valTargets = zeros([numSamples, 1]);
  }

  it('Repeated fit calls leads to no memory leak: no validation or metrics',
     async done => {
       createDenseModelAndData();

       model.compile({optimizer: 'SGD', loss: 'meanSquaredError'});
       // Use batchSize === numSamples to get exactly one batch.
       await model.fit(inputs, targets, {batchSize: numSamples, epochs: 1});
       const numTensors0 = memory().numTensors;
       for (let i = 0; i < 2; ++i) {
         await model.fit(inputs, targets, {batchSize: numSamples, epochs: 1});
         const numTensorsNow = memory().numTensors;
         if (numTensorsNow > numTensors0) {
           done.fail(
               `Memory leak detected during fit(): Leaked ` +
               `${numTensorsNow - numTensors0} tensor(s) after the ` +
               `${i + 1}-th fit() call.`);
         }
       }
       done();
     });

  it('Repeated fit calls leads to no memory leak: batchSize=1, ' +
         'no validation or metrics',
     async done => {
       createDenseModelAndData();

       model.compile({optimizer: 'SGD', loss: 'meanSquaredError'});
       const batchSize = 1;  // Use batchSize = 1.
       await model.fit(inputs, targets, {batchSize, epochs: 1});
       const numTensors0 = memory().numTensors;
       for (let i = 0; i < 2; ++i) {
         await model.fit(inputs, targets, {batchSize, epochs: 1});
         const numTensorsNow = memory().numTensors;
         if (numTensorsNow > numTensors0) {
           done.fail(
               `Memory leak detected during fit(): Leaked ` +
               `${numTensorsNow - numTensors0} tensor(s) after the ` +
               `${i + 1}-th fit() call.`);
         }
       }
       done();
     });

  it('Repeated fit calls leads to no memory leak: with metrics', async done => {
    createDenseModelAndData();

    model.compile(
        {optimizer: 'SGD', loss: 'meanSquaredError', metrics: ['mse']});
    // Use batchSize === numSamples to get exactly one batch.
    await model.fit(inputs, targets, {batchSize: numSamples, epochs: 1});
    const numTensors0 = memory().numTensors;
    for (let i = 0; i < 2; ++i) {
      await model.fit(inputs, targets, {batchSize: numSamples, epochs: 1});
      const numTensorsNow = memory().numTensors;
      if (numTensorsNow > numTensors0) {
        done.fail(
            `Memory leak detected during fit(): Leaked ` +
            `${numTensorsNow - numTensors0} tensor(s) after the ` +
            `${i + 1}-th fit() call.`);
      }
    }
    done();
  });

  it('Repeated fit calls leads to no memory leak: validationSplit',
     async done => {
       createDenseModelAndData();

       const validationSplit = 0.4;
       model.compile({optimizer: 'SGD', loss: 'meanSquaredError'});
       // Use batchSize === numSamples to get exactly one batch.
       await model.fit(
           inputs, targets,
           {batchSize: numSamples, epochs: 1, validationSplit});
       const numTensors0 = memory().numTensors;
       for (let i = 0; i < 2; ++i) {
         await model.fit(
             inputs, targets,
             {batchSize: numSamples, epochs: 1, validationSplit});
         const numTensorsNow = memory().numTensors;
         if (numTensorsNow > numTensors0) {
           done.fail(
               `Memory leak detected during fit(): Leaked ` +
               `${numTensorsNow - numTensors0} tensor(s) after the ` +
               `${i + 1}-th fit() call.`);
         }
       }
       done();
     });

  it('Repeated fit calls leads to no memory leak: validationData',
     async done => {
       createDenseModelAndData();

       const validationData: [Tensor, Tensor] = [valInputs, valTargets];
       model.compile({optimizer: 'SGD', loss: 'meanSquaredError'});
       // Use batchSize === numSamples to get exactly one batch.
       await model.fit(
           inputs, targets, {batchSize: numSamples, epochs: 1, validationData});
       const numTensors0 = memory().numTensors;
       for (let i = 0; i < 2; ++i) {
         await model.fit(
             inputs, targets,
             {batchSize: numSamples, epochs: 1, validationData});
         const numTensorsNow = memory().numTensors;
         if (numTensorsNow > numTensors0) {
           done.fail(
               `Memory leak detected during fit(): Leaked ` +
               `${numTensorsNow - numTensors0} tensor(s) after the ` +
               `${i + 1}-th fit() call.`);
         }
       }
       done();
     });

  it('Repeated fit calls leads to no memory leak: metrics & validationSplit',
     async done => {
       createDenseModelAndData();

       const validationSplit = 0.4;
       model.compile(
           {optimizer: 'SGD', loss: 'meanSquaredError', metrics: ['mse']});
       // Use batchSize === numSamples to get exactly one batch.
       await model.fit(
           inputs, targets,
           {batchSize: numSamples, epochs: 1, validationSplit});
       const numTensors0 = memory().numTensors;
       for (let i = 0; i < 2; ++i) {
         await model.fit(
             inputs, targets,
             {batchSize: numSamples, epochs: 1, validationSplit});
         const numTensorsNow = memory().numTensors;
         if (numTensorsNow > numTensors0) {
           done.fail(
               `Memory leak detected during fit(): Leaked ` +
               `${numTensorsNow - numTensors0} tensor(s) after the ` +
               `${i + 1}-th fit() call.`);
         }
       }
       done();
     });

  it('Repeated fit calls leads to no memory leak: batchSize=2, ' +
         'metrics & validationSplit',
     async done => {
       createDenseModelAndData();

       const validationSplit = 0.4;
       model.compile(
           {optimizer: 'SGD', loss: 'meanSquaredError', metrics: ['mse']});
       const batchSize = 2;  // Use batchSize < numSamples.
       const epochsPerIter = 2;
       await model.fit(
           inputs, targets, {batchSize, epochs: 1, validationSplit});
       const numTensors0 = memory().numTensors;
       for (let i = 0; i < 2; ++i) {
         const history = await model.fit(
             inputs, targets,
             {batchSize, epochs: epochsPerIter, validationSplit});
         expect(history.history['loss'].length).toEqual(epochsPerIter);
         expect(history.history['val_loss'].length).toEqual(epochsPerIter);
         expect(history.history['mse'].length).toEqual(epochsPerIter);
         expect(history.history['val_mse'].length).toEqual(epochsPerIter);
         const numTensorsNow = memory().numTensors;
         if (numTensorsNow > numTensors0) {
           done.fail(
               `Memory leak detected during fit(): Leaked ` +
               `${numTensorsNow - numTensors0} tensor(s) after the ` +
               `${i + 1}-th fit() call.`);
         }
       }
       done();
     });

  it('Fit with onEpochEnd callback: no memory leak: validation & metrics',
     async done => {
       createDenseModelAndData();

       model.compile(
           {optimizer: 'SGD', loss: 'meanSquaredError', metrics: ['mse']});
       const validationSplit = 0.4;

       // First, perform a burn-in call.
       await model.fit(
           inputs, targets,
           {batchSize: numSamples, epochs: 1, validationSplit});
       const numTensors0 = memory().numTensors;

       // Perform actual testing calls.
       const numFitCalls = 2;
       for (let n = 0; n < numFitCalls; ++n) {
         const tensorCounts: number[] = [];
         await model.fit(inputs, targets, {
           batchSize: numSamples,
           epochs: 4,
           validationSplit,
           callbacks: {
             onEpochEnd: async () => {
               tensorCounts.push(memory().numTensors);
             }
           }
         });
         expect(tensorCounts.length).toEqual(4);
         if (unique(tensorCounts).length !== 1) {
           done.fail(
               `Detected WebGL memory leak during fit() call with ` +
               `onEpochEnd callback: tensor counts: ${tensorCounts}.`);
         }
         const numTensors1 = memory().numTensors;
         if (numTensors1 > numTensors0) {
           done.fail(
               `Detected memory leak of ${numTensors1 - numTensors0} ` +
               `tensor(s) after fit() call ${n + 1} of ${numFitCalls} ` +
               `with onEpochEnd callback.`);
         }
       }
       done();
     });

  it('Fit with onBatchEnd callback: no memory leak: validation & metrics',
     async done => {
       createDenseModelAndData();

       model.compile(
           {optimizer: 'SGD', loss: 'meanSquaredError', metrics: ['mse']});
       const validationSplit = 0.4;
       const epochs = 3;
       const batchesPerEpoch = numSamples * (1 - validationSplit);

       // First, perform a burn-in call.
       await model.fit(inputs, targets, {
         batchSize: 1,
         epochs,
         validationSplit,
       });
       const numTensors0 = memory().numTensors;

       // Perform actual testing calls.
       for (let n = 0; n < 2; ++n) {
         const tensorCounts: number[] = [];
         await model.fit(inputs, targets, {
           batchSize: 1,
           epochs,
           validationSplit,
           callbacks: {
             onBatchEnd: async (batch: number, logs: Logs) => {
               tensorCounts.push(memory().numTensors);
             }
           }
         });
         for (let epochIndex = 0; epochIndex < epochs; ++epochIndex) {
           // Get the tensor counts within an epoch (i.e., from the first batch
           // till the penultimate one.) Assert that the counts are constant,
           // i.e., no increase in the tensor count within the epoch.
           // N.B.: Even though the tensor count is expected to be constant
           // across batches, across epochs, the count will increase, due to the
           // per-epoch loss and metric values stored for the returned history
           // object, which are currently downloaded via data() calls only at
           // the end of the fit() call.
           const beginBatch = batchesPerEpoch * epochIndex;
           const endBatch = batchesPerEpoch * (epochIndex + 1);
           const inEpochTensorCounts =
               tensorCounts.slice(beginBatch, endBatch - 1);
           if (unique(inEpochTensorCounts).length !== 1) {
             done.fail(
                 `Detected WebGL memory leak within epoch ${epochIndex + 1} ` +
                 `of ${epochs} of the fit() call with ` +
                 `onBatchEnd callback: tensor counts: ${inEpochTensorCounts}.`);
           }
           // Now, assert that the amount of increase in the number of tensors
           // at the end of the epoch equals the expected value.
           if (epochIndex < epochs - 1) {
             // The expected increase of 4 comes from the fact that the fit()
             // call here generates 4 additional scalars and will store them
             // till the end of the fit() call:
             //   loss, val_loss, mse and val_mse.
             expect(tensorCounts[endBatch] - tensorCounts[beginBatch])
                 .toEqual(4);
           }
         }
         expect(tensorCounts.length).toEqual(batchesPerEpoch * epochs);
         const numTensors1 = memory().numTensors;
         if (numTensors1 > numTensors0) {
           done.fail(
               `Detected memory leak of ${numTensors1 - numTensors0} ` +
               `tensor(s) after fit() call with callback.`);
         }
       }
       done();
     });
});

describeMathGPU('Model.fit: yieldEvery', () => {
  function createDummyModel(inputSize: number): tfl.Model {
    const model = tfl.sequential();
    const layerSize = 10;
    model.add(tfl.layers.dense(
        {units: layerSize, inputShape: [inputSize], activation: 'relu'}));
    model.add(tfl.layers.dense({units: 1}));
    model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});
    return model;
  }

  it('auto: 1 batches per epoch; 20 epochs; short batches', async () => {
    const presetBatchTimestamps = [0, 2, 4, 6, 8, 10];
    let counter = 0;
    spyOn(util, 'now').and.callFake(() => presetBatchTimestamps[counter++]);
    let nextFrameCallCount = 0;
    spyOn(tfc, 'nextFrame').and.callFake(async () => {
      nextFrameCallCount++;
    });

    const inputSize = 2;
    const numExamples = 10;
    const epochs = 20;
    const model = createDummyModel(inputSize);
    const xs = ones([numExamples, inputSize]);
    const ys = ones([numExamples, 1]);
    const history = await model.fit(xs, ys, {epochs, batchSize: numExamples});
    expect(history.history.loss.length).toEqual(epochs);
    // For example, there are 20 batches in total. The first several batch are
    // for measurement, during each of which nextFrame() is called. The
    // remaining 17 batches consists of 2 full collections of 8 batches. So
    // nextFrame() is expected to have been called  + 2 = 5 times in total.
    const expectedNextFrameCalls = ModelTrainingYielder.SKIP_FIRST_BATCHES +
        ModelTrainingYielder.DECISION_BATCH_COUNT +
        Math.floor(
            (epochs - ModelTrainingYielder.SKIP_FIRST_BATCHES -
             ModelTrainingYielder.DECISION_BATCH_COUNT) /
            8);
    expect(nextFrameCallCount).toEqual(expectedNextFrameCalls);
  });

  it('auto: 2 batches per epoch; 20 epochs; short batches', async () => {
    const presetBatchTimestamps = [0, 2, 4, 6, 8, 10];
    let counter = 0;
    spyOn(util, 'now').and.callFake(() => presetBatchTimestamps[counter++]);
    let nextFrameCallCount = 0;
    spyOn(tfc, 'nextFrame').and.callFake(async () => {
      nextFrameCallCount++;
    });

    const inputSize = 5;
    const numExamples = 100;
    const epochs = 10;
    const model = createDummyModel(inputSize);
    const xs = ones([numExamples, inputSize]);
    const ys = ones([numExamples, 1]);
    const history =
        await model.fit(xs, ys, {epochs, batchSize: numExamples / 2});
    expect(history.history.loss.length).toEqual(epochs);
    // There are 10 * 2 = 20 batches in total. The first 3 batch are for
    // measurement, during each of which nextFrame() is called. The
    // remaining 17 batches consists of 2 full collections of 8 batches. So
    // nextFrame() is expected to have been called 3 + 2 = 5 times in total.
    const expectedNextFrameCalls = ModelTrainingYielder.SKIP_FIRST_BATCHES +
        ModelTrainingYielder.DECISION_BATCH_COUNT +
        Math.floor(
            (epochs * 2 - ModelTrainingYielder.SKIP_FIRST_BATCHES -
             ModelTrainingYielder.DECISION_BATCH_COUNT) /
            8);
    expect(nextFrameCallCount).toEqual(expectedNextFrameCalls);
  });

  it('auto: 1 batches per epoch; 20 epochs; long batches', async () => {
    const presetBatchTimestamps = [0, 20, 40, 60, 80, 100];
    let counter = 0;
    spyOn(util, 'now').and.callFake(() => presetBatchTimestamps[counter++]);
    let nextFrameCallCount = 0;
    spyOn(tfc, 'nextFrame').and.callFake(async () => {
      nextFrameCallCount++;
    });

    const inputSize = 5;
    const numExamples = 10;
    const epochs = 4;
    const model = createDummyModel(inputSize);
    const xs = ones([numExamples, inputSize]);
    const ys = ones([numExamples, 1]);
    const history = await model.fit(xs, ys, {epochs, batchSize: numExamples});
    expect(history.history.loss.length).toEqual(epochs);
    // For long batch durations, `await nextFrame()` should have been called
    // every batch.
    expect(nextFrameCallCount).toEqual(epochs);
  });

  it('auto: 2 batches per epoch; 20 epochs; long batches', async () => {
    const presetBatchTimestamps = [0, 20, 40, 60, 80, 100];
    let counter = 0;
    spyOn(util, 'now').and.callFake(() => presetBatchTimestamps[counter++]);
    let nextFrameCallCount = 0;
    spyOn(tfc, 'nextFrame').and.callFake(async () => {
      nextFrameCallCount++;
    });

    const inputSize = 5;
    const numExamples = 10;
    const epochs = 4;
    const model = createDummyModel(inputSize);
    const xs = ones([numExamples, inputSize]);
    const ys = ones([numExamples, 1]);
    const history =
        await model.fit(xs, ys, {epochs, batchSize: numExamples / 2});
    expect(history.history.loss.length).toEqual(epochs);
    // For long batch durations, `await nextFrame()` should have been called
    // every batch.
    expect(nextFrameCallCount).toEqual(epochs * 2);
  });

  it('batch: uneven 9 batches per epoch; 2 epochs', async () => {
    const presetBatchTimestamps = [0, 2, 4, 6, 8, 10];
    let counter = 0;
    spyOn(util, 'now').and.callFake(() => presetBatchTimestamps[counter++]);
    let nextFrameCallCount = 0;
    spyOn(tfc, 'nextFrame').and.callFake(async () => {
      nextFrameCallCount++;
    });

    const inputSize = 1;
    const numExamples = 10;
    const epochs = 2;
    const model = createDummyModel(inputSize);
    const xs = ones([numExamples, inputSize]);
    const ys = ones([numExamples, 1]);
    const history =
        await model.fit(xs, ys, {epochs, batchSize: 4, yieldEvery: 'batch'});
    expect(history.history.loss.length).toEqual(epochs);
    // There are `ceil(10 / 4)` batches per epoch.
    expect(nextFrameCallCount).toEqual(Math.ceil(10 / 4) * epochs);
  });

  it('epoch: 10 batches per epoch; 2 epochs', async () => {
    let nextFrameCallCount = 0;
    spyOn(tfc, 'nextFrame').and.callFake(async () => {
      nextFrameCallCount++;
    });

    const inputSize = 5;
    const numExamples = 10;
    const epochs = 2;
    const model = createDummyModel(inputSize);
    const xs = ones([numExamples, inputSize]);
    const ys = ones([numExamples, 1]);
    const history = await model.fit(
        xs, ys, {epochs, batchSize: numExamples / 10, yieldEvery: 'epoch'});
    expect(history.history.loss.length).toEqual(epochs);
    expect(nextFrameCallCount).toEqual(epochs);
  });

  it('never: 2 batches per epoch; 20 epochs', async () => {
    let nextFrameCallCount = 0;
    spyOn(tfc, 'nextFrame').and.callFake(async () => {
      nextFrameCallCount++;
    });

    const inputSize = 5;
    const numExamples = 10;
    const epochs = 4;
    const model = createDummyModel(inputSize);
    const xs = ones([numExamples, inputSize]);
    const ys = ones([numExamples, 1]);
    const history = await model.fit(
        xs, ys, {epochs, batchSize: numExamples / 2, yieldEvery: 'never'});
    expect(history.history.loss.length).toEqual(epochs);
    // Due to yieldEvery = 'never', no `await nextFrame()` call should have
    // happened.
    expect(nextFrameCallCount).toEqual(0);
  });
});

describeMathCPUAndGPU('Model.evaluate', () => {
  const numExamples = 8;
  const inputSize = 2;
  const outputSize = 1;
  let model: tfl.Model;
  let x: Tensor;
  let y: Tensor;
  function prepModel() {
    const input = tfl.layers.input({shape: [inputSize]});
    const dense = tfl.layers.dense(
        {units: outputSize, kernelInitializer: 'ones', useBias: false});
    const output = dense.apply(input) as tfl.SymbolicTensor;
    model = new tfl.Model({inputs: input, outputs: output});
  }
  function prepData() {
    x = ones([numExamples, inputSize]);
    y = ones([numExamples, outputSize]);
  }

  it('Calling evaluate before compile leads to error', () => {
    prepModel();
    prepData();

    expect(() => model.evaluate(x, y))
        .toThrowError(/must compile a model before/);
  });

  const metricsValues: string[][] = [null, ['mse']];
  const batchSizes = [null, 4, 16];
  for (const metrics of metricsValues) {
    for (const batchSize of batchSizes) {
      const testTitle =
          `metrics=${JSON.stringify(metrics)}, batchSize=${batchSize}`;
      it(testTitle, () => {
        prepModel();
        prepData();
        model.compile({optimizer: 'sgd', loss: 'meanSquaredError', metrics});
        const losses = model.evaluate(x, y, {batchSize});
        if (metrics == null) {
          expectTensorsClose(losses as Scalar, scalar(1));
        } else {
          const lossesArray = losses as Scalar[];
          expect(lossesArray.length).toEqual(2);
          expectTensorsClose(lossesArray[0], scalar(1));
          expectTensorsClose(lossesArray[1], scalar(1));
        }
      });
    }
  }
});

describeMathCPUAndGPU('Load weights', () => {
  it('Simple functional model', () => {
    const inputTensor =
        tfl.layers.input({shape: [3], name: 'inputLayer', dtype: 'float32'});
    const denseLayer =
        tfl.layers.dense({units: 2, useBias: true, name: 'denseLayer'});
    const output = denseLayer.apply(inputTensor) as tfl.SymbolicTensor;
    const model = new tfl.Model({
      inputs: [inputTensor],
      outputs: [output],
      name: 'modelWithWeightsToLoad',
    });
    const weightsJSON = {
      'keras_version': '2.1.2',
      'backend': 'tensorflow',
      'weights': {
        'denseLayer': [
          {
            'name': 'denseLayer/kernel:0',
            'dtype': 'float32',
            'shape': [3, 2],
            'value': [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
          },
          {
            'name': 'denseLayer/bias:0',
            'dtype': 'float32',
            'shape': [2],
            'value': [-0.1, -0.2],
          },
        ],
      },
    };
    model.loadWeights(weightsJSON);

    // Run a concrete input value through the layer to check that the weights
    // are loaded properly.
    expectTensorsClose(
        model.apply(tensor2d([[1, 1, 1]], [1, 3])) as Tensor,
        tensor2d([[0.8, 1.0]], [1, 2]));
  });
});

describeMathCPUAndGPU('Model.execute', () => {
  function createFunctionalModel():
      [tfl.Model, {[name: string]: tfl.SymbolicTensor}] {
    const input1 = tfl.input({shape: [2, 3]});
    const reshape1 = tfl.layers.reshape({targetShape: [3, 2]}).apply(input1) as
        tfl.SymbolicTensor;
    const input2 = tfl.input({shape: [3, 4]});
    const concat =
        tfl.layers.concatenate({axis: -1}).apply([reshape1, input2]) as
        tfl.SymbolicTensor;
    const model = tfl.model({inputs: [input1, input2], outputs: concat});

    return [model, {input1, reshape1, input2, concat}];
  }

  function createSequentialModel(): tfl.Sequential {
    const model = tfl.sequential();
    model.add(tfl.layers.dense({
      units: 6,
      inputShape: [4],
      kernelInitializer: 'zeros',
      useBias: false
    }));
    model.add(tfl.layers.dense(
        {units: 3, kernelInitializer: 'zeros', useBias: false}));
    model.add(tfl.layers.dense(
        {units: 1, kernelInitializer: 'zeros', useBias: false}));
    return model;
  }

  it('Functional model: single output', () => {
    const [model, layers] = createFunctionalModel();
    const inputs = [zeros([1, 2, 3]), zeros([1, 3, 4])];
    const outputs = model.execute(inputs, layers['reshape1'].name) as Tensor;
    expectTensorsClose(outputs, zeros([1, 3, 2]));
  });

  it('Functional model: multiple outputs', () => {
    const [model, layers] = createFunctionalModel();
    const inputs = [zeros([1, 2, 3]), zeros([1, 3, 4])];
    const outputs = model.execute(inputs, [
      layers['reshape1'].name, layers['concat'].name, layers['input2'].name
    ]) as Tensor[];
    expectTensorsClose(outputs[0], zeros([1, 3, 2]));
    expectTensorsClose(outputs[1], zeros([1, 3, 6]));
    expectTensorsClose(outputs[2], zeros([1, 3, 4]));
  });

  it('Functional model: Dictionary of inputs', () => {
    const [model, layers] = createFunctionalModel();
    const inputName1 = model.inputs[0].name;
    const inputName2 = model.inputs[1].name;
    const inputs: NamedTensorMap = {};
    inputs[inputName1] = zeros([1, 2, 3]);
    inputs[inputName2] = zeros([1, 3, 4]);
    const outputs = model.execute(inputs, [
      layers['reshape1'].name, layers['concat'].name, layers['input2'].name
    ]) as Tensor[];
    expectTensorsClose(outputs[0], zeros([1, 3, 2]));
    expectTensorsClose(outputs[1], zeros([1, 3, 6]));
    expectTensorsClose(outputs[2], zeros([1, 3, 4]));
  });

  it('Functional model: missing input in dictionary throws Error', () => {
    const [model, layers] = createFunctionalModel();
    const inputName2 = model.inputs[1].name;
    const inputs: NamedTensorMap = {};
    inputs[inputName2] = zeros([1, 3, 4]);
    expect(() => model.execute(inputs, layers['reshape1'].name))
        .toThrowError(/No value is provided for .* input/);
  });

  it('Functional model: Incorrect number of inputs throws Error', () => {
    const [model, layers] = createFunctionalModel();
    const inputs = [zeros([1, 2, 3])];
    expect(() => model.execute(inputs, layers['reshape1'].name))
        .toThrowError(/The number of inputs provided \(1\) does not match .*2/);
  });

  it('Functional model: nonexistent tensor name throws Error', () => {
    const [model, layers] = createFunctionalModel();
    const inputs = [zeros([1, 2, 3]), zeros([1, 3, 4])];
    const nonexistentTensorName =
        layers['reshape1'].name + Math.random().toFixed(4);
    expect(() => model.execute(inputs, nonexistentTensorName))
        .toThrowError(/Cannot find SymbolicTensors for output name/);
    expect(() => model.execute(inputs, [
      layers['reshape1'].name, nonexistentTensorName
    ])).toThrowError(/Cannot find SymbolicTensors for output name/);
  });

  it('Functional model: empty outputs string throws Error', () => {
    const model = createFunctionalModel()[0];
    const inputs = [zeros([1, 2, 3]), zeros([1, 3, 4])];
    expect(() => model.execute(inputs, [])).toThrowError(/empty Array/);
  });

  it('Sequential model: singleton input', () => {
    const model = createSequentialModel();
    const input = zeros([2, 4]);
    const outputs = model.execute(input, [
      (model.layers[2].output as tfl.SymbolicTensor).name,
      (model.layers[1].output as tfl.SymbolicTensor).name,
      (model.layers[0].output as tfl.SymbolicTensor).name,
    ]) as Tensor[];
    expectTensorsClose(outputs[0], zeros([2, 1]));
    expectTensorsClose(outputs[1], zeros([2, 3]));
    expectTensorsClose(outputs[2], zeros([2, 6]));
  });

  it('Sequential model: length-1 Array input', () => {
    const model = createSequentialModel();
    const input = [zeros([2, 4])];
    const output = model.execute(
                       input,
                       (model.layers[1].output as tfl.SymbolicTensor).name,
                       ) as Tensor;
    expectTensorsClose(output, zeros([2, 3]));
  });

  it('Sequential model: length-1 dictionary input', () => {
    const model = createSequentialModel();
    const inputs: NamedTensorMap = {};
    inputs[(model.input as SymbolicTensor).name] = zeros([2, 4]);
    const output = model.execute(
                       inputs,
                       (model.layers[1].output as tfl.SymbolicTensor).name,
                       ) as Tensor;
    expectTensorsClose(output, zeros([2, 3]));
  });
});
