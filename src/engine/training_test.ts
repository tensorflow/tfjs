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

// tslint:disable:max-line-length
import {Scalar, scalar, SGDOptimizer, Tensor, tensor1d, tensor2d, tensor3d, zeros} from '@tensorflow/tfjs-core';

import * as K from '../backend/tfjs_backend';
import {CustomCallback, CustomCallbackConfig, Logs} from '../callbacks';
import * as tfl from '../index';
import {Dropout, Flatten, Reshape} from '../layers/core';
import {SimpleRNN} from '../layers/recurrent';
import {TimeDistributed} from '../layers/wrappers';
import {Regularizer} from '../regularizers';
import {DType, SymbolicTensor} from '../types';
import {pyListRepeat, stringsEqual} from '../utils/generic_utils';
import {describeMathCPU, describeMathCPUAndGPU, expectTensorsClose} from '../utils/test_utils';

import {Input, Layer} from './topology';
import {checkArrayLengths, isDataArray, isDataDict, isDataTensor, makeBatches, Model, sliceArraysByIndices, standardizeInputData} from './training';

// tslint:enable:max-line-length

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
    const inputs = [K.zeros([2, 1]), K.zeros([3, 1])];
    const targets = [K.zeros([2, 1]), K.zeros([2, 1])];
    expect(() => checkArrayLengths(inputs, targets))
        .toThrowError(/All input .* should have the same number of samples/);
  });
  it('Batch mismatch in targets', () => {
    const inputs = [K.zeros([2, 1]), K.zeros([2, 1])];
    const targets = [K.zeros([2, 1]), K.zeros([3, 1])];
    expect(() => checkArrayLengths(inputs, targets))
        .toThrowError(/All target .* should have the same number of samples/);
  });
  it('Batch mismatch between inputs and targets', () => {
    const inputs = [K.zeros([2, 1]), K.zeros([2, 1])];
    const targets = [K.zeros([3, 1]), K.zeros([3, 1])];
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
    const inputTensor =
        Input({shape: [3, 4], name: 'inputLayer1', dtype: DType.float32});
    const layer = new Reshape({targetShape: [2, 6]});
    const output = layer.apply(inputTensor) as SymbolicTensor;
    const model =
        new Model({inputs: [inputTensor], outputs: [output], name: 'model1x1'});
    const xs = K.ones([10, 3, 4]);
    const ys = model.predict(xs, {batchSize: 4}) as Tensor;
    expectTensorsClose(ys, K.ones([10, 2, 6]));
  });

  it('1 input, 1 output, tensor as input argument', () => {
    const inputTensor =
        Input({shape: [3, 4], name: 'inputLayer1', dtype: DType.float32});
    const layer = new Reshape({targetShape: [2, 6]});
    const output = layer.apply(inputTensor) as SymbolicTensor;
    const model =
        new Model({inputs: [inputTensor], outputs: [output], name: 'model1x1'});
    const xs = K.ones([10, 3, 4]);
    const ys = model.predict(xs) as Tensor;
    expectTensorsClose(ys, K.ones([10, 2, 6]));
  });

  it('1 input as Array, 1 output', () => {
    const inputTensor =
        Input({shape: [3, 4], name: 'inputLayer1', dtype: DType.float32});
    const layer = new Reshape({targetShape: [2, 6]});
    const output = layer.apply(inputTensor) as SymbolicTensor;
    const model =
        new Model({inputs: [inputTensor], outputs: [output], name: 'model1x1'});
    const xs = K.ones([10, 3, 4]);
    const ys = model.predict([xs], {batchSize: 4}) as Tensor;
    expectTensorsClose(ys, K.ones([10, 2, 6]));
  });

  it('1 input, 2 outputs', () => {
    const inputTensor =
        Input({shape: [3, 4], name: 'inputLayer2', dtype: DType.float32});
    const layer1 = new Reshape({targetShape: [2, 6]});
    const layer2 = new Flatten();
    const output1 = layer1.apply(inputTensor) as SymbolicTensor;
    const output2 = layer2.apply(output1) as SymbolicTensor;
    const model = new Model(
        {inputs: [inputTensor], outputs: [output1, output2], name: 'model1x2'});
    const xs = K.ones([10, 3, 4]);
    const ys = model.predict(xs, {batchSize: 4}) as Tensor[];
    expect(ys.length).toEqual(2);
    expectTensorsClose(ys[0], K.ones([10, 2, 6]));
    expectTensorsClose(ys[1], K.ones([10, 12]));
  });

  it('2 inputs, 2 outputs', () => {
    const inputTensor1 =
        Input({shape: [3, 4], name: 'inputLayer3', dtype: DType.float32});
    const inputTensor2 =
        Input({shape: [3, 3], name: 'inputLayer4', dtype: DType.float32});
    const layer1 = new Reshape({targetShape: [2, 6]});
    const layer2 = new Flatten();
    const output1 = layer1.apply(inputTensor1) as SymbolicTensor;
    const output2 = layer2.apply(inputTensor2) as SymbolicTensor;
    const model = new Model({
      inputs: [inputTensor1, inputTensor2],
      outputs: [output1, output2],
      name: 'model2x2'
    });
    const xs1 = K.ones([10, 3, 4]);
    const xs2 = K.ones([10, 3, 3]);
    const ys = model.predict([xs1, xs2], {batchSize: 4}) as Tensor[];
    expect(ys.length).toEqual(2);
    expectTensorsClose(ys[0], K.ones([10, 2, 6]));
    expectTensorsClose(ys[1], K.ones([10, 9]));
  });

  it('Incorrect number of inputs leads to exception: 1 vs 2', () => {
    const inputTensor =
        Input({shape: [3, 4], name: 'inputLayer_inc_1', dtype: DType.float32});
    const layer = new Reshape({targetShape: [2, 6]});
    const output = layer.apply(inputTensor) as SymbolicTensor;
    const model = new Model(
        {inputs: [inputTensor], outputs: [output], name: 'model_inc_1x1'});
    const xs1 = K.ones([10, 3, 4]);

    expect(() => model.predict([
      xs1, xs1
    ])).toThrowError(/.*Expected.*1 Tensor.*got 2 Tensor.*/);
  });

  it('Incorrect number of inputs leads to exception: 2 vs 3', () => {
    const inputTensor1 =
        Input({shape: [3, 4], name: 'inputLayer_inc_3', dtype: DType.float32});
    const inputTensor2 =
        Input({shape: [3, 3], name: 'inputLayer_inc_4', dtype: DType.float32});
    const layer1 = new Reshape({targetShape: [2, 6]});
    const layer2 = new Flatten();
    const output1 = layer1.apply(inputTensor1) as SymbolicTensor;
    const output2 = layer2.apply(inputTensor2) as SymbolicTensor;
    const model = new Model({
      inputs: [inputTensor1, inputTensor2],
      outputs: [output1, output2],
      name: 'model_inc_2x2'
    });
    const xs1 = K.ones([10, 3, 4]);

    expect(() => model.predict([
      xs1, xs1, xs1
    ])).toThrowError(/.*Expected.*2 Tensor.*got 3 Tensor.*/);
  });

  it('Incorrect input shape leads to exception', () => {
    const inputTensor =
        Input({shape: [3, 4], name: 'inputLayer_inc_1', dtype: DType.float32});
    const layer = new Reshape({targetShape: [2, 6]});
    const output = layer.apply(inputTensor) as SymbolicTensor;
    const model = new Model(
        {inputs: [inputTensor], outputs: [output], name: 'model_inc_1x1'});
    const xs1 = K.ones([2, 4, 3]);

    expect(() => model.predict(xs1))
        .toThrowError(/.*expected.* shape \[null,3,4\].*but got.*\[2,4,3\]/);
  });
});

describeMathCPUAndGPU('Model.fit', () => {
  const inputSize = 4;   // Input vector size for model with one input.
  const inputSize1 = 3;  // 1st input vector size for model with two inputs.
  const inputSize2 = 4;  // 2nd input vector size for model with two inputs.
  const numSamples = 5;  // Number of samples in a batch.

  const inputTensor =
      Input({shape: [inputSize], name: 'inputLayer1', dtype: DType.float32});
  const inputTensor1 = Input(
      {shape: [inputSize1], name: 'inputLayer1of2', dtype: DType.float32});
  const inputTensor2 = Input(
      {shape: [inputSize2], name: 'inputLayer2of2', dtype: DType.float32});

  // For model with one input.
  let model: Model;
  let inputs: Tensor;
  let targets: Tensor;

  // For model with two inputs (and two outputs).
  let twoOutputModel: Model;
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
    const output = layer.apply(inputTensor) as SymbolicTensor;
    model = new Model({inputs: [inputTensor], outputs: [output]});
    inputs = K.ones([numSamples, inputSize]);
    targets = K.ones([numSamples, 1]);
  }

  function createDenseCategoricalModelAndData(useBias = false): void {
    const layer =
        tfl.layers.dense({units: 2, useBias, kernelInitializer: 'ones'});
    const output = layer.apply(inputTensor) as SymbolicTensor;
    model = new Model({inputs: [inputTensor], outputs: [output]});
    inputs = K.ones([numSamples, inputSize]);
    targets = K.oneHot(K.ones([numSamples]), 2);
  }

  function createTwoLayerDenseModelAndData(useBias = false): [Layer, Layer] {
    const layer1 =
        tfl.layers.dense({units: 10, useBias, kernelInitializer: 'ones'});
    const layer2 =
        tfl.layers.dense({units: 1, useBias, kernelInitializer: 'ones'});
    const output = layer2.apply(layer1.apply(inputTensor)) as SymbolicTensor;
    model = new Model({inputs: [inputTensor], outputs: [output]});
    inputs = K.ones([numSamples, inputSize]);
    targets = K.ones([numSamples, 1]);
    return [layer1, layer2];
  }

  function createDenseModelWithTwoOutputsAndData(): void {
    const layer1 =
        tfl.layers.dense({units: 1, useBias: false, kernelInitializer: 'ones'});
    const layer2 =
        tfl.layers.dense({units: 1, useBias: false, kernelInitializer: 'ones'});
    const output1 = layer1.apply(inputTensor1) as SymbolicTensor;
    const output2 = layer2.apply(inputTensor2) as SymbolicTensor;
    twoOutputModel = new Model(
        {inputs: [inputTensor1, inputTensor2], outputs: [output1, output2]});
    inputs1 = K.ones([numSamples, inputSize1]);
    inputs2 = K.ones([numSamples, inputSize2]);
    targets1 = K.ones([numSamples, 1]);
    targets2 = K.ones([numSamples, 1]);
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

  it('Using only x and y input arguments', async done => {
    createDenseModelAndData();

    model.compile({optimizer: 'SGD', loss: 'meanSquaredError'});
    model.fit(inputs, targets)
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

  it('Training with Dropout layer', async done => {
    const inputSize = 2;
    const batchSize = 4;
    const input = Input({shape: [inputSize]});
    const dense1 =
        tfl.layers.dense({units: 2, kernelInitializer: 'ones', useBias: false});
    const dropout = new Dropout({rate: 0.5});
    const dense2 =
        tfl.layers.dense({units: 1, kernelInitializer: 'ones', useBias: false});
    const output =
        dense2.apply(dropout.apply(dense1.apply(input))) as SymbolicTensor;
    const model = new Model({inputs: input, outputs: output});
    model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});
    const x = K.ones([batchSize, inputSize]);
    const y = K.ones([batchSize, 1]);
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
    const simpleRNN = new SimpleRNN({
      units: outputSize,
      kernelInitializer: 'ones',
      recurrentInitializer: 'ones',
      useBias: false,
      returnSequences: true,
    });
    const timeDistributed = new TimeDistributed({
      layer: tfl.layers.dense(
          {units: outputSize, kernelInitializer: 'ones', useBias: false})
    });
    const input = Input({shape: [sequenceLength, inputSize]});
    const output =
        timeDistributed.apply(simpleRNN.apply(input)) as SymbolicTensor;
    const model = new Model({inputs: input, outputs: output});
    model.compile({
      optimizer: 'sgd',
      loss: 'categoricalCrossentropy',
      metrics: ['accuracy'],
    });
    const history = await model.fit(
        K.ones([dataSize, sequenceLength, inputSize]),
        K.ones([dataSize, sequenceLength, outputSize]), {
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
        layer1.getWeights()[0],
        K.scalarTimesArray(scalar(-0.61341441), K.ones([4, 10])));
    expectTensorsClose(
        layer2.getWeights()[0],
        K.scalarTimesArray(scalar(-1.77405429), K.ones([10, 1])));

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
        layer1.getWeights()[0],
        K.scalarTimesArray(scalar(-0.61341441), K.ones([4, 10])));
    // Expect change in the value of layer2's kernel.
    expectTensorsClose(
        layer2.getWeights()[0],
        K.scalarTimesArray(scalar(-0.11295), K.ones([10, 1])));
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

  // TODO(cais): Uncommment the test below once the 1-tensor leak during
  // //   `updateVariable` is fixed.
  // it('Repeated fit calls leads to no memory leak: no validation',
  //    async done => {
  //      createDenseModelAndData();

  //      model.compile({optimizer: 'SGD', loss: 'meanSquaredError'});
  //      // Use batchSize === numSamples to get exactly one batch.
  //      await model.fit(inputs, targets, {batchSize: numSamples, epochs: 1});
  //      const numTensors1 = memory().numTensors;
  //      await model.fit(inputs, targets, {batchSize: numSamples, epochs: 1});
  //      const numTensors2 = memory().numTensors;
  //      if (numTensors2 > numTensors1) {
  //        done.fail(
  //            `Memory leak detected during fit(): Leaked ` +
  //            `${numTensors2 - numTensors1} tensor(s) after the ` +
  //            `second fit() call.`);
  //      } else {
  //        done();
  //      }
  //    });

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
        Input({shape: [1], name: 'inputLayer1', dtype: DType.float32});
    const layer1 = tfl.layers.dense({units: 1});
    const layer2 = tfl.layers.dropout({rate: 0.5});

    // Hook the dropout layer to observe the training arg values during the
    // fit(), evaluate() and predict() calls.
    const dropoutLayerTrainingFlags: boolean[] = [];
    // tslint:disable:no-any
    const recordDropoutTrainingArgHook =
        (inputs: Tensor|Tensor[], kwargs: any) => {
          dropoutLayerTrainingFlags.push(kwargs.training as boolean);
        };
    // tslint:enable:no-any
    layer2.setCallHook(recordDropoutTrainingArgHook);

    const output = layer2.apply(layer1.apply(inputTensor)) as SymbolicTensor;
    const model = new Model({inputs: [inputTensor], outputs: [output]});
    model.compile({optimizer: 'sgd', loss: 'meanSquaredError'});
    const xs = K.ones([4, 1]);
    const ys = K.ones([4, 1]);

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

describeMathCPUAndGPU('Model.evaluate', () => {
  const numExamples = 8;
  const inputSize = 2;
  const outputSize = 1;
  let model: Model;
  let x: Tensor;
  let y: Tensor;
  function prepModel() {
    const input = Input({shape: [inputSize]});
    const dense = tfl.layers.dense(
        {units: outputSize, kernelInitializer: 'ones', useBias: false});
    const output = dense.apply(input) as SymbolicTensor;
    model = new Model({inputs: input, outputs: output});
  }
  function prepData() {
    x = K.ones([numExamples, inputSize]);
    y = K.ones([numExamples, outputSize]);
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
        Input({shape: [3], name: 'inputLayer', dtype: DType.float32});
    const denseLayer =
        tfl.layers.dense({units: 2, useBias: true, name: 'denseLayer'});
    const output = denseLayer.apply(inputTensor) as SymbolicTensor;
    const model = new Model({
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
