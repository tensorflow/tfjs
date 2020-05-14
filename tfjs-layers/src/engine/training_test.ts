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
import {CustomCallback, CustomCallbackArgs, DEFAULT_YIELD_EVERY_MS, Params} from '../base_callbacks';
import * as tfl from '../index';
import * as logs from '../logs';
import {Logs, UnresolvedLogs} from '../logs';
import {Regularizer} from '../regularizers';
import {Kwargs} from '../types';
import {pyListRepeat, stringsEqual, unique} from '../utils/generic_utils';
import {describeMathCPU, describeMathCPUAndGPU, describeMathGPU, expectTensorsClose} from '../utils/test_utils';

// TODO(bileschi): Use external version of Layer.
import {Layer, SymbolicTensor} from './topology';
import {checkArrayLengths, collectMetrics, isDataArray, isDataDict, isDataTensor, standardizeInputData} from './training';
import {makeBatches, sliceArraysByIndices} from './training_tensors';

describeMathCPU('isDataTensor', () => {
  const x = tfc.tensor2d([[3.14]]);

  it('Positive case', () => {
    expect(isDataTensor(x)).toEqual(true);
  });
  it('Negative cases', () => {
    expect(isDataTensor([x, x])).toEqual(false);
    expect(isDataTensor({'Pie': x})).toEqual(false);
    expect(isDataTensor({})).toEqual(false);
  });
});

describeMathCPU('isDataArray', () => {
  const x = tfc.tensor2d([[3.14]]);

  it('Positive case', () => {
    expect(isDataArray([x, x])).toEqual(true);
    expect(isDataArray([])).toEqual(true);
  });
  it('Negative cases', () => {
    expect(isDataArray(x)).toEqual(false);
    expect(isDataArray({'Pie': x})).toEqual(false);
    expect(isDataArray({})).toEqual(false);
  });
});

describeMathCPU('isDataDict', () => {
  const x = tfc.tensor2d([[3.14]]);
  it('Positive case', () => {
    expect(isDataDict({'Pie': x})).toEqual(true);
    expect(isDataDict({})).toEqual(true);
  });
  it('Negative cases', () => {
    expect(isDataDict(x)).toEqual(false);
    expect(isDataDict([x, x])).toEqual(false);
    expect(isDataDict([])).toEqual(false);
  });
});

describeMathCPU('standardizeInputData', () => {
  const getX = () => tfc.tensor2d([[42]]);
  const getY = () => tfc.tensor2d([[21]]);

  it('Singleton Tensor, Array of one name', () => {
    const outputs = standardizeInputData(getX(), ['Foo']);
    expect(outputs.length).toEqual(1);
    expectTensorsClose(outputs[0], getX());
  });
  it('Array of one Tensor, Array of one name', () => {
    const outputs = standardizeInputData([getX()], ['Foo']);
    expect(outputs.length).toEqual(1);
    expectTensorsClose(outputs[0], getX());
  });
  it('Array of two Tensors, Array of two names', () => {
    const outputs = standardizeInputData([getX(), getY()], ['Foo', 'Bar']);
    expect(outputs.length).toEqual(2);
    expectTensorsClose(outputs[0], getX());
    expectTensorsClose(outputs[1], getY());
  });
  it('Dict of two Tensors, Array of two names', () => {
    const outputs =
        standardizeInputData({'Foo': getX(), 'Bar': getY()}, ['Foo', 'Bar']);
    expect(outputs.length).toEqual(2);
    expectTensorsClose(outputs[0], getX());
    expectTensorsClose(outputs[1], getY());
  });
  it('Unexpected data leads to exception: singleton Tensor', () => {
    expect(() => standardizeInputData(getX(), []))
        .toThrowError(/expected no data/);
  });
  it('Unexpected data leads to exception: Array of two Tensors', () => {
    expect(() => standardizeInputData([getX(), getY()], []))
        .toThrowError(/expected no data/);
  });
  it('Unexpected data leads to exception: Dict', () => {
    expect(() => standardizeInputData({'Pie': getX()}, []))
        .toThrowError(/expected no data/);
  });
  it('Length mismatch: 1 singleton Scalar vs two names', () => {
    expect(() => standardizeInputData(getX(), ['Foo', 'Bar']))
        .toThrowError(/expects 2 Tensor.* but only received one/);
  });
  it('Length mismatch: Array of 2 Scalars vs one name', () => {
    expect(() => standardizeInputData([getX(), scalar(-42)], ['Foo']))
        .toThrowError(/Expected to see 1 Tensor/);
  });
  it('Length mismatch: Dict of 1 Scalar vs 2 names', () => {
    expect(() => standardizeInputData({'Foo': getX()}, ['Foo', 'Bar']))
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

describeMathCPUAndGPU('collectMetrics', () => {
  it('shortcut strings name', () => {
    const metrics = 'mse';
    const outputNames = ['output'];
    const collectedMetrics = collectMetrics(metrics, outputNames);
    expect(collectedMetrics.length).toEqual(1);
    expect(collectedMetrics[0].length).toEqual(1);
    expect(collectedMetrics[0][0]).toEqual('mse');
  });
  it('metric function', () => {
    const metrics = tfl.metrics.meanSquaredError;
    const outputNames = ['output'];
    const collectedMetrics = collectMetrics(metrics, outputNames);
    expect(collectedMetrics.length).toEqual(1);
    expect(collectedMetrics[0].length).toEqual(1);
    expect(collectedMetrics[0][0]).toEqual(metrics);
  });
  it('Array of shortcut string names', () => {
    const metrics = ['mse', 'crossentropy'];
    const outputNames = ['output'];
    const collectedMetrics = collectMetrics(metrics, outputNames);
    expect(collectedMetrics.length).toEqual(1);
    expect(collectedMetrics[0].length).toEqual(2);
    expect(collectedMetrics[0][0]).toEqual('mse');
    expect(collectedMetrics[0][1]).toEqual('crossentropy');
  });
  it('Array of metric functions', () => {
    const metrics = [tfl.metrics.meanSquaredError, tfl.metrics.precision];
    const outputNames = ['output'];
    const collectedMetrics = collectMetrics(metrics, outputNames);
    expect(collectedMetrics.length).toEqual(1);
    expect(collectedMetrics[0].length).toEqual(2);
    expect(collectedMetrics[0][0]).toEqual(metrics[0]);
    expect(collectedMetrics[0][1]).toEqual(metrics[1]);
  });
  it('Array of mixing shortcut string names and metric functions', () => {
    const metrics = ['mse', tfl.metrics.precision];
    const outputNames = ['output'];
    const collectedMetrics = collectMetrics(metrics, outputNames);
    expect(collectedMetrics.length).toEqual(1);
    expect(collectedMetrics[0].length).toEqual(2);
    expect(collectedMetrics[0][0]).toEqual('mse');
    expect(collectedMetrics[0][1]).toEqual(metrics[1]);
  });
  it('Dict of shortcut string names', () => {
    const metrics = {'output': 'mse'};
    const outputNames = ['output'];
    const collectedMetrics = collectMetrics(metrics, outputNames);
    expect(collectedMetrics.length).toEqual(1);
    expect(collectedMetrics[0].length).toEqual(1);
    expect(collectedMetrics[0][0]).toEqual('mse');
  });
  it('Dict of metric functions', () => {
    const metrics = {'output': tfl.metrics.meanSquaredError};
    const outputNames = ['output'];
    const collectedMetrics = collectMetrics(metrics, outputNames);
    expect(collectedMetrics.length).toEqual(1);
    expect(collectedMetrics[0].length).toEqual(1);
    expect(collectedMetrics[0][0]).toEqual(metrics['output']);
  });
  it('metrics null', () => {
    const outputNames = ['output'];
    const collectedMetrics = collectMetrics(null, outputNames);
    expect(collectedMetrics.length).toEqual(1);
    expect(collectedMetrics[0].length).toEqual(0);
  });
  it('metrics is array, length = 0', () => {
    const outputNames = ['output'];
    const collectedMetrics = collectMetrics([], outputNames);
    expect(collectedMetrics.length).toEqual(1);
    expect(collectedMetrics[0].length).toEqual(0);
  });
  it('multiple output names', () => {
    const metrics = ['mse'];
    const outputNames = ['output1', 'output2'];
    const collectedMetrics = collectMetrics(metrics, outputNames);
    expect(collectedMetrics.length).toEqual(2);
    expect(collectedMetrics[0].length).toEqual(1);
    expect(collectedMetrics[0][0]).toEqual('mse');
    expect(collectedMetrics[1].length).toEqual(1);
    expect(collectedMetrics[1][0]).toEqual('mse');
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

describeMathCPUAndGPU('LayersModel.predict', () => {
  it('1 input, 1 output', () => {
    const inputTensor = tfl.layers.input(
        {shape: [3, 4], name: 'inputLayer1', dtype: 'float32'});
    const layer = tfl.layers.reshape({targetShape: [2, 6]});
    const output = layer.apply(inputTensor) as tfl.SymbolicTensor;
    const model = new tfl.LayersModel(
        {inputs: [inputTensor], outputs: [output], name: 'model1x1'});
    const xs = ones([10, 3, 4]);
    const ys = model.predict(xs, {batchSize: 4}) as Tensor;
    expectTensorsClose(ys, ones([10, 2, 6]));
  });

  it('1D tensors as inputs', () => {
    const model = tfl.sequential();
    model.add(tfl.layers.dense({units: 1, inputShape: [1]}));

    const xs = ones([10]);  // A batch of 10.
    // Do a burn-in call first.
    tfc.dispose(model.predict(xs, {batchSize: 4}));
    const numTensors0 = memory().numTensors;
    const ys = model.predict(xs, {batchSize: 4}) as Tensor;
    expect(ys.shape).toEqual([10, 1]);
    ys.dispose();
    // Assert no memory leak.
    expect(memory().numTensors).toEqual(numTensors0);
  });

  it('1 input, 1 output, tensor as input argument', () => {
    const inputTensor = tfl.layers.input(
        {shape: [3, 4], name: 'inputLayer1', dtype: 'float32'});
    const layer = tfl.layers.reshape({targetShape: [2, 6]});
    const output = layer.apply(inputTensor) as tfl.SymbolicTensor;
    const model = new tfl.LayersModel(
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
    const model = new tfl.LayersModel(
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
    const model = new tfl.LayersModel(
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
    const model = new tfl.LayersModel({
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
    const model = new tfl.LayersModel(
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
    const model = new tfl.LayersModel({
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
    const model = new tfl.LayersModel(
        {inputs: [inputTensor], outputs: [output], name: 'model_inc_1x1'});
    const xs1 = ones([2, 4, 3]);

    expect(() => model.predict(xs1))
        .toThrowError(/.*expected.* shape \[null,3,4\].*but got.*\[2,4,3\]/);
  });

  it('Invalid batchSize value leads to Error', () => {
    const model = tfl.sequential(
        {layers: [tfl.layers.dense({units: 1, inputShape: [2]})]});
    const xs = tfc.zeros([5, 2]);
    expect(() => model.predict(xs, {batchSize: 0}))
        .toThrowError(
            /batchSize is required to be a positive integer, but got 0/);
    expect(() => model.predict(xs, {batchSize: -2}))
        .toThrowError(
            /batchSize is required to be a positive integer, but got -2/);
    expect(() => model.predict(xs, {batchSize: 3.14}))
        .toThrowError(
            /batchSize is required to be a positive integer, but got 3\.14/);
    // tslint:disable-next-line:no-any
    expect(() => model.predict(xs, {batchSize: 'a' as any}))
        .toThrowError(
            /batchSize is required to be a positive integer, but got a/);
  });
});

describeMathCPUAndGPU('LayersModel.fit', () => {
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
  let model: tfl.LayersModel;
  let inputs: Tensor;
  let targets: Tensor;

  // For model with two inputs (and two outputs).
  let twoOutputModel: tfl.LayersModel;
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
    model = new tfl.LayersModel({inputs: [inputTensor], outputs: [output]});
    inputs = ones([numSamples, inputSize]);
    targets = ones([numSamples, 1]);
  }

  function createDenseCategoricalModelAndData(useBias = false): void {
    const layer =
        tfl.layers.dense({units: 2, useBias, kernelInitializer: 'ones'});
    const output = layer.apply(inputTensor) as tfl.SymbolicTensor;
    model = new tfl.LayersModel({inputs: [inputTensor], outputs: [output]});
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
    model = new tfl.LayersModel({inputs: [inputTensor], outputs: [output]});
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
    twoOutputModel = new tfl.LayersModel(
        {inputs: [inputTensor1, inputTensor2], outputs: [output1, output2]});
    inputs1 = ones([numSamples, inputSize1]);
    inputs2 = ones([numSamples, inputSize2]);
    targets1 = ones([numSamples, 1]);
    targets2 = ones([numSamples, 1]);
  }

  it('1 input, 1 output, dense, 1 weight, string optimizer, 1 batch',
     async () => {
       createDenseModelAndData();

       model.compile({optimizer: 'SGD', loss: 'meanSquaredError'});
       // Use batchSize === numSamples to get exactly one batch.
       const history =
           await model.fit(inputs, targets, {batchSize: numSamples, epochs: 1});

       expect(history.epoch).toEqual([0]);
       const newWeightsValue = model.trainableWeights[0].read();

       const lr = 0.01;  // This is the default learning rate of SGD.
       const expectedValueArray =
           pyListRepeat([1.0 - (inputSize - 1) * 2 * lr], inputSize);
       expectTensorsClose(
           newWeightsValue, tensor2d(expectedValueArray, [inputSize, 1]));
     });

  it('1D tensor as inputs, targets and validationData', async () => {
    const model = tfl.sequential();
    model.add(tfl.layers.dense({units: 1, inputShape: [1]}));
    model.compile({optimizer: 'SGD', loss: 'meanSquaredError'});

    // Use 1D tensor shapes.
    inputs = ones([numSamples]);
    targets = ones([numSamples]);
    const valInputs = ones([numSamples]);
    const valTargets = ones([numSamples]);

    // Do a burn-in run before checking memory to give any cached
    // tensors a chance to be created first.
    await model.fit(inputs, targets, {
      batchSize: numSamples,
      epochs: 2,
      validationData: [valInputs, valTargets]
    });

    for (let i = 0; i < 2; ++i) {
      const numTensors0 = memory().numTensors;
      const history = await model.fit(inputs, targets, {
        batchSize: numSamples,
        epochs: 2,
        validationData: [valInputs, valTargets]
      });
      expect(memory().numTensors).toEqual(numTensors0);
      // Assert no memory leak.
      expect(history.epoch).toEqual([0, 1]);
      expect(history.history.loss.length).toEqual(2);
      expect(history.history.val_loss.length).toEqual(2);
    }
  });

  it('training with cosineProximity loss', async () => {
    createDenseCategoricalModelAndData();
    model.compile({optimizer: 'SGD', loss: 'cosineProximity'});
    // Use batchSize === numSamples to get exactly one batch.
    const history = await model.fit(
        inputs, targets,
        {batchSize: numSamples, epochs: 2, validationSplit: 0.2});
    expect(history.epoch).toEqual([0, 1]);
    expect(history.history.loss.length).toEqual(2);
    expect(history.history.val_loss.length).toEqual(2);
    test_util.expectArraysClose(
        history.history['loss'] as number[],
        [-0.70710688829422, -0.7077317237854004]);
    test_util.expectArraysClose(
        history.history['val_loss'] as number[],
        [-0.70710688829422, -0.7077317237854004]);
  });

  it('training with custom loss', async () => {
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
    const history = await model.fit(
        inputs, targets,
        {batchSize: numSamples, epochs: 2, validationSplit: 0.2});
    test_util.expectArraysClose(history.history['loss'] as number[], [3, 2.96]);
    test_util.expectArraysClose(
        history.history['val_loss'] as number[], [2.96, 2.92]);
  });

  it('Using only x and y input arguments', async () => {
    createDenseModelAndData();

    model.compile({optimizer: 'SGD', loss: 'meanSquaredError'});
    const history = await model.fit(inputs, targets, {epochs: 10});
    // 100 is the default number of epochs.
    expect(history.epoch.length).toEqual(10);
    for (let i = 0; i < 10; ++i) {
      expect(history.epoch[i]).toEqual(i);
    }
  });

  it('Default LayersModel.fit epochs is 1', async () => {
    createDenseModelAndData();

    model.compile({optimizer: 'SGD', loss: 'meanSquaredError'});
    const history = await model.fit(inputs, targets);
    expect(history.epoch.length).toEqual(1);
    expect(history.epoch[0]).toEqual(0);
  });

  it('1 input, 1 output, dense, 1 weight, string optimizer, 2 epochs',
     async () => {
       createDenseModelAndData();

       model.compile({optimizer: 'SGD', loss: 'meanSquaredError'});
       const history =
           await model.fit(inputs, targets, {batchSize: numSamples, epochs: 2});
       expect(history.epoch).toEqual([0, 1]);
     });

  it('1 input, 1 output, dense, 1 weight, string optimizer, 2 epochs, ' +
         '1 initialEpoch',
     async () => {
       createDenseModelAndData();

       model.compile({optimizer: 'SGD', loss: 'meanSquaredError'});
       const history = await model.fit(
           inputs, targets,
           {batchSize: numSamples, epochs: 2, initialEpoch: 1});
       expect(history.epoch).toEqual([1]);
       expect(history.history.loss.length).toEqual(1);
     });

  it('Training with Dropout layer', async () => {
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
    const model = new tfl.LayersModel({inputs: input, outputs: output});
    model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});
    const x = ones([batchSize, inputSize]);
    const y = ones([batchSize, 1]);
    await model.fit(x, y, {batchSize, epochs: 1});
  });

  it('Calling fit twice in a row leads to Error', async () => {
    createDenseModelAndData();
    model.compile({optimizer: 'SGD', loss: 'meanSquaredError'});
    // Do not use `await` in the following `model.fit` call, so that
    // the two model.fit() calls may interleave.
    const firstFit =
        model.fit(inputs, targets, {batchSize: numSamples, epochs: 8});
    let errorCaught: Error;
    try {
      await model.fit(inputs, targets);
    } catch (err) {
      errorCaught = err;
    }
    expect(errorCaught.message)
        .toEqual(
            'Cannot start training because another fit() call is ongoing.');
    await firstFit;
  });

  const validationSplits = [0.2, 0.01];
  for (const validationSplit of validationSplits) {
    const testTitle =
        '1 input, 1 output, dense, 1 weight, string optimizer, 2 epochs, ' +
        `validationSplit=${validationSplit}`;
    it(testTitle, async () => {
      createDenseModelAndData();
      model.compile({optimizer: 'SGD', loss: 'meanSquaredError'});
      const history = await model.fit(
          inputs, targets, {batchSize: numSamples, epochs: 2, validationSplit});
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
    });
  }

  it('1 input, 1 output, dense, 1 weight, string optimizer, 2 epochs, ' +
         'use validationData',
     async () => {
       createDenseModelAndData();
       model.compile({optimizer: 'SGD', loss: 'meanSquaredError'});
       const history = await model.fit(inputs, targets, {
         batchSize: numSamples,
         epochs: 2,
         validationData: [zeros(inputs.shape as [number, number]), targets]
       });
       expect(history.epoch).toEqual([0, 1]);
       const losses = history.history['loss'];
       expect(losses.length).toEqual(2);
       const valLosses = history.history['val_loss'];
       expect(valLosses.length).toEqual(2);
       expectTensorsClose(losses as number[], [9, 7.617599964141846]);
     });

  it('1 input, 1 output, dense, 1 weight, string optimizer, 2 epochs, ' +
         'validationSplit = 0.2, with additional metric',
     async () => {
       createDenseModelAndData();
       model.compile(
           {optimizer: 'SGD', loss: 'meanSquaredError', metrics: ['accuracy']});
       expect(model.metricsNames).toEqual(['loss', 'acc']);
       const history = await model.fit(inputs, targets, {
         batchSize: numSamples,
         epochs: 2,
         validationSplit: 0.2,
       });

       expect(history.epoch).toEqual([0, 1]);
       const losses = history.history['loss'];
       expect(losses.length).toEqual(2);
       const valLosses = history.history['val_loss'];
       expect(valLosses.length).toEqual(2);
       expectTensorsClose(losses as number[], [9, 7.617599964141846]);
       expectTensorsClose(
           valLosses as number[], [7.617599964141846, 6.447536945343018]);
     });

  it('Return sequences; Fit with metric', async () => {
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
    const model = new tfl.LayersModel({inputs: input, outputs: output});
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
  });

  // TODO(cais): Test metric as a "dict", for models with >1 outputs.

  const metricsToTest: string[][] = [['acc'], ['accuracy']];
  // TODO(cais): Add 'acc', 'accuracy' and assertion acc_1, acc_2.
  for (const metrics of metricsToTest) {
    const testTitle = `categoricalCrossentropy model, validationSplit = 0.2, ` +
        `${JSON.stringify(metrics)}`;
    it(testTitle, async () => {
      createDenseCategoricalModelAndData();
      model.compile(
          {optimizer: 'SGD', loss: 'categoricalCrossentropy', metrics});
      if (stringsEqual(metrics, ['acc']) ||
          stringsEqual(metrics, ['accuracy'])) {
        expect(model.metricsNames).toEqual(['loss', 'acc']);
      } else if (stringsEqual(metrics, ['acc', 'accuracy'])) {
        expect(model.metricsNames).toEqual(['loss', 'acc', 'acc']);
      }
      const history = await model.fit(
          inputs, targets,
          {batchSize: numSamples, epochs: 2, validationSplit: 0.2});
      const losses = history.history['loss'];
      expectTensorsClose(
          losses as number[], [0.6931471824645996, 0.6918979287147522]);
      const valLosses = history.history['val_loss'];
      expectTensorsClose(
          valLosses as number[], [0.6918979287147522, 0.6906517744064331]);
      const acc = history.history['acc'];
      expectTensorsClose(acc as number[], [0, 1]);
      const valAcc = history.history['val_acc'];
      expectTensorsClose(valAcc as number[], [1, 1]);
    });
  }

  it('categoricalCrossentropy model, validationSplit = 0.2, ' +
         'tf.metrics.meanSquareError metric function',
     async () => {
       createDenseCategoricalModelAndData();
       model.compile({
         optimizer: 'SGD',
         loss: 'categoricalCrossentropy',
         metrics: tfl.metrics.meanSquaredError
       });
       expect(model.metricsNames).toEqual(['loss', 'meanSquaredError']);
       const history = await model.fit(
           inputs, targets,
           {batchSize: numSamples, epochs: 2, validationSplit: 0.2});
       const losses = history.history['loss'];
       expectTensorsClose(
           losses as number[], [0.6931471824645996, 0.6918979287147522]);
       const meanSquareError = history.history['meanSquaredError'];
       expectTensorsClose(
           meanSquareError as number[], [12.5, 12.495024681091309]);
     });

  it('categoricalCrossentropy model, validationSplit = 0.2, ' +
         'tf.metrics.meanSquareError metric function && acc',
     async () => {
       createDenseCategoricalModelAndData();
       model.compile({
         optimizer: 'SGD',
         loss: 'categoricalCrossentropy',
         metrics: [tfl.metrics.meanSquaredError, 'acc']
       });
       expect(model.metricsNames).toEqual(['loss', 'meanSquaredError', 'acc']);
       const history = await model.fit(
           inputs, targets,
           {batchSize: numSamples, epochs: 2, validationSplit: 0.2});
       const losses = history.history['loss'];
       expectTensorsClose(
           losses as number[], [0.6931471824645996, 0.6918979287147522]);
       const meanSquareError = history.history['meanSquaredError'];
       expectTensorsClose(
           meanSquareError as number[], [12.5, 12.495024681091309]);
       const acc = history.history['acc'];
       expectTensorsClose(acc as number[], [0, 1]);
     });

  it('Two layers, freeze one layer', async () => {
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
  });

  it('Setting trainable of layer from fit callback', async () => {
    const model = tfl.sequential();
    model.add(tfl.layers.dense({
      units: 3,
      activation: 'relu',
      inputShape: [4],
      kernelInitializer: 'ones'
    }));
    model.add(tfl.layers.dense({units: 1, kernelInitializer: 'ones'}));
    model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});

    expect(model.trainableWeights.length).toEqual(4);
    const xs = tfc.ones([5, 4]);
    const ys = tfc.ones([5, 1]);
    const layer1KernelValues: Float32Array[] = [];
    const layer2KernelValues: Float32Array[] = [];
    await model.fit(xs, ys, {
      epochs: 3,
      callbacks: {
        onEpochEnd: async (epoch, logs) => {
          layer1KernelValues.push(
              model.layers[0].getWeights()[0].dataSync() as Float32Array);
          layer2KernelValues.push(
              model.layers[1].getWeights()[0].dataSync() as Float32Array);
          // Freeze the first dense layer after the 2nd epoch and unfreeze it
          // after the 3rd epoch.
          if (epoch === 1) {
            model.layers[0].trainable = false;
          } else if (epoch === 2) {
            model.layers[0].trainable = true;
          }
          if (epoch > 0) {
            // The 2nd dense layer is never frozen. So its kernel should
            // be updated in every training epoch.
            // TODO(cais): Use `expectArraysNotClose()` when available.
            expect(tensor1d(layer2KernelValues[epoch])
                       .sub(tensor1d(layer2KernelValues[epoch - 1]))
                       .abs()
                       .max()
                       .dataSync()[0])
                .toBeGreaterThan(0);
          }
          // The 1st dense layer is frozen after the 2nd epoch (epoch === 1),
          // and is then unfrozen after the 3rd (epoch === 2).
          // So its kernel value should not change between epoch === 1 and epoch
          // === 2.
          if (epoch === 2) {
            expect(tensor1d(layer1KernelValues[epoch])
                       .sub(tensor1d(layer1KernelValues[epoch - 1]))
                       .abs()
                       .max()
                       .dataSync()[0])
                .toEqual(0);
          } else if (epoch > 0) {
            expect(tensor1d(layer1KernelValues[epoch])
                       .sub(tensor1d(layer1KernelValues[epoch - 1]))
                       .abs()
                       .max()
                       .dataSync()[0])
                .toBeGreaterThan(0);
          }
        }
      }
    });
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
     async () => {
       createDenseModelAndData(true);

       model.compile({optimizer: 'SGD', loss: 'meanSquaredError'});
       const history =
           await model.fit(inputs, targets, {batchSize: numSamples, epochs: 1});
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
     });

  it('1 input, 1 output, dense, 1 weight, optimizer object, 1 batch',
     async () => {
       createDenseModelAndData();

       // Use a custom learning rate for SGD.
       const lr = 0.025;
       model.compile(
           {optimizer: new SGDOptimizer(lr), loss: 'meanSquaredError'});
       const history =
           await model.fit(inputs, targets, {batchSize: numSamples, epochs: 1});
       expect(history.epoch).toEqual([0]);
       const newWeightsValue = model.trainableWeights[0].read();

       const expectedValueArray =
           pyListRepeat([1.0 - (inputSize - 1) * 2 * lr], inputSize);
       expectTensorsClose(
           newWeightsValue, tensor2d(expectedValueArray, [inputSize, 1]));
     });

  // Reference Python code:
  // ```py
  // import keras
  // import numpy as np
  //
  // input1 = keras.Input(shape=[2])
  // output1 = keras.layers.Dense(1,
  //                             kernel_initializer='ones',
  //                             use_bias=False)(input1)
  // output2 = keras.layers.Dense(1,
  //                             kernel_initializer='ones',
  //                             use_bias=False)(input1)
  // model = keras.Model(input1, [output1, output2])
  //
  // model.compile(loss={model.output_names[0]: 'mean_squared_error',
  //                     model.output_names[1]: 'mean_absolute_error'},
  //               optimizer='sgd')
  //
  // xs = np.ones([2, 2])
  // ys1 = np.zeros([2, 1])
  // ys2 = np.zeros([2, 1])
  // history = model.fit(xs, [ys1, ys2], epochs=1)
  // print(history.history)
  // ```
  it('2 outputs, losses by output name', async () => {
    const input1 = tfl.input({shape: [2]});
    const output1 =
        tfl.layers.dense({units: 1, kernelInitializer: 'ones', useBias: false})
            .apply(input1) as SymbolicTensor;
    const output2 =
        tfl.layers.dense({units: 1, kernelInitializer: 'ones', useBias: false})
            .apply(input1) as SymbolicTensor;
    const model = tfl.model({inputs: input1, outputs: [output1, output2]});
    const loss: {[outputName: string]: string} = {};
    loss[model.outputNames[0]] = 'meanSquaredError';
    loss[model.outputNames[1]] = 'meanAbsoluteError';
    model.compile({loss, optimizer: 'sgd'});

    const xs = ones([2, 2]);
    const ys1 = zeros([2, 1]);
    const ys2 = zeros([2, 1]);
    const history = await model.fit(xs, [ys1, ys2], {epochs: 1});
    expect(history.history.loss[0]).toBeCloseTo(6);
    expect(history.history[`${model.outputNames[0]}_loss`][0]).toBeCloseTo(4);
    expect(history.history[`${model.outputNames[1]}_loss`][0]).toBeCloseTo(2);
  });

  it('2 inputs, 2 outputs, dense, optimizer object, 1 batch', async () => {
    createDenseModelWithTwoOutputsAndData();

    const lr = 0.01;
    twoOutputModel.compile({
      optimizer: new SGDOptimizer(lr),
      loss: ['meanSquaredError', 'meanSquaredError']
    });
    const trainableWeights = twoOutputModel.trainableWeights;
    let newWeightsValue1 = trainableWeights[0].read();
    let newWeightsValue2 = trainableWeights[1].read();
    await twoOutputModel.fit(
        [inputs1, inputs2], [targets1, targets2],
        {batchSize: numSamples, epochs: 1});

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
  });

  const isCustomCallbackArgs = [false, true];
  const isCustomCallbackArray = [false, true];
  for (const isArgs of isCustomCallbackArgs) {
    for (const isArray of isCustomCallbackArray) {
      const testTitle = `Fit with custom callback object: isConfig=${
          isArgs}, isArray=${isArray}`;
      it(testTitle, async () => {
        createDenseModelAndData();
        const trainBeginLogs: Logs[] = [];
        const trainEndLogs: Logs[] = [];
        const epochBeginEpochs: number[] = [];
        const epochEndEpochs: number[] = [];
        const batchBeginBatches: number[] = [];
        const batchEndBatches: number[] = [];
        const batchEndLosses: number[] = [];
        const epochEndLosses: number[] = [];
        const customCallbackArgs: CustomCallbackArgs = {
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
        const customCallback = isArgs ? customCallbackArgs :
                                        new CustomCallback(customCallbackArgs);
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
      });
    }
  }

  it('Using custom regularizer', async () => {
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
    const history =
        await model.fit(inputs, targets, {batchSize: numSamples, epochs: 2});
    expectTensorsClose(
        model.layers[1].getWeights()[0],
        tensor2d([0.829, 0.829, 0.829, 0.829], [4, 1]));
    expect(history.history.loss.length).toEqual(2);
    expect(history.history.loss[0]).toBeCloseTo(17);
    expect(history.history.loss[1]).toBeCloseTo(13.92);
  });

  it('Using string regularizer', async () => {
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
    const history =
        await model.fit(inputs, targets, {batchSize: numSamples, epochs: 2});
    expectTensorsClose(
        model.layers[1].getWeights()[0],
        tensor2d([0.884, 0.884, 0.884, 0.884], [4, 1]));
    expect(history.history.loss.length).toEqual(2);
    expect(history.history.loss[0]).toBeCloseTo(9.08);
    expect(history.history.loss[1]).toBeCloseTo(7.68);
  });

  it('and then set weights to new weights', async () => {
    createDenseModelAndData(false, 'l1l2');
    model.compile({optimizer: 'SGD', loss: 'meanSquaredError'});
    await model.fit(inputs, targets, {batchSize: numSamples, epochs: 2});
    const w = zeros([4, 1]);
    model.layers[1].setWeights([w]);
    expectTensorsClose(model.layers[1].getWeights()[0], w);
  });

  it('and then set weights to own weights', async () => {
    createDenseModelAndData(false, 'l1l2');
    model.compile({optimizer: 'SGD', loss: 'meanSquaredError'});
    await model.fit(inputs, targets, {batchSize: numSamples, epochs: 2});
    const w = model.layers[1].getWeights()[0];
    model.layers[1].setWeights([w]);
    expectTensorsClose(model.layers[1].getWeights()[0], w);
  });

  class CustomCallbackForTest extends tfl.CustomCallback {
    constructor(readonly recordedParams: Params[]) {
      super({
        onTrainBegin: async () => {
          recordedParams.push(this.params);
        }
      });
    }
  }

  it('Custom callback params: no validation', async () => {
    createDenseModelAndData();
    model.compile({optimizer: 'SGD', loss: 'meanSquaredError'});
    const recordedParams: Params[] = [];
    const epochs = 3;
    const batchSize = 2;
    await model.fit(inputs, targets, {
      epochs,
      batchSize,
      callbacks: new CustomCallbackForTest(recordedParams)
    });
    expect(recordedParams[0].epochs).toEqual(epochs);
    expect(recordedParams[0].initialEpoch).toEqual(0);
    expect(recordedParams[0].samples).toEqual(inputs.shape[0]);
    expect(recordedParams[0].steps).toEqual(null);
    expect(recordedParams[0].batchSize).toEqual(batchSize);
    expect(recordedParams[0].doValidation).toEqual(false);
    expect(recordedParams[0].metrics).toEqual(['loss']);
  });

  it('Custom callback params: has validation', async () => {
    createDenseModelAndData();
    model.compile({optimizer: 'SGD', loss: 'meanSquaredError'});
    const recordedParams: Params[] = [];
    const epochs = 3;
    const batchSize = 2;
    const validationSplit = 0.2;
    await model.fit(inputs, targets, {
      epochs,
      batchSize,
      validationSplit,
      callbacks: new CustomCallbackForTest(recordedParams)
    });
    expect(recordedParams[0].epochs).toEqual(epochs);
    expect(recordedParams[0].initialEpoch).toEqual(0);
    expect(recordedParams[0].samples)
        .toEqual(Math.round(inputs.shape[0] * (1 - validationSplit)));
    expect(recordedParams[0].steps).toEqual(null);
    expect(recordedParams[0].batchSize).toEqual(batchSize);
    expect(recordedParams[0].doValidation).toEqual(true);
    expect(recordedParams[0].metrics).toEqual(['loss', 'val_loss']);
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

  it('Stop training at the end of an epoch: Functional model', async () => {
    createDenseModelAndData(true);
    model.compile({optimizer: 'SGD', loss: 'meanSquaredError'});
    // Order 10 epochs of training, but the training should stop after two
    // epochs due to the callback.
    const history = await model.fit(inputs, targets, {
      batchSize: numSamples,
      epochs: 10,
      callbacks: [new StopAfterNEpochs(2)]
    });
    expect(history.history.loss.length).toEqual(2);
  });

  class StopAfterNBatches extends tfl.Callback {
    private readonly batchesToTrain: number;
    constructor(batchesToTrain: number) {
      super();
      this.batchesToTrain = batchesToTrain;
    }

    async onBatchEnd(batch: number, logs?: Logs) {
      if (batch === this.batchesToTrain - 1) {
        this.model.stopTraining = true;
      }
    }
  }

  it('Stop training at the end of a batch: Sequential model', async () => {
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
    const history = await sequentialModel.fit(
        inputs, targets,
        {batchSize: 1, epochs: 10, callbacks: [new StopAfterNBatches(2)]});
    expect(history.history.loss.length).toEqual(1);
  });

  it('Stop LayersModel.fit() using non-class object callback function',
     async () => {
       createDenseModelAndData();

       model.compile({optimizer: 'SGD', loss: 'meanSquaredError'});

       let numEpochsDone = 0;
       const epochs = 8;
       const stopAfterEpoch = 3;
       let history = await model.fit(inputs, targets, {
         epochs,
         callbacks: {
           onEpochEnd: async (epoch: number, logs?: UnresolvedLogs) => {
             numEpochsDone++;
             if (epoch === stopAfterEpoch) {
               model.stopTraining = true;
             }
           }
         }
       });
       expect(numEpochsDone).toEqual(stopAfterEpoch + 1);
       expect(history.history.loss.length).toEqual(stopAfterEpoch + 1);

       // Check that model.fit can still be called after force stopping.
       history = await model.fit(inputs, targets, {epochs: 2});
       expect(history.history.loss.length).toEqual(2);
     });

  it('Stop training resets at start of LayersModel.fit()', async () => {
    const sequentialModel = tfl.sequential();
    sequentialModel.add(tfl.layers.dense(
        {units: 1, kernelInitializer: 'ones', inputShape: [inputSize]}));
    // numSamples is 5.
    inputs = ones([numSamples, inputSize]);
    targets = ones([numSamples, 1]);
    sequentialModel.compile({optimizer: 'SGD', loss: 'meanSquaredError'});
    // Order 10 epochs of training, but the training should stop after only one
    // epochs due to the callback that orders the training to stop after two
    // batches. The first epoch should have five batches due to a batchSize
    // of 1.
    let history = await sequentialModel.fit(
        inputs, targets,
        {batchSize: 1, epochs: 10, callbacks: [new StopAfterNBatches(2)]});
    expect(history.history.loss.length).toEqual(1);

    // Running fit again should now run to completion
    history =
        await sequentialModel.fit(inputs, targets, {batchSize: 1, epochs: 10});
    expect(history.history.loss.length).toEqual(10);
  });

  it('Model.dispose() cleans up owned optimizer: Functional', async () => {
    const input1 = tfl.input({shape: [2]});
    const input2 = tfl.input({shape: [2]});
    const y1 = tfl.layers.add().apply([input1, input2]);
    const y2 = tfl.layers.concatenate().apply([input1, input2]);
    const output1 =
        tfl.layers
            .dense({units: 1, activation: 'linear', kernelInitializer: 'zeros'})
            .apply(y1) as tfl.SymbolicTensor;
    const output2 =
        tfl.layers
            .dense(
                {units: 1, activation: 'sigmoid', kernelInitializer: 'zeros'})
            .apply(y2) as tfl.SymbolicTensor;
    const model =
        tfl.model({inputs: [input1, input2], outputs: [output1, output2]});
    model.compile(
        {loss: ['meanSquaredError', 'binaryCrossentropy'], optimizer: 'adam'});

    const xs: Tensor[] = [zeros([4, 2]), zeros([4, 2])];
    const ys: Tensor[] = [zeros([4, 1]), zeros([4, 1])];

    await model.fit(xs, ys, {epochs: 1});
    const numTensors1 = memory().numTensors;

    const disposalResult = model.dispose();
    const numTensors2 = memory().numTensors;
    // The optimizerNumGlobalVariables comes from the intrinsic weights of the
    // ADAM optimizer, e.g., accBeta1, accBeta2.
    const optimizerNumGlobalVariables = 2;
    // The optimizerNumVariablesPerWeight comes from accumulatedFirstMoment and
    // accumulatedSecondMoment, which are computed per model weight.
    const optimizerNumVariablesPerWeight = 2;
    const numModelWeights = 4;
    expect(disposalResult.numDisposedVariables)
        .toEqual(
            numModelWeights + optimizerNumGlobalVariables +
            numModelWeights * optimizerNumVariablesPerWeight);
    expect(disposalResult.refCountAfterDispose).toEqual(0);
    expect(numTensors1 - numTensors2)
        .toEqual(
            numModelWeights + optimizerNumGlobalVariables +
            numModelWeights * optimizerNumVariablesPerWeight);
  });

  it('Model.dispose() cleans up owned optimizer: Sequential', async () => {
    const model = tfl.sequential();
    model.add(tfl.layers.dense({units: 1, inputShape: [2]}));
    model.compile({loss: 'meanSquaredError', optimizer: 'adam'});

    const xs = zeros([4, 2]);
    const ys = zeros([4, 1]);

    await model.fit(xs, ys, {epochs: 1});
    const numTensors1 = memory().numTensors;

    const disposalResult = model.dispose();
    const numTensors2 = memory().numTensors;

    // The optimizerNumGlobalVariables comes from the intrinsic weights of the
    // ADAM optimizer, e.g., accBeta1, accBeta2.
    const optimizerNumGlobalVariables = 2;
    // The optimizerNumVariablesPerWeight comes from accumulatedFirstMoment and
    // accumulatedSecondMoment, which are computed per model weight.
    const optimizerNumVariablesPerWeight = 2;
    const numModelWeights = 2;
    expect(disposalResult.numDisposedVariables)
        .toEqual(
            numModelWeights + optimizerNumGlobalVariables +
            numModelWeights * optimizerNumVariablesPerWeight);
    expect(disposalResult.refCountAfterDispose).toEqual(0);
    expect(numTensors1 - numTensors2)
        .toEqual(
            numModelWeights + optimizerNumGlobalVariables +
            numModelWeights * optimizerNumVariablesPerWeight);
  });

  it('Model.dispose() skips non-owned optimizer: Functional', async () => {
    const input1 = tfl.input({shape: [2]});
    const input2 = tfl.input({shape: [2]});
    const y1 = tfl.layers.add().apply([input1, input2]);
    const y2 = tfl.layers.concatenate().apply([input1, input2]);
    const output1 =
        tfl.layers
            .dense({units: 1, activation: 'linear', kernelInitializer: 'zeros'})
            .apply(y1) as tfl.SymbolicTensor;
    const output2 =
        tfl.layers
            .dense(
                {units: 1, activation: 'sigmoid', kernelInitializer: 'zeros'})
            .apply(y2) as tfl.SymbolicTensor;
    const model =
        tfl.model({inputs: [input1, input2], outputs: [output1, output2]});
    const optimizer = new tfc.AdamOptimizer(1e-3, 0.9, 0.999, 1e-6);
    model.compile(
        {loss: ['meanSquaredError', 'binaryCrossentropy'], optimizer});

    const xs: Tensor[] = [zeros([4, 2]), zeros([4, 2])];
    const ys: Tensor[] = [zeros([4, 1]), zeros([4, 1])];

    await model.fit(xs, ys, {epochs: 1});
    const numTensors1 = memory().numTensors;

    const disposalResult = model.dispose();
    const numTensors2 = memory().numTensors;
    // Only the weights of the model (not including the optimizer) should have
    // been disposed.
    expect(disposalResult.numDisposedVariables).toEqual(4);
    expect(disposalResult.refCountAfterDispose).toEqual(0);
    expect(numTensors1 - numTensors2).toEqual(4);
  });

  it('Model.dispose() skips non-owned optimizer: Sequential', async () => {
    const model = tfl.sequential();
    model.add(tfl.layers.dense({units: 1, inputShape: [2]}));
    const optimizer = new tfc.AdamOptimizer(1e-3, 0.9, 0.999, 1e-6);
    model.compile({loss: 'meanSquaredError', optimizer});

    const xs = zeros([4, 2]);
    const ys = zeros([4, 1]);

    await model.fit(xs, ys, {epochs: 1});
    const numTensors1 = memory().numTensors;

    const disposalResult = model.dispose();
    const numTensors2 = memory().numTensors;

    // Only the Model's own weights should have been disposed.
    expect(disposalResult.numDisposedVariables).toEqual(2);
    expect(disposalResult.refCountAfterDispose).toEqual(0);
    expect(numTensors1 - numTensors2).toEqual(2);
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

  it('Invalid batchSize leads to Error', async () => {
    createDenseModelAndData();
    const badBatchSizeValues: Array<number|string> = [0, -1, 3.14, 'a'];
    for (const batchSize of badBatchSizeValues) {
      let errorCaught: Error;
      try {
        // tslint:disable-next-line:no-any
        await model.fit(inputs, targets, {batchSize: batchSize as any});
      } catch (err) {
        errorCaught = err;
      }
      expect(errorCaught.message)
          .toEqual(`batchSize is required to be a positive integer, but got ${
              batchSize}`);
    }
  });
});

describeMathCPUAndGPU('LayersModel.fit with training-sensitive layers', () => {
  it('Correct training arg during fit/evaluate/predict', async () => {
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
    const model =
        new tfl.LayersModel({inputs: [inputTensor], outputs: [output]});
    model.compile({optimizer: 'sgd', loss: 'meanSquaredError'});
    const xs = ones([4, 1]);
    const ys = ones([4, 1]);

    // 1. Call fit: Dropout layer should be called twice, with training as
    // true.
    await model.fit(xs, ys, {epochs: 2, batchSize: 4});
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
  });
});

describeMathCPUAndGPU(
    'LayersModel.predict and LayersModel.evaluate: No memory leak', () => {
      const inputSize = 4;  // Input vector size for model with one input.

      const inputTensor = tfl.layers.input(
          {shape: [inputSize], name: 'inputLayer1', dtype: 'float32'});
      let model: tfl.LayersModel;
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
        model = new tfl.LayersModel({inputs: [inputTensor], outputs: [output]});
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

describeMathCPUAndGPU('LayersModel.fit: No memory leak', () => {
  const inputSize = 4;   // Input vector size for model with one input.
  const numSamples = 5;  // Number of samples in a batch.

  const inputTensor = tfl.layers.input(
      {shape: [inputSize], name: 'inputLayer1', dtype: 'float32'});
  let model: tfl.LayersModel;
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
    model = new tfl.LayersModel({inputs: [inputTensor], outputs: [output]});
    inputs = ones([numSamples, inputSize]);
    targets = ones([numSamples, 1]);
    valInputs = zeros([numSamples, inputSize]);
    valTargets = zeros([numSamples, 1]);
  }

  it('Repeated fit calls leads to no memory leak: no validation or metrics',
     async (done) => {
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
           // Get the tensor counts within an epoch (i.e., from the first
           // batch till the penultimate one.) Assert that the counts are
           // constant, i.e., no increase in the tensor count within the
           // epoch. N.B.: Even though the tensor count is expected to be
           // constant across batches, across epochs, the count will increase,
           // due to the per-epoch loss and metric values stored for the
           // returned history object, which are currently downloaded via
           // data() calls only at the end of the fit() call.
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

describeMathGPU('LayersModel.fit: yieldEvery', () => {
  function createDummyModel(inputSize: number): tfl.LayersModel {
    const model = tfl.sequential();
    const layerSize = 10;
    model.add(tfl.layers.dense(
        {units: layerSize, inputShape: [inputSize], activation: 'relu'}));
    model.add(tfl.layers.dense({units: 1}));
    model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});
    return model;
  }

  it('auto: 1 batch per epoch; 5 epochs', async () => {
    const wait = DEFAULT_YIELD_EVERY_MS;
    const timeBetweenCalls = [
      0,
      1,
      wait + 1,  // Should call.
      wait + 1,  // Should call.
      1,
      1,
    ];
    let counter = 0;
    let prevTime = 0;
    spyOn(util, 'now').and.callFake(() => {
      prevTime += timeBetweenCalls[counter++];
      return prevTime;
    });
    let nextFrameCallCount = 0;
    spyOn(tfc, 'nextFrame').and.callFake(async () => {
      nextFrameCallCount++;
    });

    const inputSize = 2;
    const numExamples = 10;
    const epochs = 5;
    const model = createDummyModel(inputSize);
    const xs = ones([numExamples, inputSize]);
    const ys = ones([numExamples, 1]);
    const onYieldEpochIds: number[] = [];
    const onYieldBatchesIds: number[] = [];
    const history = await model.fit(xs, ys, {
      epochs,
      batchSize: numExamples,
      callbacks: {
        onYield: async (epoch, batch, _logs) => {
          onYieldBatchesIds.push(batch);
          onYieldEpochIds.push(epoch);
        }
      }
    });
    expect(history.history.loss.length).toEqual(epochs);
    // There are 5 batches in total (1 batch per epoch). We expect next frame
    // to be called twice, after epoch 1 and after epoch 2.
    expect(nextFrameCallCount).toBe(2);
    expect(onYieldEpochIds).toEqual([1, 2]);
    expect(onYieldBatchesIds).toEqual([0, 0]);
  });

  it('auto: 2 batches per epoch; 4 epochs', async () => {
    const yieldEvery = DEFAULT_YIELD_EVERY_MS;
    const timeBetweenCalls = [
      0,
      1,
      yieldEvery + 1,  // Should call.
      yieldEvery + 1,  // Should call.
      1,
      yieldEvery + 1,  // SHould call.
      yieldEvery + 1,  // Should call.
      1,
      1,
    ];
    let counter = 0;
    let prevTime = 0;
    spyOn(util, 'now').and.callFake(() => {
      prevTime += timeBetweenCalls[counter++];
      return prevTime;
    });
    let nextFrameCallCount = 0;
    spyOn(tfc, 'nextFrame').and.callFake(async () => {
      nextFrameCallCount++;
    });

    const inputSize = 2;
    const numExamples = 10;
    const epochs = 4;
    const model = createDummyModel(inputSize);
    const xs = ones([numExamples, inputSize]);
    const ys = ones([numExamples, 1]);
    const onYieldEpochIds: number[] = [];
    const onYieldBatchesIds: number[] = [];
    const history = await model.fit(xs, ys, {
      epochs,
      batchSize: numExamples / 2,
      callbacks: {
        onYield: async (epoch, batch, _logs) => {
          onYieldBatchesIds.push(batch);
          onYieldEpochIds.push(epoch);
        }
      }
    });
    expect(history.history.loss.length).toEqual(epochs);
    // There are 8 batches in total (2 batches per epoch). We expect next
    // frame to be called 3 times, after (epoch 0, batch 1), (epoch 1, batch 0)
    // (epoch 2, batch 0) and (epoch 2, batch 1).
    expect(nextFrameCallCount).toBe(4);
    expect(onYieldEpochIds).toEqual([0, 1, 2, 2]);
    expect(onYieldBatchesIds).toEqual([1, 0, 0, 1]);
  });

  it('yieldEvery: 5, 1 batch per epoch; 5 epochs', async () => {
    const yieldEvery = 5;
    const timeBetweenCalls = [
      0, 1,
      yieldEvery + 1,     // Should call.
      1, yieldEvery + 1,  // Should call.
      yieldEvery + 1,     // Should call.
    ];
    let counter = 0;
    let prevTime = 0;
    spyOn(util, 'now').and.callFake(() => {
      prevTime += timeBetweenCalls[counter++];
      return prevTime;
    });
    let nextFrameCallCount = 0;
    spyOn(tfc, 'nextFrame').and.callFake(async () => {
      nextFrameCallCount++;
    });

    const inputSize = 2;
    const numExamples = 10;
    const epochs = 5;
    const model = createDummyModel(inputSize);
    const xs = ones([numExamples, inputSize]);
    const ys = ones([numExamples, 1]);
    const onYieldEpochIds: number[] = [];
    const onYieldBatchesIds: number[] = [];
    const history = await model.fit(xs, ys, {
      epochs,
      batchSize: numExamples,
      yieldEvery,
      callbacks: {
        onYield: async (epoch, batch, _logs) => {
          onYieldBatchesIds.push(batch);
          onYieldEpochIds.push(epoch);
        }
      }
    });
    expect(history.history.loss.length).toEqual(epochs);
    // There are 5 batches in total (1 batch per epoch). We expect next frame
    // to be called 3 times, after epoch 1, epoch 3 and epoch 4.
    expect(nextFrameCallCount).toBe(3);
    expect(onYieldEpochIds).toEqual([1, 3, 4]);
    expect(onYieldBatchesIds).toEqual([0, 0, 0]);
  });

  it('fails when onYield is provided, but yieldEvery is never', async done => {
    const inputSize = 2;
    const numExamples = 10;
    const epochs = 5;
    const model = createDummyModel(inputSize);
    const xs = ones([numExamples, inputSize]);
    const ys = ones([numExamples, 1]);
    try {
      await model.fit(xs, ys, {
        epochs,
        batchSize: numExamples,
        yieldEvery: 'never',
        callbacks: {onYield: async (_epoch, _batch, _logs) => {}}
      });
      done.fail('Model.fit should fail');
    } catch {
      done();
    }
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

  it('resolveScalarInLogs is not called if no custom callbacks', async () => {
    const inputSize = 1;
    const numExamples = 10;
    const batchSize = 2;
    const epochs = 2;
    const model = createDummyModel(inputSize);
    const xs = ones([numExamples, inputSize]);
    const ys = ones([numExamples, 1]);

    const spy = spyOn(logs, 'resolveScalarsInLogs').and.callThrough();
    await model.fit(xs, ys, {epochs, batchSize, yieldEvery: 'never'});
    expect(spy).not.toHaveBeenCalled();
  });
});

describeMathCPUAndGPU('LayersModel.trainOnBatch', () => {
  // Reference Python Keras code:
  // ```py
  // import keras
  // import numpy as np
  //
  // model = keras.Sequential()
  // model.add(keras.layers.Dense(
  //     1, input_shape=[3], kernel_initializer='zeros'))
  // model.compile(loss='mean_squared_error', optimizer='sgd')
  //
  // batch_size = 4
  // xs = np.ones([batch_size, 3])
  // ys = np.ones([batch_size, 1])
  //
  // for _ in range(3):
  //   loss = model.train_on_batch(xs, ys)
  //   print(loss)
  // ```
  it('Sequential MLP: correctness', async () => {
    const model = tfl.sequential();
    model.add(tfl.layers.dense(
        {units: 1, inputShape: [3], kernelInitializer: 'zeros'}));
    model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});

    const batchSize = 4;
    const xs = tfc.ones([batchSize, 3]);
    const ys = tfc.ones([batchSize, 1]);
    let loss = await model.trainOnBatch(xs, ys) as number;
    expect(loss).toBeCloseTo(1.0);
    loss = await model.trainOnBatch(xs, ys) as number;
    expect(loss).toBeCloseTo(0.8464);
    loss = await model.trainOnBatch(xs, ys) as number;
    expect(loss).toBeCloseTo(0.716393);
  });

  // Reference Python Keras code:
  // ```py
  // import keras
  // import numpy as np
  //
  // input1 = keras.Input(shape=[2])
  // input2 = keras.Input(shape=[2])
  // y1 = keras.layers.Add()([input1, input2])
  // y2 = keras.layers.Concatenate()([input1, input2])
  // output1 = keras.layers.Dense(
  //     units=1,
  //     activation='linear',
  //     kernel_initializer='zeros')(y1)
  // output2 = keras.layers.Dense(
  //     units=1,
  //     activation='sigmoid',
  //     kernel_initializer='zeros')(y2)
  // model = keras.Model(inputs=[input1, input2], outputs=[output1, output2])
  // model.compile(loss=['mean_squared_error', 'binary_crossentropy'],
  //               optimizer='sgd')
  //
  // batch_size = 4
  // xs1 = np.ones([batch_size, 2])
  // xs2 = np.ones([batch_size, 2])
  // ys1 = np.ones([batch_size, 1])
  // ys2 = np.ones([batch_size, 1])
  //
  // for _ in range(3):
  //   losses = model.train_on_batch([xs1, xs2], [ys1, ys2])
  //   print(losses)
  // ```
  it('Functional two inputs and two outputs: correctness', async () => {
    const input1 = tfl.input({shape: [2]});
    const input2 = tfl.input({shape: [2]});
    const y1 = tfl.layers.add().apply([input1, input2]);
    const y2 = tfl.layers.concatenate().apply([input1, input2]);
    const output1 =
        tfl.layers
            .dense({units: 1, activation: 'linear', kernelInitializer: 'zeros'})
            .apply(y1) as tfl.SymbolicTensor;
    const output2 =
        tfl.layers
            .dense(
                {units: 1, activation: 'sigmoid', kernelInitializer: 'zeros'})
            .apply(y2) as tfl.SymbolicTensor;
    const model =
        tfl.model({inputs: [input1, input2], outputs: [output1, output2]});
    model.compile(
        {loss: ['meanSquaredError', 'binaryCrossentropy'], optimizer: 'sgd'});

    const batchSize = 4;
    const xs1 = tfc.ones([batchSize, 2]);
    const xs2 = tfc.ones([batchSize, 2]);
    const ys1 = tfc.ones([batchSize, 1]);
    const ys2 = tfc.ones([batchSize, 1]);
    let losses = await model.trainOnBatch([xs1, xs2], [ys1, ys2]) as number[];
    expect(losses.length).toEqual(3);
    expect(losses[0]).toBeCloseTo(1.6931472);
    expect(losses[1]).toBeCloseTo(1.0);
    expect(losses[2]).toBeCloseTo(0.6931472);
    losses = await model.trainOnBatch([xs1, xs2], [ys1, ys2]) as number[];
    expect(losses.length).toEqual(3);
    expect(losses[0]).toBeCloseTo(1.3531253);
    expect(losses[1]).toBeCloseTo(0.6724);
    expect(losses[2]).toBeCloseTo(0.68072534);
    losses = await model.trainOnBatch([xs1, xs2], [ys1, ys2]) as number[];
    expect(losses.length).toEqual(3);
    expect(losses[0]).toBeCloseTo(1.1207337);
    expect(losses[1]).toBeCloseTo(0.45212176);
    expect(losses[2]).toBeCloseTo(0.66861194);
  });

  // Reference Python Keras code.
  // ```py
  // import keras
  // import numpy as np
  //
  // model = keras.Sequential()
  // model.add(keras.layers.Dense(
  //     3, input_shape=[2], activation='softmax', kernel_initializer='ones'))
  // model.compile(loss='categorical_crossentropy', optimizer='sgd')
  //
  // xs = np.array([[0.5, 0.5], [1, 1], [0, 1]], dtype=np.float32)
  // ys = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
  //
  // for _ in range(3):
  //   loss = model.train_on_batch(xs, ys)
  //   print(loss)
  // ```
  it('Categorical: Correctness and no memory leak', async () => {
    const model = tfl.sequential();
    model.add(tfl.layers.dense({
      units: 3,
      inputShape: [2],
      activation: 'softmax',
      kernelInitializer: 'ones'
    }));
    model.compile({loss: 'categoricalCrossentropy', optimizer: 'sgd'});

    const xs = tfc.tensor2d([[0.5, 0.5], [1, 1], [0, 1]]);
    const ys = tfc.tensor2d([[1, 0, 0], [0, 1, 0], [0, 0, 1]]);
    // Perform burn-in.
    model.trainOnBatch(xs, ys);
    const numTensors0 = memory().numTensors;
    for (let i = 0; i < 3; ++i) {
      const loss = await model.trainOnBatch(xs, ys);
      tfc.tidy(() => {
        if (i === 0) {
          expect(loss).toBeCloseTo(1.0986123);
        } else if (i === 1) {
          expect(loss).toBeCloseTo(1.0978721);
        } else {
          expect(loss).toBeCloseTo(1.0971345);
        }
      });
      // Assert no tensor memory leak.
      expect(memory().numTensors).toBeLessThanOrEqual(numTensors0);
    }
  });

  // Reference Python Keras code:
  // ```py
  // import keras
  // import numpy as np
  //
  // model = keras.Sequential()
  // model.add(keras.layers.Dense(
  //     units=3,
  //     input_shape=[2],
  //     activation='softmax',
  //     kernel_initializer='ones'
  // ))
  // model.compile(
  //     loss='sparse_categorical_crossentropy',
  //     optimizer='sgd',
  //     metrics=['acc'])
  // model.summary()
  //
  // xs = np.array([[0, 0.5], [0.5, 1], [0, 1]], dtype=np.float32)
  // ys = np.array([[2], [1], [0]], dtype=np.float32)
  //
  // for _ in range(3):
  //   print(model.train_on_batch(xs, ys))
  // ```
  it('Sparse categorical: w/ metrics: correctness and no leak', async () => {
    const model = tfl.sequential();
    model.add(tfl.layers.dense({
      units: 3,
      inputShape: [2],
      activation: 'softmax',
      kernelInitializer: 'ones'
    }));
    model.compile({
      loss: 'sparseCategoricalCrossentropy',
      optimizer: 'sgd',
      metrics: ['acc']
    });

    const xs = tfc.tensor2d([[0, 0.5], [0.5, 1], [0, 1]]);
    const ys = tfc.tensor2d([[2], [1], [0]]);

    for (let i = 0; i < 3; ++i) {
      const [loss, acc] = await model.trainOnBatch(xs, ys) as number[];
      if (i === 0) {
        expect(loss).toBeCloseTo(1.0986123);
        expect(acc).toBeCloseTo(0.3333333);
      } else if (i === 1) {
        expect(loss).toBeCloseTo(1.0982422);
        expect(acc).toBeCloseTo(0.6666667);
      } else if (i === 2) {
        expect(loss).toBeCloseTo(1.0978734);
        expect(acc).toBeCloseTo(0.6666667);
      }
    }
  });
});

describeMathCPUAndGPU('LayersModel.evaluate', () => {
  const numExamples = 8;
  const inputSize = 2;
  const outputSize = 1;
  let model: tfl.LayersModel;
  let x: Tensor;
  let y: Tensor;
  function prepModel() {
    const input = tfl.layers.input({shape: [inputSize]});
    const dense = tfl.layers.dense(
        {units: outputSize, kernelInitializer: 'ones', useBias: false});
    const output = dense.apply(input) as tfl.SymbolicTensor;
    model = new tfl.LayersModel({inputs: input, outputs: output});
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

  it('1D tensors as inputs and targets', () => {
    const model = tfl.sequential();
    model.add(tfl.layers.dense({units: 1, inputShape: [1]}));
    model.compile(
        {loss: 'meanSquaredError', optimizer: 'sgd', metrics: ['acc']});

    const xs = ones([10]);   // A batch of 10, as a 1D tensor.
    const ys = zeros([10]);  // A batch of 10, as a 1D tensor.
    // Do a burn-in call first.
    tfc.dispose(model.evaluate(xs, ys, {batchSize: 4}));
    const numTensors0 = memory().numTensors;
    const evalOuts = model.evaluate(xs, ys, {batchSize: 4}) as Tensor[];
    expect(evalOuts.length).toEqual(2);     // Loss and acc.
    expect(evalOuts[0].shape).toEqual([]);  // Loss as a scalar.
    expect(evalOuts[1].shape).toEqual([]);  // Acc as a scalar.
    tfc.dispose(evalOuts);
    // Assert no memory leak.
    expect(memory().numTensors).toEqual(numTensors0);
  });

  it('Invalid batchSize value leads to Error', () => {
    prepModel();
    prepData();
    expect(() => model.evaluate(x, y, {batchSize: 0}))
        .toThrowError(
            /batchSize is required to be a positive integer, but got 0/);
    expect(() => model.evaluate(x, y, {batchSize: -2}))
        .toThrowError(
            /batchSize is required to be a positive integer, but got -2/);
    expect(() => model.evaluate(x, y, {batchSize: 3.14}))
        .toThrowError(
            /batchSize is required to be a positive integer, but got 3\.14/);
    // tslint:disable-next-line:no-any
    expect(() => model.evaluate(x, y, {batchSize: 'a' as any}))
        .toThrowError(
            /batchSize is required to be a positive integer, but got a/);
  });
});

describe('LayersModel trainable setter and getter', () => {
  it('Setting trainable does not affect Layers', () => {
    const model = tfl.sequential({
      layers: [
        tfl.layers.flatten({inputShape: [2, 5]}),
        // Initially non-trainable.
        tfl.layers.dense({units: 3, activation: 'relu', trainable: false}),
        tfl.layers.dense({units: 1}),
      ]
    });

    model.trainable = false;
    expect(model.trainable).toEqual(false);
    // The trainable property of the layers should be unaffected.
    expect(model.layers[0].trainable).toEqual(true);
    expect(model.layers[1].trainable).toEqual(false);
    expect(model.layers[2].trainable).toEqual(true);

    model.trainable = true;
    expect(model.trainable).toEqual(true);
    expect(model.layers[0].trainable).toEqual(true);
    expect(model.layers[1].trainable).toEqual(false);
    expect(model.layers[2].trainable).toEqual(true);
  });

  it('Setting trainable of model sets trainable bit of Variable', async () => {
    const model = tfl.sequential();
    model.add(
        tfl.layers.dense({units: 3, activation: 'relu', inputShape: [4]}));
    model.add(tfl.layers.dense({units: 1, kernelInitializer: 'ones'}));
    model.trainable = false;
    expect(model.layers[0].weights[0].trainable).toEqual(false);
    expect(model.layers[0].weights[1].trainable).toEqual(false);
    expect(model.layers[1].weights[0].trainable).toEqual(false);
    expect(model.layers[1].weights[1].trainable).toEqual(false);
    model.trainable = true;
    expect(model.layers[0].weights[0].trainable).toEqual(true);
    expect(model.layers[0].weights[1].trainable).toEqual(true);
    expect(model.layers[1].weights[0].trainable).toEqual(true);
    expect(model.layers[1].weights[1].trainable).toEqual(true);
  });

  it('model.trainable respects layer.trainable', async () => {
    const model = tfl.sequential();
    model.add(
        tfl.layers.dense({units: 3, activation: 'relu', inputShape: [4]}));
    model.add(tfl.layers.dense({units: 1}));
    expect(model.trainableWeights.length).toEqual(4);
    model.layers[0].trainable = false;
    expect(model.trainableWeights.length).toEqual(2);
    model.trainable = false;
    expect(model.trainableWeights.length).toEqual(0);
    model.trainable = true;
    expect(model.trainableWeights.length).toEqual(2);
    model.layers[0].trainable = true;
    expect(model.trainableWeights.length).toEqual(4);
    model.trainable = false;
    expect(model.trainableWeights.length).toEqual(0);
  });
});

describeMathCPUAndGPU('LayersModel.execute', () => {
  function createFunctionalModel():
      [tfl.LayersModel, {[name: string]: tfl.SymbolicTensor}] {
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
