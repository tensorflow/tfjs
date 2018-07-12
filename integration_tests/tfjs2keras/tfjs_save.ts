/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

import * as tfl from '@tensorflow/tfjs-layers';
import * as fs from 'fs';
import {join} from 'path';

// Multi-layer perceptron (MLP).
function exportMLPModel(exportPath: string): void {
  const model = tfl.sequential();
  // Test both activations encapsulated in other layers and as standalone
  // layers.
  model.add(
      tfl.layers.dense({units: 100, inputShape: [200], activation: 'relu'}));
  model.add(tfl.layers.dense({units: 50, activation: 'elu'}));
  model.add(tfl.layers.dense({units: 24}));
  model.add(tfl.layers.activation({activation: 'elu'}));
  model.add(tfl.layers.dense({units: 8, activation: 'softmax'}));
  fs.writeFileSync(exportPath, model.toJSON());
}

// Convolutional neural network (CNN).
function exportCNNModel(exportPath: string): void {
  const model = tfl.sequential();

  // Cover separable and non-separable convoluational layers.
  model.add(tfl.layers.conv2d({
    filters: 32,
    kernelSize: [3, 3],
    strides: [2, 2],
    inputShape: [40, 40, 3],
    padding: 'valid',
  }));
  model.add(tfl.layers.batchNormalization({}));
  model.add(tfl.layers.activation({activation: 'relu'}));
  model.add(tfl.layers.dropout({rate: 0.5}));
  model.add(tfl.layers.maxPooling2d({poolSize: 2}));
  model.add(tfl.layers.separableConv2d({
    filters: 32,
    kernelSize: [4, 4],
    strides: [3, 3],
  }));
  model.add(tfl.layers.batchNormalization({}));
  model.add(tfl.layers.activation({activation: 'relu'}));
  model.add(tfl.layers.dropout({rate: 0.5}));
  model.add(tfl.layers.avgPooling2d({poolSize: [2, 2]}));
  model.add(tfl.layers.flatten({}));
  model.add(tfl.layers.dense({units: 100, activation: 'softmax'}));
  fs.writeFileSync(exportPath, model.toJSON());
}

function exportDepthwiseCNNModel(exportPath: string): void {
  const model = tfl.sequential();

  // Cover depthwise 2D convoluational layer.
  model.add(tfl.layers.depthwiseConv2d({
    depthMultiplier: 2,
    kernelSize: [3, 3],
    strides: [2, 2],
    inputShape: [40, 40, 3],
    padding: 'valid',
  }));
  model.add(tfl.layers.batchNormalization({}));
  model.add(tfl.layers.activation({activation: 'relu'}));
  model.add(tfl.layers.dropout({rate: 0.5}));
  model.add(tfl.layers.maxPooling2d({poolSize: 2}));
  model.add(tfl.layers.flatten({}));
  model.add(tfl.layers.dense({units: 100, activation: 'softmax'}));
  fs.writeFileSync(exportPath, model.toJSON());
}

// SimpleRNN with embedding.
function exportSimpleRNNModel(exportPath: string): void {
  const model = tfl.sequential();
  model.add(tfl.layers.embedding({inputDim: 100, outputDim: 20}));
  model.add(tfl.layers.simpleRNN({units: 4}));
  fs.writeFileSync(exportPath, model.toJSON());
}

// GRU with embedding.
function exportGRUModel(exportPath: string): void {
  const model = tfl.sequential();
  model.add(tfl.layers.embedding({inputDim: 100, outputDim: 20}));
  model.add(tfl.layers.gru({units: 4, goBackwards: true}));
  fs.writeFileSync(exportPath, model.toJSON());
}

// Bidirecitonal LSTM with embedding.
function exportBidirectionalLSTMModel(exportPath: string): void {
  const model = tfl.sequential();
  model.add(tfl.layers.embedding({inputDim: 100, outputDim: 20}));
  // TODO(cais): Investigate why the `tfl.layers.RNN` typing doesn't work.
  // tslint:disable-next-line:no-any
  const lstm = tfl.layers.lstm({units: 4, goBackwards: true}) as any;
  model.add(tfl.layers.bidirectional({layer: lstm, mergeMode: 'concat'}));
  fs.writeFileSync(exportPath, model.toJSON());
}

// LSTM + time-distributed layer with embedding.
function exportTimeDistributedLSTMModel(exportPath: string): void {
  const model = tfl.sequential();
  model.add(tfl.layers.embedding({inputDim: 100, outputDim: 20}));
  model.add(tfl.layers.lstm({units: 4, returnSequences: true}));
  model.add(tfl.layers.timeDistributed({
    layer:
        tfl.layers.dense({units: 2, useBias: false, activation: 'softmax'})
  }));
  fs.writeFileSync(exportPath, model.toJSON());
}

// Model with Conv1D and Pooling1D layers.
function exportOneDimensionalModel(exportPath: string): void {
  const model = tfl.sequential();
  model.add(tfl.layers.conv1d(
      {filters: 16, kernelSize: [4], inputShape: [80, 1], activation: 'relu'}));
  model.add(tfl.layers.maxPooling1d({poolSize: 3}));
  model.add(
      tfl.layers.conv1d({filters: 8, kernelSize: [3], activation: 'relu'}));
  model.add(tfl.layers.avgPooling1d({poolSize: 5}));
  model.add(tfl.layers.flatten());
  fs.writeFileSync(exportPath, model.toJSON());
}

// Functional model with two Merge layers.
function exportFunctionalMergeModel(exportPath: string): void {
  const input1 = tfl.input({shape: [2, 5]});
  const input2 = tfl.input({shape: [4, 5]});
  const input3 = tfl.input({shape: [30]});
  const reshaped1 = tfl.layers.reshape({targetShape: [10]}).apply(input1) as
      tfl.SymbolicTensor;
  const reshaped2 = tfl.layers.reshape({targetShape: [20]}).apply(input2) as
      tfl.SymbolicTensor;
  const dense1 =
      tfl.layers.dense({units: 5}).apply(reshaped1) as tfl.SymbolicTensor;
  const dense2 =
      tfl.layers.dense({units: 5}).apply(reshaped2) as tfl.SymbolicTensor;
  const dense3 =
      tfl.layers.dense({units: 5}).apply(input3) as tfl.SymbolicTensor;
  const avg =
      tfl.layers.average().apply([dense1, dense2]) as tfl.SymbolicTensor;
  const concat = tfl.layers.concatenate({axis: -1}).apply([avg, dense3]) as
      tfl.SymbolicTensor;
  const output =
      tfl.layers.dense({units: 1}).apply(concat) as tfl.SymbolicTensor;
  const model = tfl.model({inputs: [input1, input2, input3], outputs: output});
  fs.writeFileSync(exportPath, model.toJSON());
}

// TODO(cais): Once an IOHandler implementation for node.js native filesystem
//   exists, use it to serialize the weight values, in addition to the model
//   topology. Then load the weights in Python Keras and compare predict()
//   outputs.

console.log(`Using tfjs-layers version: ${tfl.version_layers}`);

if (process.argv.length < 3) {
  throw new Error('Usage: node tfjs_save.ts <test_data_dir>');
}
const testDataDir = process.argv[2];

exportMLPModel(join(testDataDir, 'mlp.json'));
exportCNNModel(join(testDataDir, 'cnn.json'));
exportDepthwiseCNNModel(join(testDataDir, 'depthwise_cnn.json'));
exportSimpleRNNModel(join(testDataDir, 'simple_rnn.json'));
exportGRUModel(join(testDataDir, 'gru.json'));
exportBidirectionalLSTMModel(join(testDataDir, 'bidirectional_lstm.json'));
exportTimeDistributedLSTMModel(join(testDataDir, 'time_distributed_lstm.json'));
exportOneDimensionalModel(join(testDataDir, 'one_dimensional.json'));
exportFunctionalMergeModel(join(testDataDir, 'functional_merge.json'));
