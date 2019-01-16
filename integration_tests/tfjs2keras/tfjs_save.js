/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

const tfc = require('@tensorflow/tfjs-core');
const tfl = require('@tensorflow/tfjs-layers');
const tfjsNode = require('@tensorflow/tfjs-node');
const fs = require('fs');
const join = require('path').join;

/**
 * Generate random input(s), get predict() output(s), and save them along with
 * the model.
 *
 * @param model The `tf.Model` instance in question. It may have one or more
 *   inputs and one or more outputs. It is assumed that for each input, only
 *   the first dimension (i.e., the batch dimension) is undetermined.
 * @param exportPathprefix The path prefix to which the model, the input and
 *   output tensors will be saved
 * @param inputIntegerMax (Optional) Maximum integer value for the input
 *   tensors. Used for models that take integer tensors as inputs.
 */
async function saveModelAndRandomInputsAndOutputs(
    model, exportPathprefix, inputIntegerMax) {
  await model.save(tfjsNode.io.fileSystem(`${exportPathprefix}`));

  const xs = [];
  const xsData = [];
  const xsShapes = [];
  for (const inputTensor of model.inputs) {
    const inputShape = inputTensor.shape;
    inputShape[0] = 1;
    if (inputShape.indexOf(null) !== -1) {
      throw new Error(
          `It is assumed that the only the first dimension of the tensor ` +
          `is undetermined, but the assumption is not satisfied for ` +
          `input shape ${JSON.stringify(inputTensor.shape)}`);
    }
    const xTensor = inputIntegerMax == null ?
        tfc.randomNormal(inputShape) :
        tfc.floor(tfc.randomUniform(inputShape, 0, inputIntegerMax));
    xs.push(xTensor);
    xsData.push(Array.from(xTensor.dataSync()));
    xsShapes.push(xTensor.shape);
  }
  fs.writeFileSync(exportPathprefix + '.xs-data.json', JSON.stringify(xsData));
  fs.writeFileSync(
      exportPathprefix + '.xs-shapes.json', JSON.stringify(xsShapes));

  const ys = model.outputs.length === 1 ?
      [model.predict(xs)] :
      model.predict(xs);
  fs.writeFileSync(
      exportPathprefix + '.ys-data.json',
      JSON.stringify((ys.map(y => Array.from(y.dataSync())))));
  fs.writeFileSync(
      exportPathprefix + '.ys-shapes.json',
      JSON.stringify(ys.map(y => y.shape)));
}

// Multi-layer perceptron (MLP).
async function exportMLPModel(exportPath) {
  const model = tfl.sequential();
  // Test both activations encapsulated in other layers and as standalone
  // layers.
  model.add(
      tfl.layers.dense({units: 100, inputShape: [200], activation: 'relu'}));
  model.add(tfl.layers.dense({units: 50, activation: 'elu'}));
  model.add(tfl.layers.dense({units: 24}));
  model.add(tfl.layers.activation({activation: 'elu'}));
  model.add(tfl.layers.dense({units: 8, activation: 'softmax'}));

  await saveModelAndRandomInputsAndOutputs(model, exportPath);
}

// Convolutional neural network (CNN).
async function exportCNNModel(exportPath) {
  const model = tfl.sequential();

  // Cover separable and non-separable convoluational layers.
  const inputShape = [40, 40, 3];
  model.add(tfl.layers.conv2d({
    filters: 32,
    kernelSize: [3, 3],
    strides: [2, 2],
    inputShape,
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

  await saveModelAndRandomInputsAndOutputs(model, exportPath);
}

async function exportDepthwiseCNNModel(exportPath) {
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

  await saveModelAndRandomInputsAndOutputs(model, exportPath);
}

// SimpleRNN with embedding.
async function exportSimpleRNNModel(exportPath) {
  const model = tfl.sequential();
  const inputDim = 100;
  model.add(tfl.layers.embedding({inputDim, outputDim: 20, inputShape: [10]}));
  model.add(tfl.layers.simpleRNN({units: 4}));

  await saveModelAndRandomInputsAndOutputs(model, exportPath, inputDim);
}

// GRU with embedding.
async function exportGRUModel(exportPath) {
  const model = tfl.sequential();
  const inputDim = 100;
  model.add(tfl.layers.embedding({inputDim, outputDim: 20, inputShape: [10]}));
  model.add(tfl.layers.gru({units: 4, goBackwards: true}));

  await saveModelAndRandomInputsAndOutputs(model, exportPath, inputDim);
}

// Bidirecitonal LSTM with embedding.
async function exportBidirectionalLSTMModel(exportPath) {
  const model = tfl.sequential();
  const inputDim = 100;
  model.add(tfl.layers.embedding({inputDim, outputDim: 20, inputShape: [10]}));
  // TODO(cais): Investigate why the `tfl.layers.RNN` typing doesn't work.
  const lstm = tfl.layers.lstm({units: 4, goBackwards: true});
  model.add(tfl.layers.bidirectional({layer: lstm, mergeMode: 'concat'}));

  await saveModelAndRandomInputsAndOutputs(model, exportPath, inputDim);
}

// LSTM + time-distributed layer with embedding.
async function exportTimeDistributedLSTMModel(exportPath) {
  const model = tfl.sequential();
  const inputDim = 100;
  model.add(tfl.layers.embedding({inputDim, outputDim: 20, inputShape: [10]}));
  model.add(tfl.layers.lstm({units: 4, returnSequences: true}));
  model.add(tfl.layers.timeDistributed({
    layer:
        tfl.layers.dense({units: 2, useBias: false, activation: 'softmax'})
  }));

  await saveModelAndRandomInputsAndOutputs(model, exportPath, inputDim);
}

// Model with Conv1D and Pooling1D layers.
async function exportOneDimensionalModel(exportPath) {
  const model = tfl.sequential();
  model.add(tfl.layers.conv1d(
      {filters: 16, kernelSize: [4], inputShape: [80, 1], activation: 'relu'}));
  model.add(tfl.layers.maxPooling1d({poolSize: 3}));
  model.add(
      tfl.layers.conv1d({filters: 8, kernelSize: [3], activation: 'relu'}));
  model.add(tfl.layers.avgPooling1d({poolSize: 5}));
  model.add(tfl.layers.flatten());

  await saveModelAndRandomInputsAndOutputs(model, exportPath);
}

// Functional model with two Merge layers.
async function exportFunctionalMergeModel(exportPath) {
  const input1 = tfl.input({shape: [2, 5]});
  const input2 = tfl.input({shape: [4, 5]});
  const input3 = tfl.input({shape: [30]});
  const reshaped1 = tfl.layers.reshape({targetShape: [10]}).apply(input1);
  const reshaped2 = tfl.layers.reshape({targetShape: [20]}).apply(input2);
  const dense1 = tfl.layers.dense({units: 5}).apply(reshaped1);
  const dense2 = tfl.layers.dense({units: 5}).apply(reshaped2);
  const dense3 = tfl.layers.dense({units: 5}).apply(input3);
  const avg = tfl.layers.average().apply([dense1, dense2]);
  const concat = tfl.layers.concatenate({axis: -1}).apply([avg, dense3]);
  const output = tfl.layers.dense({units: 1}).apply(concat);
  const model = tfl.model({inputs: [input1, input2, input3], outputs: output});

  await saveModelAndRandomInputsAndOutputs(model, exportPath);
}

console.log(`Using tfjs-core version: ${tfc.version_core}`);
console.log(`Using tfjs-layers version: ${tfl.version_layers}`);
console.log(`Using tfjs-node version: ${tfjsNode.version}`);

if (process.argv.length !== 3) {
  throw new Error('Usage: node tfjs_save.ts <test_data_dir>');
}
const testDataDir = process.argv[2];

(async function() {
  await exportMLPModel(join(testDataDir, 'mlp'));
  await exportCNNModel(join(testDataDir, 'cnn'));
  await exportDepthwiseCNNModel(join(testDataDir, 'depthwise_cnn'));
  await exportSimpleRNNModel(join(testDataDir, 'simple_rnn'));
  await exportGRUModel(join(testDataDir, 'gru'));
  await exportBidirectionalLSTMModel(join(testDataDir, 'bidirectional_lstm'));
  await exportTimeDistributedLSTMModel(
      join(testDataDir, 'time_distributed_lstm'));
  await exportOneDimensionalModel(join(testDataDir, 'one_dimensional'));
  await exportFunctionalMergeModel(join(testDataDir, 'functional_merge.json'));
})();
