/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

import {io, linalg, randomNormal, Tensor, zeros} from '@tensorflow/tfjs-core';

import * as initializers from './initializers';
import * as tfl from './index';

// tslint:disable-next-line:max-line-length
import {describeMathCPUAndGPU, describeMathGPU, expectTensorsClose} from './utils/test_utils';

describeMathCPUAndGPU('Model.save', () => {
  class IOHandlerForTest implements io.IOHandler {
    savedArtifacts: io.ModelArtifacts;

    async save(modelArtifacts: io.ModelArtifacts): Promise<io.SaveResult> {
      this.savedArtifacts = modelArtifacts;
      return {modelArtifactsInfo: null};
    }
  }

  class EmptyIOHandler implements io.IOHandler {}

  it('Saving all weights succeeds', async () => {
    const model = tfl.sequential();
    model.add(tfl.layers.dense({units: 3, inputShape: [5]}));
    const handler = new IOHandlerForTest();

    await model.save(handler);
    expect(handler.savedArtifacts.modelTopology)
        .toEqual(model.toJSON(null, false));
    expect(handler.savedArtifacts.weightSpecs.length).toEqual(2);
    expect(handler.savedArtifacts.weightSpecs[0].name.indexOf('/kernel'))
        .toBeGreaterThan(0);
    expect(handler.savedArtifacts.weightSpecs[0].shape).toEqual([5, 3]);
    expect(handler.savedArtifacts.weightSpecs[0].dtype).toEqual('float32');
    expect(handler.savedArtifacts.weightSpecs[1].name.indexOf('/bias'))
        .toBeGreaterThan(0);
    expect(handler.savedArtifacts.weightSpecs[1].shape).toEqual([3]);
    expect(handler.savedArtifacts.weightSpecs[1].dtype).toEqual('float32');
  });

  it('Saving only trainable weights succeeds', async () => {
    const model = tfl.sequential();
    model.add(tfl.layers.dense({units: 3, inputShape: [5], trainable: false}));
    model.add(tfl.layers.dense({units: 2}));
    const handler = new IOHandlerForTest();

    await model.save(handler, {trainableOnly: true});
    expect(handler.savedArtifacts.modelTopology)
        .toEqual(model.toJSON(null, false));
    // Verify that only the trainable weights (i.e., weights from the
    // 2nd, trainable Dense layer) are saved.
    expect(handler.savedArtifacts.weightSpecs.length).toEqual(2);
    expect(handler.savedArtifacts.weightSpecs[0].name.indexOf('/kernel'))
        .toBeGreaterThan(0);
    expect(handler.savedArtifacts.weightSpecs[0].shape).toEqual([3, 2]);
    expect(handler.savedArtifacts.weightSpecs[0].dtype).toEqual('float32');
    expect(handler.savedArtifacts.weightSpecs[1].name.indexOf('/bias'))
        .toBeGreaterThan(0);
    expect(handler.savedArtifacts.weightSpecs[1].shape).toEqual([2]);
    expect(handler.savedArtifacts.weightSpecs[1].dtype).toEqual('float32');
  });

  it('Saving to a handler without save method fails', async done => {
    const model = tfl.sequential();
    model.add(tfl.layers.dense({units: 3, inputShape: [5]}));
    const handler = new EmptyIOHandler();
    model.save(handler)
        .then(saveResult => {
          fail(
              'Saving with an IOHandler without `save` succeeded ' +
              'unexpectedly.');
        })
        .catch(err => {
          expect(err.message)
              .toEqual(
                  'Model.save() cannot proceed because the IOHandler ' +
                  'provided does not have the `save` attribute defined.');
          done();
        });
  });
});

describeMathGPU('Save-load round trips', () => {
  it('Sequential model, Local storage', async () => {
    const model1 = tfl.sequential();
    model1.add(
        tfl.layers.dense({units: 2, inputShape: [2], activation: 'relu'}));
    model1.add(tfl.layers.dense({units: 1, useBias: false}));

    // Use a randomly generated model path to prevent collision.
    const path = `testModel${new Date().getTime()}_${Math.random()}`;

    // First save the model to local storage.
    const modelURL = `localstorage://${path}`;
    await model1.save(modelURL);
    // Once the saving succeeds, load the model back.
    const model2 = await tfl.loadLayersModel(modelURL);
    // Verify that the topology of the model is correct.
    expect(model2.toJSON(null, false)).toEqual(model1.toJSON(null, false));

    // Check the equality of the two models' weights.
    const weights1 = model1.getWeights();
    const weights2 = model2.getWeights();
    expect(weights2.length).toEqual(weights1.length);
    for (let i = 0; i < weights1.length; ++i) {
      expectTensorsClose(weights1[i], weights2[i]);
    }
  });

  it('Functional model, IndexedDB', async () => {
    const input = tfl.input({shape: [2, 2]});
    const layer1 = tfl.layers.flatten().apply(input);
    const layer2 =
        tfl.layers.dense({units: 2}).apply(layer1) as tfl.SymbolicTensor;
    const model1 = tfl.model({inputs: input, outputs: layer2});
    // Use a randomly generated model path to prevent collision.
    const path = `testModel${new Date().getTime()}_${Math.random()}`;

    // First save the model to local storage.
    const modelURL = `indexeddb://${path}`;
    await model1.save(modelURL);
    // Once the saving succeeds, load the model back.
    const model2 = await tfl.loadLayersModel(modelURL);
    // Verify that the topology of the model is correct.
    expect(model2.toJSON(null, false)).toEqual(model1.toJSON(null, false));

    // Check the equality of the two models' weights.
    const weights1 = model1.getWeights();
    const weights2 = model2.getWeights();
    expect(weights2.length).toEqual(weights1.length);
    for (let i = 0; i < weights1.length; ++i) {
      expectTensorsClose(weights1[i], weights2[i]);
    }
  });

  it('Call predict() and fit() after load: conv2d model', async () => {
    const model = tfl.sequential();
    model.add(tfl.layers.conv2d({
      filters: 8,
      kernelSize: 4,
      inputShape: [28, 28, 1],
      padding: 'same',
      activation: 'relu'
    }));
    model.add(tfl.layers.maxPooling2d({
      poolSize: 2,
      padding: 'same',
    }));
    model.add(tfl.layers.flatten());
    model.add(tfl.layers.dense({units: 1}));

    const x = randomNormal([1, 28, 28, 1]);
    const y = model.predict(x) as Tensor;

    const path = `testModel${new Date().getTime()}_${Math.random()}`;
    const url = `indexeddb://${path}`;
    await model.save(url);
    // Load the model back.
    const modelPrime = await tfl.loadLayersModel(url);
    // Call predict() on the loaded model and assert the result
    // equals the original predict() result.
    const yPrime = modelPrime.predict(x) as Tensor;
    expectTensorsClose(y, yPrime);

    // Call compile and fit() on the loaded model.
    modelPrime.compile({optimizer: 'sgd', loss: 'meanSquaredError'});
    const trainExamples = 10;
    await modelPrime.fit(
        randomNormal([trainExamples, 28, 28, 1]),
        randomNormal([trainExamples, 1]), {epochs: 4});
  });

  it('Call predict() and fit() after load: conv1d model', async () => {
    const model = tfl.sequential();
    model.add(tfl.layers.conv1d({
      filters: 8,
      kernelSize: 4,
      inputShape: [100, 1],
      padding: 'same',
      activation: 'relu'
    }));
    model.add(tfl.layers.maxPooling1d({
      poolSize: 2,
      padding: 'same',
    }));
    model.add(tfl.layers.flatten());
    model.add(tfl.layers.dense({units: 1}));

    const x = randomNormal([1, 100, 1]);
    const y = model.predict(x) as Tensor;

    const path = `testModel${new Date().getTime()}_${Math.random()}`;
    const url = `indexeddb://${path}`;
    await model.save(url);
    // Load the model back.
    const modelPrime = await tfl.loadModel(url);
    // Call predict() on the loaded model and assert the
    // result equals the original predict() result.
    const yPrime = modelPrime.predict(x) as Tensor;
    expectTensorsClose(y, yPrime);

    // Call compile and fit() on the loaded model.
    modelPrime.compile({optimizer: 'sgd', loss: 'meanSquaredError'});
    const trainExamples = 10;
    await modelPrime.fit(
        randomNormal([trainExamples, 100, 1]), randomNormal([trainExamples, 1]),
        {epochs: 4});
  });

  it('Call predict() and fit() after load: Bidirectional LSTM', async () => {
    const model = tfl.sequential();
    const lstmUnits = 3;
    const sequenceLength = 4;
    const inputDims = 5;
    model.add(tfl.layers.bidirectional({
      layer: tfl.layers.lstm({units: lstmUnits}) as tfl.RNN,
      mergeMode: 'concat',
      inputShape: [sequenceLength, inputDims]
    }));

    const x = randomNormal([2, 4, 5]);
    const y = model.predict(x) as Tensor;

    const path = `testModel${new Date().getTime()}_${Math.random()}`;
    const url = `indexeddb://${path}`;
    await model.save(url);
    const modelPrime = await tfl.loadLayersModel(url);
    const yPrime = modelPrime.predict(x) as Tensor;
    expectTensorsClose(y, yPrime);

    // Call compile and fit() on the loaded model.
    modelPrime.compile({optimizer: 'sgd', loss: 'meanSquaredError'});
    const trainExamples = 2;
    await modelPrime.fit(
        randomNormal([trainExamples, sequenceLength, inputDims]),
        randomNormal([trainExamples, lstmUnits * 2]), {epochs: 2});
  });

  it('Load model: Fast init w/ weights: Sequential & LSTM', async () => {
    const model = tfl.sequential();
    model.add(tfl.layers.lstm({
      units: 2,
      inputShape: [3, 4],
      recurrentInitializer: 'orthogonal',
      kernelInitializer: 'orthogonal',
      biasInitializer: 'randomNormal',
    }));
    let savedArtifacts: io.ModelArtifacts;
    await model.save(io.withSaveHandler(
        async (artifacts: io.ModelArtifacts) => {
          savedArtifacts = artifacts;
          return {modelArtifactsInfo: null};
        }));
    const weights = model.getWeights();

    const getInitSpy = spyOn(initializers, 'getInitializer').and.callThrough();
    const gramSchmidtSpy = spyOn(linalg,  'gramSchmidt').and.callThrough();
    const modelPrime = await tfl.loadLayersModel(io.fromMemory(
        savedArtifacts.modelTopology, savedArtifacts.weightSpecs,
        savedArtifacts.weightData));
    const weightsPrime = modelPrime.getWeights();
    expect(weightsPrime.length).toEqual(weights.length);
    for (let i = 0; i < weights.length; ++i) {
      expectTensorsClose(weightsPrime[i], weights[i]);
    }
    // Assert that orthogonal initializer hasn't been obtained during
    // the model loading.
    expect(getInitSpy).toHaveBeenCalledWith('zeros');
    expect(gramSchmidtSpy).not.toHaveBeenCalled();
  });

  it('Loading model: Fast init w/ weights: timeDistributed', async () => {
    const model = tfl.sequential();
    model.add(tfl.layers.timeDistributed({
      inputShape: [3, 4],
      layer: tfl.layers.dense({
        units: 4,
        kernelInitializer: 'orthogonal',
        useBias: false
      })
    }));
    let savedArtifacts: io.ModelArtifacts;
    await model.save(io.withSaveHandler(
        async (artifacts: io.ModelArtifacts) => {
          savedArtifacts = artifacts;
          return {modelArtifactsInfo: null};
        }));
    const weights = model.getWeights();

    const getInitSpy = spyOn(initializers, 'getInitializer').and.callThrough();
    const gramSchmidtSpy = spyOn(linalg,  'gramSchmidt').and.callThrough();
    const modelPrime = await tfl.loadLayersModel(io.fromMemory(
        savedArtifacts.modelTopology, savedArtifacts.weightSpecs,
        savedArtifacts.weightData));
    const weightsPrime = modelPrime.getWeights();
    expect(weightsPrime.length).toEqual(weights.length);
    for (let i = 0; i < weights.length; ++i) {
      expectTensorsClose(weightsPrime[i], weights[i]);
    }
    // Assert that orthogonal initializer hasn't been obtained during
    // the model loading.
    expect(getInitSpy).toHaveBeenCalledWith('zeros');
    expect(gramSchmidtSpy).not.toHaveBeenCalled();
  });

  it('Loading model: Fast init w/ weights: bidirectional', async () => {
    const model = tfl.sequential();
    model.add(tfl.layers.bidirectional({
      inputShape: [3, 4],
      mergeMode: 'concat',
      layer: tfl.layers.lstm({
        units: 4,
        kernelInitializer: 'orthogonal',
        recurrentInitializer: 'orthogonal',
        biasInitializer: 'glorotNormal'
      }) as tfl.RNN
    }));
    let savedArtifacts: io.ModelArtifacts;
    await model.save(io.withSaveHandler(
        async (artifacts: io.ModelArtifacts) => {
          savedArtifacts = artifacts;
          return {modelArtifactsInfo: null};
        }));
    const weights = model.getWeights();

    const getInitSpy = spyOn(initializers, 'getInitializer').and.callThrough();
    const gramSchmidtSpy = spyOn(linalg,  'gramSchmidt').and.callThrough();
    const modelPrime = await tfl.loadLayersModel(io.fromMemory(
        savedArtifacts.modelTopology, savedArtifacts.weightSpecs,
        savedArtifacts.weightData));
    const weightsPrime = modelPrime.getWeights();
    expect(weightsPrime.length).toEqual(weights.length);
    for (let i = 0; i < weights.length; ++i) {
      expectTensorsClose(weightsPrime[i], weights[i]);
    }
    // Assert that orthogonal initializer hasn't been obtained during
    // the model loading.
    expect(getInitSpy).toHaveBeenCalledWith('zeros');
    expect(gramSchmidtSpy).not.toHaveBeenCalled();
  });

  it('Loading model: Fast init w/ weights: functional model', async () => {
    const input1 = tfl.input({shape: [3, 2]});
    const input2 = tfl.input({shape: [3, 2]});
    let y = tfl.layers.concatenate()
        .apply([input1, input2]) as tfl.SymbolicTensor;
    y = tfl.layers.lstm({
      units: 4,
      kernelInitializer: 'orthogonal',
      recurrentInitializer: 'orthogonal',
      biasInitializer: 'glorotNormal'
    }).apply(y) as tfl.SymbolicTensor;
    const model = tfl.model({inputs: [input1, input2], outputs: y});
    let savedArtifacts: io.ModelArtifacts;
    await model.save(io.withSaveHandler(
        async (artifacts: io.ModelArtifacts) => {
          savedArtifacts = artifacts;
          return {modelArtifactsInfo: null};
        }));
    const weights = model.getWeights();

    const getInitSpy = spyOn(initializers, 'getInitializer').and.callThrough();
    const gramSchmidtSpy = spyOn(linalg, 'gramSchmidt').and.callThrough();
    const modelPrime = await tfl.loadLayersModel(io.fromMemory(
        savedArtifacts.modelTopology, savedArtifacts.weightSpecs,
        savedArtifacts.weightData));
    const weightsPrime = modelPrime.getWeights();
    expect(weightsPrime.length).toEqual(weights.length);
    for (let i = 0; i < weights.length; ++i) {
      expectTensorsClose(weightsPrime[i], weights[i]);
    }
    // Assert that orthogonal initializer hasn't been obtained during
    // the model loading.
    expect(getInitSpy).toHaveBeenCalledWith('zeros');
    expect(gramSchmidtSpy).not.toHaveBeenCalled();
  });

  it('modelFromJSON calls correct weight initializers', async () => {
    const model = tfl.sequential();
    model.add(tfl.layers.lstm({
      units: 2,
      inputShape: [3, 4],
      recurrentInitializer: 'orthogonal',
      kernelInitializer: 'orthogonal',
      biasInitializer: 'randomNormal',
    }));
    const modelJSON = model.toJSON(null, false);

    const gramSchmidtSpy = spyOn(linalg, 'gramSchmidt').and.callThrough();
    const modelPrime =
        await tfl.models.modelFromJSON({modelTopology: modelJSON});
    // Make sure modelPrime builds.
    modelPrime.predict(zeros([2, 3, 4]));
    // Assert the orthogonal initializer has been called.
    expect(gramSchmidtSpy).toHaveBeenCalled();
  });

  it('Partial non-strict load calls weight initializers', async () => {
    const model = tfl.sequential();
    model.add(tfl.layers.lstm({
      units: 2,
      inputShape: [3, 4],
      recurrentInitializer: 'orthogonal',
      kernelInitializer: 'orthogonal',
      biasInitializer: 'randomNormal',
    }));
    let savedArtifacts: io.ModelArtifacts;
    await model.save(io.withSaveHandler(
        async (artifacts: io.ModelArtifacts) => {
          savedArtifacts = artifacts;
          return {modelArtifactsInfo: null};
        }));
    const weights = model.getWeights();

    expect(savedArtifacts.weightSpecs.length).toEqual(3);
    savedArtifacts.weightSpecs = savedArtifacts.weightSpecs.slice(0, 1);

    const gramSchmidtSpy = spyOn(linalg,  'gramSchmidt').and.callThrough();
    const strict = false;
    const modelPrime = await tfl.loadModel(io.fromMemory(
        savedArtifacts.modelTopology, savedArtifacts.weightSpecs,
        savedArtifacts.weightData), strict);
    const weightsPrime = modelPrime.getWeights();
    expect(weightsPrime.length).toEqual(weights.length);
    expectTensorsClose(weightsPrime[0], weights[0]);
    // Assert the orthogonal initializer has been called.
    expect(gramSchmidtSpy).toHaveBeenCalled();
  });

  it('loadLayersModel: non-strict load calls weight initializers', async () => {
    const model = tfl.sequential();
    model.add(tfl.layers.lstm({
      units: 2,
      inputShape: [3, 4],
      recurrentInitializer: 'orthogonal',
      kernelInitializer: 'orthogonal',
      biasInitializer: 'randomNormal',
    }));
    let savedArtifacts: io.ModelArtifacts;
    await model.save(io.withSaveHandler(
        async (artifacts: io.ModelArtifacts) => {
          savedArtifacts = artifacts;
          return {modelArtifactsInfo: null};
        }));
    const weights = model.getWeights();

    expect(savedArtifacts.weightSpecs.length).toEqual(3);
    savedArtifacts.weightSpecs = savedArtifacts.weightSpecs.slice(0, 1);

    const gramSchmidtSpy = spyOn(linalg,  'gramSchmidt').and.callThrough();
    const strict = false;
    const modelPrime = await tfl.loadLayersModel(io.fromMemory(
        savedArtifacts.modelTopology, savedArtifacts.weightSpecs,
        savedArtifacts.weightData), {strict});
    const weightsPrime = modelPrime.getWeights();
    expect(weightsPrime.length).toEqual(weights.length);
    expectTensorsClose(weightsPrime[0], weights[0]);
    // Assert the orthogonal initializer has been called.
    expect(gramSchmidtSpy).toHaveBeenCalled();
  });

  it('Load model artifact with ndarray-format scalar objects', async () => {
    // The following model config contains a scalar parameter serialized in the
    // ndarray-style format: `{"type": "ndarray", "value": 6}`.
    // tslint:disable-next-line:max-line-length
    const modelJSON = `{"class_name": "Sequential", "keras_version": "2.2.4", "config": {"layers": [{"class_name": "Dense", "config": {"kernel_initializer": {"class_name": "VarianceScaling", "config": {"distribution": "uniform", "scale": 1.0, "seed": null, "mode": "fan_avg"}}, "name": "dense_1", "kernel_constraint": null, "bias_regularizer": null, "bias_constraint": null, "dtype": "float32", "activation": "linear", "trainable": true, "kernel_regularizer": null, "bias_initializer": {"class_name": "Zeros", "config": {}}, "units": 2, "batch_input_shape": [null, 3], "use_bias": true, "activity_regularizer": null}}, {"class_name": "ReLU", "config": {"threshold": 0.0, "max_value": {"type": "ndarray", "value": 6}, "trainable": true, "name": "re_lu_1", "negative_slope": 0.0}}, {"class_name": "Dense", "config": {"kernel_initializer": {"class_name": "VarianceScaling", "config": {"distribution": "uniform", "scale": 1.0, "seed": null, "mode": "fan_avg"}}, "name": "dense_2", "kernel_constraint": null, "bias_regularizer": null, "bias_constraint": null, "activation": "linear", "trainable": true, "kernel_regularizer": null, "bias_initializer": {"class_name": "Zeros", "config": {}}, "units": 1, "use_bias": true, "activity_regularizer": null}}], "name": "sequential_1"}, "backend": "tensorflow"}`;
    const model =
        await tfl.models.modelFromJSON({modelTopology: JSON.parse(modelJSON)});
    expect(model.layers.length).toEqual(3);
    expect(model.layers[1].getConfig().maxValue).toEqual(6);

    const xs = randomNormal([5].concat(model.inputs[0].shape.slice(1)));
    const ys = model.predict(xs) as Tensor;
    expect(ys.shape).toEqual([5, 1]);
  });

  // TODO(cais): Test fast initialization of models consisting of
  //   StackedRNN layers.
});
