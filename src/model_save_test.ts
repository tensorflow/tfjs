/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

import {io, randomNormal, Tensor} from '@tensorflow/tfjs-core';

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
  it('Sequential model, Local storage', done => {
    const model1 = tfl.sequential();
    model1.add(
        tfl.layers.dense({units: 2, inputShape: [2], activation: 'relu'}));
    model1.add(tfl.layers.dense({units: 1, useBias: false}));

    // Use a randomly generated model path to prevent collision.
    const path = `testModel${new Date().getTime()}_${Math.random()}`;

    // First save the model to local storage.
    const modelURL = `localstorage://${path}`;
    model1.save(modelURL)
        .then(saveResult => {
          // Once the saving succeeds, load the model back.
          tfl.loadModel(modelURL)
              .then(model2 => {
                // Verify that the topology of the model is correct.
                expect(model2.toJSON(null, false))
                    .toEqual(model1.toJSON(null, false));

                // Check the equality of the two models' weights.
                const weights1 = model1.getWeights();
                const weights2 = model2.getWeights();
                expect(weights2.length).toEqual(weights1.length);
                for (let i = 0; i < weights1.length; ++i) {
                  expectTensorsClose(weights1[i], weights2[i]);
                }

                done();
              })
              .catch(err => {
                done.fail(err.stack);
              });
        })
        .catch(err => {
          done.fail(err.stack);
        });
  });

  it('Functional model, IndexedDB', done => {
    const input = tfl.input({shape: [2, 2]});
    const layer1 = tfl.layers.flatten().apply(input);
    const layer2 =
        tfl.layers.dense({units: 2}).apply(layer1) as tfl.SymbolicTensor;
    const model1 = tfl.model({inputs: input, outputs: layer2});
    // Use a randomly generated model path to prevent collision.
    const path = `testModel${new Date().getTime()}_${Math.random()}`;

    // First save the model to local storage.
    const modelURL = `indexeddb://${path}`;
    model1.save(modelURL)
        .then(saveResult => {
          // Once the saving succeeds, load the model back.
          tfl.loadModel(modelURL)
              .then(model2 => {
                // Verify that the topology of the model is correct.
                expect(model2.toJSON(null, false))
                    .toEqual(model1.toJSON(null, false));

                // Check the equality of the two models' weights.
                const weights1 = model1.getWeights();
                const weights2 = model2.getWeights();
                expect(weights2.length).toEqual(weights1.length);
                for (let i = 0; i < weights1.length; ++i) {
                  expectTensorsClose(weights1[i], weights2[i]);
                }

                done();
              })
              .catch(err => {
                done.fail(err.stack);
              });
        })
        .catch(err => {
          done.fail(err.stack);
        });
  });

  it('Call predict() and fit() after load: conv2d model', done => {
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
    model.save(url)
        .then(saveResult => {
          // Load the model back.
          tfl.loadModel(url)
              .then(modelPrime => {
                // Call predict() on the loaded model and assert the result
                // equals the original predict() result.
                const yPrime = modelPrime.predict(x) as Tensor;
                expectTensorsClose(y, yPrime);

                // Call compile and fit() on the loaded model.
                modelPrime.compile(
                    {optimizer: 'sgd', loss: 'meanSquaredError'});
                const trainExamples = 10;
                modelPrime
                    .fit(
                        randomNormal([trainExamples, 28, 28, 1]),
                        randomNormal([trainExamples]), {epochs: 4})
                    .then(history => {
                      done();
                    })
                    .catch(err => done.fail(err.stack));
              })
              .catch(err => done.fail(err.stack));
        })
        .catch(err => done.fail(err.stack));
  });

  it('Call predict() and fit() after load: conv1d model', done => {
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
    model.save(url)
        .then(saveResult => {
          // Load the model back.
          tfl.loadModel(url)
              .then(modelPrime => {
                // Call predict() on the loaded model and assert the
                // result equals the original predict() result.
                const yPrime = modelPrime.predict(x) as Tensor;
                expectTensorsClose(y, yPrime);

                // Call compile and fit() on the loaded model.
                modelPrime.compile(
                    {optimizer: 'sgd', loss: 'meanSquaredError'});
                const trainExamples = 10;
                modelPrime
                    .fit(
                        randomNormal([trainExamples, 100, 1]),
                        randomNormal([trainExamples]), {epochs: 4})
                    .then(history => {
                      done();
                    })
                    .catch(err => done.fail(err.stack));
              })
              .catch(err => done.fail(err.stack));
        })
        .catch(err => done.fail(err.stack));
  });

  it('Call predict() and fit() after load: Bidirectional LSTM', done => {
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
    model.save(url)
        .then(saveResult => {
          tfl.loadModel(url)
              .then(modelPrime => {
                const yPrime = modelPrime.predict(x) as Tensor;
                expectTensorsClose(y, yPrime);

                // Call compile and fit() on the loaded model.
                modelPrime.compile(
                    {optimizer: 'sgd', loss: 'meanSquaredError'});
                const trainExamples = 2;
                modelPrime
                    .fit(
                        randomNormal(
                            [trainExamples, sequenceLength, inputDims]),
                        randomNormal([trainExamples, lstmUnits * 2]),
                        {epochs: 2})
                    .then(history => {
                      done();
                    })
                    .catch(err => done.fail(err.stack));
              })
              .catch(err => done.fail(err.stack));
        })
        .catch(err => done.fail(err.stack));
  });
});
