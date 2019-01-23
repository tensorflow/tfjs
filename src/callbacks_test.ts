/**
 * @license
 * Copyright 2018 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import * as tf from '@tensorflow/tfjs';

import {getDisplayDecimalPlaces, progressBarHelper} from './callbacks';

describe('progbarLogger', () => {
  // Fake progbar class written for testing.
  class FakeProgbar {
    readonly tickConfigs: Array<{placeholderForLossesAndMetrics: string}> = [];

    constructor(readonly specs: string, readonly config?: {}) {}

    tick(tickConfig: {placeholderForLossesAndMetrics: string}) {
      this.tickConfigs.push(tickConfig);
    }
  }

  it('Model.fit with loss, no metric, no validation, verobse = 1', async () => {
    const fakeProgbars: FakeProgbar[] = [];
    spyOn(progressBarHelper, 'ProgressBar')
        .and.callFake((specs: string, config: {}) => {
          const fakeProgbar = new FakeProgbar(specs, config);
          fakeProgbars.push(fakeProgbar);
          return fakeProgbar;
        });
    const consoleMessages: string[] = [];
    spyOn(progressBarHelper, 'log').and.callFake((message: string) => {
      consoleMessages.push(message);
    });

    const model = tf.sequential();
    model.add(
        tf.layers.dense({units: 10, inputShape: [8], activation: 'relu'}));
    model.add(tf.layers.dense({units: 1}));
    model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});

    const numSamples = 14;
    const epochs = 3;
    const batchSize = 8;
    const xs = tf.randomNormal([numSamples, 8]);
    const ys = tf.randomNormal([numSamples, 1]);
    await model.fit(xs, ys, {epochs, batchSize, verbose: 1});

    // A progbar object is created for each epoch.
    expect(fakeProgbars.length).toEqual(3);
    for (const fakeProgbar of fakeProgbars) {
      const tickConfigs = fakeProgbar.tickConfigs;
      // There are ceil(14 / 8) = 2 batchs per epoch. There should be 1 tick
      // for epoch batch, plus a tick at the end of the epoch.
      expect(tickConfigs.length).toEqual(3);
      for (let i = 0; i < 2; ++i) {
        expect(Object.keys(tickConfigs[i])).toEqual([
          'placeholderForLossesAndMetrics'
        ]);
        expect(tickConfigs[i]['placeholderForLossesAndMetrics'])
            .toMatch(/^loss=.*/);
      }
      expect(tickConfigs[2]).toEqual({placeholderForLossesAndMetrics: ''});
    }
    expect(consoleMessages.length).toEqual(6);
    expect(consoleMessages[0]).toEqual('Epoch 1 / 3');
    expect(consoleMessages[1]).toMatch(/.*ms .*us\/step - loss=.*/);
    expect(consoleMessages[2]).toEqual('Epoch 2 / 3');
    expect(consoleMessages[3]).toMatch(/.*ms .*us\/step - loss=.*/);
    expect(consoleMessages[4]).toEqual('Epoch 3 / 3');
    expect(consoleMessages[5]).toMatch(/.*ms .*us\/step - loss=.*/);
  });

  it('Model.fit with loss, metric and validation, verbose = 2', async () => {
    const fakeProgbars: FakeProgbar[] = [];
    spyOn(progressBarHelper, 'ProgressBar')
        .and.callFake((specs: string, config: {}) => {
          const fakeProgbar = new FakeProgbar(specs, config);
          fakeProgbars.push(fakeProgbar);
          return fakeProgbar;
        });
    const consoleMessages: string[] = [];
    spyOn(progressBarHelper, 'log').and.callFake((message: string) => {
      consoleMessages.push(message);
    });

    const model = tf.sequential();
    model.add(
        tf.layers.dense({units: 10, inputShape: [8], activation: 'relu'}));
    model.add(tf.layers.dense({units: 1}));
    model.compile(
        {loss: 'meanSquaredError', optimizer: 'sgd', metrics: ['acc']});

    const numSamples = 40;
    const epochs = 2;
    const batchSize = 8;
    const validationSplit = 0.15;
    const xs = tf.randomNormal([numSamples, 8]);
    const ys = tf.randomNormal([numSamples, 1]);
    await model.fit(xs, ys, {epochs, batchSize, validationSplit, verbose: 2});

    // A progbar object is created for each epoch.
    expect(fakeProgbars.length).toEqual(2);
    for (const fakeProgbar of fakeProgbars) {
      const tickConfigs = fakeProgbar.tickConfigs;
      // There are 5 batchs per epoch. There should be 1 tick for epoch batch,
      // plus a tick at the end of the epoch.
      expect(tickConfigs.length).toEqual(6);
      for (let i = 0; i < 5; ++i) {
        expect(Object.keys(tickConfigs[i])).toEqual([
          'placeholderForLossesAndMetrics'
        ]);
        expect(tickConfigs[i]['placeholderForLossesAndMetrics'])
            .toMatch(/^acc=.* loss=.*/);
      }
      expect(tickConfigs[5]).toEqual({placeholderForLossesAndMetrics: ''});
    }
    expect(consoleMessages.length).toEqual(4);
    expect(consoleMessages[0]).toEqual('Epoch 1 / 2');
    expect(consoleMessages[1])
        .toMatch(/.*ms .*us\/step - acc=.* loss=.* val_acc=.* val_loss=.*/);
    expect(consoleMessages[2]).toEqual('Epoch 2 / 2');
    expect(consoleMessages[3])
        .toMatch(/.*ms .*us\/step - acc=.* loss=.* val_acc=.* val_loss=.*/);
  });

  it('Model.fit does not create ProgbarLogger if verbose is 0', async () => {
    const fakeProgbars: FakeProgbar[] = [];
    spyOn(progressBarHelper, 'ProgressBar')
        .and.callFake((specs: string, config: {}) => {
          const fakeProgbar = new FakeProgbar(specs, config);
          fakeProgbars.push(fakeProgbar);
          return fakeProgbar;
        });
    const consoleMessages: string[] = [];
    spyOn(progressBarHelper, 'log').and.callFake((message: string) => {
      consoleMessages.push(message);
    });

    const model = tf.sequential();
    model.add(
        tf.layers.dense({units: 10, inputShape: [8], activation: 'relu'}));
    model.add(tf.layers.dense({units: 1}));
    model.compile(
        {loss: 'meanSquaredError', optimizer: 'sgd', metrics: ['acc']});

    const numSamples = 40;
    const epochs = 2;
    const batchSize = 8;
    const validationSplit = 0.15;
    const xs = tf.randomNormal([numSamples, 8]);
    const ys = tf.randomNormal([numSamples, 1]);
    await model.fit(xs, ys, {epochs, batchSize, validationSplit, verbose: 0});

    expect(fakeProgbars.length).toEqual(0);
  });

  it('Model.fitDataset: batchesPerEpoch specified, verbose = 1', async () => {
    const fakeProgbars: FakeProgbar[] = [];
    spyOn(progressBarHelper, 'ProgressBar')
        .and.callFake((specs: string, config: {}) => {
          const fakeProgbar = new FakeProgbar(specs, config);
          fakeProgbars.push(fakeProgbar);
          return fakeProgbar;
        });
    const consoleMessages: string[] = [];
    spyOn(progressBarHelper, 'log').and.callFake((message: string) => {
      consoleMessages.push(message);
    });

    const epochs = 2;
    const xDataset = tf.data.array([[1, 2], [3, 4], [5, 6], [7, 8]])
                         .map(x => tf.tensor2d(x, [1, 2]));
    const yDataset =
        tf.data.array([[1], [2], [3], [4]]).map(y => tf.tensor2d(y, [1, 1]));
    const dataset = tf.data.zip([xDataset, yDataset]).repeat(epochs);

    const model = tf.sequential();
    model.add(tf.layers.dense({units: 1, inputShape: [2]}));
    model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});
    await model.fitDataset(dataset, {batchesPerEpoch: 4, epochs, verbose: 1});

    expect(consoleMessages.length).toEqual(4);
    expect(consoleMessages[0]).toEqual('Epoch 1 / 2');
    expect(consoleMessages[1]).toMatch(/.*ms .*us\/step - loss=.*/);
    expect(consoleMessages[2]).toEqual('Epoch 2 / 2');
    expect(consoleMessages[3]).toMatch(/.*ms .*us\/step - loss=.*/);
  });

  it('Model.fitDataset: batchesPerEpoch unavailable, verbose = 1', async () => {
    const fakeProgbars: FakeProgbar[] = [];
    spyOn(progressBarHelper, 'ProgressBar')
        .and.callFake((specs: string, config: {}) => {
          const fakeProgbar = new FakeProgbar(specs, config);
          fakeProgbars.push(fakeProgbar);
          return fakeProgbar;
        });
    const consoleMessages: string[] = [];
    spyOn(progressBarHelper, 'log').and.callFake((message: string) => {
      consoleMessages.push(message);
    });

    const epochs = 2;
    const xDataset = tf.data.array([[1, 2], [3, 4], [5, 6], [7, 8]])
                         .map(x => tf.tensor2d(x, [1, 2]));
    const yDataset =
        tf.data.array([[1], [2], [3], [4]]).map(y => tf.tensor2d(y, [1, 1]));
    const dataset = tf.data.zip([xDataset, yDataset]).repeat(epochs);

    const model = tf.sequential();
    model.add(tf.layers.dense({units: 1, inputShape: [2]}));
    model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});
    // `batchesPerEpoch` is not specified. Instead, `fitDataset()` relies on the
    // `done` field being `true` to terminate the epoch(s).
    await model.fitDataset(dataset, {epochs, verbose: 1});

    expect(consoleMessages.length).toEqual(4);
    expect(consoleMessages[0]).toEqual('Epoch 1 / 2');
    expect(consoleMessages[1]).toMatch(/.*ms .*us\/step - loss=.*/);
    expect(consoleMessages[2]).toEqual('Epoch 2 / 2');
    expect(consoleMessages[3]).toMatch(/.*ms .*us\/step - loss=.*/);
  });

  it('Model.fitDataset: verbose = 0 leads to no logging', async () => {
    const fakeProgbars: FakeProgbar[] = [];
    spyOn(progressBarHelper, 'ProgressBar')
        .and.callFake((specs: string, config: {}) => {
          const fakeProgbar = new FakeProgbar(specs, config);
          fakeProgbars.push(fakeProgbar);
          return fakeProgbar;
        });
    const consoleMessages: string[] = [];
    spyOn(progressBarHelper, 'log').and.callFake((message: string) => {
      consoleMessages.push(message);
    });

    const xDataset = tf.data.array([[1, 2], [3, 4], [5, 6], [7, 8]])
                         .map(x => tf.tensor2d(x, [1, 2]));
    const yDataset =
        tf.data.array([[1], [2], [3], [4]]).map(y => tf.tensor2d(y, [1, 1]));
    const dataset = tf.data.zip([xDataset, yDataset]);

    const model = tf.sequential();
    model.add(tf.layers.dense({units: 1, inputShape: [2]}));
    model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});
    const history = await model.fitDataset(dataset, {epochs: 1, verbose: 0});
    expect(history.history.loss.length).toEqual(1);
    expect(consoleMessages.length)
        .toEqual(0);  // No logging should have happened.
  });
});

describe('getDisplayDecimalPlaces', () => {
  it('Not finite', () => {
    expect(getDisplayDecimalPlaces(Infinity)).toEqual(2);
    expect(getDisplayDecimalPlaces(-Infinity)).toEqual(2);
    expect(getDisplayDecimalPlaces(NaN)).toEqual(2);
  });

  it('zero', () => {
    expect(getDisplayDecimalPlaces(0)).toEqual(2);
  });

  it('Finite and positive', () => {
    expect(getDisplayDecimalPlaces(300)).toEqual(2);
    expect(getDisplayDecimalPlaces(30)).toEqual(2);
    expect(getDisplayDecimalPlaces(1)).toEqual(2);
    expect(getDisplayDecimalPlaces(1e-2)).toEqual(4);
    expect(getDisplayDecimalPlaces(1e-3)).toEqual(5);
    expect(getDisplayDecimalPlaces(4e-3)).toEqual(5);
    expect(getDisplayDecimalPlaces(1e-6)).toEqual(8);
  });

  it('Finite and negative', () => {
    expect(getDisplayDecimalPlaces(-300)).toEqual(2);
    expect(getDisplayDecimalPlaces(-30)).toEqual(2);
    expect(getDisplayDecimalPlaces(-1)).toEqual(2);
    expect(getDisplayDecimalPlaces(-1e-2)).toEqual(4);
    expect(getDisplayDecimalPlaces(-1e-3)).toEqual(5);
    expect(getDisplayDecimalPlaces(-4e-3)).toEqual(5);
    expect(getDisplayDecimalPlaces(-1e-6)).toEqual(8);
  });
});
