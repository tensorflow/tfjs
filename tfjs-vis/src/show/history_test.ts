/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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

import {fitCallbacks, history} from './history';

describe('fitCallbacks', () => {
  beforeEach(() => {
    document.body.innerHTML = '<div id="container"></div>';
  });

  it('returns two callbacks', async () => {
    const container = {name: 'Test'};
    const callbacks = fitCallbacks(container, ['loss', 'acc']);

    expect(typeof (callbacks.onEpochEnd)).toEqual('function');
    expect(typeof (callbacks.onBatchEnd)).toEqual('function');
  });

  it('returns one callback', async () => {
    const container = {name: 'Test'};
    const callbacks = fitCallbacks(container, ['loss', 'acc'], {
      callbacks: ['onBatchEnd'],
    });

    expect(callbacks.onEpochEnd).toEqual(undefined);
    expect(typeof (callbacks.onBatchEnd)).toEqual('function');
  });

  it('onEpochEnd callback can render logs', async () => {
    const container = {name: 'Test'};
    const callbacks =
        fitCallbacks(container, ['loss', 'val_loss', 'acc', 'val_acc']);

    const l1 = {loss: 0.5, 'val_loss': 0.7};
    const l2 = {loss: 0.2, acc: 0.6, 'val_loss': 0.5, 'val_acc': 0.3};

    await callbacks.onEpochEnd(0, l1);
    expect(document.querySelectorAll('.vega-embed').length).toBe(1);
    expect(document.querySelectorAll('div[data-name="loss"]').length).toBe(1);

    await callbacks.onEpochEnd(1, l2);
    expect(document.querySelectorAll('.vega-embed').length).toBe(2);
    expect(document.querySelectorAll('div[data-name="loss"]').length).toBe(1);
    expect(document.querySelectorAll('div[data-name="acc"]').length).toBe(1);
  });

  it('onBatchEnd callback can render logs', async () => {
    const container = {name: 'Test'};
    const callbacks = fitCallbacks(container, ['loss', 'acc']);

    const l1 = {loss: 0.5};
    const l2 = {loss: 0.2, acc: 0.6};

    await callbacks.onBatchEnd(0, l1);
    expect(document.querySelectorAll('.vega-embed').length).toBe(1);
    expect(document.querySelectorAll('div[data-name="loss"]').length).toBe(1);

    await callbacks.onBatchEnd(1, l2);
    expect(document.querySelectorAll('.vega-embed').length).toBe(2);
    expect(document.querySelectorAll('div[data-name="loss"]').length).toBe(1);
    expect(document.querySelectorAll('div[data-name="acc"]').length).toBe(1);
  });
});

describe('history', () => {
  beforeEach(() => {
    document.body.innerHTML = '<div id="container"></div>';
  });

  it('renders a logs[]', async () => {
    const container = {name: 'Test'};
    const logs = [{loss: 0.5}, {loss: 0.3}];
    const metrics = ['loss'];
    await history(container, logs, metrics);

    expect(document.querySelectorAll('.vega-embed').length).toBe(1);
  });

  it('renders a logs object with multiple metrics', async () => {
    const container = {name: 'Test'};
    const logs = [{loss: 0.2, acc: 0.6}, {loss: 0.1, acc: 0.65}];
    const metrics = ['loss', 'acc'];
    await history(container, logs, metrics);

    expect(document.querySelectorAll('.vega-embed').length).toBe(2);
  });

  it('renders a history object with multiple metrics', async () => {
    const container = {name: 'Test'};
    const hist = {
      history: {
        'loss': [0.7, 0.3, 0.2],
        'acc': [0.2, 0.3, 0.21],
      }
    };
    const metrics = ['loss', 'acc'];
    await history(container, hist, metrics);

    expect(document.querySelectorAll('.vega-embed').length).toBe(2);
  });

  it('can render multiple history objects', async () => {
    const container = {name: 'Test'};
    const container2 = {name: 'Other Test'};
    const hist = {
      history: {
        'loss': [0.7, 0.3, 0.2],
        'acc': [0.2, 0.3, 0.21],
      }
    };

    await history(container, hist, ['loss']);
    await history(container2, hist, ['acc']);

    expect(document.querySelectorAll('.vega-embed').length).toBe(2);
  });
});
