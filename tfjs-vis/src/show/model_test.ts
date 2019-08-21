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

import * as tf from '@tensorflow/tfjs';

import {layer, modelSummary} from './model';

describe('modelSummary', () => {
  beforeEach(() => {
    document.body.innerHTML = '<div id="container"></div>';
  });

  it('renders a model summary', async () => {
    const container = {name: 'Test'};
    const model = tf.sequential();
    model.add(tf.layers.dense({units: 1, inputShape: [1]}));
    await modelSummary(container, model);
    expect(document.querySelectorAll('table').length).toBe(1);
    expect(document.querySelectorAll('tr').length).toBe(2);
  });
});

describe('layer', () => {
  beforeEach(() => {
    document.body.innerHTML = '<div id="container"></div>';
  });

  it('renders a layer summary', async () => {
    const container = {name: 'Test'};
    const model = tf.sequential();
    const dense = tf.layers.dense({units: 1, inputShape: [1]});
    model.add(dense);
    model.compile({optimizer: 'sgd', loss: 'meanSquaredError'});
    await layer(container, dense);
    expect(document.querySelectorAll('table').length).toBe(1);
    expect(document.querySelectorAll('tr').length).toBe(3);
  });
});
