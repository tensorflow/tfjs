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

import * as tf from '@tensorflow/tfjs-core';
import {valuesDistribution} from './tensor';

describe('perClassAccuracy', () => {
  beforeEach(() => {
    document.body.innerHTML = '<div id="container"></div>';
  });

  it('renders histogram', async () => {
    const container = {name: 'Test'};
    const tensor = tf.tensor1d([0, 0, 0, 0, 2, 3, 4]);

    await valuesDistribution(container, tensor);
    expect(document.querySelectorAll('table').length).toBe(1);
    expect(document.querySelectorAll('.vega-embed').length).toBe(1);
  });
});
