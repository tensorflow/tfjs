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

import {ConfusionMatrixData} from '../types';

import {confusionMatrix} from './confusion_matrix';

describe('renderConfusionMatrix', () => {
  let pixelRatio: number;

  beforeEach(() => {
    document.body.innerHTML = '<div id="container"></div>';
    pixelRatio = window.devicePixelRatio;
  });

  it('renders a chart', async () => {
    const data: ConfusionMatrixData = {
      values: [[4, 2, 8], [1, 7, 2], [3, 3, 20]],
      tickLabels: ['cheese', 'pig', 'font'],
    };

    const container = document.getElementById('container');
    await confusionMatrix(container, data);

    expect(document.querySelectorAll('.vega-embed').length).toBe(1);
  });

  it('renders a chart with shaded diagonal', async () => {
    const data: ConfusionMatrixData = {
      values: [[4, 2, 8], [1, 7, 2], [3, 3, 20]],
      tickLabels: ['cheese', 'pig', 'font'],
    };

    const container = document.getElementById('container');
    await confusionMatrix(container, data, {shadeDiagonal: true});

    expect(document.querySelectorAll('.vega-embed').length).toBe(1);
  });

  it('renders the chart with generated labels', async () => {
    const data: ConfusionMatrixData = {
      values: [[4, 2, 8], [1, 7, 2], [3, 3, 20]],
    };

    const container = document.getElementById('container');

    await confusionMatrix(container, data);
    expect(document.querySelectorAll('.vega-embed').length).toBe(1);
  });

  it('updates the chart', async () => {
    let data: ConfusionMatrixData = {
      values: [[4, 2, 8], [1, 7, 2], [3, 3, 20]],
      tickLabels: ['cheese', 'pig', 'font'],
    };

    const container = document.getElementById('container');

    await confusionMatrix(container, data);
    expect(document.querySelectorAll('.vega-embed').length).toBe(1);

    data = {
      values: [[43, 2, 8], [1, 7, 2], [3, 3, 20]],
      tickLabels: ['cheese', 'pig', 'font'],
    };

    await confusionMatrix(container, data);
    expect(document.querySelectorAll('.vega-embed').length).toBe(1);
  });

  it('sets width of chart', async () => {
    const data: ConfusionMatrixData = {
      values: [[4, 2, 8], [1, 7, 2], [3, 3, 20]],
      tickLabels: ['cheese', 'pig', 'font'],
    };

    const container = document.getElementById('container');
    await confusionMatrix(container, data, {width: 400});

    expect(document.querySelectorAll('.vega-embed').length).toBe(1);
    expect(document.querySelectorAll('canvas').length).toBe(1);
    expect(document.querySelector('canvas').width).toBe(400 * pixelRatio);
  });

  it('sets height of chart', async () => {
    const data: ConfusionMatrixData = {
      values: [[4, 2, 8], [1, 7, 2], [3, 3, 20]],
      tickLabels: ['cheese', 'pig', 'font'],
    };

    const container = document.getElementById('container');
    await confusionMatrix(container, data, {height: 200});

    expect(document.querySelectorAll('.vega-embed').length).toBe(1);
    expect(document.querySelectorAll('canvas').length).toBe(1);
    expect(document.querySelector('canvas').height).toBe(200 * pixelRatio);
  });
});
