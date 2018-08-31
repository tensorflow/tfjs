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

import {renderConfusionMatrix} from './confusion_matrix';

describe('renderConfusionMatrix', () => {
  beforeEach(() => {
    document.body.innerHTML = '<div id="container"></div>';
  });

  it('renders a chart', async () => {
    const data = {
      values: [[4, 2, 8], [1, 7, 2], [3, 3, 20]],
      labels: ['cheese', 'pig', 'font'],
    };

    const container = document.getElementById('container') as HTMLElement;
    await renderConfusionMatrix(data, container);

    expect(document.querySelectorAll('.vega-embed').length).toBe(1);
  });

  it('renders a chart with shaded diagonal', async () => {
    const data = {
      values: [[4, 2, 8], [1, 7, 2], [3, 3, 20]],
      labels: ['cheese', 'pig', 'font'],
    };

    const container = document.getElementById('container') as HTMLElement;
    await renderConfusionMatrix(data, container, {shadeDiagonal: true});

    expect(document.querySelectorAll('.vega-embed').length).toBe(1);
  });

  it('renders the chart with generated labels', async () => {
    const data = {
      values: [[4, 2, 8], [1, 7, 2], [3, 3, 20]],
    };

    const container = document.getElementById('container') as HTMLElement;

    await renderConfusionMatrix(data, container);
    expect(document.querySelectorAll('.vega-embed').length).toBe(1);
  });

  it('updates the chart', async () => {
    let data = {
      values: [[4, 2, 8], [1, 7, 2], [3, 3, 20]],
      labels: ['cheese', 'pig', 'font'],
    };

    const container = document.getElementById('container') as HTMLElement;

    await renderConfusionMatrix(data, container);
    expect(document.querySelectorAll('.vega-embed').length).toBe(1);

    data = {
      values: [[43, 2, 8], [1, 7, 2], [3, 3, 20]],
      labels: ['cheese', 'pig', 'font'],
    };

    await renderConfusionMatrix(data, container);
    expect(document.querySelectorAll('.vega-embed').length).toBe(1);
  });

  it('sets width of chart', async () => {
    const data = {
      values: [[4, 2, 8], [1, 7, 2], [3, 3, 20]],
      labels: ['cheese', 'pig', 'font'],
    };

    const container = document.getElementById('container') as HTMLElement;
    await renderConfusionMatrix(data, container, {width: 400});

    expect(document.querySelectorAll('.vega-embed').length).toBe(1);
    expect(document.querySelectorAll('canvas').length).toBe(1);
    expect(document.querySelector('canvas')!.style.width).toBe('400px');
  });

  it('sets height of chart', async () => {
    const data = {
      values: [[4, 2, 8], [1, 7, 2], [3, 3, 20]],
      labels: ['cheese', 'pig', 'font'],
    };

    const container = document.getElementById('container') as HTMLElement;
    await renderConfusionMatrix(data, container, {height: 200});

    expect(document.querySelectorAll('.vega-embed').length).toBe(1);
    expect(document.querySelectorAll('canvas').length).toBe(1);
    expect(document.querySelector('canvas')!.style.height).toBe('200px');
  });
});
