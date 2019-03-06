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

import {HeatmapData} from '../types';

import {heatmap} from './heatmap';

describe('renderHeatmap', () => {
  let pixelRatio: number;

  beforeEach(() => {
    document.body.innerHTML = '<div id="container"></div>';
    pixelRatio = window.devicePixelRatio;
  });

  it('renders a chart', async () => {
    const data: HeatmapData = {
      values: [[4, 2, 8], [1, 7, 2], [3, 3, 20]],
    };

    const container = document.getElementById('container') as HTMLElement;
    await heatmap(container, data);

    expect(document.querySelectorAll('.vega-embed').length).toBe(1);
  });

  it('renders a chart with a tensor', async () => {
    const values = tf.tensor2d([[4, 2, 8], [1, 7, 2], [3, 3, 20]]);
    const data: HeatmapData = {
      values,
    };

    const container = document.getElementById('container') as HTMLElement;
    await heatmap(container, data);

    expect(document.querySelectorAll('.vega-embed').length).toBe(1);

    values.dispose();
  });

  it('throws an exception with a non 2d tensor', async () => {
    const values = tf.tensor1d([4, 2, 8, 1, 7, 2, 3, 3, 20]);
    const data = {
      values,
    };

    const container = document.getElementById('container') as HTMLElement;

    let threw = false;
    try {
      // @ts-ignore — passing in the wrong datatype
      await heatmap(data, container);
    } catch (e) {
      threw = true;
    } finally {
      values.dispose();
    }
    expect(threw).toBe(true);
  });

  it('renders a chart with custom colormap', async () => {
    const data: HeatmapData = {
      values: [[4, 2, 8], [1, 7, 2], [3, 3, 20]],
    };

    const container = document.getElementById('container') as HTMLElement;
    await heatmap(container, data, {colorMap: 'greyscale'});

    expect(document.querySelectorAll('.vega-embed').length).toBe(1);
  });

  it('renders a chart with custom domain', async () => {
    const data: HeatmapData = {
      values: [[4, 2, 8], [1, 7, 2], [3, 3, 20]],
    };

    const container = document.getElementById('container') as HTMLElement;
    await heatmap(container, data, {domain: [0, 30]});

    expect(document.querySelectorAll('.vega-embed').length).toBe(1);
  });

  it('renders a chart with custom labels', async () => {
    const data: HeatmapData = {
      values: [[4, 2, 8], [1, 7, 2], [3, 3, 20]],
      xTickLabels: ['cheese', 'pig', 'font'],
      yTickLabels: ['speed', 'dexterity', 'roundness'],
    };

    const container = document.getElementById('container') as HTMLElement;
    await heatmap(container, data);

    expect(document.querySelectorAll('.vega-embed').length).toBe(1);
  });

  it('updates the chart', async () => {
    let data: HeatmapData = {
      values: [[4, 2, 8], [1, 7, 2], [3, 3, 20]],
    };

    const container = document.getElementById('container') as HTMLElement;

    await heatmap(container, data);
    expect(document.querySelectorAll('.vega-embed').length).toBe(1);

    data = {
      values: [[43, 2, 8], [1, 7, 2], [3, 3, 20]],
    };

    await heatmap(container, data);
    expect(document.querySelectorAll('.vega-embed').length).toBe(1);
  });

  it('sets width of chart', async () => {
    const data: HeatmapData = {
      values: [[4, 2, 8], [1, 7, 2], [3, 3, 20]],
    };

    const container = document.getElementById('container') as HTMLElement;
    await heatmap(container, data, {width: 400});

    expect(document.querySelectorAll('.vega-embed').length).toBe(1);
    expect(document.querySelectorAll('canvas').length).toBe(1);
    expect(document.querySelector('canvas')!.width).toBe(400 * pixelRatio);
  });

  it('sets height of chart', async () => {
    const data: HeatmapData = {
      values: [[4, 2, 8], [1, 7, 2], [3, 3, 20]],
    };

    const container = document.getElementById('container') as HTMLElement;
    await heatmap(container, data, {height: 200});

    expect(document.querySelectorAll('.vega-embed').length).toBe(1);
    expect(document.querySelectorAll('canvas').length).toBe(1);
    expect(document.querySelector('canvas')!.height).toBe(200 * pixelRatio);
  });
});
