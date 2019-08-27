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

import {barchart} from './barchart';

describe('renderBarChart', () => {
  let pixelRatio: number;

  beforeEach(() => {
    document.body.innerHTML = '<div id="container"></div>';
    pixelRatio = window.devicePixelRatio;
  });

  it('renders a bar chart', async () => {
    const data = [
      {index: 0, value: 50},
      {index: 1, value: 100},
      {index: 2, value: 230},
    ];

    const container = document.getElementById('container');
    await barchart(container, data);

    expect(document.querySelectorAll('.vega-embed').length).toBe(1);
  });

  it('re-renders a bar chart', async () => {
    const data = [
      {index: 0, value: 50},
      {index: 1, value: 100},
      {index: 2, value: 230},
    ];

    const container = document.getElementById('container');

    await barchart(container, data);
    expect(document.querySelectorAll('.vega-embed').length).toBe(1);

    await barchart(container, data);
    expect(document.querySelectorAll('.vega-embed').length).toBe(1);
  });

  it('updates a bar chart', async () => {
    let data = [
      {index: 0, value: 50},
      {index: 1, value: 100},
      {index: 2, value: 150},
    ];

    const container = document.getElementById('container');

    await barchart(container, data);
    expect(document.querySelectorAll('.vega-embed').length).toBe(1);

    data = [
      {index: 0, value: 50},
      {index: 1, value: 100},
      {index: 2, value: 150},
    ];

    await barchart(container, data);
    expect(document.querySelectorAll('.vega-embed').length).toBe(1);
  });

  it('sets width of chart', async () => {
    const data = [
      {index: 0, value: 50},
      {index: 1, value: 100},
      {index: 2, value: 230},
    ];

    const container = document.getElementById('container');
    await barchart(container, data, {width: 400});

    expect(document.querySelectorAll('.vega-embed').length).toBe(1);
    expect(document.querySelectorAll('canvas').length).toBe(1);
    expect(document.querySelector('canvas').width).toBe(400 * pixelRatio);
  });

  it('sets height of chart', async () => {
    const data = [
      {index: 0, value: 50},
      {index: 1, value: 100},
      {index: 2, value: 230},
    ];

    const container = document.getElementById('container');
    await barchart(container, data, {height: 200});

    expect(document.querySelectorAll('.vega-embed').length).toBe(1);
    expect(document.querySelectorAll('canvas').length).toBe(1);
    expect(document.querySelector('canvas').height).toBe(200 * pixelRatio);
  });
});
