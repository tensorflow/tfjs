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

import embed, {Mode, VisualizationSpec} from 'vega-embed';

import {Drawable, Point2D, VisOptions} from '../types';

import {getDrawArea} from './render_utils';

/**
 * Renders a scatter plot
 *
 * @param data Data in the following format
 *  {
 *    // A nested array of objects each with an x and y property,
 *    // one per series.
 *    // If you only have one series to render you can just pass an array
 *    // of objects with x, y properties
 *    values: {x: number, y: number}[][]
 *
 *    // An array of strings with the names of each series passed above.
 *    // Optional
 *    series: string[]
 *  }
 * @param container An HTMLElement in which to draw the chart
 * @param opts optional parameters
 * @param opts.width width of chart in px
 * @param opts.height height of chart in px
 * @param opts.xLabel label for x axis
 * @param opts.yLabel label for y axis
 */
export async function renderScatterplot(
    data: {values: Point2D[][]|Point2D[], series?: string[]},
    container: Drawable, opts: VisOptions = {}): Promise<void> {
  let _values = data.values;
  const _series = data.series == null ? [] : data.series;

  // Nest data if necessary before further processing
  _values = Array.isArray(_values[0]) ? _values as Point2D[][] :
                                        [_values] as Point2D[][];

  const values: Point2D[] = [];
  _values.forEach((seriesData, i) => {
    const seriesName: string =
        _series[i] != null ? _series[i] : `Series ${i + 1}`;
    const seriesVals =
        seriesData.map(v => Object.assign({}, v, {series: seriesName}));
    values.push(...seriesVals);
  });

  const drawArea = getDrawArea(container);
  const options = Object.assign({}, defaultOpts, opts);

  const embedOpts = {
    actions: false,
    mode: 'vega-lite' as Mode,
  };

  const spec: VisualizationSpec = {
    'width': options.width || drawArea.clientWidth,
    'height': options.height || drawArea.clientHeight,
    'padding': 5,
    'autosize': {
      'type': 'fit',
      'contains': 'padding',
      'resize': true,
    },
    'data': {'values': values},
    'mark': {'type': 'point'},
    'encoding': {
      'x': {
        'field': 'x',
        'type': options.xType,
        'title': options.xLabel,
      },
      'y': {
        'field': 'y',
        'type': options.yType,
        'title': options.yLabel,
      },
      'color': {
        'field': 'series',
        'type': 'nominal',
      },
      'shape': {
        'field': 'series',
        'type': 'nominal',
      },
      // TODO revisit when https://github.com/vega/vega-embed/issues/96 is
      // resolved
      'tooltip': {'field': 'value', 'type': options.yType}
    },
  };

  await embed(drawArea, spec, embedOpts);
  return Promise.resolve();
}

const defaultOpts = {
  xLabel: 'x',
  yLabel: 'y',
  xType: 'quantitative',
  yType: 'quantitative',
};
