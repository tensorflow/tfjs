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

import {Drawable, Point2D, XYPlotOptions} from '../types';

import {getDrawArea} from './render_utils';

/**
 * Renders a line chart
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
 * @param opts.zoomToFit a boolean indicating whether to allow non-zero
 * baselines setting this to true allows the line chart to take up more room in
 * the plot.
 * @param opts.yAxisDomain array of two numbers indicating the domain of the y
 * axis. This is overriden by zoomToFit
 */
export async function renderLinechart(
    data: {values: Point2D[][]|Point2D[], series?: string[]},
    container: Drawable, opts: XYPlotOptions = {}): Promise<void> {
  let inputArray = data.values;
  const _series = data.series == null ? [] : data.series;

  // Nest data if necessary before further processing
  inputArray = Array.isArray(inputArray[0]) ? inputArray as Point2D[][] :
                                              [inputArray] as Point2D[][];

  const values: Point2D[] = [];
  inputArray.forEach((seriesData, i) => {
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

  const yScale = (): {}|undefined => {
    if (options.zoomToFit) {
      return {'zero': false};
    } else if (options.yAxisDomain != null) {
      return {'domain': options.yAxisDomain};
    }
    return undefined;
  };

  // tslint:disable-next-line:no-any
  const encodings: any = {
    'x': {
      'field': 'x',
      'type': options.xType,
      'title': options.xLabel,
    },
    'y': {
      'field': 'y',
      'type': options.yType,
      'title': options.yLabel,
      'scale': yScale(),
    },
    'color': {
      'field': 'series',
      'type': 'nominal',
    },
  };

  // tslint:disable-next-line:no-any
  let domainFilter: any;
  if (options.yAxisDomain != null) {
    domainFilter = {'filter': {'field': 'y', 'range': options.yAxisDomain}};
  }

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
    'layer': [
      {
        // Render the main line chart
        'mark': {
          'type': 'line',
          'clip': true,
        },
        'encoding': encodings,
      },
      {
        // Render invisible points for all the the data to make selections
        // easier
        'mark': {'type': 'point'},
        // 'encoding': encodings,
        // If a custom domain is set, filter out the values that will not
        // fit we do this on the points and not the line so that the line
        // still appears clipped for values outside the domain but we can
        // still operate on an unclipped set of points.
        'transform': options.yAxisDomain ? [domainFilter] : undefined,
        'selection': {
          'nearestPoint': {
            'type': 'single',
            'on': 'mouseover',
            'nearest': true,
            'empty': 'none',
            'encodings': ['x'],
          },
        },
        'encoding': Object.assign({}, encodings, {
          'opacity': {
            'value': 0,
            'condition': {
              'selection': 'nearestPoint',
              'value': 1,
            },
          }
        }),
      },
      {
        // Render a tooltip where the selection is
        'transform': [
          {'filter': {'selection': 'nearestPoint'}},
          domainFilter
        ].filter(Boolean),  // remove undefineds from array
        'mark': {
          'type': 'text',
          'align': 'left',
          'dx': 5,
          'dy': -5,
          'color': 'black',
        },
        'encoding': Object.assign({}, encodings, {
          'text': {
            'type': options.xType,
            'field': 'y',
            'format': '.6f',
          },
          // Unset text color to improve readability
          'color': undefined,
        }),
      },
      {
        // Draw a vertical line where the selection is
        'transform': [{'filter': {'selection': 'nearestPoint'}}],
        'mark': {'type': 'rule', 'color': 'gray'},
        'encoding': {
          'x': {
            'type': options.xType,
            'field': 'x',
          }
        }
      },
    ],
  };

  await embed(drawArea, spec, embedOpts);
  return Promise.resolve();
}

const defaultOpts = {
  xLabel: 'x',
  yLabel: 'y',
  xType: 'quantitative',
  yType: 'quantitative',
  zoomToFit: false,
};
