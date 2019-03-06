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

import {format as d3Format} from 'd3-format';
import embed, {Mode, VisualizationSpec} from 'vega-embed';

import {HistogramOpts, HistogramStats, TypedArray} from '../types';
import {subSurface} from '../util/dom';
import {arrayStats} from '../util/math';

import {table} from './table';

const defaultOpts = {
  maxBins: 12,
  fontSize: 11,
};

/**
 * Renders a histogram of values
 *
 * ```js
 * const data = Array(100).fill(0)
 *   .map(x => Math.random() * 100 - (Math.random() * 50))
 *
 * // Push some special values for the stats table.
 * data.push(Infinity);
 * data.push(NaN);
 * data.push(0);
 *
 * const surface = { name: 'Histogram', tab: 'Charts' };
 * tfvis.render.histogram(surface, data);
 * ```
 *
 * @param container An `HTMLElement`|`Surface` in which to draw the histogram
 * @param data Data in the following format:
 *  `[ {value: number}, ... ]` or `[number]` or `TypedArray`
 * @param opts optional parameters
 * @param opts.width width of chart in px
 * @param opts.height height of chart in px
 * @param opts.fontSize fontSize in pixels for text in the chart
 * @param opts.maxBins maximimum number of bins to use in histogram
 * @param opts.stats summary statistics to show. These will be computed
 *    internally if no stats are passed. Pass `false` to not compute any stats.
 *    Callers are allowed to pass in their own stats as in some cases they
 *    may be able to compute them more efficiently.
 *
 *    Stats should have the following format
 *    {
 *      numVals?: number,
 *      min?: number,
 *      max?: number,
 *      numZeros?: number,
 *      numNans?: number
 *    }
 *
 */
/** @doc {heading: 'Charts', namespace: 'render'} */
export async function histogram(
    container: HTMLElement, data: Array<{value: number}>|number[]|TypedArray,
    opts: HistogramOpts = {}) {
  const values = prepareData(data);

  const options = Object.assign({}, defaultOpts, opts);

  const embedOpts = {
    actions: false,
    mode: 'vega-lite' as Mode,
    defaultStyle: false,
  };

  const histogramContainer = subSurface(container, 'histogram');
  if (opts.stats !== false) {
    const statsContainer = subSurface(container, 'stats', {
      prepend: true,
    });
    let stats: HistogramStats;

    if (opts.stats) {
      stats = opts.stats;
    } else {
      stats = arrayStats(values.map(x => x.value));
    }
    renderStats(stats, statsContainer, {fontSize: options.fontSize});
  }

  // Now that we have rendered stats we need to remove any NaNs and Infinities
  // before rendering the histogram
  const filtered = [];
  for (let i = 0; i < values.length; i++) {
    const val = values[i].value;
    if (val != null && isFinite(val)) {
      filtered.push(values[i]);
    }
  }

  const histogramSpec: VisualizationSpec = {

    'width': options.width || histogramContainer.clientWidth,
    'height': options.height || histogramContainer.clientHeight,
    'padding': 0,
    'autosize': {
      'type': 'fit',
      'contains': 'padding',
      'resize': true,
    },
    'data': {'values': filtered},
    'mark': 'bar',
    'config': {
      'axis': {
        'labelFontSize': options.fontSize,
        'titleFontSize': options.fontSize,
      },
      'text': {'fontSize': options.fontSize},
      'legend': {
        'labelFontSize': options.fontSize,
        'titleFontSize': options.fontSize,
      }
    },
    'encoding': {
      'x': {
        'bin': {'maxbins': options.maxBins},
        'field': 'value',
        'type': 'quantitative',
      },
      'y': {
        'aggregate': 'count',
        'type': 'quantitative',
      },
      'color': {
        // TODO extract to theme?
        'value': '#001B44',
      }
    }
  };

  return embed(histogramContainer, histogramSpec, embedOpts);
}

function renderStats(
    stats: HistogramStats, container: HTMLElement, opts: {fontSize: number}) {
  const format = d3Format(',.4~f');
  const pctFormat = d3Format('.4~p');

  const headers: string[] = [];
  const vals: string[] = [];

  if (stats.numVals != null) {
    headers.push('Num Vals');
    vals.push(format(stats.numVals));
  }

  if (stats.min != null) {
    headers.push('Min');
    vals.push(format(stats.min));
  }

  if (stats.max != null) {
    headers.push('Max');
    vals.push(format(stats.max));
  }

  if (stats.numZeros != null) {
    headers.push('# Zeros');
    let zeroPct = '';
    if (stats.numVals) {
      zeroPct = stats.numZeros > 0 ?
          `(${pctFormat(stats.numZeros / stats.numVals)})` :
          '';
    }

    vals.push(`${format(stats.numZeros)} ${zeroPct}`);
  }

  if (stats.numNans != null) {
    headers.push('# NaNs');
    let nanPct = '';
    if (stats.numVals) {
      nanPct = stats.numNans > 0 ?
          `(${pctFormat(stats.numNans / stats.numVals)})` :
          '';
    }

    vals.push(`${format(stats.numNans)} ${nanPct}`);
  }

  if (stats.numInfs != null) {
    headers.push('# Infinity');
    let infPct = '';
    if (stats.numVals) {
      infPct = stats.numInfs > 0 ?
          `(${pctFormat(stats.numInfs / stats.numVals)})` :
          '';
    }

    vals.push(`${format(stats.numInfs)} ${infPct}`);
  }

  table(container, {headers, values: [vals]}, opts);
}

/**
 * Formats data to the internal format used by this chart.
 */
function prepareData(data: Array<{value: number}>|number[]|
                     TypedArray): Array<{value: number}> {
  if (data.length == null) {
    throw new Error('input data must be an array');
  }

  if (data.length === 0) {
    return [];
  } else if (typeof data[0] === 'object') {
    if ((data[0] as {value: number}).value == null) {
      throw new Error('input data must have a value field');
    } else {
      return data as Array<{value: number}>;
    }
  } else {
    const ret = Array(data.length);
    for (let i = 0; i < data.length; i++) {
      ret[i] = {value: data[i]};
    }
    return ret;
  }
}
