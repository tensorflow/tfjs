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

import {Drawable, Point2D, XYPlotData, XYPlotOptions} from '../types';
import {getDefaultHeight, getDefaultWidth} from '../util/dom';
import {assert} from '../util/utils';
import {getDrawArea} from './render_utils';

/**
 * Renders a scatter plot
 *
 * ```js
 * const series1 = Array(100).fill(0)
 *   .map(y => Math.random() * 100 - (Math.random() * 50))
 *   .map((y, x) => ({ x, y, }));
 *
 * const series2 = Array(100).fill(0)
 *   .map(y => Math.random() * 100 - (Math.random() * 150))
 *   .map((y, x) => ({ x, y, }));
 *
 * const series = ['First', 'Second'];
 * const data = { values: [series1, series2], series }
 *
 * const surface = { name: 'Scatterplot', tab: 'Charts' };
 * tfvis.render.scatterplot(surface, data);
 * ```
 *
 * @doc {heading: 'Charts', namespace: 'render'}
 */
export async function scatterplot(
    container: Drawable, data: XYPlotData,
    opts: XYPlotOptions = {}): Promise<void> {
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

  if (opts.seriesColors != null) {
    assert(
        opts.seriesColors.length === _values.length,
        'Must have an equal number of series colors as there are data series');
  }

  const drawArea = getDrawArea(container);
  const options = Object.assign({}, defaultOpts, opts);

  const embedOpts = {
    actions: false,
    mode: 'vega-lite' as Mode,
    defaultStyle: false,
  };

  const xDomain = (): {}|undefined => {
    if (options.zoomToFit) {
      return {'zero': false};
    } else if (options.xAxisDomain != null) {
      return {'domain': options.xAxisDomain};
    }
    return undefined;
  };

  const yDomain = (): {}|undefined => {
    if (options.zoomToFit) {
      return {'zero': false};
    } else if (options.yAxisDomain != null) {
      return {'domain': options.yAxisDomain};
    }
    return undefined;
  };

  const spec: VisualizationSpec = {
    'width': options.width || getDefaultWidth(drawArea),
    'height': options.height || getDefaultHeight(drawArea),
    'padding': 0,
    'autosize': {
      'type': 'fit',
      'contains': 'padding',
      'resize': true,
    },
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
    //@ts-ignore
    'data': {
      'values': values,
    },
    'mark': {
      'type': 'point',
      'clip': true,
      'tooltip': {'content': 'data'},
    },
    'encoding': {
      'x': {
        'field': 'x',
        'type': options.xType,
        'title': options.xLabel,
        'scale': xDomain(),
      },
      'y': {
        'field': 'y',
        'type': options.yType,
        'title': options.yLabel,
        'scale': yDomain(),
      },
      'color': {
        'field': 'series',
        'type': 'nominal',
        'scale': {
          'range': options.seriesColors,
        }
      },
      'shape': {
        'field': 'series',
        'type': 'nominal',
      }
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
  zoomToFit: false,
  fontSize: 11,
};
