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
import {TopLevelSpec} from 'vega-lite';

import {Drawable, Point2D, XYPlotData, XYPlotOptions} from '../types';
import {getDefaultHeight, getDefaultWidth} from '../util/dom';
import {assert} from '../util/utils';

import {getDrawArea} from './render_utils';

/**
 * Renders a line chart
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
 * const surface = { name: 'Line chart', tab: 'Charts' };
 * tfvis.render.linechart(surface, data);
 * ```
 *
 * ```js
 * const series1 = Array(100).fill(0)
 *   .map(y => Math.random() * 100 + 50)
 *   .map((y, x) => ({ x, y, }));
 *
 * const data = { values: [series1] }
 *
 * // Render to visor
 * const surface = { name: 'Zoomed Line Chart', tab: 'Charts' };
 * tfvis.render.linechart(surface, data, { zoomToFit: true });
 * ```
 *
 * @doc {heading: 'Charts', namespace: 'render'}
 */
export async function linechart(
    container: Drawable, data: XYPlotData,
    opts: XYPlotOptions = {}): Promise<void> {
  // Nest data if necessary before further processing
  const _data = Array.isArray(data.values[0]) ? data.values as Point2D[][] :
                                                [data.values] as Point2D[][];
  const numValues = _data[0].length;

  // Create series names if none were passed in.
  const _series: string[] =
      data.series ? data.series : _data.map((_, i) => `Series ${i + 1}`);
  assert(
      _series.length === _data.length,
      'Must have an equal number of series labels as there are data series');

  if (opts.seriesColors != null) {
    assert(
        opts.seriesColors.length === _data.length,
        'Must have an equal number of series colors as there are data series');
  }

  const vlChartValues: VLChartValue[] = [];
  for (let valueIdx = 0; valueIdx < numValues; valueIdx++) {
    const v: VLChartValue = {
      x: valueIdx,
    };

    _series.forEach((seriesName, seriesIdx) => {
      const seriesValue = _data[seriesIdx][valueIdx].y;
      v[seriesName] = seriesValue;
      v[`${seriesName}-name`] = seriesName;
    });
    vlChartValues.push(v);
  }

  const options = Object.assign({}, defaultOpts, opts);

  const yScale = (): {}|undefined => {
    if (options.zoomToFit) {
      return {'zero': false};
    } else if (options.yAxisDomain != null) {
      return {'domain': options.yAxisDomain};
    }
    return undefined;
  };

  const sharedEncoding = {
    x: {
      field: 'x',
      type: options.xType,
      title: options.xLabel,
    },
    tooltip: [
      {field: 'x', type: 'quantitative'},
      ..._series.map(seriesName => {
        return {
          field: seriesName,
          type: 'quantitative',
        };
      }),
    ]
  };

  const lineLayers: TopLevelSpec[] = _series.map((seriesName) => {
    return {
      // data will be defined at the chart level.
      'data': undefined,
      'mark': {'type': 'line', 'clip': true},
      'encoding': {
        // Note: the encoding for 'x' is shared
        // Add a y encoding for this series
        'y': {
          'field': seriesName,
          'type': options.yType,
          'title': options.yLabel,
          'scale': yScale(),
        },
        'color': {
          'field': `${seriesName}-name`,
          'type': 'nominal',
          'legend': {'values': _series, title: null},
          'scale': {
            'range': options.seriesColors,
          }
        },
      }
    };
  });

  const tooltipLayer = {
    'mark': 'rule',
    'selection': {
      'hover': {
        'type': 'single',
        'on': 'mouseover',
        'nearest': true,
        clear: 'mouseout',
      }
    },
    'encoding': {
      'color': {
        'value': 'grey',
        'condition': {
          'selection': {'not': 'hover'},
          'value': 'transparent',
        }
      }
    }
  };

  const drawArea = getDrawArea(container);
  const spec = {
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
    'data': {'values': vlChartValues},
    'encoding': sharedEncoding,
    'layer': [
      ...lineLayers,
      tooltipLayer,
    ],
  };

  const embedOpts = {
    actions: false,
    mode: 'vega-lite' as Mode,
    defaultStyle: false,
  };

  await embed(drawArea, spec as VisualizationSpec, embedOpts);
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

interface VLChartValue {
  x: number;
  [key: string]: string|number;
}
