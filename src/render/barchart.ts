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

import embed, {Mode, Result as EmbedRes, VisualizationSpec} from 'vega-embed';

import {Drawable, VisOptions} from '../types';

import {getDrawArea, nextFrame, shallowEquals} from './render_utils';

/**
 * Renders a barchart.
 *
 * ```js
 * const data = [
 *   { index: 0, value: 50 },
 *   { index: 1, value: 100 },
 *   { index: 2, value: 150 },
 *  ];
 *
 * // Render to visor
 * const surface = { name: 'Bar chart', tab: 'Charts' };
 * tfvis.render.barchart(surface, data);
 * ```
 *
 * @param container An `HTMLElement` or `Surface` in which to draw the bar
 *    chart. Note that the chart expects to have complete control over
 *    the contents of the container and can clear its contents at will.
 * @param data Data in the following format, (an array of objects)
 *    [ {index: number, value: number} ... ]
 *
 * @param opts optional parameters
 * @param opts.width width of chart in px
 * @param opts.height height of chart in px
 * @param opts.xLabel label for x-axis, set to null to hide the
 * @param opts.yLabel label for y-axis, set to null to hide the
 * @param opts.fontSize fontSize in pixels for text in the chart
 *
 * @returns Promise - indicates completion of rendering
 *
 *
 */
/** @doc {heading: 'Charts', namespace: 'render'} */
export async function barchart(
    container: Drawable, data: Array<{index: number; value: number;}>,
    opts: VisOptions = {}): Promise<void> {
  const drawArea = getDrawArea(container);
  const values = data;
  const options = Object.assign({}, defaultOpts, opts);

  // If we have rendered this chart before with the same options we can do a
  // data only update, else  we do a regular re-render.
  if (instances.has(drawArea)) {
    const instanceInfo = instances.get(drawArea)!;
    if (shallowEquals(options, instanceInfo.lastOptions)) {
      await nextFrame();
      const view = instanceInfo.view;
      const changes = view.changeset().remove(() => true).insert(values);
      await view.change('values', changes).runAsync();
      return;
    }
  }

  const {xLabel, yLabel, xType, yType} = options;

  let xAxis: {}|null = null;
  if (xLabel != null) {
    xAxis = {title: xLabel};
  }

  let yAxis: {}|null = null;
  if (yLabel != null) {
    yAxis = {title: yLabel};
  }

  const embedOpts = {
    actions: false,
    mode: 'vega-lite' as Mode,
    defaultStyle: false,
  };

  const spec: VisualizationSpec = {
    'width': options.width || drawArea.clientWidth,
    'height': options.height || drawArea.clientHeight,
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
    'data': {'values': values, 'name': 'values'},
    'mark': 'bar',
    'encoding': {
      'x': {'field': 'index', 'type': xType, 'axis': xAxis},
      'y': {'field': 'value', 'type': yType, 'axis': yAxis}
    }
  };

  await nextFrame();
  const embedRes = await embed(drawArea, spec, embedOpts);
  instances.set(drawArea, {
    view: embedRes.view,
    lastOptions: options,
  });
}

const defaultOpts = {
  xLabel: '',
  yLabel: '',
  xType: 'ordinal',
  yType: 'quantitative',
  fontSize: 11,
};

// We keep a map of containers to chart instances in order to reuse the instance
// where possible.
const instances: Map<HTMLElement, InstanceInfo> =
    new Map<HTMLElement, InstanceInfo>();

interface InstanceInfo {
  // tslint:disable-next-line:no-any
  view: any;
  lastOptions: VisOptions;
}
