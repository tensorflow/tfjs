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

import {ConfusionMatrixData, Drawable, VisOptions,} from '../types';
import {getDrawArea} from './render_utils';

/**
 * Renders a confusion matrix.
 *
 * Can optionally exclude the diagonal from being shaded if one wants the visual
 * focus to be on the incorrect classifications. Note that if the classification
 * is perfect (i.e. only the diagonal has values) then the diagonal will always
 * be shaded.
 *
 * ```js
 * const rows = 5;
 * const cols = 5;
 * const values = [];
 * for (let i = 0; i < rows; i++) {
 *   const row = []
 *   for (let j = 0; j < cols; j++) {
 *     row.push(Math.round(Math.random() * 50));
 *   }
 *   values.push(row);
 * }
 * const data = { values };
 *
 * // Render to visor
 * const surface = { name: 'Confusion Matrix', tab: 'Charts' };
 * tfvis.render.confusionMatrix(surface, data);
 * ```
 *
 * ```js
 * // The diagonal can be excluded from shading.
 *
 * const data = {
 *   values: [[4, 2, 8], [1, 7, 2], [3, 3, 20]],
 * }
 *
 * // Render to visor
 * const surface = {
 *  name: 'Confusion Matrix with Excluded Diagonal', tab: 'Charts'
 * };
 *
 * tfvis.render.confusionMatrix(surface, data, {
 *   shadeDiagonal: false
 * });
 * ```
 *
 * @param container An `HTMLElement` or `Surface` in which to draw the chart
 * @param data Data consists of an object with a 'values' property
 *  and a 'labels' property.
 *  {
 *    // a matrix of numbers representing counts for each (label, prediction)
 *    // pair
 *    values: number[][],
 *
 *    // Human readable labels for each class in the matrix. Optional
 *    tickLabels?: string[]
 *  }
 *  e.g.
 *  {
 *    values: [[80, 23], [56, 94]],
 *    tickLabels: ['dog', 'cat'],
 *  }
 * @param opts optional parameters
 * @param opts.shadeDiagonal boolean that controls whether or not to color cells
 * on the diagonal. Defaults to true
 * @param opts.showTextOverlay boolean that controls whether or not to render
 * the values of each cell as text. Defaults to true
 * @param opts.width width of chart in px
 * @param opts.height height of chart in px
 * @param opts.fontSize fontSize in pixels for text in the chart
 *
 */
/** @doc {heading: 'Charts', namespace: 'render'} */
export async function confusionMatrix(
    container: Drawable, data: ConfusionMatrixData,
    opts: VisOptions&
    {shadeDiagonal?: boolean, showTextOverlay?: boolean} = {}): Promise<void> {
  const options = Object.assign({}, defaultOpts, opts);
  const drawArea = getDrawArea(container);

  // Format data for vega spec; an array of objects, one for for each cell
  // in the matrix.
  const values: MatrixEntry[] = [];

  const inputArray = data.values;
  const tickLabels = data.tickLabels || [];
  const generateLabels = tickLabels.length === 0;

  let nonDiagonalIsAllZeroes = true;
  for (let i = 0; i < inputArray.length; i++) {
    const label = generateLabels ? `Class ${i}` : tickLabels[i];

    if (generateLabels) {
      tickLabels.push(label);
    }

    for (let j = 0; j < inputArray[i].length; j++) {
      const prediction = generateLabels ? `Class ${j}` : tickLabels[j];

      const count = inputArray[i][j];
      if (i === j && !options.shadeDiagonal) {
        values.push({
          label,
          prediction,
          diagCount: count,
          noFill: true,
        });
      } else {
        values.push({
          label,
          prediction,
          count,
        });
        // When not shading the diagonal we want to check if there is a non
        // zero value. If all values are zero we will not color them as the
        // scale will be invalid.
        if (count !== 0) {
          nonDiagonalIsAllZeroes = false;
        }
      }
    }
  }

  if (!options.shadeDiagonal && nonDiagonalIsAllZeroes) {
    // User has specified requested not to shade the diagonal but all the other
    // values are zero. We have two choices, don't shade the anything or only
    // shade the diagonal. We choose to shade the diagonal as that is likely
    // more helpful even if it is not what the user specified.
    for (const val of values) {
      if (val.noFill === true) {
        val.noFill = false;
        val.count = val.diagCount;
      }
    }
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
    'data': {'values': values},
    'encoding': {
      'x': {
        'field': 'prediction',
        'type': 'ordinal',
        // Maintain sort order of the axis if labels is passed in
        'scale': {'domain': tickLabels},
      },
      'y': {
        'field': 'label',
        'type': 'ordinal',
        // Maintain sort order of the axis if labels is passed in
        'scale': {'domain': tickLabels},
      },
    },
    'layer': [
      {
        // The matrix
        'mark': {
          'type': 'rect',
        },
        'encoding': {
          'fill': {
            'condition': {
              'test': 'datum["noFill"] == true',
              'value': 'white',
            },
            'field': 'count',
            'type': 'quantitative',
            'scale': {'range': ['#f7fbff', '#4292c6']},
          },
          'tooltip': {
            'condition': {
              'test': 'datum["noFill"] == true',
              'field': 'diagCount',
              'type': 'nominal',
            },
            'field': 'count',
            'type': 'nominal',
          }
        },

      },
    ]
  };

  if (options.showTextOverlay) {
    spec.layer.push({
      // The text labels
      'mark': {'type': 'text', 'baseline': 'middle'},
      'encoding': {
        'text': {
          'condition': {
            'test': 'datum["noFill"] == true',
            'field': 'diagCount',
            'type': 'nominal',
          },
          'field': 'count',
          'type': 'nominal',
        },
      }
    });
  }

  await embed(drawArea, spec, embedOpts);
  return Promise.resolve();
}

const defaultOpts = {
  xLabel: null,
  yLabel: null,
  xType: 'nominal',
  yType: 'nominal',
  shadeDiagonal: true,
  fontSize: 12,
  showTextOverlay: true,
  height: 400,
};

interface MatrixEntry {
  label: string;
  prediction: string;
  count?: number;
  diagCount?: number;
  noFill?: boolean;
}
