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
import embed, {Mode, VisualizationSpec} from 'vega-embed';

import {Drawable, HeatmapData, HeatmapOptions} from '../types';
import {assert} from '../util/utils';

import {getDrawArea} from './render_utils';

/**
 * Renders a heatmap.
 *
 * ```js
 * const cols = 50;
 * const rows = 20;
 * const values = [];
 * for (let i = 0; i < cols; i++) {
 *   const col = []
 *   for (let j = 0; j < rows; j++) {
 *     col.push(i * j)
 *   }
 *   values.push(col);
 * }
 * const data = { values };
 *
 * // Render to visor
 * const surface = { name: 'Heatmap', tab: 'Charts' };
 * tfvis.render.heatmap(surface, data);
 * ```
 *
 * ```js
 * const data = {
 *   values: [[4, 2, 8, 20], [1, 7, 2, 10], [3, 3, 20, 13]],
 *   xTickLabels: ['cheese', 'pig', 'font'],
 *   yTickLabels: ['speed', 'smoothness', 'dexterity', 'mana'],
 * }
 *
 * // Render to visor
 * const surface = { name: 'Heatmap w Custom Labels', tab: 'Charts' };
 * tfvis.render.heatmap(surface, data);
 * ```
 *
 */
/** @doc {heading: 'Charts', namespace: 'render'} */
export async function heatmap(
    container: Drawable, data: HeatmapData,
    opts: HeatmapOptions = {}): Promise<void> {
  const options = Object.assign({}, defaultOpts, opts);
  const drawArea = getDrawArea(container);

  let inputValues = data.values;
  if (options.rowMajor) {
    let originalShape: number[];
    let transposed: tf.Tensor2D;
    if (inputValues instanceof tf.Tensor) {
      originalShape = inputValues.shape;
      transposed = inputValues.transpose();
    } else {
      originalShape = [inputValues.length, inputValues[0].length];
      transposed =
          tf.tidy(() => tf.tensor2d(inputValues as number[][]).transpose());
    }

    assert(
        transposed.rank === 2,
        'Input to renderHeatmap must be a 2d array or Tensor2d');

    // Download the intermediate tensor values and
    // dispose the transposed tensor.
    inputValues = await transposed.array();
    transposed.dispose();

    const transposedShape = [inputValues.length, inputValues[0].length];
    assert(
        originalShape[0] === transposedShape[1] &&
            originalShape[1] === transposedShape[0],
        `Unexpected transposed shape. Original ${originalShape} : Transposed ${
            transposedShape}`);
  }

  // Format data for vega spec; an array of objects, one for for each cell
  // in the matrix.
  const values: MatrixEntry[] = [];
  const {xTickLabels, yTickLabels} = data;

  // These two branches are very similar but we want to do the test once
  // rather than on every element access
  if (inputValues instanceof tf.Tensor) {
    assert(
        inputValues.rank === 2,
        'Input to renderHeatmap must be a 2d array or Tensor2d');

    const shape = inputValues.shape;
    if (xTickLabels) {
      assert(
          shape[0] === xTickLabels.length,
          `Length of xTickLabels (${
              xTickLabels.length}) must match number of rows
          (${shape[0]})`);
    }

    if (yTickLabels) {
      assert(
          shape[1] === yTickLabels.length,
          `Length of yTickLabels (${
              yTickLabels.length}) must match number of columns
          (${shape[1]})`);
    }

    // This is a slightly specialized version of TensorBuffer.get, inlining it
    // avoids the overhead of a function call per data element access and is
    // specialized to only deal with the 2d case.
    const inputArray = await inputValues.data();
    const [numRows, numCols] = shape;

    for (let row = 0; row < numRows; row++) {
      const x = xTickLabels ? xTickLabels[row] : row;
      for (let col = 0; col < numCols; col++) {
        const y = yTickLabels ? yTickLabels[col] : col;

        const index = (row * numCols) + col;
        const value = inputArray[index];

        values.push({x, y, value});
      }
    }
  } else {
    if (xTickLabels) {
      assert(
          inputValues.length === xTickLabels.length,
          `Number of rows (${inputValues.length}) must match
          number of xTickLabels (${xTickLabels.length})`);
    }

    const inputArray = inputValues as number[][];
    for (let row = 0; row < inputArray.length; row++) {
      const x = xTickLabels ? xTickLabels[row] : row;
      if (yTickLabels) {
        assert(
            inputValues[row].length === yTickLabels.length,
            `Number of columns in row ${row} (${inputValues[row].length})
            must match length of yTickLabels (${yTickLabels.length})`);
      }
      for (let col = 0; col < inputArray[row].length; col++) {
        const y = yTickLabels ? yTickLabels[col] : col;
        const value = inputArray[row][col];
        values.push({x, y, value});
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
      },
      'scale': {'bandPaddingInner': 0, 'bandPaddingOuter': 0},
    },
    'data': {'values': values},
    'mark': 'rect',
    'encoding': {
      'x': {
        'field': 'x',
        'type': options.xType,
        // Maintain sort order of the axis if labels is passed in
        'scale': {'domain': xTickLabels},
        'title': options.xLabel,
      },
      'y': {
        'field': 'y',
        'type': options.yType,
        // Maintain sort order of the axis if labels is passed in
        'scale': {'domain': yTickLabels},
        'title': options.yLabel,
      },
      'fill': {
        'field': 'value',
        'type': 'quantitative',
      },
    }
  };

  let colorRange: string[]|string;
  switch (options.colorMap) {
    case 'blues':
      colorRange = ['#f7fbff', '#4292c6'];
      break;
    case 'greyscale':
      colorRange = ['#000000', '#ffffff'];
      break;
    case 'viridis':
    default:
      colorRange = 'viridis';
      break;
  }

  if (colorRange !== 'viridis') {
    const fill = spec.encoding!.fill;
    // @ts-ignore
    fill.scale = {'range': colorRange};
  }

  if (options.domain) {
    const fill = spec.encoding!.fill;
    // @ts-ignore
    if (fill.scale != null) {
      // @ts-ignore
      fill.scale = Object.assign({}, fill.scale, {'domain': options.domain});
    } else {
      // @ts-ignore
      fill.scale = {'domain': options.domain};
    }
  }

  await embed(drawArea, spec, embedOpts);
}

const defaultOpts = {
  xLabel: null,
  yLabel: null,
  xType: 'ordinal',
  yType: 'ordinal',
  colorMap: 'viridis',
  fontSize: 12,
  domain: null,
  rowMajor: false,
};

interface MatrixEntry {
  x: string|number;
  y: string|number;
  value: number;
}
