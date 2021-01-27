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

import * as tf from '@tensorflow/tfjs-core';
import embed, {Mode, VisualizationSpec} from 'vega-embed';

import {Drawable, HeatmapData, HeatmapOptions} from '../types';
import {getDefaultHeight, getDefaultWidth} from '../util/dom';
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
 *
 * @doc {heading: 'Charts', namespace: 'render'}
 */
export async function heatmap(
    container: Drawable, data: HeatmapData,
    opts: HeatmapOptions = {}): Promise<void> {
  const options = Object.assign({}, defaultOpts, opts);
  const drawArea = getDrawArea(container);

  let inputValues = data.values;
  if (options.rowMajor) {
    inputValues = await convertToRowMajor(data.values);
  }

  // Data validation
  const {xTickLabels, yTickLabels} = data;
  if (xTickLabels != null) {
    const dimension = 0;
    assertLabelsMatchShape(inputValues, xTickLabels, dimension);
  }

  // Note that we will only do a check on the first element of the second
  // dimension. We do not protect users against passing in a ragged array.
  if (yTickLabels != null) {
    const dimension = 1;
    assertLabelsMatchShape(inputValues, yTickLabels, dimension);
  }

  //
  // Format data for vega spec; an array of objects, one for for each cell
  // in the matrix.
  //
  // If custom labels are passed in for xTickLabels or yTickLabels we need
  // to make sure they are 'unique' before mapping them to visual properties.
  // We therefore append the index of the label to the datum that will be used
  // for that label in the x or y axis. We could do this in all cases but choose
  // not to to avoid unnecessary string operations.
  //
  // We use IDX_SEPARATOR to demarcate the added index
  const IDX_SEPARATOR = '@tfidx@';

  const values: MatrixEntry[] = [];
  if (inputValues instanceof tf.Tensor) {
    assert(
        inputValues.rank === 2,
        'Input to renderHeatmap must be a 2d array or Tensor2d');

    // This is a slightly specialized version of TensorBuffer.get, inlining it
    // avoids the overhead of a function call per data element access and is
    // specialized to only deal with the 2d case.
    const inputArray = await inputValues.data();
    const [numRows, numCols] = inputValues.shape;

    for (let row = 0; row < numRows; row++) {
      const x = xTickLabels ? `${xTickLabels[row]}${IDX_SEPARATOR}${row}` : row;
      for (let col = 0; col < numCols; col++) {
        const y =
            yTickLabels ? `${yTickLabels[col]}${IDX_SEPARATOR}${col}` : col;

        const index = (row * numCols) + col;
        const value = inputArray[index];

        values.push({x, y, value});
      }
    }
  } else {
    const inputArray = inputValues;
    for (let row = 0; row < inputArray.length; row++) {
      const x = xTickLabels ? `${xTickLabels[row]}${IDX_SEPARATOR}${row}` : row;
      for (let col = 0; col < inputArray[row].length; col++) {
        const y =
            yTickLabels ? `${yTickLabels[col]}${IDX_SEPARATOR}${col}` : col;
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
      },
      'scale': {'bandPaddingInner': 0, 'bandPaddingOuter': 0},
    },
    //@ts-ignore
    'data': {'values': values},
    'mark': {'type': 'rect', 'tooltip': true},
    'encoding': {
      'x': {
        'field': 'x',
        'type': options.xType,
        'title': options.xLabel,
        'sort': false,
      },
      'y': {
        'field': 'y',
        'type': options.yType,
        'title': options.yLabel,
        'sort': false,
      },
      'fill': {
        'field': 'value',
        'type': 'quantitative',
      }
    }
  };

  //
  // Format custom labels to remove the appended indices
  //
  const suffixPattern = `${IDX_SEPARATOR}\\d+$`;
  const suffixRegex = new RegExp(suffixPattern);
  if (xTickLabels) {
    // @ts-ignore
    spec.encoding.x.axis = {
      'labelExpr': `replace(datum.value, regexp(/${suffixPattern}/), '')`,
    };
  }

  if (yTickLabels) {
    // @ts-ignore
    spec.encoding.y.axis = {
      'labelExpr': `replace(datum.value, regexp(/${suffixPattern}/), '')`,
    };
  }

  // Customize tooltip formatting to remove the appended indices
  if (xTickLabels || yTickLabels) {
    //@ts-ignore
    embedOpts.tooltip = {
      sanitize: (value: string|number) => {
        const valueString = String(value);
        return valueString.replace(suffixRegex, '');
      }
    };
  }

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
    //@ts-ignore
    const fill = spec.encoding.fill;
    // @ts-ignore
    fill.scale = {'range': colorRange};
  }

  if (options.domain) {
    //@ts-ignore
    const fill = spec.encoding.fill;
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

async function convertToRowMajor(inputValues: number[][]|
                                 tf.Tensor2D): Promise<number[][]> {
  let originalShape: number[];
  let transposed: tf.Tensor2D;
  if (inputValues instanceof tf.Tensor) {
    originalShape = inputValues.shape;
    transposed = inputValues.transpose();
  } else {
    originalShape = [inputValues.length, inputValues[0].length];
    transposed = tf.tidy(() => tf.tensor2d(inputValues).transpose());
  }

  assert(
      transposed.rank === 2,
      'Input to renderHeatmap must be a 2d array or Tensor2d');

  // Download the intermediate tensor values and
  // dispose the transposed tensor.
  const transposedValues = await transposed.array();
  transposed.dispose();

  const transposedShape = [transposedValues.length, transposedValues[0].length];
  assert(
      originalShape[0] === transposedShape[1] &&
          originalShape[1] === transposedShape[0],
      `Unexpected transposed shape. Original ${originalShape} : Transposed ${
          transposedShape}`);
  return transposedValues;
}

function assertLabelsMatchShape(
    inputValues: number[][]|tf.Tensor2D, labels: string[], dimension: 0|1) {
  const shape = inputValues instanceof tf.Tensor ?
      inputValues.shape :
      [inputValues.length, inputValues[0].length];
  if (dimension === 0) {
    assert(
        shape[0] === labels.length,
        `Length of xTickLabels (${labels.length}) must match number of rows` +
            ` (${shape[0]})`);
  } else if (dimension === 1) {
    assert(
        shape[1] === labels.length,
        `Length of yTickLabels (${
            labels.length}) must match number of columns (${shape[1]})`);
  }
}

const defaultOpts: HeatmapOptions = {
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
