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
// TODO use type only imports for these types once we are on a version of
// ts that supports this.
// https://www.typescriptlang.org/docs/handbook/release-notes/typescript-3-8.html#type-only-imports-and-export
// tslint:disable-next-line: no-imports-from-dist
import {Layer} from '@tensorflow/tfjs-layers/dist/engine/topology';
// tslint:disable-next-line: no-imports-from-dist
import {LayersModel} from '@tensorflow/tfjs-layers/dist/engine/training';

import {histogram} from '../render/histogram';
import {getDrawArea} from '../render/render_utils';
import {table} from '../render/table';
import {Drawable, HistogramStats} from '../types';
import {subSurface} from '../util/dom';
import {tensorStats} from '../util/math';

/**
 * Renders a summary of a tf.Model. Displays a table with layer information.
 *
 * ```js
 * const model = tf.sequential({
 *  layers: [
 *    tf.layers.dense({inputShape: [784], units: 32, activation: 'relu'}),
 *    tf.layers.dense({units: 10, activation: 'softmax'}),
 *  ]
 * });
 *
 * const surface = { name: 'Model Summary', tab: 'Model Inspection'};
 * tfvis.show.modelSummary(surface, model);
 * ```
 *
 * @doc {
 *  heading: 'Models & Tensors',
 *  subheading: 'Model Inspection',
 *  namespace: 'show'
 * }
 */
export async function modelSummary(container: Drawable, model: LayersModel) {
  const drawArea = getDrawArea(container);
  const summary = getModelSummary(model);

  const headers = [
    'Layer Name',
    'Output Shape',
    '# Of Params',
    'Trainable',
  ];

  const values = summary.layers.map(
      l =>
          [l.name,
           l.outputShape,
           l.parameters,
           l.trainable,
  ]);

  table(drawArea, {headers, values});
}

/**
 * Renders summary information about a layer and a histogram of parameters in
 * that layer.
 *
 * ```js
 * const model = tf.sequential({
 *  layers: [
 *    tf.layers.dense({inputShape: [784], units: 32, activation: 'relu'}),
 *    tf.layers.dense({units: 10, activation: 'softmax'}),
 *  ]
 * });
 *
 * const surface = { name: 'Layer Summary', tab: 'Model Inspection'};
 * tfvis.show.layer(surface, model.getLayer(undefined, 1));
 * ```
 *
 * @doc {
 *  heading: 'Models & Tensors',
 *  subheading: 'Model Inspection',
 *  namespace: 'show'
 * }
 */
export async function layer(container: Drawable, layer: Layer) {
  const drawArea = getDrawArea(container);
  const details = await getLayerDetails(layer);

  const headers = [
    'Weight Name',
    'Shape',
    'Min',
    'Max',
    '# Params',
    '# Zeros',
    '# NaNs',
    '# Infinity',
  ];

  // Show layer summary
  const weightsInfoSurface = subSurface(drawArea, 'layer-weights-info');
  const detailValues = details.map(
      l =>
          [l.name, l.shape, l.stats.min, l.stats.max, l.weight.size,
           l.stats.numZeros, l.stats.numNans, l.stats.numInfs]);

  table(weightsInfoSurface, {headers, values: detailValues});

  const histogramSelectorSurface = subSurface(drawArea, 'select-layer');
  const layerValuesHistogram = subSurface(drawArea, 'param-distribution');

  const handleSelection = async (layerName: string) => {
    const layer = details.filter(d => d.name === layerName)[0];
    const weights = await layer.weight.data();

    histogram(
        layerValuesHistogram, weights, {height: 150, width: 460, stats: false});
  };

  addHistogramSelector(
      details.map(d => d.name), histogramSelectorSurface, handleSelection);
}

//
// Helper functions
//

function getModelSummary(model: LayersModel) {
  return {
    layers: model.layers.map(getLayerSummary),
  };
}

/*
 * Gets summary information/metadata about a layer.
 */
function getLayerSummary(layer: Layer): LayerSummary {
  let outputShape: string;
  if (Array.isArray(layer.outputShape[0])) {
    const shapes = (layer.outputShape as number[][]).map(s => formatShape(s));
    outputShape = `[${shapes.join(', ')}]`;
  } else {
    outputShape = formatShape(layer.outputShape as number[]);
  }

  return {
    name: layer.name,
    trainable: layer.trainable,
    parameters: layer.countParams(),
    outputShape,
  };
}

interface LayerSummary {
  name: string;
  trainable: boolean;
  parameters: number;
  outputShape: string;
}

/*
 * Gets summary stats and shape for all weights in a layer.
 */
async function getLayerDetails(layer: Layer): Promise<Array<
    {name: string, stats: HistogramStats, shape: string, weight: tf.Tensor}>> {
  const weights = layer.getWeights();
  const layerVariables = layer.weights;
  const statsPromises = weights.map(tensorStats);
  const stats = await Promise.all(statsPromises);
  const shapes = weights.map(w => w.shape);
  return weights.map((weight, i) => ({
                       name: layerVariables[i].name,
                       stats: stats[i],
                       shape: formatShape(shapes[i]),
                       weight,
                     }));
}

function formatShape(shape: number[]): string {
  const oShape: Array<number|string> = shape.slice();
  if (oShape.length === 0) {
    return 'Scalar';
  }
  if (oShape[0] === null) {
    oShape[0] = 'batch';
  }
  return `[${oShape.join(',')}]`;
}

function addHistogramSelector(
    items: string[], parent: HTMLElement,
    // tslint:disable-next-line:no-any
    selectionHandler: (item: string) => any) {
  const select = `
    <select>
      ${items.map((i) => `<option value=${i}>${i}</option>`)}
    </select>
  `;

  const button = `<button>Show Values Distribution for:</button>`;
  const content = `<div>${button}${select}</div>`;

  parent.innerHTML = content;

  // Add listeners
  const buttonEl = parent.querySelector('button');
  const selectEl = parent.querySelector('select');

  buttonEl.addEventListener('click', () => {
    selectionHandler(selectEl.selectedOptions[0].label);
  });
}
