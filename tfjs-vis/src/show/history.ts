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

import {Logs} from '@tensorflow/tfjs-layers';

import {linechart} from '../render/linechart';
import {getDrawArea, nextFrame} from '../render/render_utils';
import {Drawable, Point2D, XYPlotOptions} from '../types';
import {subSurface} from '../util/dom';

/**
 * Renders a tf.Model training 'History'.
 *
 * ```js
 * const model = tf.sequential({
 *  layers: [
 *    tf.layers.dense({inputShape: [784], units: 32, activation: 'relu'}),
 *    tf.layers.dense({units: 10, activation: 'softmax'}),
 *  ]
 * });
 *
 * model.compile({
 *   optimizer: 'sgd',
 *   loss: 'categoricalCrossentropy',
 *   metrics: ['accuracy']
 * });
 *
 * const data = tf.randomNormal([100, 784]);
 * const labels = tf.randomUniform([100, 10]);
 *
 * function onBatchEnd(batch, logs) {
 *   console.log('Accuracy', logs.acc);
 * }
 *
 * const surface = { name: 'show.history', tab: 'Training' };
 * // Train for 5 epochs with batch size of 32.
 * const history = await model.fit(data, labels, {
 *    epochs: 5,
 *    batchSize: 32
 * });
 *
 * tfvis.show.history(surface, history, ['loss', 'acc']);
 * ```
 *
 * ```js
 * const model = tf.sequential({
 *  layers: [
 *    tf.layers.dense({inputShape: [784], units: 32, activation: 'relu'}),
 *    tf.layers.dense({units: 10, activation: 'softmax'}),
 *  ]
 * });
 *
 * model.compile({
 *   optimizer: 'sgd',
 *   loss: 'categoricalCrossentropy',
 *   metrics: ['accuracy']
 * });
 *
 * const data = tf.randomNormal([100, 784]);
 * const labels = tf.randomUniform([100, 10]);
 *
 * function onBatchEnd(batch, logs) {
 *   console.log('Accuracy', logs.acc);
 * }
 *
 * const surface = { name: 'show.history live', tab: 'Training' };
 * // Train for 5 epochs with batch size of 32.
 * const history = [];
 * await model.fit(data, labels, {
 *    epochs: 5,
 *    batchSize: 32,
 *    callbacks: {
 *      onEpochEnd: (epoch, log) => {
 *        history.push(log);
 *        tfvis.show.history(surface, history, ['loss', 'acc']);
 *      }
 *    }
 * });
 * ```
 *
 * @param history A history like object. Either a tfjs-layers `History` object
 *  or an array of tfjs-layers `Logs` objects.
 * @param metrics An array of strings for each metric to plot from the history
 *  object. Using this allows you to control which metrics appear on the same
 *  plot.
 * @param opts Optional parameters for the line charts.
 *
 * @doc {heading: 'Models & Tensors', subheading: 'Model Training', namespace:
 * 'show'}
 */
export async function history(
    container: Drawable, history: HistoryLike, metrics: string[],
    opts: HistoryOptions = {}): Promise<void> {
  // Get the draw surface
  const drawArea = getDrawArea(container);

  // We organize the data from the history object into discrete plot data
  // objects so that we can group together appropriate metrics into single
  // multi-series charts.
  const plots: HistoryPlotData = {};
  for (const metric of metrics) {
    if (!(/val_/.test(metric))) {
      // Non validation metric
      const values = getValues(history, metric, metrics.indexOf(metric));
      initPlot(plots, metric);
      plots[metric].series.push(metric);
      plots[metric].values.push(values);
    } else {
      // Validation metrics are grouped with their equivalent non validation
      // metrics. Note that the corresponding non validation metric may not
      // actually be included but we still want to use it as a plot name.
      const nonValidationMetric = metric.replace('val_', '');
      initPlot(plots, nonValidationMetric);
      const values = getValues(history, metric, metrics.indexOf(metric));
      plots[nonValidationMetric].series.push(metric);
      plots[nonValidationMetric].values.push(values);
    }
  }

  // Render each plot specified above to a new subsurface.
  // A plot may have multiple series.
  const plotNames = Object.keys(plots);
  const options =
      Object.assign({}, {xLabel: 'Iteration', yLabel: 'Value'}, opts);

  const renderPromises = [];
  for (const name of plotNames) {
    const subContainer = subSurface(drawArea, name);
    const series = plots[name].series;
    const values = plots[name].values;

    if (series.every(seriesName => Boolean(seriesName.match('acc')))) {
      // Set a domain of 0-1 if all the series in this plot are related to
      // accuracy. Can be overridden by setting zoomToFitAccuracy to true.
      if (options.zoomToFitAccuracy) {
        options.zoomToFit = true;
      } else {
        options.yAxisDomain = [0, 1];
        delete options.zoomToFit;
      }
    }

    const done = linechart(subContainer, {values, series}, options);
    renderPromises.push(done);
  }
  await Promise.all(renderPromises);
}

interface HistoryOptions extends XYPlotOptions {
  zoomToFitAccuracy?: boolean;
}

type HistoryLike = Logs[]|Logs[][]|{
  history: {
    [key: string]: number[],
  }
};

interface HistoryPlotData {
  [name: string]: {
    series: string[],
    values: Point2D[][],
  };
}

function initPlot(plot: HistoryPlotData, name: string) {
  if (plot[name] == null) {
    plot[name] = {series: [], values: []};
  }
}

/*
 * Extracts a list of Point2D's suitable for plotting from a HistoryLike for
 * a single metric.
 * @param history a HistoryLike object
 * @param metric the metric to extract from the logs
 * @param metricIndex this is needed because the historylike can be a nested
 * list of logs for multiple metrics, this index lets us extract the correct
 * list.
 */
function getValues(
    history: HistoryLike, metric: string, metricIndex: number): Point2D[] {
  if (Array.isArray(history)) {
    // If we were passed a nested array we want to get the correct list
    // for this given metric, metrix index gives us this list.
    const metricHistory =
        (Array.isArray(history[0]) ? history[metricIndex] : history) as Logs[];
    const points: Point2D[] = [];

    for (let i = 0; i < metricHistory.length; i++) {
      const log = metricHistory[i];
      points.push({x: i, y: log[metric]});
    }
    return points;
  } else {
    return history.history[metric].map((y, x) => ({x, y}));
  }
}

/**
 * Returns a collection of callbacks to pass to tf.Model.fit. Callbacks are
 * returned for the following events, `onBatchEnd` & `onEpochEnd`.
 *
 * ```js
 * const model = tf.sequential({
 *  layers: [
 *    tf.layers.dense({inputShape: [784], units: 32, activation: 'relu'}),
 *    tf.layers.dense({units: 10, activation: 'softmax'}),
 *  ]
 * });
 *
 * model.compile({
 *   optimizer: 'sgd',
 *   loss: 'categoricalCrossentropy',
 *   metrics: ['accuracy']
 * });
 *
 * const data = tf.randomNormal([100, 784]);
 * const labels = tf.randomUniform([100, 10]);
 *
 * function onBatchEnd(batch, logs) {
 *   console.log('Accuracy', logs.acc);
 * }
 *
 * const surface = { name: 'show.fitCallbacks', tab: 'Training' };
 * // Train for 5 epochs with batch size of 32.
 * await model.fit(data, labels, {
 *    epochs: 5,
 *    batchSize: 32,
 *    callbacks: tfvis.show.fitCallbacks(surface, ['loss', 'acc']),
 * });
 * ```
 *
 * @param metrics List of metrics to plot.
 * @param opts Optional parameters
 *
 * @doc {heading: 'Models & Tensors', subheading: 'Model Training', namespace:
 * 'show'}
 */
export function fitCallbacks(
    container: Drawable, metrics: string[],
    opts: FitCallbackOptions = {}): FitCallbackHandlers {
  const accumulators: FitCallbackLogs = {};
  const callbackNames = opts.callbacks || ['onEpochEnd', 'onBatchEnd'];
  const drawArea = getDrawArea(container);

  const historyOpts = Object.assign({}, opts);
  delete historyOpts.callbacks;
  function makeCallbackFor(callbackName: string) {
    return async (_: number, log: Logs) => {
      // Set a nicer x axis name where possible
      if ((/batch/i).test(callbackName)) {
        historyOpts.xLabel = 'Batch';
      } else if ((/epoch/i).test(callbackName)) {
        historyOpts.xLabel = 'Epoch';
      }

      // Because of how the _ (iteration) numbers are given in the layers api
      // we have to store each metric for each callback in different arrays else
      // we cannot get accurate 'global' batch numbers for onBatchEnd.

      // However at render time we want to be able to combine metrics for a
      // given callback. So here we make a nested list of metrics, the first
      // level are arrays for each callback, the second level contains arrays
      // (of logs) for each metric within that callback.

      const metricLogs: Logs[][] = [];
      const presentMetrics: string[] = [];
      for (const metric of metrics) {
        // not all logs have all kinds of metrics.
        if (log[metric] != null) {
          presentMetrics.push(metric);

          const accumulator =
              getAccumulator(accumulators, callbackName, metric);
          accumulator.push({[metric]: log[metric]});
          metricLogs.push(accumulator);
        }
      }

      const subContainer =
          subSurface(drawArea, callbackName, {title: callbackName});
      history(subContainer, metricLogs, presentMetrics, historyOpts);
      await nextFrame();
    };
  }

  const callbacks: FitCallbackHandlers = {};
  callbackNames.forEach((name: string) => {
    callbacks[name] = makeCallbackFor(name);
  });
  return callbacks;
}

interface FitCallbackHandlers {
  [key: string]: (iteration: number, log: Logs) => Promise<void>;
}

interface FitCallbackLogs {
  [callback: string]: {
    [metric: string]: Logs[],
  };
}

interface FitCallbackOptions extends HistoryOptions {
  /**
   * Array of callback names. Valid options
   * are 'onEpochEnd' and 'onBatchEnd'. Defaults to ['onEpochEnd',
   * 'onBatchEnd'].
   */
  callbacks?: string[];
}

function getAccumulator(
    accumulators: FitCallbackLogs, callback: string, metric: string): Logs[] {
  if (accumulators[callback] == null) {
    accumulators[callback] = {};
  }
  if (accumulators[callback][metric] == null) {
    accumulators[callback][metric] = [];
  }
  return accumulators[callback][metric];
}
