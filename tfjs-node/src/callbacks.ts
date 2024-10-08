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

import {CustomCallback, LayersModel, Logs, nextFrame, util} from '@tensorflow/tfjs';
import * as path from 'path';
import * as ProgressBar from 'progress';

import {summaryFileWriter, SummaryFileWriter} from './tensorboard';

type LogFunction = (message: string) => void;
// A helper class created for testing with the jasmine `spyOn` method, which
// operates only on member methods of objects.
// tslint:disable-next-line:no-any
export const progressBarHelper: {ProgressBar: any, log: LogFunction} = {
  ProgressBar,
  log: console.log
};

/**
 * Terminal-based progress bar callback for tf.Model.fit().
 */
export class ProgbarLogger extends CustomCallback {
  private numTrainBatchesPerEpoch: number;
  private progressBar: ProgressBar;
  private currentEpochBegin: number;
  private epochDurationMillis: number;
  private usPerStep: number;
  private batchesInLatestEpoch: number;

  private terminalWidth: number;

  private readonly RENDER_THROTTLE_MS = 50;

  /**
   * Constructor of LoggingCallback.
   */
  constructor() {
    super({
      onTrainBegin: async (logs?: Logs) => {
        const samples = this.params.samples as number;
        const batchSize = this.params.batchSize as number;
        const steps = this.params.steps as number;
        if (samples != null || steps != null) {
          this.numTrainBatchesPerEpoch =
              samples != null ? Math.ceil(samples / batchSize) : steps;
        } else {
          // Undetermined number of batches per epoch, e.g., due to
          // `fitDataset()` without `batchesPerEpoch`.
          this.numTrainBatchesPerEpoch = 0;
        }
      },
      onEpochBegin: async (epoch: number, logs?: Logs) => {
        progressBarHelper.log(`Epoch ${epoch + 1} / ${this.params.epochs}`);
        this.currentEpochBegin = util.now();
        this.epochDurationMillis = null;
        this.usPerStep = null;
        this.batchesInLatestEpoch = 0;
        this.terminalWidth = process.stderr.columns;
      },
      onBatchEnd: async (batch: number, logs?: Logs) => {
        this.batchesInLatestEpoch++;
        if (batch === 0) {
          this.progressBar = new progressBarHelper.ProgressBar(
              'eta=:eta :bar :placeholderForLossesAndMetrics', {
                width: Math.floor(0.5 * this.terminalWidth),
                total: this.numTrainBatchesPerEpoch + 1,
                head: `>`,
                renderThrottle: this.RENDER_THROTTLE_MS
              });
        }
        const maxMetricsStringLength =
            Math.floor(this.terminalWidth * 0.5 - 12);
        const tickTokens = {
          placeholderForLossesAndMetrics:
              this.formatLogsAsMetricsContent(logs, maxMetricsStringLength)
        };
        if (this.numTrainBatchesPerEpoch === 0) {
          // Undetermined number of batches per epoch.
          this.progressBar.tick(0, tickTokens);
        } else {
          this.progressBar.tick(tickTokens);
        }
        await nextFrame();
        if (batch === this.numTrainBatchesPerEpoch - 1) {
          this.epochDurationMillis = util.now() - this.currentEpochBegin;
          this.usPerStep = this.params.samples != null ?
              this.epochDurationMillis / (this.params.samples as number) * 1e3 :
              this.epochDurationMillis / this.batchesInLatestEpoch * 1e3;
        }
      },
      onEpochEnd: async (epoch: number, logs?: Logs) => {
        if (this.epochDurationMillis == null) {
          // In cases where the number of batches per epoch is not determined,
          // the calculation of the per-step duration is done at the end of the
          // epoch. N.B., this includes the time spent on validation.
          this.epochDurationMillis = util.now() - this.currentEpochBegin;
          this.usPerStep =
              this.epochDurationMillis / this.batchesInLatestEpoch * 1e3;
        }
        this.progressBar.tick({placeholderForLossesAndMetrics: ''});
        const lossesAndMetricsString = this.formatLogsAsMetricsContent(logs);
        progressBarHelper.log(
            `${this.epochDurationMillis.toFixed(0)}ms ` +
            `${this.usPerStep.toFixed(0)}us/step - ` +
            `${lossesAndMetricsString}`);
        await nextFrame();
      },
    });
  }

  private formatLogsAsMetricsContent(logs: Logs, maxMetricsLength?: number):
      string {
    let metricsContent = '';
    const keys = Object.keys(logs).sort();
    for (const key of keys) {
      if (this.isFieldRelevant(key)) {
        const value = logs[key];
        metricsContent += `${key}=${getSuccinctNumberDisplay(value)} `;
      }
    }

    if (maxMetricsLength != null && metricsContent.length > maxMetricsLength) {
      // Cut off metrics strings that are too long to avoid new lines being
      // constantly created.
      metricsContent = metricsContent.slice(0, maxMetricsLength - 3) + '...';
    }
    return metricsContent;
  }

  private isFieldRelevant(key: string) {
    return key !== 'batch' && key !== 'size';
  }
}

const BASE_NUM_DIGITS = 2;
const MAX_NUM_DECIMAL_PLACES = 4;

/**
 * Get a succinct string representation of a number.
 *
 * Uses decimal notation if the number isn't too small.
 * Otherwise, use engineering notation.
 *
 * @param x Input number.
 * @return Succinct string representing `x`.
 */
export function getSuccinctNumberDisplay(x: number): string {
  const decimalPlaces = getDisplayDecimalPlaces(x);
  return decimalPlaces > MAX_NUM_DECIMAL_PLACES ?
      x.toExponential(BASE_NUM_DIGITS) :
      x.toFixed(decimalPlaces);
}

/**
 * Determine the number of decimal places to display.
 *
 * @param x Number to display.
 * @return Number of decimal places to display for `x`.
 */
export function getDisplayDecimalPlaces(x: number): number {
  if (!Number.isFinite(x) || x === 0 || x > 1 || x < -1) {
    return BASE_NUM_DIGITS;
  } else {
    return BASE_NUM_DIGITS - Math.floor(Math.log10(Math.abs(x)));
  }
}

export interface TensorBoardCallbackArgs {
  /**
   * The frequency at which loss and metric values are written to logs.
   *
   * Currently supported options are:
   *
   * - 'batch': Write logs at the end of every batch of training, in addition
   *   to the end of every epoch of training.
   * - 'epoch': Write logs at the end of every epoch of training.
   *
   * Note that writing logs too often slows down the training.
   *
   * Default: 'epoch'.
   */
  updateFreq?: 'batch'|'epoch';

  /**
   * The frequency (in epochs) at which to compute activation and weight
   * histograms for the layers of the model.
   *
   * If set to 0, histograms won't be computed.
   *
   * Validation data (or split) must be specified for histogram visualizations.
   *
   * Default: 0.
   */
  histogramFreq?: number;
}

/**
 * Callback for logging to TensorBoard during training.
 *
 * Users are expected to access this class through the `tensorBoardCallback()`
 * factory method instead.
 */
export class TensorBoardCallback extends CustomCallback {
  private model: LayersModel = null;
  private trainWriter: SummaryFileWriter;
  private valWriter: SummaryFileWriter;
  private batchesSeen: number;
  private readonly args: TensorBoardCallbackArgs;

  constructor(readonly logdir = './logs', args?: TensorBoardCallbackArgs) {
    super({
      onBatchEnd: async (batch: number, logs?: Logs) => {
        this.batchesSeen++;
        if (this.args.updateFreq !== 'epoch') {
          this.logMetrics(logs, 'batch_', this.batchesSeen);
        }
      },
      onEpochEnd: async (epoch: number, logs?: Logs) => {
        this.logMetrics(logs, 'epoch_', epoch + 1);
        if (this.args.histogramFreq > 0 &&
            epoch % this.args.histogramFreq === 0) {
          this.logWeights(epoch);
        }
      },
      onTrainEnd: async (logs?: Logs) => {
        if (this.trainWriter != null) {
          this.trainWriter.flush();
        }
        if (this.valWriter != null) {
          this.valWriter.flush();
        }
      }
    });

    this.args = args == null ? {} : args;
    if (this.args.updateFreq == null) {
      this.args.updateFreq = 'epoch';
    }
    util.assert(
        ['batch', 'epoch'].indexOf(this.args.updateFreq) !== -1,
        () => `Expected updateFreq to be 'batch' or 'epoch', but got ` +
            `${this.args.updateFreq}`);
    if (this.args.histogramFreq == null) {
      this.args.histogramFreq = 0;
    }
    util.assert(
        Number.isInteger(this.args.histogramFreq) &&
            this.args.histogramFreq >= 0,
        () => `Expected histogramFreq to be a positive integer, but got ` +
            `${this.args.histogramFreq}`);
    this.batchesSeen = 0;
  }

  setModel(model: LayersModel): void {
    // This method is inherited from BaseCallback. To avoid cyclical imports,
    // that class uses Container instead of LayersModel, and uses a run-time
    // check to make sure the model is a LayersModel.
    // Since this subclass isn't imported by tfjs-layers, we can safely use type
    // the parameter as a LayersModel.
    this.model = model;
  }

  private logMetrics(logs: Logs, prefix: string, step: number) {
    for (const key in logs) {
      if (key === 'batch' || key === 'size' || key === 'num_steps') {
        continue;
      }

      const VAL_PREFIX = 'val_';
      if (key.startsWith(VAL_PREFIX)) {
        this.ensureValWriterCreated();
        const scalarName = prefix + key.slice(VAL_PREFIX.length);
        this.valWriter.scalar(scalarName, logs[key], step);
      } else {
        this.ensureTrainWriterCreated();
        this.trainWriter.scalar(`${prefix}${key}`, logs[key], step);
      }
    }
  }

  private logWeights(step: number) {
    for (const weights of this.model.weights) {
      this.trainWriter.histogram(weights.name, weights.read(), step);
    }
  }

  private ensureTrainWriterCreated() {
    this.trainWriter = summaryFileWriter(path.join(this.logdir, 'train'));
  }

  private ensureValWriterCreated() {
    this.valWriter = summaryFileWriter(path.join(this.logdir, 'val'));
  }
}

/**
 * Callback for logging to TensorBoard during training.
 *
 * Writes the loss and metric values (if any) to the specified log directory
 * (`logdir`) which can be ingested and visualized by TensorBoard.
 * This callback is usually passed as a callback to `tf.Model.fit()` or
 * `tf.Model.fitDataset()` calls during model training. The frequency at which
 * the values are logged can be controlled with the `updateFreq` field of the
 * configuration object (2nd argument).
 *
 * Usage example:
 * ```js
 * // Constructor a toy multilayer-perceptron regressor for demo purpose.
 * const model = tf.sequential();
 * model.add(
 *     tf.layers.dense({units: 100, activation: 'relu', inputShape: [200]}));
 * model.add(tf.layers.dense({units: 1}));
 * model.compile({
 *   loss: 'meanSquaredError',
 *   optimizer: 'sgd',
 *   metrics: ['MAE']
 * });
 *
 * // Generate some random fake data for demo purpose.
 * const xs = tf.randomUniform([10000, 200]);
 * const ys = tf.randomUniform([10000, 1]);
 * const valXs = tf.randomUniform([1000, 200]);
 * const valYs = tf.randomUniform([1000, 1]);
 *
 * // Start model training process.
 * await model.fit(xs, ys, {
 *   epochs: 100,
 *   validationData: [valXs, valYs],
 *    // Add the tensorBoard callback here.
 *   callbacks: tf.node.tensorBoard('/tmp/fit_logs_1')
 * });
 * ```
 *
 * Then you can use the following commands to point tensorboard
 * to the logdir:
 *
 * ```sh
 * pip install tensorboard  # Unless you've already installed it.
 * tensorboard --logdir /tmp/fit_logs_1
 * ```
 *
 * @param logdir Directory to which the logs will be written.
 * @param args Optional configuration arguments.
 * @returns An instance of `TensorBoardCallback`, which is a subclass of
 *   `tf.CustomCallback`.
 *
 * @doc {heading: 'TensorBoard', namespace: 'node'}
 */
export function tensorBoard(
    logdir = './logs', args?: TensorBoardCallbackArgs): TensorBoardCallback {
  return new TensorBoardCallback(logdir, args);
}
