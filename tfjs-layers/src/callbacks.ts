/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

/* Original source: keras/callbacks.py */

import {BaseCallback} from './base_callbacks';
import {Container} from './engine/container';
import {LayersModel} from './engine/training';
import {NotImplementedError} from './errors';
import {Logs, resolveScalarsInLogs} from './logs';

export abstract class Callback extends BaseCallback {
  /** Instance of `keras.models.Model`. Reference of the model being trained. */
  model: LayersModel = null;

  setModel(model: Container): void {
    if (!(model instanceof LayersModel)) {
      throw new Error('model must be a LayersModel, not some other Container');
    }
    this.model = model;
  }
}

export interface EarlyStoppingCallbackArgs {
  /**
   * Quantity to be monitored.
   *
   * Defaults to 'val_loss'.
   */
  monitor?: string;

  /**
   * Minimum change in the monitored quantity to qualify as improvement,
   * i.e., an absolute change of less than `minDelta` will count as no
   * improvement.
   *
   * Defaults to 0.
   */
  minDelta?: number;

  /**
   * Number of epochs with no improvement after which training will be stopped.
   *
   * Defaults to 0.
   */
  patience?: number;

  /** Verbosity mode. */
  verbose?: number;

  /**
   * Mode: one of 'min', 'max', and 'auto'.
   * - In 'min' mode, training will be stopped when the quantity monitored has
   *   stopped decreasing.
   * - In 'max' mode, training will be stopped when the quantity monitored has
   *   stopped increasing.
   * - In 'auto' mode, the direction is inferred automatically from the name of
   *   the monitored quantity.
   *
   * Defaults to 'auto'.
   */
  mode?: 'auto'|'min'|'max';

  /**
   * Baseline value of the monitored quantity.
   *
   * If specified, training will be stopped if the model doesn't show
   * improvement over the baseline.
   */
  baseline?: number;

  /**
   * Whether to restore model weights from the epoch with the best value
   * of the monitored quantity. If `False`, the model weights obtained at the
   * at the last step of training are used.
   *
   * **`True` is not supported yet.**
   */
  restoreBestWeights?: boolean;
}

function less(currVal: number, prevVal: number) {
  return currVal < prevVal;
}

function greater(currVal: number, prevVal: number) {
  return currVal > prevVal;
}

/**
 * A Callback that stops training when a monitored quantity has stopped
 * improving.
 */
export class EarlyStopping extends Callback {
  protected readonly monitor: string;
  protected readonly minDelta: number;
  protected readonly patience: number;
  protected readonly baseline: number;
  protected readonly verbose: number;
  protected readonly mode: 'auto'|'min'|'max';

  protected monitorFunc: (currVal: number, prevVal: number) => boolean;

  private wait: number;
  private stoppedEpoch: number;
  private best: number;

  constructor(args?: EarlyStoppingCallbackArgs) {
    super();
    if (args == null) {
      args = {};
    }
    if (args.restoreBestWeights) {
      throw new NotImplementedError(
          'restoreBestWeights = True is not implemented in EarlyStopping yet.');
    }

    this.monitor = args.monitor || 'val_loss';
    this.minDelta = Math.abs(args.minDelta || 0);
    this.patience = args.patience || 0;
    this.verbose = args.verbose || 0;
    this.mode = args.mode || 'auto';
    this.baseline = args.baseline;

    if (['auto', 'min', 'max'].indexOf(this.mode) === -1) {
      console.warn(
          `EarlyStopping mode '${this.mode}' is invalid. ` +
          `Falling back to mode 'auto'.`);
      this.mode = 'auto';
    }

    if (this.mode === 'min') {
      this.monitorFunc = less;
    } else if (this.mode === 'max') {
      this.monitorFunc = greater;
    } else {
      // For mode === 'auto'.
      if (this.monitor.indexOf('acc') !== -1) {
        this.monitorFunc = greater;
      } else {
        this.monitorFunc = less;
      }
    }

    if (this.monitorFunc === less) {
      this.minDelta *= -1;
    }
  }

  async onTrainBegin(logs?: Logs) {
    this.wait = 0;
    this.stoppedEpoch = 0;
    if (this.baseline != null) {
      this.best = this.baseline;
    } else {
      this.best = this.monitorFunc === less ? Infinity : -Infinity;
    }
  }

  async onEpochEnd(epoch: number, logs?: Logs) {
    await resolveScalarsInLogs(logs);
    const current = this.getMonitorValue(logs);
    if (current == null) {
      return;
    }

    if (this.monitorFunc(current - this.minDelta, this.best)) {
      this.best = current;
      this.wait = 0;
      // TODO(cais): Logic for restoreBestWeights.
    } else {
      this.wait++;
      if (this.wait >= this.patience) {
        this.stoppedEpoch = epoch;
        this.model.stopTraining = true;
      }
      // TODO(cais): Logic for restoreBestWeights.
    }
  }

  async onTrainEnd(logs?: Logs) {
    if (this.stoppedEpoch > 0 && this.verbose) {
      console.log(`Epoch ${this.stoppedEpoch}: early stopping.`);
    }
  }

  private getMonitorValue(logs: Logs) {
    if (logs == null) {
      logs = {};
    }
    const monitorValue = logs[this.monitor];
    if (monitorValue == null) {
      console.warn(
          `Metric for EarlyStopping ${this.monitor} is not available. ` +
          `Available metrics are: ${Object.keys(logs)}`);
    }
    return monitorValue;
  }
}

/**
 * Factory function for a Callback that stops training when a monitored
 * quantity has stopped improving.
 *
 * Early stopping is a type of regularization, and protects model against
 * overfitting.
 *
 * The following example based on fake data illustrates how this callback
 * can be used during `tf.LayersModel.fit()`:
 *
 * ```js
 * const model = tf.sequential();
 * model.add(tf.layers.dense({
 *   units: 3,
 *   activation: 'softmax',
 *   kernelInitializer: 'ones',
 *   inputShape: [2]
 * }));
 * const xs = tf.tensor2d([1, 2, 3, 4], [2, 2]);
 * const ys = tf.tensor2d([[1, 0, 0], [0, 1, 0]], [2, 3]);
 * const xsVal = tf.tensor2d([4, 3, 2, 1], [2, 2]);
 * const ysVal = tf.tensor2d([[0, 0, 1], [0, 1, 0]], [2, 3]);
 * model.compile(
 *     {loss: 'categoricalCrossentropy', optimizer: 'sgd', metrics: ['acc']});
 *
 * // Without the EarlyStopping callback, the val_acc value would be:
 * //   0.5, 0.5, 0.5, 0.5, ...
 * // With val_acc being monitored, training should stop after the 2nd epoch.
 * const history = await model.fit(xs, ys, {
 *   epochs: 10,
 *   validationData: [xsVal, ysVal],
 *   callbacks: tf.callbacks.earlyStopping({monitor: 'val_acc'})
 * });
 *
 * // Expect to see a length-2 array.
 * console.log(history.history.val_acc);
 * ```
 */
/**
 * @doc {
 *   heading: 'Callbacks',
 *   namespace: 'callbacks'
 * }
 */
export function earlyStopping(args?: EarlyStoppingCallbackArgs) {
  return new EarlyStopping(args);
}

export const callbacks = {earlyStopping};
