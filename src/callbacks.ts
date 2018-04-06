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

import {Scalar, Tensor, tidy} from '@tensorflow/tfjs-core';

import * as K from './backend/tfjs_backend';
import {Model} from './engine/training';
import * as generic_utils from './utils/generic_utils';

/**
 * Logs in which values can be either numbers or Tensors (Scalars).
 *
 * Used internally.
 */
export type UnresolvedLogs = {
  [key: string]: number|Scalar;
};


/**
 * Logs in which values can only be numbers.
 *
 * Used when calling client-provided custom callbacks.
 */
export type Logs = {
  [key: string]: number;
};

export type Params = {
  [key: string]: number|string|boolean|number[]|string[]|boolean[];
};

/**
 * Abstract base class used to build new callbacks.
 *
 * The `logs` dictionary that callback methods take as argument will contain
 * keys for quantities relevant to the current batch or epoch.
 *
 * Currently, the `.fit()` method of the `Sequential` model class
 * will include the following quantities in the `logs` that
 * it passes to its callbacks:
 *
 * onEpochEnd: Logs include `acc` and `loss`, and optionally include `valLoss`
 *   (if validation is enabled in `fit`), and `valAcc` (if validation and
 *   accuracy monitoring are enabled).
 * onBatchBegin: Logs include `size`, the number of samples in the current
 *   batch.
 * onBatchEnd: Logs include `loss`, and optionally `acc` (if accuracy monitoring
 *   is enabled).
 */
export abstract class Callback {
  // TODO(michaelterry): This type is a best guess.
  validationData: Tensor|Tensor[] = null;
  /** Instance of `keras.models.Model`. Reference of the model being trained. */
  model: Model = null;
  /**
   * Training parameters (eg. verbosity, batch size, number of epochs...).
   */
  params: Params;

  setParams(params: Params): void {
    this.params = params;
  }

  setModel(model: Model): void {
    this.model = model;
  }

  async onEpochBegin(epoch: number, logs?: UnresolvedLogs) {}

  async onEpochEnd(epoch: number, logs?: UnresolvedLogs) {}

  async onBatchBegin(batch: number, logs?: UnresolvedLogs) {}

  async onBatchEnd(batch: number, logs?: UnresolvedLogs) {}

  async onTrainBegin(logs?: UnresolvedLogs) {}

  async onTrainEnd(logs?: UnresolvedLogs) {}
}

/**
 * Container abstracting a list of callbacks.
 */
export class CallbackList {
  callbacks: Callback[];
  queueLength: number;

  // TODO(cais): When the need arises, uncomment the following lines and
  // implement the queue for time values.
  // private deltaTBatch: number;
  // private deltaTsBatchBegin: Array<number>;
  // private deltaTsBatchEnd: Array<number>;

  /**
   * Constructor of CallbackList.
   * @param callbacks Array of `Callback` instances.
   * @param queueLength Queue length for keeping running statistics over
   *   callback execution time.
   */
  constructor(callbacks?: Callback[], queueLength = 10) {
    // TODO(cais): Make use of queueLength when implementing the queue for time
    // values.
    if (callbacks == null) {
      callbacks = [];
    }
    this.callbacks = callbacks;
    this.queueLength = queueLength;
  }

  append(callback: Callback): void {
    this.callbacks.push(callback);
  }

  setParams(params: Params): void {
    for (const callback of this.callbacks) {
      callback.setParams(params);
    }
  }

  setModel(model: Model): void {
    for (const callback of this.callbacks) {
      callback.setModel(model);
    }
  }

  /**
   * Called at the start of an epoch.
   * @param epoch Index of epoch.
   * @param logs Dictionary of logs.
   */
  async onEpochBegin(epoch: number, logs?: UnresolvedLogs) {
    if (logs == null) {
      logs = {};
    }
    for (const callback of this.callbacks) {
      await callback.onEpochBegin(epoch, logs);
    }
  }

  /**
   * Called at the end of an epoch.
   * @param epoch Index of epoch.
   * @param logs Dictionary of logs.
   */
  async onEpochEnd(epoch: number, logs?: UnresolvedLogs) {
    if (logs == null) {
      logs = {};
    }
    for (const callback of this.callbacks) {
      await callback.onEpochEnd(epoch, logs);
    }
  }

  /**
   * Called  right before processing a batch.
   * @param batch Index of batch within the current epoch.
   * @param logs Dictionary of logs.
   */
  async onBatchBegin(batch: number, logs?: UnresolvedLogs) {
    if (logs == null) {
      logs = {};
    }
    for (const callback of this.callbacks) {
      await callback.onBatchBegin(batch, logs);
    }
  }

  /**
   * Called at the end of a batch.
   * @param batch Index of batch within the current epoch.
   * @param logs Dictionary of logs.
   */
  async onBatchEnd(batch: number, logs?: UnresolvedLogs) {
    if (logs == null) {
      logs = {};
    }
    for (const callback of this.callbacks) {
      await callback.onBatchEnd(batch, logs);
    }
  }

  /**
   * Called at the beginning of training.
   * @param logs Dictionary of logs.
   */
  async onTrainBegin(logs?: UnresolvedLogs) {
    if (logs == null) {
      logs = {};
    }
    for (const callback of this.callbacks) {
      await callback.onTrainBegin(logs);
    }
  }

  /**
   * Called at the end of training.
   * @param logs Dictionary of logs.
   */
  async onTrainEnd(logs?: UnresolvedLogs) {
    if (logs == null) {
      logs = {};
    }
    for (const callback of this.callbacks) {
      await callback.onTrainEnd(logs);
    }
  }
}

/**
 * Callback that accumulates epoch averages of metrics.
 *
 * This callback is automatically applied to every Model.
 */
export class BaseLogger extends Callback {
  private seen: number;
  private totals: UnresolvedLogs;

  constructor() {
    super();
  }

  async onEpochBegin(epoch: number, logs?: UnresolvedLogs) {
    this.seen = 0;
    this.totals = {};
  }

  async onBatchEnd(batch: number, logs?: UnresolvedLogs) {
    if (logs == null) {
      logs = {};
    }
    const batchSize = logs['size'] == null ? 0 : logs['size'] as number;
    this.seen += batchSize;
    for (const key in logs) {
      const value = logs[key];
      if (typeof value === 'number') {
        if (!this.totals.hasOwnProperty(key)) {
          this.totals[key] = 0;
        }
        this.totals[key] = this.totals[key] as number + value * batchSize;
      } else {
        if (!this.totals.hasOwnProperty(key)) {
          this.totals[key] = K.getScalar(0);
        }
        // TODO(cais): Do not leak tidy from TensorFlow.js Core.
        tidy(() => {
          this.totals[key] =
              K.scalarPlusArray(
                  this.totals[key] as Scalar,
                  K.multiply(value, K.getScalar(batchSize))) as Scalar;
          K.keep(this.totals[key] as Scalar);
        });
      }
    }
  }

  async onEpochEnd(epoch: number, logs?: UnresolvedLogs) {
    if (logs != null) {
      for (const key of this.params['metrics'] as string[]) {
        if (this.totals[key] == null) {
          continue;
        }
        if (typeof this.totals[key] === 'number') {
          logs[key] = this.totals[key] as number / this.seen;
        } else {
          tidy(() => {
            logs[key] =
                K.scalarTimesArray(
                    K.divide(K.getScalar(1), K.getScalar(this.seen)) as Scalar,
                    this.totals[key] as Scalar) as Scalar;
            K.keep(logs[key] as Scalar);
          });
        }
      }
    }
  }
}

/**
 * Turn any Scalar values in a Logs object into actual number values.
 *
 * @param logs The `Logs` object to be resolved in place.
 */
export async function resolveScalarsInLogs(logs: UnresolvedLogs) {
  if (logs == null) {
    return;
  }
  const promises: Array<Promise<Float32Array|Int32Array|Uint8Array>> = [];
  const keys: string[] = [];
  for (const key in logs) {
    const value = logs[key];
    if (typeof value !== 'number') {
      const valueScalar = value as Scalar;
      promises.push(valueScalar.data());
      keys.push(key);
    }
  }
  const values = await Promise.all(promises);
  for (let i = 0; i < values.length; ++i) {
    logs[keys[i]] = values[i][0];
  }
}

/**
 * Dispose all Tensors in an UnresolvedLogs object.
 *
 * @param logs An `UnresolvedLogs` object potentially containing `Tensor`s in
 *   places where the values can be `Tensor` or `number`.
 */
export function disposeTensorsInLogs(logs: UnresolvedLogs) {
  if (logs == null) {
    return;
  }
  for (const key in logs) {
    const value = logs[key];
    if (typeof value !== 'number') {
      value.dispose();
    }
  }
}

/**
 * Callback that records events into a `History` object. This callback is
 * automatically applied to every TF.js Layers model. The `History` object gets
 * returned by the `fit` method of models.
 */
export class History extends Callback {
  epoch: number[];
  history: {[key: string]: Array<number|Tensor>};

  async onTrainBegin(logs?: UnresolvedLogs) {
    this.epoch = [];
    this.history = {};
  }

  async onEpochEnd(epoch: number, logs?: UnresolvedLogs) {
    if (logs == null) {
      logs = {};
    }
    this.epoch.push(epoch);
    for (const key in logs) {
      if (this.history[key] == null) {
        this.history[key] = [];
      }
      this.history[key].push(logs[key]);
    }
  }

  /**
   * Await the values of all losses and metrics.
   */
  async syncData() {
    const promises: Array<Promise<Float32Array|Int32Array|Uint8Array>> = [];
    const keys: string[] = [];
    const indices: number[] = [];
    for (const key in this.history) {
      const valueArray = this.history[key];
      for (let i = 0; i < valueArray.length; ++i) {
        if (typeof valueArray[i] !== 'number') {
          const valueScalar = valueArray[i] as Tensor;
          promises.push(valueScalar.data());
          keys.push(key);
          indices.push(i);
        }
      }
    }
    const values = await Promise.all(promises);
    for (let n = 0; n < values.length; ++n) {
      (this.history[keys[n]][indices[n]] as Tensor).dispose();
      this.history[keys[n]][indices[n]] = values[n][0];
    }
  }
}

export interface CustomCallbackConfig {
  onTrainBegin?: (logs?: Logs) => Promise<void>;
  onTrainEnd?: (logs?: Logs) => Promise<void>;
  onEpochBegin?: (epoch: number, logs?: Logs) => Promise<void>;
  onEpochEnd?: (epoch: number, logs?: Logs) => Promise<void>;
  onBatchBegin?: (batch: number, logs?: Logs) => Promise<void>;
  onBatchEnd?: (batch: number, logs?: Logs) => Promise<void>;
}

/**
 * Custom callback for training.
 */
export class CustomCallback extends Callback {
  protected readonly trainBegin: (logs?: Logs) => Promise<void>;
  protected readonly trainEnd: (logs?: Logs) => Promise<void>;
  protected readonly epochBegin: (epoch: number, logs?: Logs) => Promise<void>;
  protected readonly epochEnd: (epoch: number, logs?: Logs) => Promise<void>;
  protected readonly batchBegin: (batch: number, logs?: Logs) => Promise<void>;
  protected readonly batchEnd: (batch: number, logs?: Logs) => Promise<void>;

  constructor(config: CustomCallbackConfig) {
    super();
    this.trainBegin = config.onTrainBegin;
    this.trainEnd = config.onTrainEnd;
    this.epochBegin = config.onEpochBegin;
    this.epochEnd = config.onEpochEnd;
    this.batchBegin = config.onBatchBegin;
    this.batchEnd = config.onBatchEnd;
  }

  async onEpochBegin(epoch: number, logs?: UnresolvedLogs): Promise<void> {
    if (this.epochBegin != null) {
      await resolveScalarsInLogs(logs);
      await this.epochBegin(epoch, logs as Logs);
    }
  }

  async onEpochEnd(epoch: number, logs?: UnresolvedLogs): Promise<void> {
    if (this.epochEnd != null) {
      await resolveScalarsInLogs(logs);
      await this.epochEnd(epoch, logs as Logs);
    }
  }

  async onBatchBegin(batch: number, logs?: UnresolvedLogs): Promise<void> {
    if (this.batchBegin != null) {
      await resolveScalarsInLogs(logs);
      await this.batchBegin(batch, logs as Logs);
    }
  }

  async onBatchEnd(batch: number, logs?: UnresolvedLogs): Promise<void> {
    if (this.batchEnd != null) {
      await resolveScalarsInLogs(logs);
      await this.batchEnd(batch, logs as Logs);
    }
  }

  async onTrainBegin(logs?: UnresolvedLogs): Promise<void> {
    if (this.trainBegin != null) {
      await resolveScalarsInLogs(logs);
      await this.trainBegin(logs as Logs);
    }
  }

  async onTrainEnd(logs?: UnresolvedLogs): Promise<void> {
    if (this.trainEnd != null) {
      await resolveScalarsInLogs(logs);
      await this.trainEnd(logs as Logs);
    }
  }
}

/**
 * Standardize callbacks or configurations of them to an Array of callbacks.
 */
export function standardizeCallbacks(callbacks: Callback|Callback[]|
                                     CustomCallbackConfig|
                                     CustomCallbackConfig[]): Callback[] {
  if (callbacks == null) {
    return null;
  }
  if (callbacks instanceof Callback) {
    return [callbacks as Callback];
  }
  if (Array.isArray(callbacks) && callbacks[0] instanceof Callback) {
    return callbacks as Callback[];
  }
  // Convert custom callback configs to custom callback objects.
  const callbackConfigs =
      generic_utils.toList(callbacks) as CustomCallbackConfig[];
  return callbackConfigs.map(
      callbackConfig => new CustomCallback(callbackConfig));
}
