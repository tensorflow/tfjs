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

import {add, div, keep, mul, nextFrame, Scalar, Tensor, tidy, util} from '@tensorflow/tfjs-core';

import {Container} from './engine/container';
import {ValueError} from './errors';
import {Logs, resolveScalarsInLogs, UnresolvedLogs} from './logs';
import * as generic_utils from './utils/generic_utils';

/** Verbosity logging level when fitting a model. */
export enum ModelLoggingVerbosity {
  SILENT = 0,
  VERBOSE = 1
}

/** How often to yield to the main thread when training (in ms). */
export const DEFAULT_YIELD_EVERY_MS = 125;

export type Params = {
  [key: string]: number|string|boolean|number[]|string[]|boolean[];
};

export type YieldEveryOptions = 'auto'|'batch'|'epoch'|'never'|number;

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
export abstract class BaseCallback {
  // TODO(michaelterry): This type is a best guess.
  validationData: Tensor|Tensor[] = null;
  /**
   * Training parameters (eg. verbosity, batch size, number of epochs...).
   */
  params: Params;

  setParams(params: Params): void {
    this.params = params;
  }

  async onEpochBegin(epoch: number, logs?: UnresolvedLogs) {}

  async onEpochEnd(epoch: number, logs?: UnresolvedLogs) {}

  async onBatchBegin(batch: number, logs?: UnresolvedLogs) {}

  async onBatchEnd(batch: number, logs?: UnresolvedLogs) {}

  async onTrainBegin(logs?: UnresolvedLogs) {}

  async onTrainEnd(logs?: UnresolvedLogs) {}

  // LayersModel needs to call Callback.setModel(), but cannot actually depend
  // on Callback because that creates a cyclic dependency.  Providing this no-op
  // method on BaseCallback breaks the cycle: this way LayersModel can depend on
  // BaseCallback but not on Callback.  The argument is typed as `Container`
  // (the superclass of LayersModel) to avoid recapitulating the cycle. Callback
  // overrides this method and enforces that the argument is really a
  // LayersModel.
  setModel(model: Container): void {
    // Do nothing. Use Callback instead of BaseCallback to track the model.
  }
}

/**
 * Container abstracting a list of callbacks.
 */
export class CallbackList {
  callbacks: BaseCallback[];
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
  constructor(callbacks?: BaseCallback[], queueLength = 10) {
    // TODO(cais): Make use of queueLength when implementing the queue for time
    // values.
    if (callbacks == null) {
      callbacks = [];
    }
    this.callbacks = callbacks;
    this.queueLength = queueLength;
  }

  append(callback: BaseCallback): void {
    this.callbacks.push(callback);
  }

  setParams(params: Params): void {
    for (const callback of this.callbacks) {
      callback.setParams(params);
    }
  }

  setModel(model: Container): void {
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
 * This callback is automatically applied to every LayersModel.
 */
export class BaseLogger extends BaseCallback {
  private seen: number;
  private totals: UnresolvedLogs;

  constructor() {
    super();
  }

  async onEpochBegin(epoch: number) {
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
        let oldTotalsToDispose: Scalar;
        if (key in this.totals) {
          oldTotalsToDispose = this.totals[key] as Scalar;
        } else {
          this.totals[key] = 0;
        }
        this.totals[key] = tidy(
            () => add((this.totals[key] as Scalar), mul(value, batchSize)) as
                Scalar);
        if (oldTotalsToDispose != null) {
          oldTotalsToDispose.dispose();
        }
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
            logs[key] = mul(div(1, this.seen) as Scalar,
                            this.totals[key] as Scalar) as Scalar;
            (this.totals[key] as Tensor).dispose();
            keep(logs[key] as Scalar);
          });
        }
      }
    }
  }
}

/**
 * Callback that records events into a `History` object. This callback is
 * automatically applied to every TF.js Layers model. The `History` object
 * gets returned by the `fit` method of models.
 */
export class History extends BaseCallback {
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
      const tensorToDispose = this.history[keys[n]][indices[n]] as Tensor;
      tensorToDispose.dispose();
      this.history[keys[n]][indices[n]] = values[n][0];
    }
  }
}

export interface CustomCallbackArgs {
  onTrainBegin?: (logs?: Logs) => void | Promise<void>;
  onTrainEnd?: (logs?: Logs) => void | Promise<void>;
  onEpochBegin?: (epoch: number, logs?: Logs) => void | Promise<void>;
  onEpochEnd?: (epoch: number, logs?: Logs) => void | Promise<void>;
  onBatchBegin?: (batch: number, logs?: Logs) => void | Promise<void>;
  onBatchEnd?: (batch: number, logs?: Logs) => void | Promise<void>;
  onYield?: (epoch: number, batch: number, logs: Logs) => void | Promise<void>;
}

/**
 * Custom callback for training.
 */
export class CustomCallback extends BaseCallback {
  protected readonly trainBegin: (logs?: Logs) => void | Promise<void>;
  protected readonly trainEnd: (logs?: Logs) => void | Promise<void>;
  protected readonly epochBegin:
      (epoch: number, logs?: Logs) => void | Promise<void>;
  protected readonly epochEnd:
      (epoch: number, logs?: Logs) => void | Promise<void>;
  protected readonly batchBegin:
      (batch: number, logs?: Logs) => void | Promise<void>;
  protected readonly batchEnd:
      (batch: number, logs?: Logs) => void | Promise<void>;
  protected readonly yield:
      (epoch: number, batch: number, logs: Logs) => void | Promise<void>;

  private yieldEvery: YieldEveryOptions;
  private currentEpoch = 0;

  constructor(args: CustomCallbackArgs, yieldEvery?: YieldEveryOptions) {
    super();
    this.yieldEvery = yieldEvery || 'auto';
    if (this.yieldEvery === 'auto') {
      this.yieldEvery = DEFAULT_YIELD_EVERY_MS;
    }
    if (this.yieldEvery === 'never' && args.onYield != null) {
      throw new Error(
          'yieldEvery is `never` but you provided an `onYield` callback. ' +
          'Either change `yieldEvery` or remove the callback');
    }
    if (util.isNumber(this.yieldEvery)) {
      // Decorate `maybeWait` so it will be called at most once every
      // `yieldEvery` ms.
      this.maybeWait = generic_utils.debounce(
          this.maybeWait.bind(this), this.yieldEvery as number);
    }
    this.trainBegin = args.onTrainBegin;
    this.trainEnd = args.onTrainEnd;
    this.epochBegin = args.onEpochBegin;
    this.epochEnd = args.onEpochEnd;
    this.batchBegin = args.onBatchBegin;
    this.batchEnd = args.onBatchEnd;
    this.yield = args.onYield;
  }

  async maybeWait(epoch: number, batch: number, logs: UnresolvedLogs) {
    const ps: Array<void|Promise<void>> = [];
    if (this.yield != null) {
      await resolveScalarsInLogs(logs);
      ps.push(this.yield(epoch, batch, logs as Logs));
    }
    ps.push(nextFrame());
    await Promise.all(ps);
  }

  async onEpochBegin(epoch: number, logs?: UnresolvedLogs): Promise<void> {
    this.currentEpoch = epoch;
    if (this.epochBegin != null) {
      await resolveScalarsInLogs(logs);
      await this.epochBegin(epoch, logs as Logs);
    }
  }

  async onEpochEnd(epoch: number, logs?: UnresolvedLogs): Promise<void> {
    const ps: Array<void|Promise<void>> = [];
    if (this.epochEnd != null) {
      await resolveScalarsInLogs(logs);
      ps.push(this.epochEnd(epoch, logs as Logs));
    }
    if (this.yieldEvery === 'epoch') {
      ps.push(nextFrame());
    }
    await Promise.all(ps);
  }

  async onBatchBegin(batch: number, logs?: UnresolvedLogs): Promise<void> {
    if (this.batchBegin != null) {
      await resolveScalarsInLogs(logs);
      await this.batchBegin(batch, logs as Logs);
    }
  }

  async onBatchEnd(batch: number, logs?: UnresolvedLogs): Promise<void> {
    const ps: Array<void|Promise<void>> = [];
    if (this.batchEnd != null) {
      await resolveScalarsInLogs(logs);
      ps.push(this.batchEnd(batch, logs as Logs));
    }
    if (this.yieldEvery === 'batch') {
      ps.push(nextFrame());
    } else if (util.isNumber(this.yieldEvery)) {
      ps.push(this.maybeWait(this.currentEpoch, batch, logs));
    }
    await Promise.all(ps);
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
export function standardizeCallbacks(
    callbacks: BaseCallback|BaseCallback[]|CustomCallbackArgs|
    CustomCallbackArgs[],
    yieldEvery: YieldEveryOptions): BaseCallback[] {
  if (callbacks == null) {
    callbacks = {} as BaseCallback;
  }
  if (callbacks instanceof BaseCallback) {
    return [callbacks as BaseCallback];
  }
  if (Array.isArray(callbacks) && callbacks[0] instanceof BaseCallback) {
    return callbacks as BaseCallback[];
  }
  // Convert custom callback configs to custom callback objects.
  const callbackConfigs =
      generic_utils.toList(callbacks) as CustomCallbackArgs[];
  return callbackConfigs.map(
      callbackConfig => new CustomCallback(callbackConfig, yieldEvery));
}

export declare type BaseCallbackConstructor = {
  new (): BaseCallback
};

/**
 * A global registry for callback constructors to be used during
 * LayersModel.fit().
 */
export class CallbackConstructorRegistry {
  private static constructors:
      {[verbosityLevel: number]: BaseCallbackConstructor[]} = {};

  /**
   * Blocks public access to constructor.
   */
  private constructor() {}

  /**
   * Register a tf.LayersModel.fit() callback constructor.
   *
   * The registered callback constructor will be used to instantiate
   * callbacks for every tf.LayersModel.fit() call afterwards.
   *
   * @param verbosityLevel Level of verbosity at which the `callbackConstructor`
   *   is to be reigstered.
   * @param callbackConstructor A no-arg constructor for `tf.Callback`.
   * @throws Error, if the same callbackConstructor has been registered before,
   *   either at the same or a different `verbosityLevel`.
   */
  static registerCallbackConstructor(
      verbosityLevel: number, callbackConstructor: BaseCallbackConstructor) {
    util.assert(
        verbosityLevel >= 0 && Number.isInteger(verbosityLevel),
        () => `Verbosity level is expected to be an integer >= 0, ` +
            `but got ${verbosityLevel}`);
    CallbackConstructorRegistry.checkForDuplicate(callbackConstructor);
    if (CallbackConstructorRegistry.constructors[verbosityLevel] == null) {
      CallbackConstructorRegistry.constructors[verbosityLevel] = [];
    }
    CallbackConstructorRegistry.constructors[verbosityLevel].push(
        callbackConstructor);
  }

  private static checkForDuplicate(callbackConstructor:
                                       BaseCallbackConstructor) {
    for (const levelName in CallbackConstructorRegistry.constructors) {
      const constructors = CallbackConstructorRegistry.constructors[+levelName];
      constructors.forEach(ctor => {
        if (ctor === callbackConstructor) {
          throw new ValueError('Duplicate callback constructor.');
        }
      });
    }
  }

  /**
   * Clear all registered callback constructors.
   */
  protected static clear() {
    CallbackConstructorRegistry.constructors = {};
  }

  /**
   * Create callbacks using the registered callback constructors.
   *
   * Given `verbosityLevel`, all constructors registered at that level or above
   * will be called and the instantiated callbacks will be used.
   *
   * @param verbosityLevel: Level of verbosity.
   */
  static createCallbacks(verbosityLevel: number): BaseCallback[] {
    const constructors: BaseCallbackConstructor[] = [];
    for (const levelName in CallbackConstructorRegistry.constructors) {
      const level = +levelName;
      if (verbosityLevel >= level) {
        constructors.push(...CallbackConstructorRegistry.constructors[level]);
      }
    }
    return constructors.map(ctor => new ctor());
  }
}

export function configureCallbacks(
    callbacks: BaseCallback[], verbose: ModelLoggingVerbosity, epochs: number,
    initialEpoch: number, numTrainSamples: number, stepsPerEpoch: number,
    batchSize: number, doValidation: boolean,
    callbackMetrics: string[]): {callbackList: CallbackList, history: History} {
  const history = new History();
  const actualCallbacks: BaseCallback[] = [
    new BaseLogger(), ...CallbackConstructorRegistry.createCallbacks(verbose)
  ];
  if (callbacks != null) {
    actualCallbacks.push(...callbacks);
  }
  actualCallbacks.push(history);
  const callbackList = new CallbackList(actualCallbacks);

  // TODO(cais): Figure out when this LayersModel instance can have a
  // dynamically
  //   set property called 'callback_model' as in PyKeras.

  callbackList.setParams({
    epochs,
    initialEpoch,
    samples: numTrainSamples,
    steps: stepsPerEpoch,
    batchSize,
    verbose,
    doValidation,
    metrics: callbackMetrics,
  });
  return {callbackList, history};
}
