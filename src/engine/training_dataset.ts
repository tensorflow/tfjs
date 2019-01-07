/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

/**
 * Interfaces and methods for training models using TensorFlow.js datasets.
 */

import * as tfc from '@tensorflow/tfjs-core';
import {TensorContainer} from '@tensorflow/tfjs-core/dist/tensor_types';

import {getScalar} from '../backend/state';
import {BaseCallback, configureCallbacks, CustomCallbackConfig, History, ModelLoggingVerbosity, standardizeCallbacks, YieldEveryOptions} from '../base_callbacks';
import {NotImplementedError, ValueError} from '../errors';
import {disposeTensorsInLogs, UnresolvedLogs} from '../logs';
import {singletonOrArray, toList} from '../utils/generic_utils';

import {Dataset, LazyIterator, TensorMap, TensorOrTensorMap} from './dataset_stub';

/**
 * Interface for configuring model training based on a dataset object.
 */
export interface ModelFitDatasetConfig<T extends TensorContainer> {
  /**
   * (Optional) Total number of steps (batches of samples) before
   * declaring one epoch finished and starting the next epoch. It should
   * typically be equal to the number of samples of your dataset divided by
   * the batch size, so that `fitDataset`() call can utilize the entire dataset.
   * If it is not provided, use `done` return value in `iterator.next()` as
   * signal to finish an epoch.
   */
  batchesPerEpoch?: number;

  /**
   * The number of times to iterate over the training dataset.
   *
   * An integer.
   */
  epochs: number;

  /**
   * Verbosity level.
   *
   * Expected to be 0, 1, or 2. Default: 1.
   *
   * 0 - No printed message during fit() call.
   * 1 - In Node.js (tfjs-node), prints the progress bar, together with
   *     real-time updates of loss and metric values and training speed.
   *     In the browser: no action. This is the default.
   * 2 - Not implemented yet.
   */
  verbose?: ModelLoggingVerbosity;

  /**
   * List of callbacks to be called during training.
   * Can consist of one or more of the following fields: `onTrainBegin`,
   * `onTrainEnd`, `onEpochBegin`, `onEpochEnd`, `onBatchBegin`, `onBatchEnd`.
   */
  callbacks?: BaseCallback[]|CustomCallbackConfig|CustomCallbackConfig[];

  /**
   * Data on which to evaluate the loss and any model
   * metrics at the end of each epoch. The model will not be trained on this
   * data. This could be any of the following:
   *
   *   - an Array of `tf.Tensor` objects: [xVal, yVal]
   *   - an Array of `tf.Tensor` objects:
   *       [xVal, yVal, valSampleWeights] (not implemented yet).
   *   - a dataset object.
   *
   * If `validationData` is an Array of Tensor objects, the `tf.Tensor` will be
   * sliced into batches during validation, using the parameter
   * `validationBatchSize` (which defaults to 32). The entirety of the
   * `tf.Tensor` objects will be used in the validation.
   *
   * If `validationData` is a dataset object, and the `validationBatches`
   * parameter is specified, the validation will use `validationBatches` batches
   * drawn from the dataset object. If `validationBatches` parameter is not
   * specified, the validation will stop when the dataset is exhausted.
   *
   * The model will not be trained on this data.
   */
  validationData?:
      [
        tfc.Tensor|tfc.Tensor[]|TensorMap, tfc.Tensor|tfc.Tensor[]|TensorMap
      ]|[tfc.Tensor | tfc.Tensor[] | TensorMap,
         tfc.Tensor|tfc.Tensor[]|TensorMap, tfc.Tensor|tfc.Tensor[]|TensorMap]|
      Dataset<T>;

  /**
   * Optional batch size for validation.
   *
   * Used only if `validationData` is an array of `tf.Tensor` objects, i.e., not
   * a dataset object.
   *
   * If not specified, its value defaults to 32.
   */
  validationBatchSize?: number;

  /**
   * (Optional) Only relevant if `validationData` is specified and is a dataset
   * object.
   *
   * Total number of batches of samples to draw from `validationData` for
   * validation purpose before stopping at the end of every epoch. If not
   * specified, `evaluateDataset` will use `iterator.next().done` as signal to
   * stop validation.
   */
  validationBatches?: number;

  /**
   * Configures the frequency of yielding the main thread to other tasks.
   *
   * In the browser environment, yielding the main thread can improve the
   * responsiveness of the page during training. In the Node.js environment,
   * it can ensure tasks queued in the event loop can be handled in a timely
   * manner.
   *
   * - The value can be one of the following strings:
   *   - 'auto': automatically determine how frequently the yielding happens
   *     by measuring the duration of each batch of training (default).
   *   - 'batch': yield every batch.
   *   - 'epoch': yield every epoch.
   *   - 'never': never yield. (But yielding can still happen through `await
   *      nextFrame()` calls in custom callbacks.)
   */
  yieldEvery?: YieldEveryOptions;

  /**
   * Epoch at which to start training (useful for resuming a previous training
   * run).
   */
  initialEpoch?: number;
}

/**
 * Interface for configuring model evaluation based on a dataset object.
 */
export interface ModelEvaluateDatasetConfig {
  /**
   * Number of batches to draw from the dataset object before ending the
   * evaluation.
   */
  batches?: number;

  /**
   * Verbosity mode.
   */
  verbose?: ModelLoggingVerbosity;
}

// Default batch size used during tensor-based validation.
const DEFAULT_VALIDATION_BATCH_SIZE = 32;

/**
 * Standardize the output of a dataset iterator for use by Model.fitDataset().
 *
 * @param model: A `tf.Model` object.
 * @param iteratorOut The output of a dataset iterator. It is required to be
 *   an array of two tensor containers. Each of the two elements of the array
 *   must be a single `tf.Tensor` or a map from string names to `tf.Tensor`s.
 * @returns A flat array of `tf.Tensor` objects: the input `tf.Tensor`s followed
 *   by the target `tf.Tensor`s.
 */
function standardizeDataIteratorOutput(
    // Type `model` as `any` here to avoid circular dependency w/ training.ts.
    // tslint:disable-next-line:no-any
    model: any, iteratorOut: TensorContainer): tfc.Tensor[] {
  if (model.outputs.length > 1) {
    throw new NotImplementedError(
        `Support for training a model with multiple output tensors with ` +
        `a dataset object is not implemented yet.`);
  }

  tfc.util.assert(
      Array.isArray(iteratorOut) && iteratorOut.length === 2,
      'Dataset iterator for fitDataset() is expected to generate ' +
          'an Array of length 2: `[xs, ys]`, but instead generates ' +
          iteratorOut);
  // TODO(cais): If there are multiple inputs or outputs, make sure
  //   they all have the same batch size.
  iteratorOut = iteratorOut as [TensorOrTensorMap, TensorOrTensorMap];
  const ys = iteratorOut[1] as tfc.Tensor;
  let xs = iteratorOut[0] as TensorOrTensorMap;
  if (xs instanceof tfc.Tensor) {
    tfc.util.assert(
        model.inputs.length === 1,
        `Model has multiple ${model.inputs.length} inputs, hence it ` +
            `expects the input dataset to generate a dictionary of tensors ` +
            ` (with keys ${JSON.stringify(model.inputNames)}, ` +
            `but received a single tensor.`);
    tfc.util.assert(
        xs.shape[0] === ys.shape[0],
        `Mismatch in batch size between x and y tensors (${xs.shape[0]} vs. ` +
            `${ys.shape[0]})`);
    return [xs, ys];
  } else {
    let batchSize: number;
    xs = xs as TensorMap;
    const flattendXs: tfc.Tensor[] = [];
    // Check that all the required keys are available and all the batch sizes
    // are equal.
    for (const inputName of model.inputNames) {
      if (xs[inputName] == null) {
        throw new ValueError(
            `The feature data generated by the dataset lacks the required ` +
            `input key '${inputName}'.`);
      }
      flattendXs.push(xs[inputName]);
      if (batchSize == null) {
        batchSize = xs[inputName].shape[0];
      } else {
        tfc.util.assert(
            xs[inputName].shape[0] === batchSize,
            `Mismatch in batch size between x and y tensors ` +
                `(${xs[inputName].shape[0]} vs. ${ys.shape[0]})`);
      }
    }
    return flattendXs.concat(ys);
  }

  // TODO(cais): Handle case in which ys is a TensorMap.
}

function standardizeTensorValidationData<T extends TensorContainer>(
    data:
        [
          tfc.Tensor|tfc.Tensor[], tfc.Tensor|tfc.Tensor[]
        ]|[tfc.Tensor | tfc.Tensor[], tfc.Tensor | tfc.Tensor[],
           tfc.Tensor | tfc.Tensor[]]):
    {xs: tfc.Tensor|tfc.Tensor[], ys: tfc.Tensor|tfc.Tensor[]} {
  if (data.length === 3) {
    throw new NotImplementedError(
        'Validation with sample weights is not implemented yet.');
  }
  return {xs: data[0], ys: data[1]};
}

export async function fitDataset<T extends TensorContainer>(
    // Type `model` as `any` here to avoid circular dependency w/ training.ts.
    // tslint:disable-next-line:no-any
    model: any, dataset: Dataset<T>,
    config: ModelFitDatasetConfig<T>): Promise<History> {
  const hasBatchesPerEpoch = config.batchesPerEpoch != null;
  tfc.util.assert(
      model.optimizer != null,
      'You must compile a model before training/testing. Use ' +
          'Model.compile(modelCompileConfig).');

  tfc.util.assert(
      config != null,
      `For fitDataset(), the 2nd argument (config) is required, ` +
          `but it is not provided in this call.`);
  tfc.util.assert(
      config.epochs != null && config.epochs > 0 &&
          Number.isInteger(config.epochs),
      `For fitDataset(), config.epochs is expected to be a positive ` +
          `integer, but got ${config.epochs}`);
  tfc.util.assert(
      !hasBatchesPerEpoch ||
          (config.batchesPerEpoch > 0 &&
           Number.isInteger(config.batchesPerEpoch)),
      `For fitDataset(), config.batchesPerEpoch is expected to be a ` +
          `positive integer if specified, but got ${config.batchesPerEpoch}`);
  tfc.util.assert(
      // tslint:disable-next-line:no-any
      (config as any)['validationSplit'] == null,
      '`validationSplit` is not supported by `fitDataset()`. ' +
          'Use validationData instead.');

  if (model.isTraining) {
    throw new Error(
        'Cannot start training because another fit() call is ongoing.');
  }
  model.isTraining = true;

  try {
    const doValidation = config.validationData != null;
    let valXs: tfc.Tensor|tfc.Tensor[];
    let valYs: tfc.Tensor|tfc.Tensor[];
    if (doValidation) {
      if (isDatasetObject(config.validationData)) {
        tfc.util.assert(
            config.validationBatches == null ||
                (config.validationBatches > 0 &&
                 Number.isInteger(config.validationBatches)),
            `For fitDataset() with dataset-based validation, ` +
                `config.validationBatches is expected not to be provided, ` +
                `or to be a positive integer, ` +
                `but got ${config.validationBatches}`);
      } else {
        const validationData = standardizeTensorValidationData(
            config.validationData as
                    [tfc.Tensor | tfc.Tensor[], tfc.Tensor | tfc.Tensor[]] |
            [
              tfc.Tensor | tfc.Tensor[], tfc.Tensor | tfc.Tensor[],
              tfc.Tensor | tfc.Tensor[]
            ]);
        valXs = validationData.xs;
        valYs = validationData.ys;
      }
    }

    const trainFunction = model.makeTrainFunction();
    const outLabels = model.getDedupedMetricsNames() as string[];

    let callbackMetrics: string[];
    if (doValidation) {
      callbackMetrics =
          outLabels.slice().concat(outLabels.map(n => 'val_' + n));
    } else {
      callbackMetrics = outLabels.slice();
    }

    const callbacks = standardizeCallbacks(config.callbacks);
    const {callbackList, history} = configureCallbacks(
        callbacks, config.yieldEvery, config.verbose, config.epochs, null, null,
        config.batchesPerEpoch,
        null,  // Batch size determined by the dataset itself.
        doValidation, callbackMetrics);
    callbackList.setModel(model);
    model.history = history;

    await callbackList.onTrainBegin();
    model.stopTraining_ = false;
    let epoch = config.initialEpoch == null ? 0 : config.initialEpoch;

    let dataIterator = await dataset.iterator();
    while (epoch < config.epochs) {
      const epochLogs: UnresolvedLogs = {};
      await callbackList.onEpochBegin(epoch);
      let stepsDone = 0;
      let batchIndex = 0;
      if (!hasBatchesPerEpoch) {
        dataIterator = await dataset.iterator();
      }
      while (hasBatchesPerEpoch ? stepsDone < config.batchesPerEpoch : true) {
        const iteratorOut = await dataIterator.next();

        // If `batchesPerEpoch` is specified, the dataset should not be
        // exhausted until all epoches are done.
        if (hasBatchesPerEpoch && iteratorOut.done) {
          console.warn(
              'You provided `batchesPerEpoch` as ' +
              `${config.batchesPerEpoch}, ` +
              'but your dataset iterator ran out of data after ' +
              `${stepsDone} batches; ` +
              'interrupting training. Make sure that your ' +
              'dataset can generate at least `batchesPerEpoch * epochs` ' +
              'batches (in this case, ' +
              `${config.batchesPerEpoch * config.epochs} batches). ` +
              'You may need to use the repeat() function when building ' +
              'your dataset.');
          break;
        }


        if (iteratorOut.value != null) {
          const xsAndYs =
              standardizeDataIteratorOutput(model, iteratorOut.value);
          const batchLogs: UnresolvedLogs = {};
          batchLogs['batch'] = batchIndex;
          batchLogs['size'] = xsAndYs[0].shape[0];

          await callbackList.onBatchBegin(batchIndex, batchLogs);

          // Train on batch.
          // TODO(cais): Take care of models with multiple outputs.
          const outs = trainFunction(xsAndYs);
          tfc.dispose(xsAndYs);
          for (let i = 0; i < outLabels.length; ++i) {
            const label = outLabels[i];
            const out = outs[i];
            batchLogs[label] = out;
            tfc.keep(out);
          }

          await callbackList.onBatchEnd(batchIndex, batchLogs);
          disposeTensorsInLogs(batchLogs);

          batchIndex++;
          stepsDone++;
        }

        if (hasBatchesPerEpoch ? stepsDone >= config.batchesPerEpoch :
                                 iteratorOut.done) {
          // Epoch finished. Perform validation.
          if (doValidation) {
            let valOuts: tfc.Scalar[];
            if (isDatasetObject(config.validationData)) {
              valOuts = toList(await model.evaluateDataset(
                  config.validationData, {batches: config.validationBatches}));
            } else {
              valOuts = toList(model.evaluate(valXs, valYs, {
                batchSize: config.validationBatchSize == null ?
                    DEFAULT_VALIDATION_BATCH_SIZE :
                    config.validationBatchSize,
                verbose: 0
              }));
            }
            for (let i = 0; i < model.metricsNames.length; ++i) {
              epochLogs[`val_${model.metricsNames[i]}`] = valOuts[i];
            }
          }
          // Call `break` to exit one epoch lopp after validation is done. If
          // config.batchesPerEpoch is specified, an epoch while loop will stop
          // when `stepsDone >= config.batchesPerEpoch`. When
          // config.batchesPerEpoch is not provided, the following `break` is
          // required to exit the while lopp after dataset is exhausted.
          break;
        }

        if (model.stopTraining_) {
          break;
        }
      }
      await callbackList.onEpochEnd(epoch, epochLogs);
      epoch++;
      if (model.stopTraining_) {
        break;
      }
    }
    await callbackList.onTrainEnd();
    await model.history.syncData();
    return model.history;
  } finally {
    model.isTraining = false;
  }
}

// Check if provided object is a Dataset object by checking it's .iterator
// element.
function isDatasetObject<T extends TensorContainer>(
    dataset:
        [
          tfc.Tensor|tfc.Tensor[]|TensorMap, tfc.Tensor|tfc.Tensor[]|TensorMap
        ]|[tfc.Tensor | tfc.Tensor[] | TensorMap,
           tfc.Tensor | tfc.Tensor[] | TensorMap,
           tfc.Tensor | tfc.Tensor[] | TensorMap]|Dataset<T>): boolean {
  return (typeof (dataset as Dataset<T>).iterator === 'function');
}

// Check if provided object is a LazyIterator object by checking it's .next
// element.
function isLazyIteratorObject<T extends TensorContainer>(
    iterator: Dataset<T>|LazyIterator<T>): boolean {
  return (typeof (iterator as LazyIterator<T>).next === 'function');
}

export async function evaluateDataset<T extends TensorContainer>(
    // Type `model` as `any` here to avoid circular dependency w/ training.ts.
    // tslint:disable-next-line:no-any
    model: any, dataset: Dataset<T>|LazyIterator<T>,
    config: ModelEvaluateDatasetConfig): Promise<tfc.Scalar|tfc.Scalar[]> {
  const hasBatches = config.batches != null;
  const f = model.testFunction;
  const outs: tfc.Scalar[] = [];
  if (config.verbose > 0) {
    throw new NotImplementedError('Verbose mode is not implemented yet.');
  }

  tfc.util.assert(
      !hasBatches || (config.batches > 0 && Number.isInteger(config.batches)),
      'Test loop expects `batches` to be a positive integer, but ' +
          `received ${JSON.stringify(config.batches)}`);
  const dataIterator = isLazyIteratorObject(dataset) ?
      dataset as LazyIterator<T>:
      await (dataset as Dataset<T>).iterator();
  // Keeps track of number of examples used in this evaluation.
  let numExamples = 0;
  let batch = 0;
  while (hasBatches ? batch < config.batches : true) {
    const iteratorOut = await dataIterator.next();
    if (iteratorOut.value) {
      // TODO(cais): Once real dataset is available, use
      //   `map(x => standardizeDataIteratorOutput(model, x).map(f)`.
      const xsAndYs = standardizeDataIteratorOutput(model, iteratorOut.value);
      const batchOuts = tfc.tidy(() => f(xsAndYs));
      tfc.dispose(xsAndYs);

      if (batch === 0) {
        for (let i = 0; i < batchOuts.length; ++i) {
          outs.push(getScalar(0));
        }
      }
      const batchSize = xsAndYs[0].shape[0];
      for (let i = 0; i < batchOuts.length; ++i) {
        const batchOut = batchOuts[i];
        const oldScalar = outs[i];
        outs[i] = tfc.tidy(
            () => tfc.add(outs[i], tfc.mul(getScalar(batchSize), batchOut)) as
                tfc.Scalar);
        if (batch > 0) {
          tfc.dispose(oldScalar);
        }
      }
      tfc.dispose(batchOuts);
      numExamples += batchSize;

      ++batch;
    }
    if (iteratorOut.done) {
      if (hasBatches) {
        console.warn(
            'Your dataset iterator ran out of data during evaluateDataset(). ' +
            'Interrupting evalution. Make sure that your ' +
            'dataset can generate at least `batches` ' +
            `batches (in this case, ${config.batches} batches). ` +
            'You may need to use the repeat() function when building ' +
            'your dataset.');
      }
      break;
    }
  }
  for (let i = 0; i < outs.length; ++i) {
    const oldScalar = outs[i];
    outs[i] =
        tfc.tidy(() => tfc.div(outs[i], getScalar(numExamples)) as tfc.Scalar);
    tfc.dispose(oldScalar);
  }

  return singletonOrArray(outs);
}
