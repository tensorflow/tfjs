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
import {scalar} from '@tensorflow/tfjs-core';
import {BaseCallback, configureCallbacks, CustomCallbackArgs, History, ModelLoggingVerbosity, standardizeCallbacks, YieldEveryOptions} from '../base_callbacks';
import {NotImplementedError, ValueError} from '../errors';
import {disposeTensorsInLogs, UnresolvedLogs} from '../logs';
import {TensorOrArrayOrMap} from '../types';
import {singletonOrArray, toList} from '../utils/generic_utils';

import {Dataset, LazyIterator} from './dataset_stub';

/**
 * Interface for configuring model training based on a dataset object.
 */
export interface ModelFitDatasetArgs<T> {
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
  callbacks?: BaseCallback[]|CustomCallbackArgs|CustomCallbackArgs[];

  /**
   * Data on which to evaluate the loss and any model
   * metrics at the end of each epoch. The model will not be trained on this
   * data. This could be any of the following:
   *
   *   - An array `[xVal, yVal]`, where the two values may be `tf.Tensor`,
   *     an array of Tensors, or a map of string to Tensor.
   *   - Similarly, an array ` [xVal, yVal, valSampleWeights]`
   *     (not implemented yet).
   *   - a `Dataset` object with elements of the form `{xs: xVal, ys: yVal}`,
   *     where `xs` and `ys` are the feature and label tensors, respectively.
   *
   * If `validationData` is an Array of Tensor objects, each `tf.Tensor` will be
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
  validationData?: [
    TensorOrArrayOrMap, TensorOrArrayOrMap
  ]|[TensorOrArrayOrMap, TensorOrArrayOrMap, TensorOrArrayOrMap]|Dataset<T>;

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

export interface FitDatasetElement {
  xs: TensorOrArrayOrMap;
  ys: TensorOrArrayOrMap;
}

/**
 * Interface for configuring model evaluation based on a dataset object.
 */
export interface ModelEvaluateDatasetArgs {
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
 * Standardize the output of a dataset iterator for use by
 * LayersModel.fitDataset().
 *
 * @param model: A `tf.LayersModel` object.
 * @param iteratorOut The output of a dataset iterator. It is required to be
 *   an object of the form `{xs: TensorOrArrayOrMap, ys: TensorOrArrayOrMap}`,
 *   where `TensorOrArrayOrMap` is a single `tf.Tensor`, a `tf.Tensor[]`, or a
 *   flat map from string names to `tf.Tensor`s.
 * @returns A flat array of `tf.Tensor` objects: the input `tf.Tensor`s followed
 *   by the target `tf.Tensor`s.  When `tf.Tensor`s are provided as a map, the
 *   order in the resulting array is taken from the `inputNames` and
 *   `outputNames` of the model.
 */
function standardizeDataIteratorOutput(
    // Type `model` as `any` here to avoid circular dependency w/ training.ts.
    // tslint:disable-next-line:no-any
    model: any, iteratorOut: {}): tfc.Tensor[] {
  let xs: TensorOrArrayOrMap;
  let ys: TensorOrArrayOrMap;

  const iteratorOutObj = iteratorOut as FitDatasetElement;
  xs = iteratorOutObj['xs'];
  ys = iteratorOutObj['ys'];
  tfc.util.assert(
      xs != null && ys != null,
      () => 'A Dataset iterator for fitDataset() is expected to generate ' +
          'objects of the form `{xs: xVal, ys: yVal}`, where the two ' +
          'values may be `tf.Tensor`, an array of Tensors, or a map of ' +
          'string to Tensor.  The provided Dataset instead generates ' +
          iteratorOut);


  const flattenedXs: tfc.Tensor[] =
      flattenTensorOrArrayOrMap('input', model.inputNames, xs);
  const flattenedYs: tfc.Tensor[] =
      flattenTensorOrArrayOrMap('output', model.outputNames, ys);

  const batchSize: number = flattenedXs[0].shape[0];

  tfc.util.assert(
      flattenedXs.length === model.inputs.length,
      () => `LayersModel has ${model.inputs.length} inputs, but the dataset ` +
          `provides ${flattenedXs.length} inputs.  (Expected input keys: ` +
          `${JSON.stringify(model.inputNames)})`);

  tfc.util.assert(
      flattenedYs.length === model.outputs.length,
      () =>
          `LayersModel has ${model.outputs.length} outputs, but the dataset ` +
          `provides ${flattenedYs.length} outputs.  (Expected output keys: ` +
          `${JSON.stringify(model.outputNames)})`);

  for (const xIndex in flattenedXs) {
    tfc.util.assert(
        flattenedXs[xIndex].shape[0] === batchSize,
        () => `Batch size mismatch: input ` +
            `${model.inputNames[xIndex]} has ${
                  flattenedXs[xIndex].shape[0]}; ` +
            `expected  ${batchSize} based on input ${model.inputNames[0]}.`);
  }

  for (const yIndex in flattenedYs) {
    tfc.util.assert(
        flattenedYs[yIndex].shape[0] === batchSize,
        () => `Batch size mismatch: output ` +
            `${model.outputNames[yIndex]} has ${
                  flattenedYs[yIndex].shape[0]}; ` +
            `expected  ${batchSize} based on input ${model.inputNames[0]}.`);
  }

  return flattenedXs.concat(flattenedYs);
}

function flattenTensorOrArrayOrMap(
    inputOrOutput: string, names: string[], values: TensorOrArrayOrMap) {
  if (values instanceof tfc.Tensor) {
    return [values];
  } else if (Array.isArray(values)) {
    tfc.util.assert(
        values.length === names.length,
        () => `Received an array of ${values.length} Tensors, but expected ${
            names.length} to match the ${inputOrOutput} keys ${names}.`);
    return values;
  } else {
    const result: tfc.Tensor[] = [];
    // Check that all the required keys are available.
    for (const name of names) {
      if (values[name] == null) {
        throw new ValueError(
            `The feature data generated by the dataset lacks the required ` +
            `${inputOrOutput} key '${name}'.`);
      }
      result.push(values[name]);
    }
    return result;
  }
}

function standardizeTensorValidationData<T>(
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

export async function fitDataset<T>(
    // Type `model` as `any` here to avoid circular dependency w/ training.ts.
    // tslint:disable-next-line:no-any
    model: any, dataset: Dataset<T>,
    args: ModelFitDatasetArgs<T>): Promise<History> {
  const hasBatchesPerEpoch = args.batchesPerEpoch != null;
  tfc.util.assert(
      model.optimizer != null,
      () => 'You must compile a model before training/testing. Use ' +
          'LayersModel.compile(modelCompileConfig).');

  tfc.util.assert(
      args != null,
      () => `For fitDataset(), the 2nd argument (config) is required, ` +
          `but it is not provided in this call.`);
  tfc.util.assert(
      args.epochs != null && args.epochs > 0 && Number.isInteger(args.epochs),
      () => `For fitDataset(), config.epochs is expected to be a positive ` +
          `integer, but got ${args.epochs}`);
  tfc.util.assert(
      !hasBatchesPerEpoch ||
          (args.batchesPerEpoch > 0 && Number.isInteger(args.batchesPerEpoch)),
      () => `For fitDataset(), config.batchesPerEpoch is expected to be a ` +
          `positive integer if specified, but got ${args.batchesPerEpoch}`);
  tfc.util.assert(
      // tslint:disable-next-line:no-any
      (args as any)['validationSplit'] == null,
      () => '`validationSplit` is not supported by `fitDataset()`. ' +
          'Use validationData instead.');

  if (model.isTraining) {
    throw new Error(
        'Cannot start training because another fit() call is ongoing.');
  }
  model.isTraining = true;

  try {
    const doValidation = args.validationData != null;
    let valXs: tfc.Tensor|tfc.Tensor[];
    let valYs: tfc.Tensor|tfc.Tensor[];
    if (doValidation) {
      if (isDatasetObject(args.validationData)) {
        tfc.util.assert(
            args.validationBatches == null ||
                (args.validationBatches > 0 &&
                 Number.isInteger(args.validationBatches)),
            () => `For fitDataset() with dataset-based validation, ` +
                `config.validationBatches is expected not to be provided, ` +
                `or to be a positive integer, ` +
                `but got ${args.validationBatches}`);
      } else {
        const validationData = standardizeTensorValidationData(
            args.validationData as
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

    const callbacks = standardizeCallbacks(args.callbacks);
    const verbose = args.verbose == null ? 1 : args.verbose;
    const {callbackList, history} = configureCallbacks(
        callbacks, args.yieldEvery, verbose, args.epochs, null, null,
        getStepsPerEpoch(dataset, args),
        null,  // Batch size determined by the dataset itself.
        doValidation, callbackMetrics);
    callbackList.setModel(model);
    model.history = history;

    await callbackList.onTrainBegin();
    model.stopTraining_ = false;
    let epoch = args.initialEpoch == null ? 0 : args.initialEpoch;

    let dataIterator = await dataset.iterator();
    while (epoch < args.epochs) {
      const epochLogs: UnresolvedLogs = {};
      await callbackList.onEpochBegin(epoch);
      let stepsDone = 0;
      let batchIndex = 0;
      if (!hasBatchesPerEpoch) {
        dataIterator = await dataset.iterator();
      }
      while (hasBatchesPerEpoch ? stepsDone < args.batchesPerEpoch : true) {
        const iteratorOut = await dataIterator.next();

        // If `batchesPerEpoch` is specified, the dataset should not be
        // exhausted until all epoches are done.
        if (hasBatchesPerEpoch && iteratorOut.done) {
          console.warn(
              'You provided `batchesPerEpoch` as ' +
              `${args.batchesPerEpoch}, ` +
              'but your dataset iterator ran out of data after ' +
              `${stepsDone} batches; ` +
              'interrupting training. Make sure that your ' +
              'dataset can generate at least `batchesPerEpoch * epochs` ' +
              'batches (in this case, ' +
              `${args.batchesPerEpoch * args.epochs} batches). ` +
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

        if (hasBatchesPerEpoch ? stepsDone >= args.batchesPerEpoch :
                                 iteratorOut.done) {
          // Epoch finished. Perform validation.
          if (doValidation) {
            let valOuts: tfc.Scalar[];
            if (isDatasetObject(args.validationData)) {
              valOuts = toList(await model.evaluateDataset(
                  args.validationData, {batches: args.validationBatches}));
            } else {
              valOuts = toList(model.evaluate(valXs, valYs, {
                batchSize: args.validationBatchSize == null ?
                    DEFAULT_VALIDATION_BATCH_SIZE :
                    args.validationBatchSize,
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

/** Helper function that determines number of steps (batches) per epoch. */
function getStepsPerEpoch<T>(
    dataset: Dataset<T>, args: ModelFitDatasetArgs<T>): number {
  // Attempt to determine # of batches in an epoch.
  let stepsPerEpoch: number = null;
  if (args.batchesPerEpoch != null) {
    stepsPerEpoch = args.batchesPerEpoch;
  } else if (Number.isFinite(dataset.size)) {
    stepsPerEpoch = dataset.size;
  }
  return stepsPerEpoch;
}

// Check if provided object is a Dataset object by checking it's .iterator
// element.
function isDatasetObject<T>(
    dataset:
        [
          TensorOrArrayOrMap, TensorOrArrayOrMap
        ]|[TensorOrArrayOrMap, TensorOrArrayOrMap, TensorOrArrayOrMap]|
    Dataset<T>): boolean {
  return (typeof (dataset as Dataset<T>).iterator === 'function');
}

// Check if provided object is a LazyIterator object by checking it's .next
// element.
function isLazyIteratorObject<T>(iterator: Dataset<T>|
                                 LazyIterator<T>): boolean {
  return (typeof (iterator as LazyIterator<T>).next === 'function');
}

export async function evaluateDataset<T>(
    // Type `model` as `any` here to avoid circular dependency w/ training.ts.
    // tslint:disable-next-line:no-any
    model: any, dataset: Dataset<T>|LazyIterator<T>,
    args: ModelEvaluateDatasetArgs): Promise<tfc.Scalar|tfc.Scalar[]> {
  args = args || {};
  const hasBatches = args.batches != null;
  const f = model.testFunction;
  let outs: tfc.Scalar[] = [];
  if (args.verbose > 0) {
    throw new NotImplementedError('Verbose mode is not implemented yet.');
  }

  tfc.util.assert(
      !hasBatches || (args.batches > 0 && Number.isInteger(args.batches)),
      () => 'Test loop expects `batches` to be a positive integer, but ' +
          `received ${JSON.stringify(args.batches)}`);
  const dataIterator = isLazyIteratorObject(dataset) ?
      dataset as LazyIterator<T>:
      await (dataset as Dataset<T>).iterator();
  // Keeps track of number of examples used in this evaluation.
  let numExamples = 0;
  let batch = 0;

  while (hasBatches ? batch < args.batches : true) {
    const iteratorOut = await dataIterator.next();
    outs = tfc.tidy(() => {
      if (iteratorOut.value) {
        // TODO(cais): Once real dataset is available, use
        //   `map(x => standardizeDataIteratorOutput(model, x).map(f)`.
        const xsAndYs = standardizeDataIteratorOutput(model, iteratorOut.value);
        const batchOuts = tfc.tidy(() => f(xsAndYs));
        tfc.dispose(xsAndYs);

        if (batch === 0) {
          for (let i = 0; i < batchOuts.length; ++i) {
            outs.push(scalar(0));
          }
        }

        const batchSize = xsAndYs[0].shape[0];
        for (let i = 0; i < batchOuts.length; ++i) {
          const batchOut = batchOuts[i];
          const oldScalar = outs[i];
          outs[i] = tfc.tidy(
              () =>
                  tfc.add(outs[i], tfc.mul(batchSize, batchOut)) as tfc.Scalar);
          if (batch > 0) {
            tfc.dispose(oldScalar);
          }
        }
        tfc.dispose(batchOuts);
        numExamples += batchSize;

        ++batch;
      }
      return outs;
    });

    if (iteratorOut.done) {
      if (hasBatches) {
        console.warn(
            'Your dataset iterator ran out of data during evaluateDataset(). ' +
            'Interrupting evalution. Make sure that your ' +
            'dataset can generate at least `batches` ' +
            `batches (in this case, ${args.batches} batches). ` +
            'You may need to use the repeat() function when building ' +
            'your dataset.');
      }
      break;
    }
  }

  for (let i = 0; i < outs.length; ++i) {
    const oldScalar = outs[i];
    outs[i] = tfc.div(outs[i], numExamples);
    tfc.dispose(oldScalar);
  }

  return singletonOrArray(outs);
}
