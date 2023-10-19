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
 * Interfaces and methods for training models using tf.Tensor objects.
 */

import * as tfc from '@tensorflow/tfjs-core';
import {Tensor, Tensor1D} from '@tensorflow/tfjs-core';
import {expandDims, gather, sliceAlongFirstAxis} from '../backend/tfjs_backend';
import {BaseCallback, CustomCallbackArgs, ModelLoggingVerbosity, YieldEveryOptions} from '../base_callbacks';
import {ClassWeight, ClassWeightMap} from './training_utils';

/**
 * Interface configuration model training based on data as `tf.Tensor`s.
 */
export interface ModelFitArgs {
  /**
   * Number of samples per gradient update. If unspecified, it
   * will default to 32.
   */
  batchSize?: number;

  /**
   * Integer number of times to iterate over the training data arrays.
   */
  epochs?: number;

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
  verbose?: ModelLoggingVerbosity | 2;

  /**
   * List of callbacks to be called during training.
   * Can have one or more of the following callbacks:
   *   - `onTrainBegin(logs)`: called when training starts.
   *   - `onTrainEnd(logs)`: called when training ends.
   *   - `onEpochBegin(epoch, logs)`: called at the start of every epoch.
   *   - `onEpochEnd(epoch, logs)`: called at the end of every epoch.
   *   - `onBatchBegin(batch, logs)`: called at the start of every batch.
   *   - `onBatchEnd(batch, logs)`: called at the end of every batch.
   *   - `onYield(epoch, batch, logs)`: called every `yieldEvery` milliseconds
   *      with the current epoch, batch and logs. The logs are the same
   *      as in `onBatchEnd()`. Note that `onYield` can skip batches or
   *      epochs. See also docs for `yieldEvery` below.
   */
  callbacks?: BaseCallback[]|CustomCallbackArgs|CustomCallbackArgs[];

  /**
   * Float between 0 and 1: fraction of the training data
   * to be used as validation data. The model will set apart this fraction of
   * the training data, will not train on it, and will evaluate the loss and
   * any model metrics on this data at the end of each epoch.
   * The validation data is selected from the last samples in the `x` and `y`
   * data provided, before shuffling.
   */
  validationSplit?: number;

  /**
   * Data on which to evaluate the loss and any model
   * metrics at the end of each epoch. The model will not be trained on this
   * data. This could be a tuple [xVal, yVal] or a tuple [xVal, yVal,
   * valSampleWeights]. The model will not be trained on this data.
   * `validationData` will override `validationSplit`.
   */
  validationData?: [
    Tensor|Tensor[], Tensor|Tensor[]
  ]|[Tensor | Tensor[], Tensor|Tensor[], Tensor|Tensor[]];

  /**
   * Whether to shuffle the training data before each epoch. Has
   * no effect when `stepsPerEpoch` is not `null`.
   */
  shuffle?: boolean;

  /**
   * Optional object mapping class indices (integers) to
   * a weight (float) to apply to the model's loss for the samples from this
   * class during training. This can be useful to tell the model to "pay more
   * attention" to samples from an under-represented class.
   *
   * If the model has multiple outputs, a class weight can be specified for
   * each of the outputs by setting this field an array of weight object
   * or an object that maps model output names (e.g., `model.outputNames[0]`)
   * to weight objects.
   */
  classWeight?: ClassWeight|ClassWeight[]|ClassWeightMap;

  /**
   * Optional array of the same length as x, containing
   * weights to apply to the model's loss for each sample. In the case of
   * temporal data, you can pass a 2D array with shape (samples,
   * sequenceLength), to apply a different weight to every timestep of every
   * sample. In this case you should make sure to specify
   * sampleWeightMode="temporal" in compile().
   */
  sampleWeight?: Tensor;

  /**
   * Epoch at which to start training (useful for resuming a previous training
   * run). When this is used, `epochs` is the index of the "final epoch".
   * The model is not trained for a number of iterations given by `epochs`,
   * but merely until the epoch of index `epochs` is reached.
   */
  initialEpoch?: number;

  /**
   * Total number of steps (batches of samples) before
   * declaring one epoch finished and starting the next epoch. When training
   * with Input Tensors such as TensorFlow data tensors, the default `null` is
   * equal to the number of unique samples in your dataset divided by the
   * batch size, or 1 if that cannot be determined.
   */
  stepsPerEpoch?: number;

  /**
   * Only relevant if `stepsPerEpoch` is specified. Total number of steps
   * (batches of samples) to validate before stopping.
   */
  validationSteps?: number;

  /**
   * Configures the frequency of yielding the main thread to other tasks.
   *
   * In the browser environment, yielding the main thread can improve the
   * responsiveness of the page during training. In the Node.js environment,
   * it can ensure tasks queued in the event loop can be handled in a timely
   * manner.
   *
   * The value can be one of the following:
   *   - `'auto'`: The yielding happens at a certain frame rate (currently set
   *               at 125ms). This is the default.
   *   - `'batch'`: yield every batch.
   *   - `'epoch'`: yield every epoch.
   *   - any `number`: yield every `number` milliseconds.
   *   - `'never'`: never yield. (yielding can still happen through `await
   *      nextFrame()` calls in custom callbacks.)
   */
  yieldEvery?: YieldEveryOptions;
}

export function checkBatchSize(batchSize: number) {
  tfc.util.assert(
      batchSize > 0 && Number.isInteger(batchSize),
      () => `batchSize is required to be a positive integer, but got ${
          batchSize}`);
}

/**
 * Slice a Tensor or an Array of Tensors, by start and stop indices.
 *
 * Porting Note: The `_slice_arrays` function in PyKeras is covered by this
 *   function and `sliceArraysByIndices()` together.
 *
 * @param arrays: the input.
 * @param start: the starting index (inclusive).
 * @param stop: the stopping index (exclusive).
 * @returns The result of the slicing. If `arrays` is an `Array` of
 *   `tf.Tensor`s, the slicing will be applied to all elements of the `Array`
 *   in the same way.
 */
export function sliceArrays(
    arrays: Tensor|Tensor[], start: number, stop: number): Tensor|Tensor[] {
  if (arrays == null) {
    return [null];
  } else if (Array.isArray(arrays)) {
    return arrays.map(array => sliceAlongFirstAxis(array, start, stop - start));
  } else {  // Tensor.
    return sliceAlongFirstAxis(arrays, start, stop - start);
  }
}

/**
 * Slice a Tensor or an Array of Tensors, by random-order indices.
 *
 * Porting Note: The `_slice_arrays` function in PyKeras is covered by this
 *   function and `sliceArrays()` together.
 *
 * @param arrays The input `tf.Tensor` or `Array` of `tf.Tensor`s to slice.
 *   If an `Array` of `tf.Tensor`s, all `tf.Tensor`s will be sliced in the
 *   same fashion.
 * @param indices The indices to use for slicing along the first (batch)
 *   dimension.
 * @returns Result(s) of the slicing.
 */
export function sliceArraysByIndices(
    arrays: Tensor|Tensor[], indices: Tensor1D): Tensor|Tensor[] {
  return tfc.tidy(() => {
    if (arrays == null) {
      return null;
    } else if (Array.isArray(arrays)) {
      return arrays.map(
          array => (sliceArraysByIndices(array, indices) as Tensor));
    } else {
      // TODO(cais): indices should be a pre-constructed Tensor1D to avoid
      //   tensor1d() calls.
      return gather(
          arrays,
          indices.dtype === 'int32' ? indices : tfc.cast(indices, 'int32'));
    }
  });
}

/**
 * Returns a list of batch indices (tuples of indices).
 * @param size: Integer, total size of the data to slice into batches.
 * @param batchSize: Integer, batch size.
 * @returns An Array of [batchStart, batchEnd] tuples. batchStart is
 *   inclusive; batchEnd is exclusive. I.e., each batch consists of indices x
 *   that satisfy batchStart <= x < batchEnd.
 */
export function makeBatches(
    size: number, batchSize: number): Array<[number, number]> {
  const output: Array<[number, number]> = [];
  let batchStart = 0;
  let batchEnd: number = null;
  while (batchStart < size) {
    batchEnd = batchStart + batchSize;
    if (batchEnd >= size) {
      batchEnd = size;
    }
    output.push([batchStart, batchEnd]);
    batchStart = batchEnd;
  }
  return output;
}

/**
 * Ensure tensors all have a rank of at least 2.
 *
 * If a tensor has a rank of 1, it is dimension-expanded to rank 2.
 * If any tensor has a rank of 0 (i.e., is a scalar), an error will be thrown.
 */
export function ensureTensorsRank2OrHigher(tensors: Tensor|Tensor[]): Tensor[] {
  const outs: Tensor[] = [];
  if (tensors instanceof Tensor) {
    tensors = [tensors];
  }

  // Make Tensors at least 2D.
  for (let i = 0; i < tensors.length; ++i) {
    const tensor = tensors[i];
    if (tensor.rank === 1) {
      outs.push(expandDims(tensor, 1));
    } else if (tensor.rank === 0) {
      throw new Error(
          'Expected tensor to be at least 1D, but received a 0D tensor ' +
          '(scalar).');
    } else {
      outs.push(tensor);
    }
  }
  return outs;
}

/**
 * Compare a set of tensors with a reference (old) set, discard the ones
 * in the new set that are not present in the reference set.
 *
 * This method is used for memory clenaup during calls such as
 * LayersModel.fit().
 *
 * @param tensors New set which may contain Tensors not present in
 *   `refTensors`.
 * @param refTensors Reference Tensor set.
 */
// TODO(cais, kangyizhang): Deduplicate with tfjs-data.
export function disposeNewTensors(
    tensors: Tensor|Tensor[]|{[inputName: string]: Tensor},
    refTensors: Tensor|Tensor[]|{[inputName: string]: Tensor}): void {
  if (tensors == null) {
    return;
  }
  const oldTensorIds: number[] = [];
  if (refTensors instanceof Tensor) {
    oldTensorIds.push(refTensors.id);
  } else if (Array.isArray(refTensors)) {
    refTensors.forEach(t => oldTensorIds.push(t.id));
  } else if (refTensors != null) {
    // `oldTensors` is a map from string name to Tensor.
    for (const name in refTensors) {
      const oldTensor = refTensors[name];
      oldTensorIds.push(oldTensor.id);
    }
  }

  const tensorsToDispose: Tensor[] = [];
  if (tensors instanceof Tensor) {
    if (oldTensorIds.indexOf(tensors.id) === -1) {
      tensorsToDispose.push(tensors);
    }
  } else if (Array.isArray(tensors)) {
    tensors.forEach(t => {
      if (oldTensorIds.indexOf(t.id) === -1) {
        tensorsToDispose.push(t);
      }
    });
  } else if (tensors != null) {
    // `oldTensors` is a map from string name to Tensor.
    for (const name in tensors) {
      const tensor = tensors[name];
      if (oldTensorIds.indexOf(tensor.id) === -1) {
        tensorsToDispose.push(tensor);
      }
    }
  }

  tensorsToDispose.forEach(t => {
    if (!t.isDisposed) {
      t.dispose();
    }
  });
}
