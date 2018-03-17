/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

/* Original Source: engine/training.py */

// tslint:disable:max-line-length
import * as dl from 'deeplearn';
import {doc, Optimizer, Scalar, Tensor, Tensor1D, tensor1d} from 'deeplearn';
import * as _ from 'underscore';

import * as K from '../backend/deeplearnjs_backend';
import {BaseLogger, Callback, CallbackList, CustomCallbackConfig, History, Logs, standardizeCallbacks} from '../callbacks';
import {NotImplementedError, RuntimeError, ValueError} from '../errors';
import * as losses from '../losses';
import * as Metrics from '../metrics';
import * as optimizers from '../optimizers';
import {LayerVariable, LossOrMetricFn, Shape} from '../types';
import {ClassNameMap, count, singletonOrArray} from '../utils/generic_utils';

import {execute, FeedDict} from './executor';
import {Container, ContainerConfig} from './topology';
// tslint:enable:max-line-length

/**
 * Helper function for polymorphic input data: 1. singleton Tensor.
 */
export function isDataTensor(x: Tensor|Tensor[]|{[inputName: string]: Tensor}|
                             {[inputName: string]: Tensor[]}): boolean {
  return x instanceof Tensor;
}

/**
 * Helper function for polymorphic input data: 2. Array of Tensor.
 */
export function isDataArray(x: Tensor|Tensor[]|
                            {[inputName: string]: Tensor}): boolean {
  return Array.isArray(x);
}

/**
 * Helper function for polymorphic input data: 3. "dict" of Tensor.
 */
export function isDataDict(x: Tensor|Tensor[]|
                           {[inputName: string]: Tensor}): boolean {
  return !isDataTensor(x) && !isDataArray(x);
}

/**
 * Normalizes inputs and targets provided by users.
 * @param data User-provided input data (polymorphic).
 * @param names An Array of expected Tensor names.
 * @param shapes Optional Array of expected Tensor shapes.
 * @param checkBatchAxis Whether to check that the batch axis of the arrays
 *   match  the expected value found in `shapes`.
 * @param exceptionPrefix String prefix used for exception formatting.
 * @returns List of standardized input Tensors (one Tensor per model input).
 * @throws ValueError: in case of improperly formatted user data.
 */
export function standardizeInputData(
    data: Tensor|Tensor[]|{[inputName: string]: Tensor}, names: string[],
    shapes?: Shape[], checkBatchAxis = true, exceptionPrefix = ''): Tensor[] {
  if (names == null || names.length === 0) {
    // Check for the case where the model expected no data, but some data got
    // sent.
    if (data != null) {
      let gotUnexpectedData = false;
      if (isDataArray(data) && (data as Tensor[]).length > 0) {
        gotUnexpectedData = true;
      } else if (isDataDict(data)) {
        for (const key in data) {
          if (data.hasOwnProperty(key)) {
            gotUnexpectedData = true;
            break;
          }
        }
      } else {
        // `data` is a singleton Tensor in this case.
        gotUnexpectedData = true;
      }
      if (gotUnexpectedData) {
        throw new ValueError(
            `Error when checking model ${exceptionPrefix} expected no data, ` +
            `but got ${data}`);
      }
    }
    return [];
  }
  if (data == null) {
    return names.map(name => null);
  }

  let arrays: Tensor[];
  if (isDataDict(data)) {
    data = data as {[inputName: string]: Tensor};
    arrays = [];
    for (const name of names) {
      if (data[name] == null) {
        throw new ValueError(
            `No data provided for "${name}". Need data for each key in: ` +
            `${names}`);
      }
      arrays.push(data[name]);
    }
  } else if (isDataArray(data)) {
    data = data as Tensor[];
    if (data.length !== names.length) {
      throw new ValueError(
          `Error when checking model ${exceptionPrefix}: the Array of ` +
          `Tensors that you are passing to your model is not the size the ` +
          `model expected. Expected to see ${names.length} Tensor(s), but ` +
          `instead got the following list of Tensor(s): ${data}`);
    }
    arrays = data;
  } else {
    data = data as Tensor;
    if (names.length > 1) {
      throw new ValueError(
          `The model ${exceptionPrefix} expects ${names.length} Tensor(s), ` +
          `but only received one Tensor. Found: Tensor with shape ${
              data.shape}`);
    }
    arrays = [data];
  }

  // Make Tensors at least 2D.
  for (let i = 0; i < names.length; ++i) {
    const array = arrays[i];
    if (array.shape.length === 1) {
      arrays[i] = K.expandDims(array, 1);
    }
  }

  // Check shape compatibility.
  if (shapes != null) {
    for (let i = 0; i < names.length; ++i) {
      if (shapes[i] == null) {
        continue;
      }
      const array = arrays[i];
      if (array.shape.length !== shapes[i].length) {
        throw new ValueError(
            `Error when checking ${exceptionPrefix}: expected ${names[i]} ` +
            `to have ${shapes[i].length} dimension(s). but got array with ` +
            `shape ${array.shape}`);
      }
      for (let j = 0; j < shapes[i].length; ++j) {
        if (j === 0 && !checkBatchAxis) {
          // Skip the first (batch) axis.
          continue;
        }
        const dim = array.shape[j];
        const refDim = shapes[i][j];
        if (refDim != null && refDim >= 0 && dim !== refDim) {
          throw new ValueError(
              `Error when checking ${exceptionPrefix}: expected ${names[i]} ` +
              `to have shape [${shapes[i]}], but got array with shape ` +
              `[${array.shape}].`);
        }
      }
    }
  }
  return arrays;
}

/**
 * User input validation for Tensors.
 * @param inputs `Array` of `Tensor`s for inputs.
 * @param targets `Array` of `Tensor`s for targets.
 * @param weights Optional `Array` of `Tensor`s for sample weights.
 * @throws ValueError: in case of incorrectly formatted data.
 */
export function checkArrayLengths(
    inputs: Tensor[], targets: Tensor[], weights?: Tensor[]) {
  const setX = _.unique(inputs.map(input => input.shape[0]));
  setX.sort();
  const setY = _.unique(targets.map(target => target.shape[0]));
  setY.sort();
  // TODO(cais): Check `weights` as well.
  if (setX.length > 1) {
    throw new ValueError(
        `All input Tensors (x) should have the same number of samples. ` +
        `Got array shapes: ` +
        `${JSON.stringify(inputs.map(input => input.shape))}`);
  }
  if (setY.length > 1) {
    throw new ValueError(
        `All target Tensors (y) should have the same number of samples. ` +
        `Got array shapes: ` +
        `${JSON.stringify(targets.map(target => target.shape))}`);
  }
  if (setX.length > 0 && setY.length > 0 && !_.isEqual(setX, setY)) {
    throw new ValueError(
        `Input Tensors should have the same number of samples as target ` +
        `Tensors. Found ${setX[0]} input sample(s) and ${setY[0]} target ` +
        `sample(s).`);
  }
}

/**
 * Validation on the compatibility of targes and loss functions.
 *
 * This helps prevent users from using loss functions incorrectly.
 *
 * @param targets `Array` of `Tensor`s of targets.
 * @param lossFns `Array` of loss functions.
 * @param outputShapes `Array` of shapes of model outputs.
 */
function checkLossAndTargetCompatibility(
    targets: Tensor[], lossFns: LossOrMetricFn[], outputShapes: Shape[]) {
  // TODO(cais): Dedicated test coverage?
  const keyLosses = [
    losses.meanSquaredError, losses.binaryCrossentropy,
    losses.categoricalCrossentropy
  ];
  for (let i = 0; i < targets.length; ++i) {
    const y = targets[i];
    const loss = lossFns[i];
    const shape = outputShapes[i];
    if (loss == null) {
      continue;
    }
    if (loss === losses.categoricalCrossentropy) {
      if (y.shape[y.shape.length - 1] === 1) {
        throw new ValueError(
            `You are passing a target array of shape ${y.shape} while using ` +
            `a loss 'categorical_crossentropy'. 'categorical_crossentropy'` +
            `expects targets to be binary matrices (1s and 0s) of shape ` +
            `[samples, classes].`);
        // TODO(cais): Example code in error message.
      }
    }
    if (_.contains(keyLosses, loss)) {
      const slicedYShape = y.shape.slice(1);
      const slicedShape = shape.slice(1);
      for (let j = 0; j < slicedYShape.length; ++j) {
        const targetDim = slicedYShape[j];
        const outDim = slicedShape[j];
        if (outDim != null && targetDim !== outDim) {
          throw new ValueError(
              `A target Tensor with shape ${y.shape} was passed for an ` +
              `output of shape ${shape}, while using a loss function that ` +
              `expects targets to have the same shape as the output.`);
        }
      }
    }
  }
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
 * Slice an Tensor or an Array of Tensors, by start and stop indices.
 *
 * Porting Note: The `_slice_arrays` function in PyKeras is covered by this
 *   function and `sliceArraysByIndices()` together.
 *
 * @param arrays: the input.
 * @param start: the starting index (inclusive).
 * @param stop: the stopping index (exclusive).
 * @returns The result of the slicing. If `arrays` is an `Array` of
 *   `Tensor`s, the slicing will be applied to all elements of the `Array`
 *   in the same way.
 */
function sliceArrays(
    arrays: Tensor|Tensor[], start: number, stop: number): Tensor|Tensor[] {
  if (arrays == null) {
    return [null];
  } else if (Array.isArray(arrays)) {
    return arrays.map(
        array => K.sliceAlongFirstAxis(array, start, stop - start));
  } else {  // Tensor.
    return K.sliceAlongFirstAxis(arrays, start, stop - start);
  }
}

/**
 * Slice an Tensor or an Array of Tensors, by random-order indices.
 *
 * Porting Note: The `_slice_arrays` function in PyKeras is covered by this
 *   function and `sliceArrays()` together.
 *
 * @param arrays The input `Tensor` or `Array` of `Tensor`s to slice.
 *   If an `Array` of `Tensor`s, all `Tensor`s will be sliced in the
 *   same fashion.
 * @param indices The indices to use for slicing along the first (batch)
 *   dimension.
 * @returns Result(s) of the slicing.
 */
export function sliceArraysByIndices(
    arrays: Tensor|Tensor[], indices: Tensor1D): Tensor|Tensor[] {
  if (arrays == null) {
    return null;
  } else if (Array.isArray(arrays)) {
    return arrays.map(
        array => (sliceArraysByIndices(array, indices) as Tensor));
  } else {
    // TODO(cais): indices should be oa pre-constructed Tensor1D to avoid
    //   tensor1d() calls.
    return K.gather(arrays, indices);
  }
}

/**
 * Check inputs provided by the user.
 *
 * Porting Note: This corresponds to _standardize_input_data() in Python
 *   Keras. Because of the strong typing in TF.js, we do not need to convert
 *   the data. Specifically:
 *   1) in PyKeras, `data` can be `DataFrame` instances from pandas, for
 *      example. We don't need to worry about that here because there is no
 *      widely popular javascript/typesdcript equivalent of pandas (so far).
 *      If one becomes available in the future, we can add support.
 *   2) in PyKeras, inputs can be Python dict. But here we are stipulating
 * that the data is either a single `Tensor` or an Array of `Tensor`s. We
 * may add support for `Object` data inputs in the future when the need
 * arises.
 *
 * Instead, we perform basic checks for number of parameters and shapes.
 *
 * @param data: The input data.
 * @param names: Name for the inputs, from the model.
 * @param shapes: Expected shapes for the input data, from the model.
 * @param checkBatchAxis: Whether the size along the batch axis (i.e., the
 *   first dimension) will be checked for matching.
 * @param exceptionPrefix: Execption prefix message, used in generating error
 *   messages.
 * @throws ValueError: on incorrect number of inputs or mismatches in shapes.
 */
function checkInputData(
    data: Tensor|Tensor[], names: string[], shapes?: Shape[],
    checkBatchAxis = true, exceptionPrefix = '') {
  let arrays: Tensor[];
  if (Array.isArray(data)) {
    if (data.length !== names.length) {
      throw new ValueError(
          `Error when checking model ${exceptionPrefix}: the Array of ` +
          `Tensors that you are passing to your model is not the size the ` +
          `the model expected. Expected to see ${names.length} Tensor(s),` +
          ` but instead got ${data.length} Tensors(s).`);
    }
    arrays = data;
  } else {
    if (names.length > 1) {
      throw new ValueError(
          `The model expects ${names.length} ${exceptionPrefix} Tensors, ` +
          `but only received one Tensor. Found: array with shape ` +
          `${JSON.stringify(data.shape)}.`);
    }
    arrays = [data];
  }

  if (shapes != null) {
    for (let i = 0; i < names.length; ++i) {
      if (shapes[i] == null) {
        continue;
      }
      const array = arrays[i];
      if (array.shape.length !== shapes[i].length) {
        throw new ValueError(
            `Error when checking ${exceptionPrefix}: expected ${names[i]} ` +
            `to have ${shapes[i].length} dimension(s), but got array with ` +
            `shape ${JSON.stringify(array.shape)}`);
      }
      for (let j = 0; j < shapes[i].length; ++j) {
        if (j === 0 && !checkBatchAxis) {
          continue;
        }
        const dim = array.shape[j];
        const refDim = shapes[i][j];
        if (refDim != null) {
          if (refDim !== dim) {
            throw new ValueError(
                `Error when checking ${exceptionPrefix}: expected ` +
                `${names[i]} to have shape ${JSON.stringify(shapes[i])} but ` +
                `got array with shape ${JSON.stringify(array.shape)}.`);
          }
        }
      }
    }
  }
}

/**
 * Maps metric functions to model outputs.
 * @param metrics An `Array` or dict (`Object`) of metric functions.
 * @param outputNames An `Array` of the names of model outputs.
 * @returns An `Array` (one entry per model output) of `Array` of metric
 *   functions. For instance, if the model has 2 outputs, and for the first
 *   output we want to compute `binaryAccuracy` and `binaryCrossentropy`,
 *   and just `binaryAccuracy` for the second output, the `Array` would look
 *   like:
 *     `[[binaryAccuracy, binaryCrossentropy],  [binaryAccuracy]]`
 * @throws TypeError: if `null` or `undefined` value is provided.
 */
function collectMetrics(
    metrics: string[]|{[outputName: string]: string | string[]},
    outputNames: string[]): string[][] {
  if (metrics == null || Array.isArray(metrics) && metrics.length === 0) {
    return outputNames.map(name => []);
  }
  if (Array.isArray(metrics)) {
    // We then apply all metrics to all outputs.
    return outputNames.map(name => metrics);
  } else if (metrics != null) {
    // In this case, metrics is a dict.
    const nestedMetrics: string[][] = [];
    for (const name of outputNames) {
      let outputMetrics: string|string[] =
          metrics.hasOwnProperty(name) ? metrics[name] : [];
      if (!Array.isArray(outputMetrics)) {
        outputMetrics = [outputMetrics];
      }
      nestedMetrics.push(outputMetrics as string[]);
    }
    return nestedMetrics;
  } else {
    throw new TypeError(
        'Type of metrics argument not understood. Expected an Array or ' +
        'Object, found: ' + metrics);
  }
}

/** Verbosity logging level when fitting a model. */
export enum ModelLoggingVerbosity {
  SILENT = 0,
  VERBOSE = 1
}


export interface ModelPredictConfig {
  /**
   * Batch size (Integer). If unspecified, it will default to 32.
   */
  batchSize?: number;

  /**
   * Verbosity mode. Defaults to false.
   */
  verbose?: boolean;
}

export interface ModelEvaluateConfig {
  /**
   * Batch size (Integer). If unspecified, it will default to 32.
   */
  batchSize?: number;

  /**
   * Verbosity mode.
   */
  verbose?: ModelLoggingVerbosity;

  /**
   * Tensor of weights to weight the contribution of different samples to the
   * loss and metrics.
   */
  sampleWeight?: Tensor;

  /**
   * integer: total number of steps (batches of samples)
   * before declaring the evaluation round finished. Ignored with the default
   * value of `undefined`.
   */
  steps?: number;
}

/**
 * Interface for specifying data to fit a model to data.
 */
export interface ModelFitConfig {
  /**
   * Number of samples per gradient update. If unspecified, it
   * will default to 32.
   */
  batchSize?: number;

  /** The number of times to iterate over the training data arrays. */
  epochs?: number;

  verbose?: ModelLoggingVerbosity;

  /** List of callbacks to be called during training. */
  callbacks?: Callback[]|CustomCallbackConfig|CustomCallbackConfig[];

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
   * Optional dictionary mapping class indices (integers) to
   * a weight (float) to apply to the model's loss for the samples from this
   * class during training. This can be useful to tell the model to "pay more
   * attention" to samples from an under-represented class.
   */
  classWeight?: {[classIndex: string]: number};

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
   * run).
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
}

/**
 * Configuration for calls to `Model.compile()`.
 */
export interface ModelCompileConfig {
  /**
   * An instance of `tf.train.Optimizer` or a string name for an Optimizer.
   */
  optimizer: string|Optimizer;

  /**
   * String (name of objective function) or objective function.
   * If the model has multiple outputs, you can use a different loss
   * on each output by passing a dictionary or an Array of losses.
   * The loss value that will be minimized by the model will then be the sum
   * of all individual losses.
   */
  loss: string|string[]|{[outputName: string]: string};

  /**
   * List of metrics to be evaluated by the model during training and testing.
   * Typically you will use `metrics=['accuracy']`.
   * To specify different metrics for different outputs of a multi-output
   * model, you could also pass a dictionary,
   */
  metrics?: string[]|{[outputName: string]: string};

  // TODO(cais): Add lossWeights, sampleWeightMode, weightedMetrics, and
  //   targetTensors.
}

/**
 * The `Model` class adds training & evaluation routines to a `Container`.
 *
 * A `Model` is the basic unit of training, inference and evaluation in
 * TensorFlow.js. To create a `Model, use `model`.
 *
 * When creating a `Model`, specify its input(s) and output(s). Inputs are
 * `SymbolicTensor`s provided by `Input` layers. Outputs are `SymbolicTensor`s
 * provided by other layers that perform mathematical and neural-network
 * operations.
 *
 * For example, the following code snippet defines a model consisting of
 * two `dense` layers, with of 10 and 4 units, respectively.
 *
 * ```js
 * // Define input, which has a size of 5 (not including batch dimension).
 * const input = tf.input({shape: [5]});
 *
 * // First dense layer uses relu activation.
 * const denseLayer1 = tf.layers.dense({units: 10, activation: 'relu'});
 * // Second dense layer uses softmax activation.
 * const denseLayer2 = tf.layers.dense({units: 2, activation: 'softmax'});
 *
 * // Obtain the output symbolic tensor by applying the layers on the input.
 * const output = denseLayer2.apply(denseLayer1.apply(input));
 *
 * // Create the model based on the inputs.
 * const model = tf.model({inputs: input, outputs: output});
 *
 * // The model can be used for training, evaluation and prediction.
 * // For example, the following line runs prediction with the model on
 * // some fake data.
 * (await model.predict(tf.ones([2, 5]))).print();
 * ```
 *
 * See also:
 *   `Sequential`, `loadModel`.
 */
@doc({heading: 'Models', subheading: 'Classes'})
export class Model extends Container {
  optimizer: optimizers.LayersOptimizer;
  loss: string|string[]|{[outputName: string]: string};
  lossFunctions: LossOrMetricFn[];

  // TOOD(cais): These private variables should probably not have the string
  //   'feed' in their names, because we are not dealing with a symbolic
  //   backend.
  private feedOutputShapes: Shape[];
  private feedLossFns: LossOrMetricFn[];
  private collectedTrainableWeights: LayerVariable[];
  private testFunction: (data: Tensor[]) => Scalar[];
  history: History;

  metrics: string[]|{[outputName: string]: string};
  metricsNames: string[];
  // Porting Note: `metrics_tensors` in PyKeras is a symbolic tensor. But given
  //   the imperative nature of tfjs-core, `metricsTensors` is a
  //   TypeScript function here.
  //   Also note that due to the imperative nature of tfjs-core, `metricsTensor`
  //   here needs an output index to keep track of which output of the Model
  //   a metric belongs to. This is unlike `metrics_tensors` in PyKeras,
  //   which is a `list` of symbolic tensors, each of which has implicit
  //   "knowledge" of the outputs it depends on.
  metricsTensors: Array<[LossOrMetricFn, number]>;

  constructor(config: ContainerConfig) {
    super(config);
  }

  @doc({heading: 'Models', subheading: 'Classes'})
  compile(config: ModelCompileConfig): void {
    if (config.loss == null) {
      config.loss = [];
    }
    this.loss = config.loss;

    const optimizerConstructor = optimizers.get(config.optimizer);
    if (typeof config.optimizer === 'string') {
      this.optimizer = new optimizerConstructor({});
    } else {
      this.optimizer = new optimizerConstructor(config.optimizer);
      // TODO(cais): This should call a factory method.
    }
    // TODO(cais): Add lossWeights.
    // TODO(cais): Add sampleWeightMode.

    // Prepare loss functions.
    let lossFunctions: LossOrMetricFn[] = [];
    if (!Array.isArray(config.loss) && typeof config.loss !== 'string') {
      config.loss = config.loss as {[outputName: string]: string};
      for (const name in config.loss) {
        if (!_.contains(this.outputNames, name)) {
          throw new ValueError(
              `Unknown entry in loss dictionary: "${name}". Only expect the ` +
              `following keys: ${this.outputNames}`);
        }
      }
      for (const name in this.outputNames) {
        if (config.loss[name] == null) {
          console.warn(
              `Output "${name}" is missing from loss dictionary. We assume ` +
              `this was done on purpose, and we will not be expecting data ` +
              `to be passed to ${name} during training`);
        }
        lossFunctions.push(losses.get(config.loss[name]));
      }
    } else if (Array.isArray(config.loss)) {
      if (config.loss.length !== this.outputs.length) {
        throw new ValueError(
            `When passing an Array as loss, it should have one entry per ` +
            `model output. The model has ${this.outputs.length} output(s), ` +
            `but you passed loss=${config.loss}.`);
      }
      lossFunctions = config.loss.map(l => losses.get(l));
    } else {
      const lossFunction = losses.get(config.loss);
      this.outputs.map(layer => {
        lossFunctions.push(lossFunction);
      });
    }

    this.lossFunctions = lossFunctions;

    this.feedOutputNames = [];
    this.feedOutputShapes = [];
    this.feedLossFns = [];
    for (let i = 0; i < this.outputs.length; ++i) {
      // TODO(cais): Logic for skipping target(s).
      const shape = this.internalOutputShapes[i];
      const name = this.outputNames[i];
      this.feedOutputNames.push(name);
      this.feedOutputShapes.push(shape);
      this.feedLossFns.push(this.lossFunctions[i]);
    }

    // TODO(cais): Add logic for weighted losses.
    // TODO(cais): Add logic for output masks.
    // TOOD(cais): Add logic for sample weights.
    const skipTargetIndices: number[] = [];

    // Prepare metrics.
    this.metrics = config.metrics;
    // TODO(cais): Add weightedMetrics.
    this.metricsNames = ['loss'];
    this.metricsTensors = [];

    // Compute total loss.
    // Porting Note: In PyKeras, metrics_tensors are symbolic tensor objects.
    //   Here, metricsTensors are TypeScript functions. This difference is due
    //   to the difference in symbolic/imperative property of the backends.
    K.nameScope('loss', () => {
      for (let i = 0; i < this.outputs.length; ++i) {
        if (skipTargetIndices.indexOf(i) !== -1) {
          continue;
        }
        // TODO(cais): Add weightedLoss, sampleWeight and mask.
        //   The following line should be weightedLoss
        const weightedLoss = this.lossFunctions[i];
        if (this.outputs.length > 1) {
          this.metricsTensors.push([weightedLoss, i]);
          this.metricsNames.push(this.outputNames[i] + '_loss');
        }
      }

      // TODO(cais): Add regularizer penalties.
    });

    const nestedMetrics = collectMetrics(config.metrics, this.outputNames);
    // TODO(cais): Add nestedWeightedMetrics.

    /**
     * Helper function used in loop below.
     */
    const appendMetric =
        (outputIndex: number, metricName: string,
         metricTensor: LossOrMetricFn) => {
          if (this.outputNames.length > 1) {
            metricName = this.outputNames[outputIndex] + '_' + metricName;
          }
          this.metricsNames.push(metricName);
          this.metricsTensors.push([metricTensor, outputIndex]);
        };

    K.nameScope('metric', () => {
      for (let i = 0; i < this.outputs.length; ++i) {
        if (skipTargetIndices.indexOf(i) !== -1) {
          continue;
        }
        const outputMetrics = nestedMetrics[i];
        // TODO(cais): Add weights and outputWeightedMetrics.

        // TODO(cais): Add optional arg `weights` to the following function.
        const handleMetrics = (metrics: string[]) => {
          const metricNamePrefix = '';
          let metricName: string;
          let accFn: LossOrMetricFn;
          let weightedMetricFn: LossOrMetricFn;
          //  TOOD(cais): Use 'weights_' for weighted metrics.

          for (const metric of metrics) {
            if (['accuracy', 'acc', 'crossentropy', 'ce'].indexOf(metric) !==
                -1) {
              const outputShape = this.internalOutputShapes[i];

              if (outputShape[outputShape.length - 1] === 1 ||
                  this.lossFunctions[i] === losses.binaryCrossentropy) {
                // case: binary accuracy/crossentropy.
                if (['accuracy', 'acc'].indexOf(metric) !== -1) {
                  accFn = Metrics.binaryAccuracy;
                } else if (['crossentropy', 'ce'].indexOf(metric) !== -1) {
                  accFn = Metrics.binaryCrossentropy;
                }
              } else if (
                  this.lossFunctions[i] ===
                  losses.sparseCategoricalCrossentropy) {
                // case: categorical accuracy / crossentropy with sparse
                // targets.
                if (['accuracy', 'acc'].indexOf(metric) !== -1) {
                  accFn = Metrics.sparseCategoricalAccuracy;
                } else if (['crossentropy', 'ce'].indexOf(metric) !== -1) {
                  accFn = Metrics.sparseCategoricalCrossentropy;
                }
              } else {
                // case: categorical accuracy / crossentropy.
                if (['accuracy', 'acc'].indexOf(metric) !== -1) {
                  accFn = Metrics.categoricalAccuracy;
                } else if (['crossentropy', 'ce'].indexOf(metric) !== -1) {
                  accFn = Metrics.categoricalCrossentropy;
                }
              }
              let suffix: string;
              if (['accuracy', 'acc'].indexOf(metric) !== -1) {
                suffix = 'acc';
              } else if (['crossentropy', 'ce'].indexOf(metric) !== -1) {
                suffix = 'ce';
              }
              // TODO(cais): Add weighting actually.
              weightedMetricFn = accFn;
              metricName = metricNamePrefix + suffix;
            } else {
              const metricFn = Metrics.get(metric);
              // TODO(cais): Add weighting actually.
              weightedMetricFn = metricFn;
              metricName = metricNamePrefix + metric;
            }

            // TODO(cais): Add weighting and masking to metricResult.
            let metricResult: LossOrMetricFn;
            K.nameScope(metricName, () => {
              metricResult = weightedMetricFn;
            });
            appendMetric(i, metricName, metricResult);
          }
        };

        handleMetrics(outputMetrics);
        // TODO(cais): Call handleMetrics with weights.
      }
    });

    // Porting Notes: Given the imperative backend of tfjs-core,
    //   there is no need for constructing the symbolic graph and placeholders.
    this.collectedTrainableWeights = this.trainableWeights;
  }

  /**
   * Check trainable weights count consistency.
   *
   * This will raise a warning if `this.trainableWeights` and
   * `this.collectedTrainableWeights` are inconsistent (i.e., have different
   * numbers of parameters).
   * Inconsistency will typically arise when one modifies `model.trainable`
   * without calling `model.compile()` again.
   */
  private checkTrainableWeightsConsistency(): void {
    if (this.collectedTrainableWeights == null) {
      return;
    }
    if (this.trainableWeights.length !==
        this.collectedTrainableWeights.length) {
      console.warn(
          'Discrepancy between trainableweights and collected trainable ' +
          'weights. Did you set `model.trainable` without calling ' +
          '`model.compile()` afterwards?');
    }
  }

  /**
   * Returns the loss value & metrics values for the model in test mode.
   *
   * Loss and metrics are specified during `compile()`, which needs to happen
   * before calls to `evaluate()`.
   *
   * Computation is done in batches.
   *
   * ```js
   * const model = tf.sequential({
   *   layers: [tf.layers.dense({units: 1, inputShape: [10]})]
   * });
   * model.compile({optimizer: 'sgd', loss: 'meanSquaredError'});
   * const result = await model.evaluate(tf.ones([8, 10]), tf.ones([8, 1]), {
   *   batchSize: 4,
   * });
   * result.print();
   * ```
   *
   * @param x `Tensor` of test data, or an `Array` of `Tensor`s if the model has
   *   multiple inputs.
   * @param y `Tensor` of target data, or an `Array` of `Tensor`s if the model
   *   has multiple outputs.
   * @param config A `ModelEvaluateConfig`, containing optional fields.
   *
   * @return `Scalar` test loss (if the model has a single output and no
   *   metrics) or `Array` of `Scalar`s (if the model has multiple outputs
   *   and/or metrics), as a `Promise`. The attribute `model.metricsNames`
   *   will give you the display labels for the scalar outputs.
   */
  @doc({heading: 'Models', subheading: 'Classes', configParamIndices: [2]})
  async evaluate(
      x: Tensor|Tensor[], y: Tensor|Tensor[], config: ModelEvaluateConfig = {}):
      Promise<Scalar|Scalar[]> {
    const batchSize = config.batchSize == null ? 32 : config.batchSize;

    // TODO(cais): Standardize `config.sampleWeights` as well.
    // Validate user data.
    const standardizedOuts = this.standardizeUserData(x, y, true, batchSize);
    // TODO(cais): If uses `useLearningPhase`, set the corresponding element of
    //   the input to 0.
    const ins = standardizedOuts[0].concat(standardizedOuts[1]);
    this.makeTestFunction();
    const f = this.testFunction;
    const testOuts =
        this.testLoop(f, ins, batchSize, config.verbose, config.steps);
    return singletonOrArray(testOuts);
  }

  /**
   * Get number of samples provided for training, evaluation or prediction.
   *
   * @param ins Input `Tensor`.
   * @param batchSize Integer batch size, optional.
   * @param steps Total number of steps (batches of samples) before declaring
   *   loop finished. Optional.
   * @param stepsName The public API's parameter name for `steps`.
   * @returns Number of samples provided.
   */
  private checkNumSamples(
      ins: Tensor|Tensor[], batchSize?: number, steps?: number,
      stepsName = 'steps'): number {
    let numSamples: number;
    if (steps != null) {
      numSamples = null;
      if (batchSize != null) {
        throw new ValueError(
            `If ${stepsName} is set, batchSize must be null or undefined.` +
            `Got batchSize = ${batchSize}`);
      }
    } else if (ins != null) {
      if (Array.isArray(ins)) {
        numSamples = ins[0].shape[0];
      } else {
        numSamples = ins.shape[0];
      }
    } else {
      throw new ValueError(
          `Either the input data should have a defined shape, or ` +
          `${stepsName} shoud be specified.`);
    }
    return numSamples;
  }

  /**
   * Helper method to loop over some data in batches.
   *
   * Porting Note: Not using the functional approach in the Python equivalent
   *   due to the imperative backend.
   * Porting Note: Does not support step mode currently.
   *
   * @param ins: input data
   * @param batchSize: integer batch size.
   * @param verbose: verbosity model
   * @returns: Predictions as `Tensor` (if a single output) or an `Array` of
   *   `Tensor` (if multipe outputs).
   */
  private predictLoop(ins: Tensor|Tensor[], batchSize = 32, verbose = false):
      Tensor|Tensor[] {
    const numSamples = this.checkNumSamples(ins);
    if (verbose) {
      throw new NotImplementedError(
          'Verbose predictLoop() is not implemented yet.');
    }

    // Sample-based predictions.
    // Porting Note: Tensor currently does not support sliced assignments as
    //   in numpy, e.g., x[1:3] = y. Therefore we use concatenation while
    //   iterating over the batches.

    const batches = makeBatches(numSamples, batchSize);
    const outs: Tensor[] = [];
    // TODO(cais): Can the scope() be pushed down inside the for loop?
    for (let batchIndex = 0; batchIndex < batches.length; ++batchIndex) {
      const batchOuts = dl.tidy(() => {
        const batchStart = batches[batchIndex][0];
        const batchEnd = batches[batchIndex][1];
        // TODO(cais): Take care of the case of the last element is a flag for
        //   training/test.
        const insBatch = sliceArrays(ins, batchStart, batchEnd);

        // Construct the feeds for execute();
        const feeds = [];
        if (Array.isArray(insBatch)) {
          for (let i = 0; i < insBatch.length; ++i) {
            feeds.push({key: this.inputs[i], value: insBatch[i]});
          }
        } else {
          feeds.push({key: this.inputs[0], value: insBatch});
        }
        const feedDict = new FeedDict(feeds);
        return execute(this.outputs, feedDict) as Tensor[];
      });
      if (batchIndex === 0) {
        // Pre-allocate.
        for (const batchOut of batchOuts) {
          outs.push(batchOut);
        }
      } else {
        for (let i = 0; i < batchOuts.length; ++i) {
          outs[i] = K.concatAlongFirstAxis(outs[i], batchOuts[i]);
        }
      }
    }
    return singletonOrArray(outs);
  }

  /**
   * Generates output predictions for the input samples.
   *
   * Computation is done in batches.
   *
   * Note: the "step" mode of predict() is currently not supported.
   *   This is because the TensorFow.js core backend is imperative only.
   *
   * ```js
   * const model = tf.sequential({
   *   layers: [tf.layers.dense({units: 1, inputShape: [10]})]
   * });
   * model.predict(tf.ones([8, 10]), {batchSize: 4}).print();
   * ```
   *
   * @param x The input data, as an Tensor, or an `Array` of `Tensor`s if
   *   the model has multiple inputs.
   * @param conifg A `ModelPredictConfig` object containing optional fields.
   *
   * @return Prediction results as a `Promise` of `Tensor`(s).
   *
   * @exception ValueError In case of mismatch between the provided input data
   *   and the model's expectations, or in case a stateful model receives a
   *   number of samples that is not a multiple of the batch size.
   */
  @doc({heading: 'Models', subheading: 'Classes', configParamIndices: [1]})
  async predict(x: Tensor|Tensor[], config: ModelPredictConfig = {}):
      Promise<Tensor|Tensor[]> {
    checkInputData(x, this.inputNames, this.feedInputShapes, false);
    // TODO(cais): Take care of stateful models.
    //   if (this.stateful) ...
    // TODO(cais): Take care of the learning_phase boolean flag.
    //   if (this.useLearningPhase) ...
    const batchSize = config.batchSize == null ? 32 : config.batchSize;
    return this.predictLoop(x, batchSize);
  }

  /**
   * Returns predictions for a single batch of samples.
   *
   * @param x: Input samples, as an Tensor
   * @return Tensor(s) of predictions
   */
  @doc({heading: 'Models', subheading: 'Classes'})
  predictOnBatch(x: Tensor): Tensor|Tensor[] {
    checkInputData(x, this.inputNames, this.feedInputShapes, true);
    // TODO(cais): Take care of the learning_phase boolean flag.
    //   if (this.useLearningPhase) ...
    return this.predictLoop(x, x.shape[0]);
  }

  protected standardizeUserData(
      x: Tensor|Tensor[]|{[inputName: string]: Tensor},
      y: Tensor|Tensor[]|{[inputName: string]: Tensor}, checkBatchAxis = true,
      batchSize?: number): [Tensor[], Tensor[], Tensor[]] {
    // TODO(cais): Add sampleWeight, classWeight
    if (this.optimizer == null) {
      throw new RuntimeError(
          'You must compile a model before training/testing. Use ' +
          'Model.compile(modelCompileConfig).');
    }
    const outputShapes: Shape[] = [];
    for (let i = 0; i < this.feedOutputShapes.length; ++i) {
      const outputShape = this.feedOutputShapes[i];
      const lossFn = this.feedLossFns[i];
      if (lossFn === losses.sparseCategoricalCrossentropy) {
        outputShapes.push(
            outputShape.slice(0, outputShape.length - 1).concat([1]));
      } else {
        // Porting Note: Because of strong typing `lossFn` must be a function.
        outputShapes.push(outputShape);
      }
    }
    x = standardizeInputData(
            x, this.feedInputNames, this.feedInputShapes, false, 'input') as
        Tensor[];
    y = standardizeInputData(
            y, this.feedOutputNames, outputShapes, false, 'target') as Tensor[];
    // TODO(cais): Standardize sampleWeights & classWeights.
    checkArrayLengths(x, y, null);
    // TOOD(cais): Check sampleWeights as well.
    checkLossAndTargetCompatibility(y, this.feedLossFns, this.feedOutputShapes);
    if (this.stateful && batchSize != null && batchSize > 0) {
      if (x[0].shape[0] % batchSize !== 0) {
        throw new ValueError(
            `In a stateful network, you should only pass inputs with a ` +
            `number of samples that is divisible by the batch size ` +
            `${batchSize}. Found: ${x[0].shape[0]} sample(s).`);
      }
    }
    // TODO(cais): Deal with the case of model.stateful == true.
    return [x, y, null];
  }

  /**
   * Abstract fit function for `f(ins)`.
   * @param f A Function returning a list of tensors. For training, this
   *   function is expected to perform the updates to the variables.
   * @param ins List of tensors to be fed to `f`.
   * @param outLabels List of strings, display names of the outputs of `f`.
   * @param batchSize Integer batch size or `== null` if unknown.
   * @param epochs Number of times to iterate over the data.
   * @param verbose Verbosity mode: 0, 1, or 2.
   * @param callbacks List of callbacks to be called during training.
   * @param valF Function to call for validation.
   * @param valIns List of tensors to be fed to `valF`.
   * @param shuffle Whether to shuffle the data at the beginning of every epoch.
   * @param callbackMetrics List of strings, the display names of the metrics
   *   passed to the callbacks. They should be the concatenation of the
   *   display names of the outputs of `f` and the list of display names
   *   of the outputs of `valF`.
   * @param initialEpoch Epoch at which to start training (useful for
   *   resuming a previous training run).
   * @param stepsPerEpoch Total number of steps (batches on samples) before
   *   declaring one epoch finished and starting the next epoch. Ignored with
   *   the default value of `undefined` or `null`.
   * @param validationSteps Number of steps to run validation for (only if
   *   doing validation from data tensors). Not applicable for tfjs-layers.
   * @returns A `History` object.
   */
  private async fitLoop(
      f: (data: Tensor[]) => Scalar[], ins: Tensor[], outLabels?: string[],
      batchSize?: number, epochs?: number, verbose?: number,
      callbacks?: Callback[], valF?: (data: Tensor[]) => Scalar[],
      valIns?: Tensor[], shuffle?: boolean|string, callbackMetrics?: string[],
      initialEpoch = 0, stepsPerEpoch?: number,
      validationSteps?: number): Promise<History> {
    if (batchSize == null) {
      batchSize = 32;
    }
    if (epochs == null) {
      epochs = 100;
    }
    if (shuffle == null) {
      shuffle = true;
    }
    if (initialEpoch == null) {
      initialEpoch = 0;
    }

    // TODO(cais): Change const to let below when implementing validation.
    let doValidation = false;
    if (valF != null && valIns != null) {
      doValidation = true;
      // TODO(cais): verbose message.
    }
    if (validationSteps != null) {
      doValidation = true;
      if (stepsPerEpoch == null) {
        throw new ValueError(
            'Can only use `validationSteps` when doing step-wise training, ' +
            'i.e., `stepsPerEpoch` must be set.');
      }
    }

    const numTrainSamples =
        this.checkNumSamples(ins, batchSize, stepsPerEpoch, 'steps_per_epoch');
    let indexArray: number[];
    if (numTrainSamples != null) {
      indexArray = _.range(numTrainSamples);
    }

    this.history = new History();
    if (callbacks == null) {
      callbacks = [new BaseLogger()];
    } else {
      callbacks = [new BaseLogger() as Callback].concat(callbacks);
    }
    callbacks = callbacks.concat([this.history]);

    if (verbose > 0) {
      throw new NotImplementedError('Verbose mode is not implemented yet.');
    }
    const callbackList = new CallbackList(callbacks);

    // TODO(cais): Figure out when this Model instance can have a dynamically
    //   set property called 'callback_model' as in PyKeras.
    callbackList.setModel(this);
    callbackList.setParams({
      epochs,
      steps: stepsPerEpoch,
      verbose,
      doValidation,
      metrics: callbackMetrics,
    });
    // TODO(cais): Take care of the PyKeras logic of stop_training.
    await callbackList.onTrainBegin();
    // TODO(cais): Take care of callbacks.validation_data as in PyKeras.

    // TODO(cais): Pre-convert feeds for performance as in PyKeras.

    for (let epoch = initialEpoch; epoch < epochs; ++epoch) {
      await callbackList.onEpochBegin(epoch);
      const epochLogs: Logs = {};
      let epochIndexArray = indexArray;
      if (stepsPerEpoch != null) {
        throw new NotImplementedError(
            'stepsPerEpoch mode is not implemented yet.');
      } else {
        if (shuffle === 'batch') {
          throw new NotImplementedError(
              'batch shuffling is not implemneted yet');
        } else if (shuffle) {
          epochIndexArray = _.shuffle(indexArray);
        }
        // Convert the potentially shuffled indices to Tensor1D, to avoid the
        // cost of repeated creation of Array1Ds later on.
        const epochIndexArray1D = tensor1d(epochIndexArray);

        const batches = makeBatches(numTrainSamples, batchSize);
        for (let batchIndex = 0; batchIndex < batches.length; ++batchIndex) {
          // TODO(cais): dl.tidy() should not be leaked from the backend.
          //   Wrap it with a backend function called mathScope.
          const batchLogs: Logs = {};
          await callbackList.onBatchBegin(batchIndex, batchLogs);
          dl.tidy(() => {
            const batchStart = batches[batchIndex][0];
            const batchEnd = batches[batchIndex][1];
            const batchIds = K.sliceAlongFirstAxis(
                                 epochIndexArray1D, batchStart,
                                 batchEnd - batchStart) as Tensor1D;
            batchLogs['batch'] = batchIndex;
            batchLogs['size'] = batchEnd - batchStart;

            // TODO(cais): In ins, train flag can be a number, instead of an
            //   Tensor? Do we need to handle this in tfjs-layers?
            const insBatch = sliceArraysByIndices(ins, batchIds) as Tensor[];
            const outs = f(insBatch);
            for (let i = 0; i < outLabels.length; ++i) {
              const label = outLabels[i];
              const out = outs[i];
              batchLogs[label] = out;
              K.keep(out);
              // TODO(cais): Use scope() to avoid ownership.
            }

            // TODO(cais): Logic for early stopping using
            //   callback_model.stop_training as in PyKeras.

            if (batchIndex === batches.length - 1) {  // Last batch.
              if (doValidation) {
                const valOuts = this.testLoop(valF, valIns, batchSize);
                // Porting Notes: In tfjs-layers, valOuts is always an Array.
                for (let i = 0; i < outLabels.length; ++i) {
                  const label = outLabels[i];
                  const out = valOuts[i];
                  K.keep(out);
                  // TODO(cais): Use scope() to avoid ownership.
                  epochLogs['val_' + label] = out;
                }
              }
            }
            return outs;
          });
          await callbackList.onBatchEnd(batchIndex, batchLogs);
          // TODO(cais): return outs as list of Tensor.
        }
      }
      // TODO(cais): Run validation at the end of the epoch.
      await callbackList.onEpochEnd(epoch, epochLogs);
      // TODO(cais): Logic for early stopping using
      //   callback_model.stop_training as in PyKeras.
    }
    await callbackList.onTrainEnd();

    await this.history.syncData();
    return this.history;
  }

  /**
   * Loop over some test data in batches.
   * @param f A Function returning a list of tensors.
   * @param ins Array of tensors to be fed to `f`.
   * @param batchSize Integer batch size or `null` / `undefined`.
   * @param verbose verbosity mode.
   * @param steps Total number of steps (batches of samples) before declaring
   *   test finished. Ignored with the default value of `null` / `undefined`.
   * @returns Array of Scalars.
   */
  private testLoop(
      f: (data: Tensor[]) => Scalar[], ins: Tensor[], batchSize?: number,
      verbose = 0, steps?: number): Scalar[] {
    const numSamples = this.checkNumSamples(ins, batchSize, steps, 'steps');
    const outs: Scalar[] = [];
    if (verbose === 1) {
      throw new NotImplementedError('Verbose mode is not implemented yet.');
    }
    // TODO(cais): Use `indicesForConversionToDense' to prevent slow down.
    if (steps != null) {
      throw new NotImplementedError(
          'steps mode in testLoop() is not implemented yet');
    } else {
      const batches = makeBatches(numSamples, batchSize);
      const indexArray = tensor1d(_.range(numSamples));
      for (let batchIndex = 0; batchIndex < batches.length; ++batchIndex) {
        const batchStart = batches[batchIndex][0];
        const batchEnd = batches[batchIndex][1];
        const batchIds =
            K.sliceAlongFirstAxis(
                indexArray, batchStart, batchEnd - batchStart) as Tensor1D;
        // TODO(cais): In ins, train flag can be a number, instead of an
        //   Tensor? Do we need to handle this in tfjs-layers?
        const insBatch = sliceArraysByIndices(ins, batchIds) as Scalar[];
        const batchOuts = f(insBatch);
        if (batchIndex === 0) {
          for (let i = 0; i < batchOuts.length; ++i) {
            outs.push(K.getScalar(0));
          }
        }
        for (let i = 0; i < batchOuts.length; ++i) {
          const batchOut = batchOuts[i];
          outs[i] =
              K.add(
                  outs[i],
                  K.scalarTimesArray(
                      K.getScalar(batchEnd - batchStart), batchOut)) as Scalar;
        }
      }
      for (let i = 0; i < outs.length; ++i) {
        outs[i] = K.divide(outs[i], K.getScalar(numSamples)) as Scalar;
      }
    }
    return outs;
  }

  private getDedupedMetricsNames(): string[] {
    const outLabels = this.metricsNames;
    // Rename duplicated metrics names (can happen with an output layer shared
    // among multiple dataflows).
    const dedupedOutLabels = [];
    for (let i = 0; i < outLabels.length; ++i) {
      const label = outLabels[i];
      let newLabel = label;
      if (count(outLabels, label) > 1) {
        const dupIndex = count(outLabels.slice(0, i), label);
        newLabel += `_${dupIndex}`;
      }
      dedupedOutLabels.push(newLabel);
    }
    return dedupedOutLabels;
  }

  private makeTestFunction() {
    this.testFunction = (data: Tensor[]) => {
      return dl.tidy(() => {
        const valOutputs: Scalar[] = [];
        let totalLoss: Scalar;
        const inputs = data.slice(0, this.inputs.length);
        const targets = data.slice(
            this.inputs.length, this.inputs.length + this.outputs.length);
        const feeds = [];
        for (let i = 0; i < this.inputs.length; ++i) {
          feeds.push({key: this.inputs[i], value: inputs[i]});
        }
        const feedDict = new FeedDict(feeds);
        const outputs = execute(this.outputs, feedDict) as Tensor[];
        // Compute total loss.
        for (let i = 0; i < this.lossFunctions.length; ++i) {
          const lossFunction = this.lossFunctions[i];
          // TODO(cais): Add sample weighting and replace the simple averaging.
          const loss = K.mean(lossFunction(targets[i], outputs[i])) as Scalar;
          if (i === 0) {
            totalLoss = loss;
          } else {
            totalLoss = K.add(totalLoss, loss) as Scalar;
          }
          valOutputs.push(totalLoss);
        }
        // Compute the metrics.
        for (let i = 0; i < this.metricsTensors.length; ++i) {
          const metric = this.metricsTensors[i][0];  // TODO(cais): Restore.
          const outputIndex = this.metricsTensors[i][1];
          // TODO(cais): Replace K.mean() with a proper weighting function.
          const meanMetric =
              K.mean(metric(targets[outputIndex], outputs[outputIndex]));
          valOutputs.push(meanMetric as Scalar);
        }
        return valOutputs;
      });
    };
  }

  /**
   * Trains the model for a fixed number of epochs (iterations on a dataset).
   *
   * ```js
   * const model = tf.sequential({
   *   layers: [tf.layers.dense({units: 1, inputShape: [10]})]
   * });
   * model.compile({optimizer: 'sgd', loss: 'meanSquaredError'});
   * const history = await model.fit(tf.ones([8, 10]), tf.ones([8, 1]), {
   *   batchSize: 4,
   *   epochs: 3
   * });
   *
   * @param x `Tensor` of training data, or an array of `Tensor`s if the model
   *   has multiple inputs. If all inputs in the model are named, you can also
   *   pass a dictionary mapping input names to `Tensor`s.
   * @param y `Tensor` of target (label) data, or an array of `Tensor`s if the
   *   model has multiple outputs. If all outputs in the model are named, you
   *  can also pass a dictionary mapping output names to `Tensor`s.
   * @param config A `ModelFitConfig`, containing optional fields.
   *
   * @return A `History` instance. Its `history` attribute contains all
   *   information collected during training.
   *
   * @exception ValueError In case of mismatch between the provided input data
   *   and what the model expects.
   */
  @doc({heading: 'Models', subheading: 'Classes', configParamIndices: [2]})
  async fit(
      x: Tensor|Tensor[]|{[inputName: string]: Tensor},
      y: Tensor|Tensor[]|{[inputName: string]: Tensor},
      config: ModelFitConfig = {}): Promise<History> {
    const batchSize = config.batchSize == null ? 32 : config.batchSize;

    // Validate user data.
    // TOOD(cais): Add sampleWeight and  classWeight.
    const standardizedOuts = this.standardizeUserData(x, y, false, batchSize);
    let inputs = standardizedOuts[0];
    let targets = standardizedOuts[1];
    // TODO(cais): Make use of sampleWeights in standardizedOuts[2] when
    //   available.

    // Prepare validation data.
    let doValidation = false;
    let valX: Tensor|Tensor[];
    let valY: Tensor|Tensor[];
    let valIns: Tensor[];
    if (config.validationData != null && config.validationData.length > 0) {
      doValidation = true;
      if (config.validationData.length === 2) {
        // config.validationData consists of valX and valY.
        valX = config.validationData[0];
        valY = config.validationData[1];
      } else if (config.validationData.length === 3) {
        throw new NotImplementedError(
            'validationData including sample weights is not supported yet.');
      } else {
        throw new ValueError(
            `When passing validation data, it must contain 2 (valX, valY) ` +
            `or 3 (valX, valY, valSampleWeight) items, however it contains ` +
            `${config.validationData.length} items`);
      }

      const valStandardized =
          this.standardizeUserData(valX, valY, true, batchSize);
      valX = valStandardized[0] as Tensor[];
      valY = valStandardized[1] as Tensor[];
      // TODO(cais): Use validation sample weights in valStandardized[2] once
      //   it becomes available.
      valIns = valX.concat(valY);
      // TODO(cais): Add useLearningPhase data properly.
    } else if (
        config.validationSplit != null && config.validationSplit > 0 &&
        config.validationSplit < 1) {
      doValidation = true;
      // Porting Note: In tfjs-layers, inputs[0] is always an Tensor.
      const splitAt =
          Math.floor(inputs[0].shape[0] * (1 - config.validationSplit));
      const originalBatchSize = inputs[0].shape[0];
      valX = sliceArrays(inputs, splitAt, originalBatchSize) as Tensor[];
      inputs = sliceArrays(inputs, 0, splitAt) as Tensor[];
      valY = sliceArrays(targets, splitAt, originalBatchSize) as Tensor[];
      targets = sliceArrays(targets, 0, splitAt) as Tensor[];
      // TODO(cais): Once sampleWeights becomes available, slice it to get
      //   valSampleWeights.
      valIns = valX.concat(valY);
      // TODO(cais): Add useLearningPhase data properly.
    } else if (config.validationSteps != null) {
      doValidation = true;
      // TODO(cais): Add useLearningPhase.
    }

    const ins = inputs.concat(targets);

    this.checkTrainableWeightsConsistency();

    // TODO(cais): Handle use_learning_phase and learning_phase?

    // Porting Note: Here we see a key deviation of tfjs-layers from Keras.
    //  Due to the imperative nature of tfjs-layers' backend (deeplearn.js),
    //  we do not construct symbolic computation graphs to embody the training
    //  process. Instead, we define a function that performs the training
    //  action.
    //  In PyKeras, the data (inputs and targets) are fed through graph
    //  placeholders. In tfjs-layers, the data are fed as function arguments.
    //  Since the function are defined below in the scope, we don't have
    //  equivalents of PyKeras's `_make_train_funciton`.

    // Creat a function that performs the following actions:
    //   1) computes the losses,
    //   2) add them to get the total loss,
    //   3) call the optimizer computes the gradients of the Model's trainable
    //      weights w.r.t. the total loss and update the variables.
    //   4) calculate the metrics
    //   5) return the values of the losses and metrics.
    const trainFunction = (data: Tensor[]) => {
      const losses: Tensor[] = [];
      const lossValues: Scalar[] = [];

      const inputs = data.slice(0, this.inputs.length);
      const targets = data.slice(
          this.inputs.length, this.inputs.length + this.outputs.length);

      const metricsValues: Scalar[] = [];

      // Create a function that computes the total loss based on the inputs.
      // This function is used for obtaining gradients through backprop.
      const totalLossFunction = () => {
        const feeds = [];
        for (let i = 0; i < this.inputs.length; ++i) {
          feeds.push({key: this.inputs[i], value: inputs[i]});
        }
        const feedDict = new FeedDict(feeds);
        const outputs =
            execute(this.outputs, feedDict, {'training': true}) as Tensor[];
        // TODO(cais): Take care of the case of multiple outputs from a
        //   single layer?

        let totalLoss: Tensor;
        for (let i = 0; i < this.lossFunctions.length; ++i) {
          const lossFunction = this.lossFunctions[i];
          const loss = lossFunction(targets[i], outputs[i]);
          losses.push(loss);
          // TODO(cais): push Scalar instead.
          const meanLoss = K.mean(loss) as Scalar;
          K.keep(meanLoss);
          // TODO(cais): Use a scope() instead, to avoid ownership.
          lossValues.push(meanLoss);
          if (i === 0) {
            totalLoss = loss;
          } else {
            totalLoss = K.add(totalLoss, loss);
          }
        }

        // Compute the metrics.
        // TODO(cais): These should probably be calculated outside
        //   totalLossFunction to benefit speed?
        for (let i = 0; i < this.metricsTensors.length; ++i) {
          const metric = this.metricsTensors[i][0];
          const outputIndex = this.metricsTensors[i][1];
          // TODO(cais): Replace K.mean() with a proper weighting function.
          const meanMetric =
              K.mean(metric(targets[outputIndex], outputs[outputIndex])) as
              Scalar;
          K.keep(meanMetric);
          // TODO(cais): Use a scope() instead, to avoid ownership.
          metricsValues.push(meanMetric);
        }

        totalLoss = K.mean(totalLoss);
        return totalLoss as Scalar;
      };

      this.optimizer.updateVariables(
          totalLossFunction, this.collectedTrainableWeights);

      // TODO(cais): Append the metrics when they are available.
      // Calculate the total loss value.
      let totalLossValue = lossValues[0];
      for (let i = 1; i < lossValues.length; ++i) {
        totalLossValue =
            K.scalarPlusArray(totalLossValue, lossValues[i]) as Scalar;
      }

      return [totalLossValue].concat(metricsValues);
    };

    const outLabels = this.getDedupedMetricsNames();

    let valFunction: (data: Tensor[]) => Scalar[];
    let callbackMetrics: string[];
    if (doValidation) {
      this.makeTestFunction();
      valFunction = this.testFunction;
      callbackMetrics =
          outLabels.slice().concat(outLabels.map(n => 'val_' + n));
    } else {
      valFunction = null;
      valIns = [];
      callbackMetrics = outLabels.slice();
    }

    const callbacks = standardizeCallbacks(config.callbacks);
    return this.fitLoop(
        trainFunction, ins, outLabels, batchSize, config.epochs, config.verbose,
        callbacks, valFunction, valIns, config.shuffle, callbackMetrics, null,
        null, null);
    // TOOD(cais): Add value to outLabels.
    // TODO(cais): Add initialEpoch.
  }
}

ClassNameMap.register('Model', Model);
