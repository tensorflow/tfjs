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

import * as tfc from '@tensorflow/tfjs-core';
import {io, ModelPredictConfig as ModelPredictArgs, NamedTensorMap, Optimizer, Scalar, scalar, serialization, Tensor, Tensor1D, tensor1d, util} from '@tensorflow/tfjs-core';
import * as K from '../backend/tfjs_backend';
import {History, ModelLoggingVerbosity} from '../base_callbacks';
import {nameScope} from '../common';
import {NotImplementedError, RuntimeError, ValueError} from '../errors';
import {Shape} from '../keras_format/common';
import * as losses from '../losses';
import * as Metrics from '../metrics';
import * as optimizers from '../optimizers';
import {LossOrMetricFn} from '../types';
import {count, pyListRepeat, singletonOrArray, unique} from '../utils/generic_utils';
import {printSummary} from '../utils/layer_utils';
import {range} from '../utils/math_utils';
import {LayerVariable} from '../variables';
import {version} from '../version';
import {Container, ContainerArgs} from './container';
import {Dataset} from './dataset_stub';
import {execute, FeedDict} from './executor';
import {DisposeResult, SymbolicTensor} from './topology';
import {evaluateDataset, fitDataset, ModelEvaluateDatasetArgs, ModelFitDatasetArgs} from './training_dataset';
import {checkBatchSize, disposeNewTensors, ensureTensorsRank2OrHigher, fitTensors, makeBatches, ModelFitArgs, sliceArrays, sliceArraysByIndices} from './training_tensors';

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

  arrays = ensureTensorsRank2OrHigher(arrays);

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
 * @param inputs `Array` of `tf.Tensor`s for inputs.
 * @param targets `Array` of `tf.Tensor`s for targets.
 * @param weights Optional `Array` of `tf.Tensor`s for sample weights.
 * @throws ValueError: in case of incorrectly formatted data.
 */
export function checkArrayLengths(
    inputs: Tensor[], targets: Tensor[], weights?: Tensor[]) {
  const setX = unique(inputs.map(input => input.shape[0]));
  setX.sort();
  const setY = unique(targets.map(target => target.shape[0]));
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
  if (setX.length > 0 && setY.length > 0 && !util.arraysEqual(setX, setY)) {
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
 * @param targets `Array` of `tf.Tensor`s of targets.
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
    if (keyLosses.indexOf(loss) !== -1) {
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
 * that the data is either a single `tf.Tensor` or an Array of `tf.Tensor`s. We
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

export interface ModelEvaluateArgs {
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
 * Configuration for calls to `LayersModel.compile()`.
 */
export interface ModelCompileArgs {
  /**
   * An instance of `tf.train.Optimizer` or a string name for an Optimizer.
   */
  optimizer: string|Optimizer;

  /**
   * Object function(s) or name(s) of object function(s).
   * If the model has multiple outputs, you can use a different loss
   * on each output by passing a dictionary or an Array of losses.
   * The loss value that will be minimized by the model will then be the sum
   * of all individual losses.
   */
  loss: string|string[]|{[outputName: string]: string}|LossOrMetricFn|
      LossOrMetricFn[]|{[outputName: string]: LossOrMetricFn};

  /**
   * List of metrics to be evaluated by the model during training and testing.
   * Typically you will use `metrics=['accuracy']`.
   * To specify different metrics for different outputs of a multi-output
   * model, you could also pass a dictionary.
   */
  metrics?: string[]|{[outputName: string]: string};

  // TODO(cais): Add lossWeights, sampleWeightMode, weightedMetrics, and
  //   targetTensors.
}

const LAYERS_MODEL_FORMAT_NAME = 'layers-model';

/**
 * A `tf.LayersModel` is a directed, acyclic graph of `tf.Layer`s plus methods
 * for training, evaluation, prediction and saving.
 *
 * `tf.LayersModel` is the basic unit of training, inference and evaluation in
 * TensorFlow.js. To create a `tf.LayersModel`, use `tf.LayersModel`.
 *
 * See also:
 *   `tf.Sequential`, `tf.loadLayersModel`.
 */
/** @doc {heading: 'Models', subheading: 'Classes'} */
export class LayersModel extends Container implements tfc.InferenceModel {
  // The class name is 'Model' rather than 'LayersModel' for backwards
  // compatibility since this class name shows up in the serialization format.
  /** @nocollapse */
  static className = 'Model';
  protected optimizer_: Optimizer;
  // Whether the model instance owns the optimizer: `true` if and only if
  // `optimizer` is created from a string parameter during `compile()` call.
  protected isOptimizerOwned: boolean;

  loss: string|string[]|{[outputName: string]: string}|LossOrMetricFn|
      LossOrMetricFn[]|{[outputName: string]: LossOrMetricFn};
  lossFunctions: LossOrMetricFn[];

  // TODO(cais): These private variables should probably not have the string
  //   'feed' in their names, because we are not dealing with a symbolic
  //   backend.
  private feedOutputShapes: Shape[];
  private feedLossFns: LossOrMetricFn[];
  private collectedTrainableWeights: LayerVariable[];
  private testFunction: (data: Tensor[]) => Scalar[];
  history: History;

  // A public property that can be set by Callbacks to order early stopping
  // during `fit()` calls.
  protected stopTraining_: boolean;
  protected isTraining: boolean;

  metrics: string[]|{[outputName: string]: string};
  metricsNames: string[];
  // Porting Note: `metrics_tensors` in PyKeras is a symbolic tensor. But given
  //   the imperative nature of tfjs-core, `metricsTensors` is a
  //   TypeScript function here.
  //   Also note that due to the imperative nature of tfjs-core, `metricsTensor`
  //   here needs an output index to keep track of which output of the
  //   LayersModel a metric belongs to. This is unlike `metrics_tensors` in
  //   PyKeras, which is a `list` of symbolic tensors, each of which has
  //   implicit "knowledge" of the outputs it depends on.
  metricsTensors: Array<[LossOrMetricFn, number]>;

  constructor(args: ContainerArgs) {
    super(args);
    this.isTraining = false;
  }

  /**
   * Print a text summary of the model's layers.
   *
   * The summary includes
   * - Name and type of all layers that comprise the model.
   * - Output shape(s) of the layers
   * - Number of weight parameters of each layer
   * - If the model has non-sequential-like topology, the inputs each layer
   *   receives
   * - The total number of trainable and non-trainable parameters of the model.
   *
   * ```js
   * const input1 = tf.input({shape: [10]});
   * const input2 = tf.input({shape: [20]});
   * const dense1 = tf.layers.dense({units: 4}).apply(input1);
   * const dense2 = tf.layers.dense({units: 8}).apply(input2);
   * const concat = tf.layers.concatenate().apply([dense1, dense2]);
   * const output =
   *     tf.layers.dense({units: 3, activation: 'softmax'}).apply(concat);
   *
   * const model = tf.model({inputs: [input1, input2], outputs: output});
   * model.summary();
   * ```
   *
   * @param lineLength Custom line length, in number of characters.
   * @param positions Custom widths of each of the columns, as either
   *   fractions of `lineLength` (e.g., `[0.5, 0.75, 1]`) or absolute number
   *   of characters (e.g., `[30, 50, 65]`). Each number corresponds to
   *   right-most (i.e., ending) position of a column.
   * @param printFn Custom print function. Can be used to replace the default
   *   `console.log`. For example, you can use `x => {}` to mute the printed
   *   messages in the console.
   */
  /** @doc {heading: 'Models', subheading: 'Classes'} */
  summary(
      lineLength?: number, positions?: number[],
      printFn:
          // tslint:disable-next-line:no-any
      (message?: any, ...optionalParams: any[]) => void = console.log) {
    if (!this.built) {
      throw new ValueError(
          `This model has never been called, thus its weights have not been ` +
          `created yet. So no summary can be displayed. Build the model ` +
          `first (e.g., by calling it on some test data).`);
    }
    printSummary(this, lineLength, positions, printFn);
  }

  /**
   * Configures and prepares the model for training and evaluation.  Compiling
   * outfits the model with an optimizer, loss, and/or metrics.  Calling `fit`
   * or `evaluate` on an un-compiled model will throw an error.
   *
   * @param args a `ModelCompileArgs` specifying the loss, optimizer, and
   * metrics to be used for fitting and evaluating this model.
   */
  /**
   * @doc {heading: 'Models', subheading: 'Classes'}
   */
  compile(args: ModelCompileArgs): void {
    if (args.loss == null) {
      args.loss = [];
    }
    this.loss = args.loss;

    if (typeof args.optimizer === 'string') {
      this.optimizer_ = optimizers.getOptimizer(args.optimizer);
      this.isOptimizerOwned = true;
    } else {
      if (!(args.optimizer instanceof Optimizer)) {
        throw new ValueError(
            `User-defined optimizer must be an instance of tf.Optimizer.`);
      }
      this.optimizer_ = args.optimizer;
      this.isOptimizerOwned = false;
    }

    // TODO(cais): Add lossWeights.
    // TODO(cais): Add sampleWeightMode.

    // Prepare loss functions.
    let lossFunctions: LossOrMetricFn[] = [];
    if (!Array.isArray(args.loss) && typeof args.loss !== 'string' &&
        typeof args.loss !== 'function') {
      args.loss = args.loss as {[outputName: string]: string};
      for (const name in args.loss) {
        if (this.outputNames.indexOf(name) === -1) {
          throw new ValueError(
              `Unknown entry in loss dictionary: "${name}". ` +
              `Only expected the following keys: ${this.outputNames}`);
        }
      }
      for (const name of this.outputNames) {
        if (args.loss[name] == null) {
          console.warn(
              `Output "${name}" is missing from loss dictionary. We assume ` +
              `this was done on purpose, and we will not be expecting data ` +
              `to be passed to ${name} during training`);
        }
        lossFunctions.push(losses.get(args.loss[name]));
      }
    } else if (Array.isArray(args.loss)) {
      if (args.loss.length !== this.outputs.length) {
        throw new ValueError(
            `When passing an Array as loss, it should have one entry per ` +
            `model output. The model has ${this.outputs.length} output(s), ` +
            `but you passed loss=${args.loss}.`);
      }
      const theLosses = args.loss as Array<string|LossOrMetricFn>;
      lossFunctions = theLosses.map(l => losses.get(l));
    } else {
      const lossFunction = losses.get(args.loss);
      this.outputs.forEach(_ => {
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
    // TODO(cais): Add logic for sample weights.
    const skipTargetIndices: number[] = [];

    // Prepare metrics.
    this.metrics = args.metrics;
    // TODO(cais): Add weightedMetrics.
    this.metricsNames = ['loss'];
    this.metricsTensors = [];

    // Compute total loss.
    // Porting Note: In PyKeras, metrics_tensors are symbolic tensor objects.
    //   Here, metricsTensors are TypeScript functions. This difference is due
    //   to the difference in symbolic/imperative property of the backends.
    nameScope('loss', () => {
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

      // Porting Note: Due to the imperative nature of the backend, we calculate
      //   the regularizer penalties in the totalLossFunction, instead of here.
    });

    const nestedMetrics = collectMetrics(args.metrics, this.outputNames);
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

    nameScope('metric', () => {
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
          //  TODO(cais): Use 'weights_' for weighted metrics.

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
            nameScope(metricName, () => {
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
  protected checkTrainableWeightsConsistency(): void {
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
   * const result = model.evaluate(
   *     tf.ones([8, 10]), tf.ones([8, 1]), {batchSize: 4});
   * result.print();
   * ```
   *
   * @param x `tf.Tensor` of test data, or an `Array` of `tf.Tensor`s if the
   * model has multiple inputs.
   * @param y `tf.Tensor` of target data, or an `Array` of `tf.Tensor`s if the
   * model has multiple outputs.
   * @param args A `ModelEvaluateArgs`, containing optional fields.
   *
   * @return `Scalar` test loss (if the model has a single output and no
   *   metrics) or `Array` of `Scalar`s (if the model has multiple outputs
   *   and/or metrics). The attribute `model.metricsNames`
   *   will give you the display labels for the scalar outputs.
   */
  /**
   * @doc {heading: 'Models', subheading: 'Classes'}
   */
  evaluate(
      x: Tensor|Tensor[], y: Tensor|Tensor[],
      args: ModelEvaluateArgs = {}): Scalar|Scalar[] {
    const batchSize = args.batchSize == null ? 32 : args.batchSize;
    checkBatchSize(batchSize);

    // TODO(cais): Standardize `config.sampleWeights` as well.
    // Validate user data.
    const standardizedOuts = this.standardizeUserData(x, y, true, batchSize);
    try {
      // TODO(cais): If uses `useLearningPhase`, set the corresponding element
      // of the input to 0.
      const ins = standardizedOuts[0].concat(standardizedOuts[1]);
      this.makeTestFunction();
      const f = this.testFunction;
      const testOuts =
          this.testLoop(f, ins, batchSize, args.verbose, args.steps);
      return singletonOrArray(testOuts);
    } finally {
      disposeNewTensors(standardizedOuts[0], x);
      disposeNewTensors(standardizedOuts[1], y);
    }
  }

  // TODO(cais): Add code snippet below once real dataset objects are
  //   available.
  /**
   * Evaluate model using a dataset object.
   *
   * Note: Unlike `evaluate()`, this method is asynchronous (`async`);
   *
   * @param dataset A dataset object. Its `iterator()` method is expected
   *   to generate a dataset iterator object, the `next()` method of which
   *   is expected to produce data batches for evaluation. The return value
   *   of the `next()` call ought to contain a boolean `done` field and a
   *   `value` field. The `value` field is expected to be an array of two
   *   `tf.Tensor`s or an array of two nested `tf.Tensor` structures. The former
   *   case is for models with exactly one input and one output (e.g..
   *   a sequential model). The latter case is for models with multiple
   *   inputs and/or multiple outputs. Of the two items in the array, the
   *   first is the input feature(s) and the second is the output target(s).
   * @param args A configuration object for the dataset-based evaluation.
   * @returns Loss and metric values as an Array of `Scalar` objects.
   */
  /**
   * @doc {heading: 'Models', subheading: 'Classes'}
   */
  async evaluateDataset(dataset: Dataset<{}>, args?: ModelEvaluateDatasetArgs):
      Promise<Scalar|Scalar[]> {
    this.makeTestFunction();
    return evaluateDataset(this, dataset, args);
  }

  /**
   * Get number of samples provided for training, evaluation or prediction.
   *
   * @param ins Input `tf.Tensor`.
   * @param batchSize Integer batch size, optional.
   * @param steps Total number of steps (batches of samples) before
   * declaring loop finished. Optional.
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
   * Execute internal tensors of the model with input data feed.
   * @param inputs Input data feed. Must match the inputs of the model.
   * @param outputs Names of the output tensors to be fetched. Must match
   *   names of the SymbolicTensors that belong to the graph.
   * @returns Fetched values for `outputs`.
   */
  execute(inputs: Tensor|Tensor[]|NamedTensorMap, outputs: string|string[]):
      Tensor|Tensor[] {
    if (Array.isArray(outputs) && outputs.length === 0) {
      throw new ValueError(
          '`outputs` is an empty Array, which is not allowed.');
    }

    const outputsIsArray = Array.isArray(outputs);
    const outputNames = (outputsIsArray ? outputs as string[] :
                                          [outputs as string]) as string[];
    const outputSymbolicTensors = this.retrieveSymbolicTensors(outputNames);

    // Format the input into a FeedDict.
    const feedDict = new FeedDict();
    if (inputs instanceof Tensor) {
      inputs = [inputs as Tensor];
    }
    if (Array.isArray(inputs)) {
      if ((inputs as Tensor[]).length !== this.inputs.length) {
        throw new ValueError(
            `The number of inputs provided (${(inputs as Tensor[]).length}) ` +
            `does not match the number of inputs of this model ` +
            `(${this.inputs.length}).`);
      }
      for (let i = 0; i < this.inputs.length; ++i) {
        feedDict.add(this.inputs[i], (inputs as Tensor[])[i]);
      }
    } else {
      for (const input of this.inputs) {
        const tensorValue = (inputs as NamedTensorMap)[input.name];
        if (tensorValue == null) {
          throw new ValueError(
              `No value is provided for the model's input ${input.name}`);
        }
        feedDict.add(input, tensorValue);
      }
    }

    // Run execution.
    const executeOutputs = execute(outputSymbolicTensors, feedDict) as Tensor[];
    return outputsIsArray ? executeOutputs : executeOutputs[0];
  }

  /**
   * Retrieve the model's internal symbolic tensors from symbolic-tensor names.
   */
  private retrieveSymbolicTensors(symbolicTensorNames: string[]):
      SymbolicTensor[] {
    const outputSymbolicTensors: SymbolicTensor[] =
        pyListRepeat(null, symbolicTensorNames.length);
    let outputsRemaining = symbolicTensorNames.length;
    for (const layer of this.layers) {
      const layerOutputs: SymbolicTensor[] = Array.isArray(layer.output) ?
          layer.output as SymbolicTensor[] :
          [layer.output as SymbolicTensor];
      const layerOutputNames = layerOutputs.map(output => output.name);
      for (let i = 0; i < symbolicTensorNames.length; ++i) {
        const index = layerOutputNames.indexOf(symbolicTensorNames[i]);
        if (index !== -1) {
          outputSymbolicTensors[i] = layerOutputs[index];
          outputsRemaining--;
        }
        if (outputsRemaining === 0) {
          break;
        }
      }
      if (outputsRemaining === 0) {
        break;
      }
    }

    if (outputsRemaining > 0) {
      const remainingNames: string[] = [];
      outputSymbolicTensors.forEach((tensor, i) => {
        if (tensor == null) {
          remainingNames.push(symbolicTensorNames[i]);
        }
      });
      throw new ValueError(
          `Cannot find SymbolicTensors for output name(s): ` +
          `${JSON.stringify(remainingNames)}`);
    }
    return outputSymbolicTensors;
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
   * @returns: Predictions as `tf.Tensor` (if a single output) or an `Array` of
   *   `tf.Tensor` (if multipe outputs).
   */
  private predictLoop(ins: Tensor|Tensor[], batchSize = 32, verbose = false):
      Tensor|Tensor[] {
    return tfc.tidy(() => {
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
      const outsBatches: Tensor[][] = this.outputs.map(output => []);

      // TODO(cais): Can the scope() be pushed down inside the for loop?
      for (let batchIndex = 0; batchIndex < batches.length; ++batchIndex) {
        const batchOuts = tfc.tidy(() => {
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
        batchOuts.forEach((batchOut, i) => outsBatches[i].push(batchOut));
      }
      return singletonOrArray(
          outsBatches.map(batches => tfc.concat(batches, 0)));
    });
  }

  /**
   * Generates output predictions for the input samples.
   *
   * Computation is done in batches.
   *
   * Note: the "step" mode of predict() is currently not supported.
   *   This is because the TensorFlow.js core backend is imperative only.
   *
   * ```js
   * const model = tf.sequential({
   *   layers: [tf.layers.dense({units: 1, inputShape: [10]})]
   * });
   * model.predict(tf.ones([8, 10]), {batchSize: 4}).print();
   * ```
   *
   * @param x The input data, as an Tensor, or an `Array` of `tf.Tensor`s if
   *   the model has multiple inputs.
   * @param args A `ModelPredictArgs` object containing optional fields.
   *
   * @return Prediction results as a `tf.Tensor`(s).
   *
   * @exception ValueError In case of mismatch between the provided input data
   *   and the model's expectations, or in case a stateful model receives a
   *   number of samples that is not a multiple of the batch size.
   */
  /**
   * @doc {heading: 'Models', subheading: 'Classes'}
   */
  predict(x: Tensor|Tensor[], args: ModelPredictArgs = {}): Tensor|Tensor[] {
    const xsRank2OrHigher = ensureTensorsRank2OrHigher(x);
    checkInputData(
        xsRank2OrHigher, this.inputNames, this.feedInputShapes, false);
    try {
      // TODO(cais): Take care of stateful models.
      //   if (this.stateful) ...
      // TODO(cais): Take care of the learning_phase boolean flag.
      //   if (this.useLearningPhase) ...
      const batchSize = args.batchSize == null ? 32 : args.batchSize;
      checkBatchSize(batchSize);
      return this.predictLoop(xsRank2OrHigher, batchSize);
    } finally {
      disposeNewTensors(xsRank2OrHigher, x);
    }
  }

  /**
   * Returns predictions for a single batch of samples.
   *
   * ```js
   * const model = tf.sequential({
   *   layers: [tf.layers.dense({units: 1, inputShape: [10]})]
   * });
   * model.predictOnBatch(tf.ones([8, 10])).print();
   * ```
   * @param x: Input samples, as an Tensor
   * @return Tensor(s) of predictions
   */
  /** @doc {heading: 'Models', subheading: 'Classes'} */
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
    if (this.optimizer_ == null) {
      throw new RuntimeError(
          'You must compile a model before training/testing. Use ' +
          'LayersModel.compile(modelCompileArgs).');
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
    // TODO(cais): Check sampleWeights as well.
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
   * Loop over some test data in batches.
   * @param f A Function returning a list of tensors.
   * @param ins Array of tensors to be fed to `f`.
   * @param batchSize Integer batch size or `null` / `undefined`.
   * @param verbose verbosity mode.
   * @param steps Total number of steps (batches of samples) before
   * declaring test finished. Ignored with the default value of `null` /
   * `undefined`.
   * @returns Array of Scalars.
   */
  private testLoop(
      f: (data: Tensor[]) => Scalar[], ins: Tensor[], batchSize?: number,
      verbose = 0, steps?: number): Scalar[] {
    return tfc.tidy(() => {
      const numSamples = this.checkNumSamples(ins, batchSize, steps, 'steps');
      const outs: Scalar[] = [];
      if (verbose > 0) {
        throw new NotImplementedError('Verbose mode is not implemented yet.');
      }
      // TODO(cais): Use `indicesForConversionToDense' to prevent slow down.
      if (steps != null) {
        throw new NotImplementedError(
            'steps mode in testLoop() is not implemented yet');
      } else {
        const batches = makeBatches(numSamples, batchSize);
        const indexArray = tensor1d(range(0, numSamples));
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
              outs.push(scalar(0));
            }
          }
          for (let i = 0; i < batchOuts.length; ++i) {
            const batchOut = batchOuts[i];
            outs[i] =
                tfc.add(outs[i], tfc.mul(batchEnd - batchStart, batchOut)) as
                Scalar;
          }
        }
        for (let i = 0; i < outs.length; ++i) {
          outs[i] = tfc.div(outs[i], numSamples) as Scalar;
        }
      }
      return outs;
    });
  }

  protected getDedupedMetricsNames(): string[] {
    const outLabels = this.metricsNames;
    // Rename duplicated metrics names (can happen with an output layer
    // shared among multiple dataflows).
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

  /**
   * Creates a function that performs the following actions:
   *
   * 1. computes the losses
   * 2. sums them to get the total loss
   * 3. call the optimizer computes the gradients of the LayersModel's
   *    trainable weights w.r.t. the total loss and update the variables
   * 4. calculates the metrics
   * 5. returns the values of the losses and metrics.
   */
  protected makeTrainFunction(): (data: Tensor[]) => Scalar[] {
    return (data: Tensor[]) => {
      const losses: Tensor[] = [];
      const lossValues: Scalar[] = [];

      const inputs = data.slice(0, this.inputs.length);
      const targets = data.slice(
          this.inputs.length, this.inputs.length + this.outputs.length);

      const metricsValues: Scalar[] = [];

      // Create a function that computes the total loss based on the
      // inputs. This function is used for obtaining gradients through
      // backprop.
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
          const meanLoss = tfc.mean(loss) as Scalar;
          // TODO(cais): Use a scope() instead, to avoid ownership.
          lossValues.push(meanLoss);
          if (i === 0) {
            totalLoss = loss;
          } else {
            totalLoss = tfc.add(totalLoss, loss);
          }
        }

        // Compute the metrics.
        // TODO(cais): These should probably be calculated outside
        //   totalLossFunction to benefit speed?
        for (let i = 0; i < this.metricsTensors.length; ++i) {
          const metric = this.metricsTensors[i][0];
          const outputIndex = this.metricsTensors[i][1];
          // TODO(cais): Replace K.mean() with a proper weighting
          // function.
          const meanMetric =
              tfc.mean(metric(targets[outputIndex], outputs[outputIndex])) as
              Scalar;
          tfc.keep(meanMetric);
          // TODO(cais): Use a scope() instead, to avoid ownership.
          metricsValues.push(meanMetric);
        }

        totalLoss = tfc.mean(totalLoss);

        // Add regularizer penalties.
        this.calculateLosses().forEach(regularizerLoss => {
          totalLoss = tfc.add(totalLoss, regularizerLoss);
        });

        return totalLoss as Scalar;
      };

      const variables = this.collectedTrainableWeights.map(
          param => param.read() as tfc.Variable);
      const returnCost = true;
      const totalLossValue =
          this.optimizer_.minimize(totalLossFunction, returnCost, variables);

      return [totalLossValue].concat(metricsValues);
    };
  }

  /**
   * Create a function which, when invoked with an array of `tf.Tensor`s as a
   * batch of inputs, returns the prespecified loss and metrics of the model
   * under the batch of input data.
   */
  private makeTestFunction() {
    this.testFunction = (data: Tensor[]) => {
      return tfc.tidy(() => {
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
          // TODO(cais): Add sample weighting and replace the simple
          // averaging.
          const loss = tfc.mean(lossFunction(targets[i], outputs[i])) as Scalar;
          if (i === 0) {
            totalLoss = loss;
          } else {
            totalLoss = tfc.add(totalLoss, loss) as Scalar;
          }
          valOutputs.push(totalLoss);
        }
        // Compute the metrics.
        for (let i = 0; i < this.metricsTensors.length; ++i) {
          const metric = this.metricsTensors[i][0];
          const outputIndex = this.metricsTensors[i][1];
          // TODO(cais): Replace K.mean() with a proper weighting function.
          const meanMetric =
              tfc.mean(metric(targets[outputIndex], outputs[outputIndex]));
          valOutputs.push(meanMetric as Scalar);
        }
        return valOutputs;
      });
    };
  }

  /**
   * Trains the model for a fixed number of epochs (iterations on a
   * dataset).
   *
   * ```js
   * const model = tf.sequential({
   *     layers: [tf.layers.dense({units: 1, inputShape: [10]})]
   * });
   * model.compile({optimizer: 'sgd', loss: 'meanSquaredError'});
   * for (let i = 1; i < 5 ; ++i) {
   *   const h = await model.fit(tf.ones([8, 10]), tf.ones([8, 1]), {
   *       batchSize: 4,
   *       epochs: 3
   *   });
   *   console.log("Loss after Epoch " + i + " : " + h.history.loss[0]);
   * }
   * ```
   *
   * @param x `tf.Tensor` of training data, or an array of `tf.Tensor`s if the
   * model has multiple inputs. If all inputs in the model are named, you
   * can also pass a dictionary mapping input names to `tf.Tensor`s.
   * @param y `tf.Tensor` of target (label) data, or an array of `tf.Tensor`s if
   * the model has multiple outputs. If all outputs in the model are named,
   * you can also pass a dictionary mapping output names to `tf.Tensor`s.
   * @param args A `ModelFitArgs`, containing optional fields.
   *
   * @return A `History` instance. Its `history` attribute contains all
   *   information collected during training.
   *
   * @exception ValueError In case of mismatch between the provided input
   * data and what the model expects.
   */
  /**
   * @doc {heading: 'Models', subheading: 'Classes'}
   */
  async fit(
      x: Tensor|Tensor[]|{[inputName: string]: Tensor},
      y: Tensor|Tensor[]|{[inputName: string]: Tensor},
      args: ModelFitArgs = {}): Promise<History> {
    return fitTensors(this, x, y, args);
  }

  // TODO(cais): Add code snippet below when it's possible to instantiate
  //   actual dataset objects.
  /**
   * Trains the model using a dataset object.
   *
   * @param dataset A dataset object. Its `iterator()` method is expected
   *   to generate a dataset iterator object, the `next()` method of which
   *   is expected to produce data batches for training. The return value
   *   of the `next()` call ought to contain a boolean `done` field and a
   *   `value` field. The `value` field is expected to be an array of two
   *   `tf.Tensor`s or an array of two nested `tf.Tensor` structures. The former
   *   case is for models with exactly one input and one output (e.g..
   *   a sequential model). The latter case is for models with multiple
   *   inputs and/or multiple outputs.
   *   Of the two items in the array, the first is the input feature(s) and
   *   the second is the output target(s).
   * @param args A `ModelFitDatasetArgs`, containing optional fields.
   *
   * @return A `History` instance. Its `history` attribute contains all
   *   information collected during training.
   */
  /**
   * @doc {heading: 'Models', subheading: 'Classes'}
   */
  async fitDataset<T>(dataset: Dataset<T>, args: ModelFitDatasetArgs<T>):
      Promise<History> {
    return fitDataset(this, dataset, args);
  }

  /**
   * Runs a single gradient update on a single batch of data.
   *
   * This method differs from `fit()` and `fitDataset()` in the following
   * regards:
   *   - It operates on exactly one batch of data.
   *   - It returns only the loss and matric values, instead of
   *     returning the batch-by-batch loss and metric values.
   *   - It doesn't support fine-grained options such as verbosity and
   *     callbacks.
   *
   * @param x Input data. It could be one of the following:
   *   - A `tf.Tensor`, or an Array of `tf.Tensor`s (in case the model has
   *     multiple inputs).
   *   - An Object mapping input names to corresponding `tf.Tensor` (if the
   *     model has named inputs).
   * @param y Target darta. It could be either a `tf.Tensor` a multiple
   *   `tf.Tensor`s. It should be consistent with `x`.
   * @returns Training loss or losses (in case the model has
   *   multiple outputs), along with metrics (if any), as numbers.
   */
  /**
   * @doc {heading: 'Models', subheading: 'Classes'}
   */
  async trainOnBatch(
      x: Tensor|Tensor[]|{[inputName: string]: Tensor},
      y: Tensor|Tensor[]|
      {[inputName: string]: Tensor}): Promise<number|number[]> {
    // TODO(cais): Support sampleWeight and classWeight.
    // TODO(cais): Support Dataset objects.
    const standardizeOut = this.standardizeUserData(x, y);
    const inputs = standardizeOut[0];
    const targets = standardizeOut[1];
    const trainFunction = this.makeTrainFunction();
    const losses = trainFunction(inputs.concat(targets));
    const lossValues: number[] = [];
    for (const loss of losses) {
      const v = await loss.data();
      lossValues.push(v[0]);
    }
    tfc.dispose(losses);
    return singletonOrArray(lossValues);
  }

  /**
   * Extract weight values of the model.
   *
   * @param config: An instance of `io.SaveConfig`, which specifies
   * model-saving options such as whether only trainable weights are to be
   * saved.
   * @returns A `NamedTensorMap` mapping original weight names (i.e.,
   *   non-uniqueified weight names) to their values.
   */
  protected getNamedWeights(config?: io.SaveConfig): NamedTensorMap {
    const namedWeights: NamedTensorMap = {};

    const trainableOnly = config != null && config.trainableOnly;
    const weights = trainableOnly ? this.trainableWeights : this.weights;
    const weightValues = this.getWeights(trainableOnly);
    for (let i = 0; i < weights.length; ++i) {
      if (trainableOnly && !weights[i].trainable) {
        // Optionally skip non-trainable weights.
        continue;
      }
      namedWeights[weights[i].originalName] = weightValues[i];
    }
    return namedWeights;
  }

  /**
   * Setter used for force stopping of LayersModel.fit() (i.e., training).
   *
   * Example:
   *
   * ```js
   * const input = tf.input({shape: [10]});
   * const output = tf.layers.dense({units: 1}).apply(input);
   * const model = tf.model({inputs: [input], outputs: [output]});
   * model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});
   * const xs = tf.ones([8, 10]);
   * const ys = tf.zeros([8, 1]);
   *
   * const history = await model.fit(xs, ys, {
   *   epochs: 10,
   *   callbacks: {
   *     onEpochEnd: async (epoch, logs) => {
   *       if (epoch === 2) {
   *         model.stopTraining = true;
   *       }
   *     }
   *   }
   * });
   *
   * // There should be only 3 values in the loss array, instead of 10
   * values,
   * // due to the stopping after 3 epochs.
   * console.log(history.history.loss);
   * ```
   */
  set stopTraining(stop: boolean) {
    this.stopTraining_ = stop;
  }

  get stopTraining(): boolean {
    return this.stopTraining_;
  }

  get optimizer(): Optimizer {
    return this.optimizer_;
  }

  set optimizer(optimizer: Optimizer) {
    if (this.optimizer_ !== optimizer) {
      this.optimizer_ = optimizer;
      this.isOptimizerOwned = false;
    }
  }

  dispose(): DisposeResult {
    const result = super.dispose();
    if (result.refCountAfterDispose === 0 && this.optimizer != null &&
        this.isOptimizerOwned) {
      const numTensorsBeforeOptmizerDisposal = tfc.memory().numTensors;
      this.optimizer_.dispose();
      result.numDisposedVariables +=
          numTensorsBeforeOptmizerDisposal - tfc.memory().numTensors;
    }
    return result;
  }

  /**
   * Save the configuration and/or weights of the LayersModel.
   *
   * An `IOHandler` is an object that has a `save` method of the proper
   * signature defined. The `save` method manages the storing or
   * transmission of serialized data ("artifacts") that represent the
   * model's topology and weights onto or via a specific medium, such as
   * file downloads, local storage, IndexedDB in the web browser and HTTP
   * requests to a server. TensorFlow.js provides `IOHandler`
   * implementations for a number of frequently used saving mediums, such as
   * `tf.io.browserDownloads` and `tf.io.browserLocalStorage`. See `tf.io`
   * for more details.
   *
   * This method also allows you to refer to certain types of `IOHandler`s
   * as URL-like string shortcuts, such as 'localstorage://' and
   * 'indexeddb://'.
   *
   * Example 1: Save `model`'s topology and weights to browser [local
   * storage](https://developer.mozilla.org/en-US/docs/Web/API/Window/localStorage);
   * then load it back.
   *
   * ```js
   * const model = tf.sequential(
   *     {layers: [tf.layers.dense({units: 1, inputShape: [3]})]});
   * console.log('Prediction from original model:');
   * model.predict(tf.ones([1, 3])).print();
   *
   * const saveResults = await model.save('localstorage://my-model-1');
   *
   * const loadedModel = await tf.loadLayersModel('localstorage://my-model-1');
   * console.log('Prediction from loaded model:');
   * loadedModel.predict(tf.ones([1, 3])).print();
   * ```
   *
   * Example 2. Saving `model`'s topology and weights to browser
   * [IndexedDB](https://developer.mozilla.org/en-US/docs/Web/API/IndexedDB_API);
   * then load it back.
   *
   * ```js
   * const model = tf.sequential(
   *     {layers: [tf.layers.dense({units: 1, inputShape: [3]})]});
   * console.log('Prediction from original model:');
   * model.predict(tf.ones([1, 3])).print();
   *
   * const saveResults = await model.save('indexeddb://my-model-1');
   *
   * const loadedModel = await tf.loadLayersModel('indexeddb://my-model-1');
   * console.log('Prediction from loaded model:');
   * loadedModel.predict(tf.ones([1, 3])).print();
   * ```
   *
   * Example 3. Saving `model`'s topology and weights as two files
   * (`my-model-1.json` and `my-model-1.weights.bin`) downloaded from
   * browser.
   *
   * ```js
   * const model = tf.sequential(
   *     {layers: [tf.layers.dense({units: 1, inputShape: [3]})]});
   * const saveResults = await model.save('downloads://my-model-1');
   * ```
   *
   * Example 4. Send  `model`'s topology and weights to an HTTP server.
   * See the documentation of `tf.io.browserHTTPRequest` for more details
   * including specifying request parameters and implementation of the
   * server.
   *
   * ```js
   * const model = tf.sequential(
   *     {layers: [tf.layers.dense({units: 1, inputShape: [3]})]});
   * const saveResults = await model.save('http://my-server/model/upload');
   * ```
   *
   * @param handlerOrURL An instance of `IOHandler` or a URL-like,
   * scheme-based string shortcut for `IOHandler`.
   * @param config Options for saving the model.
   * @returns A `Promise` of `SaveResult`, which summarizes the result of
   * the saving, such as byte sizes of the saved artifacts for the model's
   *   topology and weight values.
   */
  /**
   * @doc {heading: 'Models', subheading: 'Classes'}
   */
  async save(handlerOrURL: io.IOHandler|string, config?: io.SaveConfig):
      Promise<io.SaveResult> {
    if (typeof handlerOrURL === 'string') {
      const handlers = io.getSaveHandlers(handlerOrURL);
      if (handlers.length === 0) {
        throw new ValueError(
            `Cannot find any save handlers for URL '${handlerOrURL}'`);
      } else if (handlers.length > 1) {
        throw new ValueError(
            `Found more than one (${handlers.length}) save handlers for ` +
            `URL '${handlerOrURL}'`);
      }
      handlerOrURL = handlers[0];
    }
    if (handlerOrURL.save == null) {
      throw new ValueError(
          'LayersModel.save() cannot proceed because the IOHandler ' +
          'provided does not have the `save` attribute defined.');
    }

    const weightDataAndSpecs =
        await io.encodeWeights(this.getNamedWeights(config));

    const returnString = false;
    const unusedArg: {} = null;
    const modelConfig = this.toJSON(unusedArg, returnString);

    return handlerOrURL.save({
      modelTopology: modelConfig,
      weightData: weightDataAndSpecs.data,
      weightSpecs: weightDataAndSpecs.specs,
      format: LAYERS_MODEL_FORMAT_NAME,
      generatedBy: `TensorFlow.js tfjs-layers v${version}`,
      convertedBy: null,
    });
  }
}
serialization.registerClass(LayersModel);
