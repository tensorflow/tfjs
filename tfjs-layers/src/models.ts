/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

/* Original source keras/models.py */

import {dispose, io, NamedTensorMap, Optimizer, Scalar, serialization, Tensor, util} from '@tensorflow/tfjs-core';

import {getUid} from './backend/state';
import {History} from './base_callbacks';
import {Dataset} from './engine/dataset_stub';
import {Input} from './engine/input_layer';
import {getSourceInputs, Layer, Node, SymbolicTensor} from './engine/topology';
import {LayersModel, ModelCompileArgs, ModelEvaluateArgs} from './engine/training';
import {ModelEvaluateDatasetArgs, ModelFitDatasetArgs} from './engine/training_dataset';
import {ModelFitArgs} from './engine/training_tensors';
import {NotImplementedError, RuntimeError, ValueError} from './errors';
import {Shape} from './keras_format/common';
import {TrainingConfig} from './keras_format/training_config';
import {PyJsonDict} from './keras_format/types';
import {deserialize} from './layers/serialization';
import {Kwargs, NamedTensor} from './types';
import * as generic_utils from './utils/generic_utils';
import {convertPythonicToTs} from './utils/serialization_utils';
import {getExactlyOneShape} from './utils/types_utils';

/**
 * Parses a JSON model configuration file and returns a model instance.
 *
 * ```js
 * // This example shows how to serialize a model using `toJSON()` and
 * // deserialize it as another model using `tf.models.modelFromJSON()`.
 * // Note: this example serializes and deserializes only the topology
 * // of the model; the weights of the loaded model will be different
 * // from those of the the original model, due to random weight
 * // initialization.
 * // To load the topology and weights of a model, use `tf.loadLayersModel()`.
 * const model1 = tf.sequential();
 * model1.add(tf.layers.repeatVector({inputShape: [2], n: 4}));
 * // Serialize `model1` as a JSON object.
 * const model1JSON = model1.toJSON(null, false);
 * model1.summary();
 *
 * const model2 = await tf.models.modelFromJSON(model1JSON);
 * model2.summary();
 * ```
 *
 *  @param modelAndWeightsConfig JSON object or string encoding a model and
 *       weights configuration. It can also be only the topology JSON of the
 *       model, in which case the weights will not be loaded.
 *  @param custom_objects Optional dictionary mapping names
 *       (strings) to custom classes or functions to be
 *       considered during deserialization.
 * @returns A TensorFlow.js Layers `tf.LayersModel` instance (uncompiled).
 */
export async function modelFromJSON(
    modelAndWeightsConfig: ModelAndWeightsConfig|PyJsonDict,
    customObjects?: serialization.ConfigDict): Promise<LayersModel> {
  if (!('modelTopology' in modelAndWeightsConfig)) {
    modelAndWeightsConfig = {modelTopology: modelAndWeightsConfig};
  }
  modelAndWeightsConfig = modelAndWeightsConfig as ModelAndWeightsConfig;

  let modelTopology = modelAndWeightsConfig.modelTopology;
  if (modelTopology['model_config'] != null) {
    // If the model-topology JSON contains a 'model_config' field, then it is
    // a full model JSON (e.g., from `keras.Model.save()`), which contains
    // not only the model's architecture in its 'model_config' field, but
    // additional information such as the model's optimizer. We use only the
    // 'model_config' field currently.
    modelTopology = modelTopology['model_config'] as PyJsonDict;
  }
  const tsConfig =
      convertPythonicToTs(modelTopology) as serialization.ConfigDict;
  const model = deserialize(tsConfig, customObjects) as LayersModel;

  if (modelAndWeightsConfig.weightsManifest != null) {
    // Load the weight values keyed by the original tensor names in the model
    // file that was loaded.  These should match the keys of the weight
    // manifest.
    const weightValues = await io.loadWeights(
        modelAndWeightsConfig.weightsManifest, modelAndWeightsConfig.pathPrefix,
        model.weights.map(weight => weight.originalName));

    // Map the weights to the unique tensor names generated during model loading
    const uniqueWeightValues: NamedTensorMap = {};
    for (const weight of model.weights) {
      uniqueWeightValues[weight.originalName] =
          weightValues[weight.originalName];
    }

    model.loadWeights(uniqueWeightValues);
    // Dispose temporary weight values.
    dispose(weightValues);
  }
  return model;
}

/**
 * Options for loading a saved mode in TensorFlow.js format.
 */
export interface ModelAndWeightsConfig {
  /**
   * A JSON object or JSON string containing the model config.
   *
   * This can be either of the following two formats:
   *   - A model archiecture-only config,  i.e., a format consistent with the
   *     return value of`keras.Model.to_json()`.
   *   - A full model config, containing not only model architecture, but also
   *     training options and state, i.e., a format consistent with the return
   *     value of `keras.models.save_model()`.
   */
  modelTopology: PyJsonDict;

  /**
   * A weights manifest in TensorFlow.js format.
   */
  weightsManifest?: io.WeightsManifestConfig;

  /**
   * Path to prepend to the paths in `weightManifest` before fetching.
   *
   * The path may optionally end in a slash ('/').
   */
  pathPrefix?: string;
}

// TODO(nielsene): Remove after: https://github.com/tensorflow/tfjs/issues/400
export interface ModelPredictArgs {
  /**
   * Optional. Batch size (Integer). If unspecified, it will default to 32.
   */
  batchSize?: number;

  /**
   * Optional. Verbosity mode. Defaults to false.
   */
  verbose?: boolean;
}

/**
 * Load a model, including its topology and optionally weights.  See the
 * Tutorial named "How to import a Keras Model" for usage examples.
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
 * Example 3. Load a model from user-selected files from HTML
 * [file input
 * elements](https://developer.mozilla.org/en-US/docs/Web/HTML/Element/input/file).
 *
 * ```js
 * // Note: this code snippet will not work without the HTML elements in the
 * //   page
 * const jsonUpload = document.getElementById('json-upload');
 * const weightsUpload = document.getElementById('weights-upload');
 *
 * const model = await tf.loadLayersModel(
 *     tf.io.browserFiles([jsonUpload.files[0], weightsUpload.files[0]]));
 * ```
 *
 * Example 4. Load a model from an HTTP server.
 *
 * ```js
 * const model = await
 *     tf.loadLayersModel('https://storage.googleapis.com/tfjs-models/tfjs/iris_v1/model.json');
 * model.summary();
 * ```
 *
 * @param pathOrIOHandler Can be either of the two formats
 *   1. A string path to the `ModelAndWeightsConfig` JSON describing
 *      the model in the canonical TensorFlow.js format. This path will be
 *      interpreted as a relative HTTP path, to which `fetch` will be used to
 *      request the model topology and weight manifest JSON.
 *      The content of the JSON file is assumed to be a JSON object with the
 *      following fields and values:
 *      - 'modelTopology': A JSON object that can be either of:
 *        1. a model architecture JSON consistent with the format of the return
 *            value of `keras.Model.to_json()`
 *        2. a full model JSON in the format of `keras.models.save_model()`.
 *      - 'weightsManifest': A TensorFlow.js weights manifest.
 *      See the Python converter function `save_model()` for more details.
 *      It is also assumed that model weights can be accessed from relative
 *      paths described by the `paths` fields in weights manifest.
 *   2. An `tf.io.IOHandler` object that loads model artifacts with its `load`
 *      method.
 * @param options Optional configuration arguments for the model loading,
 *   including:
 *   - `strict`: Require that the provided weights exactly match those required
 *     by the layers.  Default true.  Passing false means that both extra
 *     weights and missing weights will be silently ignored.
 *   - `onProgress`: A progress callback of the form:
 *     `(fraction: number) => void`. This callback can be used to monitor the
 *     model-loading process.
 * @returns A `Promise` of `tf.LayersModel`, with the topology and weights
 *     loaded.
 */
export async function loadLayersModelInternal(
    pathOrIOHandler: string|io.IOHandler,
    options?: io.LoadOptions): Promise<LayersModel> {
  if (options == null) {
    options = {};
  }
  if (typeof pathOrIOHandler === 'string') {
    const handlers = io.getLoadHandlers(pathOrIOHandler, options);
    if (handlers.length === 0) {
      // For backward compatibility: if no load handler can be found,
      // assume it is a relative http path.
      // TODO(cais): Reformat the args into a single `LoadOptions` once the core
      // is refactored.
      handlers.push(io.browserHTTPRequest(pathOrIOHandler, options));
    } else if (handlers.length > 1) {
      throw new ValueError(
          `Found more than one (${handlers.length}) load handlers for ` +
          `URL '${pathOrIOHandler}'`);
    }
    pathOrIOHandler = handlers[0];
  }
  return loadLayersModelFromIOHandler(pathOrIOHandler, undefined, options);
}

/**
 * Load a model and optionally its weights, using an IOHandler object.
 *
 * @param handler The instance of `IOHandler` to be used during the model
 *   loading.
 * @param customObjects Any optional custom objects to be used during model
 *   loading.
 * @param strict Whether the weight loading will be done in strict mode.
 *   Default: `true`.
 */
export async function loadLayersModelFromIOHandler(
    handler: io.IOHandler, customObjects?: serialization.ConfigDict,
    options?: io.LoadOptions): Promise<LayersModel> {
  if (options == null) {
    options = {};
  }
  if (handler.load == null) {
    throw new ValueError(
        'Cannot proceed with model loading because the IOHandler provided ' +
        'does not have the `load` method implemented.');
  }
  const artifacts = await handler.load();
  let modelTopology = artifacts.modelTopology as PyJsonDict;
  if (modelTopology['model_config'] != null) {
    modelTopology = modelTopology['model_config'] as PyJsonDict;
  }

  const strict = options.strict == null ? true : options.strict;
  // If weights are provided and the weight-loading mode is strict, use
  // fast weight initialization. This skips costly initializers such as
  // 'orthogonal' and saves unnecessary computation in cases where
  // the initialized weight values will immediately be overwritten by
  // loaded weight values.
  const fastWeightInit =
      artifacts.weightData != null && artifacts.weightSpecs != null && strict;
  const model =
      deserialize(
          convertPythonicToTs(modelTopology) as serialization.ConfigDict,
          customObjects, fastWeightInit) as LayersModel;

  const trainingConfig = artifacts.trainingConfig as TrainingConfig;
  if (trainingConfig != null) {
    model.loadTrainingConfig(trainingConfig);
  }
  if (artifacts.userDefinedMetadata != null) {
    model.setUserDefinedMetadata(artifacts.userDefinedMetadata);
  }

  // If weightData is present, load the weights into the model.
  if (artifacts.weightData != null) {
    // Loading weights requires weightSpecs.
    if (artifacts.weightSpecs == null) {
      throw new ValueError(
          'LayersModel artifacts contains weight data, but not weight specs. ' +
          'Therefore loading of weights cannot proceed.');
    }

    const {modelWeights, optimizerWeights} = decodeModelAndOptimizerWeights(
        artifacts.weightData, artifacts.weightSpecs);
    model.loadWeights(modelWeights, strict);

    if (model.optimizer != null && optimizerWeights.length > 0) {
      await model.optimizer.setWeights(optimizerWeights);
    }

    // Dispose temporary weight values.
    dispose(modelWeights);
    dispose(optimizerWeights.map(w => w.tensor));
  }
  return model;
}

function decodeModelAndOptimizerWeights(
    buffer: ArrayBuffer, specs: io.WeightsManifestEntry[]):
    {modelWeights: NamedTensorMap, optimizerWeights: NamedTensor[]} {
  const name2Tensor = io.decodeWeights(buffer, specs);
  const modelWeights: NamedTensorMap = {};
  const optimizerWeights: NamedTensor[] = [];
  specs.forEach(spec => {
    if (spec.group === 'optimizer') {
      optimizerWeights.push({name: spec.name, tensor: name2Tensor[spec.name]});
    } else {
      modelWeights[spec.name] = name2Tensor[spec.name];
    }
  });
  return {modelWeights, optimizerWeights};
}

/**
 * Configuration for a Sequential model.
 */
export interface SequentialArgs {
  /** Stack of layers for the model. */
  layers?: Layer[];

  /** The name of this model. */
  name?: string;
}

/**
 * A model with a stack of layers, feeding linearly from one to the next.
 *
 * `tf.sequential` is a factory function that creates an instance of
 * `tf.Sequential`.
 *
 * ```js
 *  // Define a model for linear regression.
 *  const model = tf.sequential();
 *  model.add(tf.layers.dense({units: 1, inputShape: [1]}));
 *
 *  // Prepare the model for training: Specify the loss and the optimizer.
 *  model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});
 *
 *  // Generate some synthetic data for training.
 *  const xs = tf.tensor2d([1, 2, 3, 4], [4, 1]);
 *  const ys = tf.tensor2d([1, 3, 5, 7], [4, 1]);
 *
 *  // Train the model using the data then do inference on a data point the
 *  // model hasn't seen:
 *  await model.fit(xs, ys);
 *  model.predict(tf.tensor2d([5], [1, 1])).print();
 * ```
 *
 * @doc {heading: 'Models', subheading: 'Classes'}
 */
export class Sequential extends LayersModel {
  /** @nocollapse */
  static className = 'Sequential';
  private model: LayersModel;
  constructor(args?: SequentialArgs) {
    super({inputs: [], outputs: []});
    args = args || {};

    this.trainable = true;
    this.built = false;

    // Set model name.
    this.name = (args.name != null) ? args.name : getUid('sequential_');

    // Add to the model any layers passed to the constructor.
    if (args.layers != null) {
      for (const layer of args.layers) {
        this.add(layer);
      }
    }
  }

  // Helper function to Sequential.add  Throws if the new output shape will be
  // invalid.
  private checkShape(layer: Layer) {
    const shape = layer.inboundNodes[0].outputTensors[0].shape;
    if (shape.some(x => x < 0)) {
      throw new ValueError(
          'Negative dimension size caused by adding layer ' +
          `${layer.name} with input shape [` +
          `${layer.inboundNodes[0].inputTensors[0].shape}]`);
    }
  }

  /**
   * Adds a layer instance on top of the layer stack.
   *
   * ```js
   *  const model = tf.sequential();
   *  model.add(tf.layers.dense({units: 8, inputShape: [1]}));
   *  model.add(tf.layers.dense({units: 4, activation: 'relu6'}));
   *  model.add(tf.layers.dense({units: 1, activation: 'relu6'}));
   *  // Note that the untrained model is random at this point.
   *  model.predict(tf.randomNormal([10, 1])).print();
   * ```
   * @param layer Layer instance.
   *
   * @exception ValueError In case the `layer` argument does not know its
   * input shape.
   * @exception ValueError In case the `layer` argument has multiple output
   *   tensors, or is already connected somewhere else (forbidden in
   *   `Sequential` models).
   *
   * @doc {heading: 'Models', subheading: 'Classes'}
   */
  add(layer: Layer): void {
    const isLayerModelInstance =
        layer instanceof Sequential || layer instanceof LayersModel;
    let modelLayer: LayersModel;
    if (isLayerModelInstance) {
      modelLayer = layer as LayersModel;
      if (modelLayer.outputs.length !== 1) {
        throw new ValueError(
            'All layers in a Sequential model ' +
            'should have a single output tensor. ' +
            'For multi-output layers, ' +
            'use the functional API.');
      }
      if (modelLayer.inputs.length !== 1) {
        throw new ValueError(
            'All layers in a Sequential model ' +
            'should have a single input tensor. ' +
            'For multi-input layers, ' +
            'use the functional API.');
      }
    }

    if (this.outputs.length === 0) {
      // first layer in model: check that it is an input layer
      if (layer.inboundNodes.length === 0) {
        // create an input layer
        if (layer.batchInputShape == null) {
          throw new ValueError(
              'The first layer in a Sequential model must ' +
              'get an `inputShape` or `batchInputShape` argument.');
        }
        // Instantiate the input layer.
        const x = Input({
          batchShape: layer.batchInputShape,
          dtype: layer.dtype,
          name: layer.name + '_input'
        });
        // This will build the current layer and create the node connecting
        // the current layer to the input layer we just created.
        layer.apply(x);
      }

      if (isLayerModelInstance) {
        this.outputs = modelLayer.outputs;
        this.inputs = modelLayer.inputs;
      } else {
        if (layer.inboundNodes.length !== 1) {
          throw new ValueError(
              'A layer added to a Sequential model must not already be ' +
              `connected somewhere else. LayersModel received layer ${
                  layer.name} ` +
              `which has ${layer.inboundNodes.length} pre-existing inbound ` +
              'connections.');
        }

        if (layer.inboundNodes[0].outputTensors.length !== 1) {
          throw new ValueError(
              'All layers in a Sequential model ' +
              'should have a single output tensor. ' +
              'For multi-output layers, ' +
              'use the functional API.');
        }
        this.checkShape(layer);
        this.outputs = [layer.inboundNodes[0].outputTensors[0]];
        this.inputs = getSourceInputs(this.outputs[0]);
      }

      this.inboundNodes = [];
      // We create an input node, which we will keep updated
      // as we add more layers.
      // (This call has side effects.)
      // tslint:disable-next-line:no-unused-expression
      new Node({
        outboundLayer: this,
        inboundLayers: [],
        nodeIndices: [],
        tensorIndices: [],
        inputTensors: this.inputs,
        outputTensors: this.outputs,
        // no model-level masking for now
        inputMasks: generic_utils.pyListRepeat(null, this.inputs.length),
        outputMasks: [null],
        inputShapes: this.inputs.map(x => x.shape),
        outputShapes: this.outputs[0].shape
      });
    } else {
      const outputTensor = layer.apply(this.outputs[0]);
      if (Array.isArray(outputTensor)) {
        throw new TypeError(
            'All layers in a Sequential model ' +
            'should have a single output tensor. ' +
            'For multi-output layers, ' +
            'use the functional API.');
      }
      this.checkShape(layer);
      this.outputs = [outputTensor as SymbolicTensor];
      // update self.inbound_nodes
      this.inboundNodes[0].outputTensors = this.outputs;
      this.inboundNodes[0].outputShapes = [this.outputs[0].shape];
    }

    this.layers.push(layer);
    this.built = false;
  }

  /**
   * Removes the last layer in the model.
   *
   * @exception TypeError if there are no layers in the model.
   */
  pop(): void {
    if (this.layers.length === 0) {
      throw new TypeError('There are no layers in the model.');
    }

    this.layers.pop();
    if (this.layers.length === 0) {
      this.outputs = [];
      this.inboundNodes = [];
      this.outboundNodes = [];
    } else {
      const lastLayerIndex = this.layers.length - 1;
      this.layers[lastLayerIndex].outboundNodes = [];
      this.outputs = [this.layers[lastLayerIndex].output as SymbolicTensor];
      // update self.inbound_nodes
      this.inboundNodes[0].outputTensors = this.outputs;
      this.inboundNodes[0].outputShapes = [this.outputs[0].shape];
    }
  }

  call(inputs: Tensor|Tensor[], kwargs: Kwargs): Tensor|Tensor[] {
    if (this.model == null) {
      this.build();
    }
    return this.model.call(inputs, kwargs);
  }

  build(inputShape?: Shape|Shape[]) {
    // Call `getExactlyOneShape` without using its return value,
    // to verify that exactly one input shape is provided.
    getExactlyOneShape(inputShape);

    if (this.inputs.length === 0 || this.outputs.length === 0) {
      throw new TypeError(
          'Sequential model cannot be built: model is empty.' +
          ' Add some layers first.');
    }
    // actually create the model
    this.model = new LayersModel({
      inputs: this.inputs,
      outputs: this.outputs[0],
      name: this.name + '_model'
    });
    this.model.trainable = this.trainable;

    // mirror model attributes
    this.supportsMasking = this.model.supportsMasking;
    // TODO(michaelterry): Add caches
    this.inputLayers = this.model.inputLayers;
    this.inputLayersNodeIndices = this.model.inputLayersNodeIndices;
    this.inputLayersTensorIndices = this.model.inputLayersTensorIndices;
    this.outputLayers = this.model.outputLayers;
    this.outputLayersNodeIndices = this.model.outputLayersNodeIndices;
    this.outputLayersTensorIndices = this.model.outputLayersTensorIndices;
    this.nodesByDepth = this.model.nodesByDepth;
    this.containerNodes = this.model.containerNodes;
    this.outputNames = this.model.outputNames;
    this.inputNames = this.model.inputNames;
    // TODO(michaelterry): Add feedInputNames, feedInputs, if needed.
    // TODO(michaelterry): Add callbackModel if needed.
    this.built = true;
  }

  countParams(): number {
    if (!this.built) {
      this.build();
    }
    return super.countParams();
  }

  /**
   * Print a text summary of the Sequential model's layers.
   *
   * The summary includes
   * - Name and type of all layers that comprise the model.
   * - Output shape(s) of the layers
   * - Number of weight parameters of each layer
   * - The total number of trainable and non-trainable parameters of the
   * model.
   *
   * ```js
   * const model = tf.sequential();
   * model.add(
   *     tf.layers.dense({units: 100, inputShape: [10], activation: 'relu'}));
   * model.add(tf.layers.dense({units: 1, activation: 'sigmoid'}));
   *
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
   *
   * @doc {heading: 'Models', subheading: 'Classes'}
   */
  summary(
      lineLength?: number, positions?: number[],
      printFn:
          // tslint:disable-next-line:no-any
      (message?: any, ...optionalParams: any[]) => void = console.log) {
    if (!this.built) {
      this.build();
    }
    super.summary(lineLength, positions, printFn);
  }

  /**
   * Sets the weights of the model.
   *
   * @param weights Should be a list of Tensors with shapes and types matching
   *   the output of `model.getWeights()`.
   */
  setWeights(weights: Tensor[]): void {
    if (this.model == null) {
      this.build();
    }
    this.model.setWeights(weights);
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
   * const result = model.evaluate(tf.ones([8, 10]), tf.ones([8, 1]), {
   *   batchSize: 4,
   * });
   * result.print();
   * ```
   *
   * @param x `tf.Tensor` of test data, or an `Array` of `tf.Tensor`s if the
   * model has multiple inputs.
   * @param y `tf.Tensor` of target data, or an `Array` of `tf.Tensor`s if the
   * model has multiple outputs.
   * @param args A `ModelEvaluateConfig`, containing optional fields.
   *
   * @return `Scalar` test loss (if the model has a single output and no
   *   metrics) or `Array` of `Scalar`s (if the model has multiple outputs
   *   and/or metrics). The attribute `model.metricsNames`
   *   will give you the display labels for the scalar outputs.
   *
   * @doc {heading: 'Models', subheading: 'Classes'}
   */
  evaluate(
      x: Tensor|Tensor[], y: Tensor|Tensor[],
      args: ModelEvaluateArgs = {}): Scalar|Scalar[] {
    if (!this.built) {
      throw new RuntimeError(
          'The model needs to be compiled before being used.');
    }
    return this.model.evaluate(x, y, args);
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
   *
   * @doc {heading: 'Models', subheading: 'Classes'}
   */
  async evaluateDataset(dataset: Dataset<{}>, args: ModelEvaluateDatasetArgs):
      Promise<Scalar|Scalar[]> {
    if (!this.built) {
      throw new RuntimeError(
          'The model needs to be compiled before being used.');
    }
    return this.model.evaluateDataset(dataset, args);
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
   * model.predict(tf.ones([2, 10])).print();
   * ```
   *
   * @param x The input data, as a Tensor, or an `Array` of `tf.Tensor`s if
   *   the model has multiple inputs.
   * @param conifg A `ModelPredictConfig` object containing optional fields.
   *
   * @return `tf.Tensor`(s) of predictions.
   *
   * @exception ValueError In case of mismatch between the provided input data
   *   and the model's expectations, or in case a stateful model receives a
   *   number of samples that is not a multiple of the batch size.
   *
   * @doc {heading: 'Models', subheading: 'Classes'}
   */
  predict(x: Tensor|Tensor[], args: ModelPredictArgs = {}): Tensor|Tensor[] {
    if (this.model == null) {
      this.build();
    }
    return this.model.predict(x, args);
  }

  /**
   * Returns predictions for a single batch of samples.
   *
   * @param x: Input samples, as a Tensor, or list of Tensors (if the model
   *   has multiple inputs).
   * @return Tensor(s) of predictions
   */
  predictOnBatch(x: Tensor): Tensor|Tensor[] {
    if (this.model == null) {
      this.build();
    }
    return this.model.predictOnBatch(x);
  }

  /**
   * See `LayersModel.compile`.
   *
   * @param args
   */
  compile(args: ModelCompileArgs): void {
    this.build();
    this.model.compile(args);
    this.optimizer_ = this.model.optimizer;
    // tslint:disable-next-line:no-any
    this.isOptimizerOwned = (this.model as any).isOptimizerOwned;
    this.loss = this.model.loss;
    this.metrics = this.model.metrics;
    // TODO(cais): Add this.lossWeights, this.sampleWeightMode,
    //   this.weightedMetrics, this.targets.
    this.metricsTensors = this.model.metricsTensors;
    this.metricsNames = this.model.metricsNames;
    // TODO(cais): Add sampleWeights.
  }

  get optimizer(): Optimizer {
    return this.model == null ? undefined : this.model.optimizer;
  }

  set optimizer(optimizer: Optimizer) {
    this.model.optimizer = optimizer;
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
   * console.log(history.history.loss[0]);
   * ```
   *
   * @param x `tf.Tensor` of training data, or an array of `tf.Tensor`s if the
   * model has multiple inputs. If all inputs in the model are named, you can
   * also pass a dictionary mapping input names to `tf.Tensor`s.
   * @param y `tf.Tensor` of target (label) data, or an array of `tf.Tensor`s if
   * the model has multiple outputs. If all outputs in the model are named, you
   *  can also pass a dictionary mapping output names to `tf.Tensor`s.
   * @param args  A `ModelFitConfig`, containing optional fields.
   *
   * @return A `History` instance. Its `history` attribute contains all
   *   information collected during training.
   *
   * @exception ValueError In case of mismatch between the provided input data
   *   and what the model expects.
   *
   * @doc {heading: 'Models', subheading: 'Classes'}
   */
  async fit(
      x: Tensor|Tensor[]|{[inputName: string]: Tensor},
      y: Tensor|Tensor[]|{[inputName: string]: Tensor},
      args: ModelFitArgs = {}): Promise<History> {
    if (!this.built) {
      throw new RuntimeError(
          'The model needs to be compiled before ' +
          'being used.');
    }
    return this.model.fit(x, y, args);
  }

  /**
   * Trains the model using a dataset object.
   *
   * ```js
   * const xArray = [
   *   [1, 1, 1, 1, 1, 1, 1, 1, 1],
   *   [1, 1, 1, 1, 1, 1, 1, 1, 1],
   *   [1, 1, 1, 1, 1, 1, 1, 1, 1],
   *   [1, 1, 1, 1, 1, 1, 1, 1, 1],
   * ];
   * const yArray = [1, 1, 1, 1];
   * // Create a dataset from the JavaScript array.
   * const xDataset = tf.data.array(xArray);
   * const yDataset = tf.data.array(yArray);
   * // Zip combines the `x` and `y` Datasets into a single Dataset, the
   * // iterator of which will return an object containing of two tensors,
   * // corresponding to `x` and `y`.  The call to `batch(4)` will bundle
   * // four such samples into a single object, with the same keys now pointing
   * // to tensors that hold 4 examples, organized along the batch dimension.
   * // The call to `shuffle(4)` causes each iteration through the dataset to
   * // happen in a different order.  The size of the shuffle window is 4.
   * const xyDataset = tf.data.zip({xs: xDataset, ys: yDataset})
   *     .batch(4)
   *     .shuffle(4);
   * const model = tf.sequential({
   *   layers: [tf.layers.dense({units: 1, inputShape: [9]})]
   * });
   * model.compile({optimizer: 'sgd', loss: 'meanSquaredError'});
   * const history = await model.fitDataset(xyDataset, {
   *   epochs: 4,
   *   callbacks: {onEpochEnd: (epoch, logs) => console.log(logs.loss)}
   * });
   * ```
   *
   * @param dataset A dataset object. Its `iterator()` method is expected to
   *   generate a dataset iterator object, the `next()` method of which is
   *   expected to produce data batches for evaluation. The return value of the
   *   `next()` call ought to contain a boolean `done` field and a `value`
   *   field.
   *
   *   The `value` field is expected to be an object of with fields
   *   `xs` and `ys`, which point to the feature tensor and the target tensor,
   *   respectively. This case is for models with exactly one input and one
   *   output (e.g.. a sequential model). For example:
   *   ```js
   *   {value: {xs: xsTensor, ys: ysTensor}, done: false}
   *   ```
   *
   *   If the model has multiple inputs, the `xs` field of `value` should
   *   be an object mapping input names to their respective feature tensors.
   *   For example:
   *   ```js
   *   {
   *     value: {
   *       xs: {
   *         input_1: xsTensor1,
   *         input_2: xsTensor2
   *       },
   *       ys: ysTensor
   *     },
   *     done: false
   *   }
   *   ```
   *   If the model has multiple outputs, the `ys` field of `value` should
   *   be an object mapping output names to their respective target tensors.
   *   For example:
   *   ```js
   *   {
   *     value: {
   *       xs: xsTensor,
   *       ys: {
   *         output_1: ysTensor1,
   *         output_2: ysTensor2
   *       },
   *     },
   *     done: false
   *   }
   *   ```
   * @param args A `ModelFitDatasetArgs`, containing optional fields.
   *
   * @return A `History` instance. Its `history` attribute contains all
   *   information collected during training.
   *
   * @doc {heading: 'Models', subheading: 'Classes', ignoreCI: true}
   */
  async fitDataset<T>(dataset: Dataset<T>, args: ModelFitDatasetArgs<T>):
      Promise<History> {
    if (!this.built) {
      throw new RuntimeError(
          'The model needs to be compiled before ' +
          'being used.');
    }
    return this.model.fitDataset(dataset, args);
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
   *
   * @doc {heading: 'Models', subheading: 'Classes'}
   */
  async trainOnBatch(
      x: Tensor|Tensor[]|{[inputName: string]: Tensor},
      y: Tensor|Tensor[]|
      {[inputName: string]: Tensor}): Promise<number|number[]> {
    return this.model.trainOnBatch(x, y);
  }

  /* See parent class for JsDoc */
  /** @nocollapse */
  static fromConfig<T extends serialization.Serializable>(
      cls: serialization.SerializableConstructor<T>,
      config: serialization.ConfigDict,
      customObjects = {} as serialization.ConfigDict,
      fastWeightInit = false): T {
    let configArray: serialization.ConfigDictArray;
    let extraModelConfig: serialization.ConfigDict = {};
    if (config instanceof Array) {
      if (!(config[0].className != null) ||
          config[0]['className'] === 'Merge') {
        throw new ValueError('Legacy serialization format not supported yet.');
      }
      configArray = config;
    } else {
      util.assert(
          config['layers'] != null,
          () =>
              `When the config data for a Sequential model is not an Array, ` +
              `it must be an Object that contains the 'layers' field.`);
      configArray = config['layers'] as serialization.ConfigDictArray;
      delete config['layers'];
      extraModelConfig = config;
    }

    const model = new cls(extraModelConfig);
    if (!(model instanceof Sequential)) {
      throw new NotImplementedError(
          `Sequential.fromConfig called on non-Sequential input: ${model}`);
    }
    for (const conf of configArray) {
      const customObjects: serialization.ConfigDict = undefined;
      const layer = deserialize(
                        conf as serialization.ConfigDict, customObjects,
                        fastWeightInit) as Layer;
      if (fastWeightInit) {
        layer.setFastWeightInitDuringBuild(true);
      }
      model.add(layer);
    }
    return model;
  }

  /**
   * Setter used for force stopping of LayersModel.fit() (i.e., training).
   *
   * Example:
   *
   * ```js
   * const model = tf.sequential();
   * model.add(tf.layers.dense({units: 1, inputShape: [10]}));
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
   * // There should be only 3 values in the loss array, instead of 10 values,
   * // due to the stopping after 3 epochs.
   * console.log(history.history.loss);
   * ```
   */
  set stopTraining(stop: boolean) {
    // TODO(cais): When refactoring to remove the composition pattern happens,
    // remove this method overriding.
    if (this.model == null) {
      throw new ValueError(
          'Cannot set the stopTraining property of a sequential model before ' +
          'it is compiled.');
    }
    this.model.stopTraining = stop;
  }

  get stopTraining(): boolean {
    if (this.model == null) {
      throw new ValueError(
          'Cannot get the stopTraining property of a sequential model before ' +
          'it is compiled.');
    }
    return this.model.stopTraining;
  }

  // TODO(cais): Override get trainableWeights() here

  // tslint:disable-next-line:no-any
  getConfig(): any {
    // NOTE(cais): We override the return type of getConfig() to `any` here,
    //   because the `Sequential` class is a special case among `Container`
    //   subtypes in that its getConfig() method returns an Array (not a
    //   dict).
    const layers: serialization.ConfigDict[] = [];
    for (const layer of this.layers) {
      const dict: serialization.ConfigDict = {};
      dict['className'] = layer.getClassName();
      dict['config'] = layer.getConfig();
      layers.push(dict);
    }
    return {name: this.name, layers};
  }
}
serialization.registerClass(Sequential);
