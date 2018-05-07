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

// tslint:disable:max-line-length
import {doc, io, Scalar, serialization, Tensor} from '@tensorflow/tfjs-core';

import * as K from './backend/tfjs_backend';
import {History} from './callbacks';
import {getSourceInputs, Input, Layer, Node} from './engine/topology';
import {Model, ModelCompileConfig, ModelEvaluateConfig, ModelFitConfig, ModelPredictConfig} from './engine/training';
import {NotImplementedError, RuntimeError, ValueError} from './errors';
import {deserialize} from './layers/serialization';
import {NamedTensorMap, Shape} from './types';
import {JsonDict, SymbolicTensor} from './types';
import * as generic_utils from './utils/generic_utils';
import {convertPythonicToTs} from './utils/serialization_utils';
// tslint:enable:max-line-length

/**
 * Parses a JSON model configuration file and returns a model instance.
 *  @param modelAndWeightsConfig JSON object or string encoding a model and
 *       weights configuration.
 *  @param custom_objects Optional dictionary mapping names
 *       (strings) to custom classes or functions to be
 *       considered during deserialization.
 * @returns A TensorFlow.js Layers `Model` instance (uncompiled).
 */
export async function modelFromJSON(
    modelAndWeightsConfig: ModelAndWeightsConfig,
    customObjects?: serialization.ConfigDict): Promise<Model> {
  let modelTopology = modelAndWeightsConfig.modelTopology;
  if (modelTopology['model_config'] != null) {
    // If the model-topology JSON contains a 'model_config' field, then it is
    // a full model JSON (e.g., from `keras.Model.save()`), which contains
    // not only the model's architecture in its 'model_config' field, but
    // additional information such as the model's optimizer. We use only the
    // 'model_config' field currently.
    modelTopology = modelTopology['model_config'] as JsonDict;
  }
  const tsConfig =
      convertPythonicToTs(modelTopology) as serialization.ConfigDict;
  const model = deserialize(tsConfig, customObjects) as Model;

  if (modelAndWeightsConfig.weightsManifest != null) {
    // Load the weight values keyed by the original tensor names in the model
    // file that was loaded.  These should match the keys of the weight
    // manifest.
    const weightValues =
        await io.loadWeights(
            modelAndWeightsConfig.weightsManifest,
            modelAndWeightsConfig.pathPrefix,
            model.weights.map(weight => weight.originalName)) as NamedTensorMap;

    // Map the weights to the unique tensor names generated during model loading
    const uniqueWeightValues: NamedTensorMap = {};
    for (const weight of model.weights) {
      uniqueWeightValues[weight.originalName] =
          weightValues[weight.originalName];
    }

    const skipMismatches: boolean = null;
    const isNamedTensorMap = true;
    model.loadWeights(uniqueWeightValues, skipMismatches, isNamedTensorMap);
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
  modelTopology: JsonDict;

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

/**
 * Load a model, including its topology and optionally weights.  See the
 * Tutorial named "How to import a Keras Model" for usage examples.
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
 * paths described by the `paths` fields in weights manifest.
 *   2. An `tf.io.IOHandler` object that loads model artifacts with its `load`
 *      method.
 *
 * @returns A `Promise` of `Model`, with the topology and weights loaded.
 */
export async function loadModelInternal(pathOrIOHandler: string|
                                        io.IOHandler): Promise<Model> {
  return (typeof pathOrIOHandler === 'string') ?
      loadModelFromPath(pathOrIOHandler) :
      loadModelFromIOHandler(pathOrIOHandler as io.IOHandler);
}

/**
 * Load a model and optionally its weights, using an IOHandler object.
 */
export async function loadModelFromIOHandler(
    handler: io.IOHandler,
    customObjects?: serialization.ConfigDict): Promise<Model> {
  if (handler.load == null) {
    throw new ValueError(
        'Cannot proceed with model loading because the IOHandler provided ' +
        'does not have the `load` method implemented.');
  }
  const artifacts = await handler.load();
  const model = deserialize(
                    convertPythonicToTs(
                        artifacts.modelTopology as serialization.ConfigDict) as
                        serialization.ConfigDict,
                    customObjects) as Model;

  // If weightData is present, load the weights into the model.
  if (artifacts.weightData != null) {
    // Loading weights requires weightSpecs.
    if (artifacts.weightSpecs == null) {
      throw new ValueError(
          'Model artifacts contains weight data, but not weight specs. ' +
          'Therefore loading of weights cannot proceed.');
    }

    const skipMismatch = false;
    const isNamedTensorMap = true;
    model.loadWeights(
        io.decodeWeights(artifacts.weightData, artifacts.weightSpecs),
        skipMismatch, isNamedTensorMap);
  }
  return model;
}

// tslint:disable:max-line-length
/**
 * Load a model, including its topology and optionally weights.  See the
 * Tutorial named "How to import a Keras Model" for usage examples.
 *
 * @param modelConfigPath A path to the `ModelAndWeightsConfig` JSON describing
 * the model in the canonical TensorFlow.js format.
 *
 *   The content of the JSON file is assumed to be a JSON object with the
 *   following fields and values:
 *   - 'modelTopology': A JSON object that can be either of:
 *     1. a model architecture JSON consistent with the format of the return
 *      value of `keras.Model.to_json()`
 *     2. a full model JSON in the format of
 *        [`keras.Model.save()`](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model).
 *   - 'weightsManifest': A TensorFlow.js weights manifest.
 *   See the Python converter function
 *   [`save_keras_model()`](https://js.tensorflow.org/tutorials/import-keras.html)
 * for more details.
 *
 *   It is also assumed that model weights can be accessed from relative paths
 *     described by the `paths` fields in weights manifest.
 *
 * @returns A `Promise` of `Model`, with the topology and weights loaded.
 */
// tslint:enable:max-line-length
// TODO(cais): Add link to the core's documentation of `WeightManifestConfig`.
export async function loadModelFromPath(modelConfigPath: string):
    Promise<Model> {
  // TODO(soergel): accept a Request object too, not just a url string.
  const modelConfigRequest = await fetch(modelConfigPath);
  const modelConfig = await modelConfigRequest.json() as ModelAndWeightsConfig;
  if (modelConfig['modelTopology'] == null) {
    throw new ValueError(
        'Missing field "modelTopology" from model JSON at path' +
        modelConfigPath);
  }
  // TODO(cais): Remove this check as it's okay to load just the topology of a
  // model.
  if (modelConfig['weightsManifest'] == null) {
    throw new ValueError(
        'Missing field "weightsManifest" from model JSON at path' +
        modelConfigPath);
  }
  modelConfig.pathPrefix =
      modelConfigPath.substring(0, modelConfigPath.lastIndexOf('/'));

  return modelFromJSON(modelConfig);
}

/**
 * Configuration for a Sequential model.
 */
export interface SequentialConfig {
  /** Stack of layers for the model. */
  layers?: Layer[];

  /** The name of this model. */
  name?: string;
}

/**
 * A model with a stack of layers, feeding linearly from one to the next.
 *
 * `sequential` is a factory function that creates an instance of
 * `Sequential`.
 */
@doc({heading: 'Models', subheading: 'Classes'})
export class Sequential extends Model {
  static className = 'Sequential';
  private model: Model;
  private _updatable: boolean;
  constructor(config?: SequentialConfig) {
    super({inputs: [], outputs: []});
    config = config || {};

    this.trainable = true;
    this._updatable = true;
    this.built = false;

    // Set model name.
    this.name = (config.name != null) ? config.name : K.getUid('sequential_');

    // Add to the model any layers passed to the constructor.
    if (config.layers != null) {
      for (const layer of config.layers) {
        this.add(layer);
      }
    }
  }

  /**
   * Adds a layer instance on top of the layer stack.
   *
   * @param layer Layer instance.
   *
   * @exception ValueError In case the `layer` argument does not know its input
   *   shape.
   * @exception ValueError In case the `layer` argument has multiple output
   *   tensors, or is already connected somewhere else (forbidden in
   *   `Sequential` models).
   */
  @doc({heading: 'Models', subheading: 'Classes'})
  add(layer: Layer): void {
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

      if (layer.inboundNodes.length !== 1) {
        throw new ValueError(
            'A layer added to a Sequential model must not already be ' +
            `connected somewhere else. Model received layer ${layer.name} ` +
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

      this.outputs = [layer.inboundNodes[0].outputTensors[0]];
      this.inputs = getSourceInputs(this.outputs[0]);

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

  // tslint:disable-next-line:no-any
  call(inputs: Tensor|Tensor[], kwargs: any): Tensor|Tensor[] {
    if (this.model == null) {
      this.build();
    }
    return this.model.call(inputs, kwargs);
  }

  build(inputShape?: Shape) {
    if (this.inputs.length === 0 || this.outputs.length === 0) {
      throw new TypeError(
          'Sequential model cannot be built: model is empty.' +
          ' Add some layers first.');
    }
    // actually create the model
    this.model = new Model({
      inputs: this.inputs,
      outputs: this.outputs[0],
      name: this.name + '_model'
    });
    this.model.trainable = this.trainable;
    this.model.updatable = this.updatable;

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

  get updatable(): boolean {
    return this._updatable;
  }

  set updatable(value: boolean) {
    if (this.built) {
      this.model.updatable = value;
    }
    this._updatable = value;
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
   * @param x `Tensor` of test data, or an `Array` of `Tensor`s if the model has
   *   multiple inputs.
   * @param y `Tensor` of target data, or an `Array` of `Tensor`s if the model
   *   has multiple outputs.
   * @param config A `ModelEvaluateConfig`, containing optional fields.
   *
   * @return `Scalar` test loss (if the model has a single output and no
   *   metrics) or `Array` of `Scalar`s (if the model has multiple outputs
   *   and/or metrics). The attribute `model.metricsNames`
   *   will give you the display labels for the scalar outputs.
   */
  @doc({heading: 'Models', subheading: 'Classes', configParamIndices: [2]})
  evaluate(
      x: Tensor|Tensor[], y: Tensor|Tensor[], config: ModelEvaluateConfig = {}):
      Scalar|Scalar[] {
    if (!this.built) {
      throw new RuntimeError(
          'The model needs to be compiled before being used.');
    }
    return this.model.evaluate(x, y, config);
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
   * @param x The input data, as an Tensor, or an `Array` of `Tensor`s if
   *   the model has multiple inputs.
   * @param conifg A `ModelPredictConfig` object containing optional fields.
   *
   * @return `Tensor`(s) of predictions.
   *
   * @exception ValueError In case of mismatch between the provided input data
   *   and the model's expectations, or in case a stateful model receives a
   *   number of samples that is not a multiple of the batch size.
   */
  @doc({heading: 'Models', subheading: 'Classes', configParamIndices: [1]})
  predict(x: Tensor|Tensor[], config: ModelPredictConfig = {}): Tensor
      |Tensor[] {
    if (this.model == null) {
      this.build();
    }
    return this.model.predict(x, config);
  }

  /**
   * Returns predictions for a single batch of samples.
   *
   * @param x: Input samples, as an Tensor, or list of Tensors (if the model
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
   * See `Model.compile`.
   *
   * @param config
   */
  compile(config: ModelCompileConfig): void {
    this.build();
    this.model.compile(config);
    this.optimizer = this.model.optimizer;
    this.loss = this.model.loss;
    this.metrics = this.model.metrics;
    // TODO(cais): Add this.lossWeights, this.sampleWeightMode,
    //   this.weightedMetrics, this.targets.
    this.metricsTensors = this.model.metricsTensors;
    this.metricsNames = this.model.metricsNames;
    // TODO(cais): Add sampleWeights.
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
   * console.log(history.history.loss);
   * ```
   *
   * @param x `Tensor` of training data, or an array of `Tensor`s if the model
   *   has multiple inputs. If all inputs in the model are named, you can also
   *   pass a dictionary mapping input names to `Tensor`s.
   * @param y `Tensor` of target (label) data, or an array of `Tensor`s if the
   *   model has multiple outputs. If all outputs in the model are named, you
   *  can also pass a dictionary mapping output names to `Tensor`s.
   * @param config  A `ModelFitConfig`, containing optional fields.
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
    if (!this.built) {
      throw new RuntimeError(
          'The model needs to be compiled before ' +
          'being used.');
    }
    return this.model.fit(x, y, config);
  }

  /* See parent class for JsDoc */
  static fromConfig<T extends serialization.Serializable>(
      cls: serialization.SerializableConstructor<T>,
      config: serialization.ConfigDict): T {
    const model = new cls({});
    if (!(model instanceof Sequential)) {
      throw new ValueError(
          `Sequential.fromConfig called on non-Sequential input: ${model}`);
    }
    if (!(config instanceof Array)) {
      throw new ValueError(
          `Sequential.fromConfig called without an array of configs`);
    }
    if (!(config[0].className != null) || config[0]['className'] === 'Merge') {
      throw new ValueError('Legacy serialization format not supported yet.');
    }
    for (const conf of config as serialization.ConfigDictArray) {
      const layer = deserialize(conf as serialization.ConfigDict) as Layer;
      model.add(layer);
    }
    return model;
  }

  // TODO(cais): Override get trainableWeights() here

  /**
   * Extract weight values of the model.
   *
   * @param config: An instance of `io.SaveConfig`, which specifies model-saving
   *   options such as whether only trainable weights are to be saved.
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
   * Save the configuration and/or weights of the Model.
   *
   * An `IOHandler` is an object that has a `save` method of the proper
   * signature defined. The `save` method manages the storing or transmission of
   * serialized data ("artifacts") that represent the model's topology and
   * weights onto or via a specific medium, such as file downloads, local
   * storage, IndexedDB in the web browser and HTTP requests to a server.
   * TensorFlow.js provides `IOHandler` implementations for a number of
   * frequently used saving mediums, such as `tf.io.browserDownloads` and
   * `tf.io.browserLocalStorage`. See `tf.io` for more details.
   *
   * This method also allows you to refer to certain types of `IOHandler`s as
   * URL-like string shortcuts, such as 'localstorage://' and 'indexeddb://'.
   *
   * @param handlerOrURL An instance of `IOHandler` or a URL-like, scheme-based
   *   string shortcut for `IOHandler`.
   * @param config Options for saving the model.
   * @returns A `Promise` of `SaveResult`, which summarizes the result of the
   *   saving, such as byte sizes of the saved artifacts for the model's
   *   topology and weight values.
   */
  async save(handlerOrURL: io.IOHandler|string, config?: io.SaveConfig):
      Promise<io.SaveResult> {
    if (typeof handlerOrURL === 'string') {
      throw new NotImplementedError(
          'String URLs support in Model.save() is not implemented yet.');
    }
    if (handlerOrURL.save == null) {
      throw new ValueError(
          'Model.save() cannot proceed because the IOHandler provided does ' +
          'not have the `save` attribute defined.');
    }

    const weightDataAndSpecs =
        await io.encodeWeights(this.getNamedWeights(config));

    const returnString = false;
    const modelConfig = this.toJSON(null, returnString);

    return handlerOrURL.save({
      modelTopology: modelConfig,
      weightData: weightDataAndSpecs.data,
      weightSpecs: weightDataAndSpecs.specs
    });
  }

  // tslint:disable-next-line:no-any
  getConfig(): any {
    // NOTE(cais): We override the return type of getConfig() to `any` here,
    //   because the `Sequential` class is a special case among `Container`
    //   subtypes in that its getConfig() method returns an Array (not a dict).
    const config: serialization.ConfigDict[] = [];
    for (const layer of this.layers) {
      config.push({
        className: layer.getClassName(),
        config: layer.getConfig(),
      });
    }
    return config;
  }
}
serialization.SerializationMap.register(Sequential);
