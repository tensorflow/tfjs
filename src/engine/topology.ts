/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

/* Original source: keras/engine/topology.py */

// tslint:disable:max-line-length
import {doc, Scalar, serialization, Tensor, tidy, util} from '@tensorflow/tfjs-core';

import * as K from '../backend/tfjs_backend';
import {Constraint} from '../constraints';
import {AttributeError, NotImplementedError, RuntimeError, ValueError} from '../errors';
import {Initializer} from '../initializers';
import {deserialize as deserializeLayer} from '../layers/serialization';
import {Regularizer} from '../regularizers';
import {DType, JsonDict, LayerVariable, NamedTensorMap, RegularizerFn, Shape, SymbolicTensor, TensorInterface} from '../types';
import * as generic_utils from '../utils/generic_utils';
import {convertTsToPythonic} from '../utils/serialization_utils';
import {version as layersVersion} from '../version';
// tslint:enable:max-line-length

// TODO(michaelterry): This is a stub until it's defined.
export type Op = (x: LayerVariable) => LayerVariable;

/**
 * Constructor arguments for InputSpec.
 */
export interface InputSpecConfig {
  /** Expected datatype of the input. */
  dtype?: DType;
  /** Expected shape of the input (may include null for unchecked axes). */
  shape?: Shape;
  /** Expected rank of the input. */
  ndim?: number;
  /** Maximum rank of the input. */
  maxNDim?: number;
  /** Minimum rank of the input. */
  minNDim?: number;
  /** Dictionary mapping integer axes to a specific dimension value. */
  axes?: {[axis: number]: number};
}

/**
 * Specifies the ndim, dtype and shape of every input to a layer.
 *
 * Every layer should expose (if appropriate) an `inputSpec` attribute:
 * a list of instances of InputSpec (one per input tensor).
 *
 * A null entry in a shape is compatible with any dimension,
 * a null shape is compatible with any shape.
 */
export class InputSpec {
  /** Expected datatype of the input. */
  dtype?: DType;
  /** Expected shape of the input (may include null for unchecked axes). */
  shape?: Shape;
  /** Expected rank of the input. */
  ndim?: number;
  /** Maximum rank of the input. */
  maxNDim?: number;
  /** Minimum rank of the input. */
  minNDim?: number;
  /** Dictionary mapping integer axes to a specific dimension value. */
  axes?: {[axis: number]: number};

  constructor(config: InputSpecConfig) {
    this.dtype = config.dtype;
    this.shape = config.shape;
    /*
      TODO(michaelterry): Could throw error if ndim and shape are both defined
        (then backport).
    */
    if (config.shape != null) {
      this.ndim = config.shape.length;
    } else {
      this.ndim = config.ndim;
    }
    this.maxNDim = config.maxNDim;
    this.minNDim = config.minNDim;
    this.axes = config.axes || {};
  }
}

/**
 * Constructor arguments for Node.
 */
export interface NodeConfig {
  /**
   * The layer that takes `inputTensors` and turns them into `outputTensors`.
   * (the node gets created when the `call` method of the layer is called).
   */
  outboundLayer: Layer;
  /**
   * A list of layers, the same length as `inputTensors`, the layers from where
   * `inputTensors` originate.
   */
  inboundLayers: Layer[];
  /**
   * A list of integers, the same length as `inboundLayers`. `nodeIndices[i]` is
   * the origin node of `inputTensors[i]` (necessary since each inbound layer
   * might have several nodes, e.g. if the layer is being shared with a
   * different data stream).
   */
  nodeIndices: number[];
  /**
   * A list of integers, the same length as `inboundLayers`. `tensorIndices[i]`
   * is the index of `inputTensors[i]` within the output of the inbound layer
   * (necessary since each inbound layer might have multiple tensor outputs,
   * with each one being independently manipulable).
   */
  tensorIndices: number[];
  /** List of input tensors. */
  inputTensors: SymbolicTensor[];
  /** List of output tensors. */
  outputTensors: SymbolicTensor[];
  /** List of input masks (a mask can be a tensor, or null). */
  inputMasks: Tensor[];
  /** List of output masks (a mask can be a tensor, or null). */
  outputMasks: Tensor[];
  /** List of input shape tuples. */
  inputShapes: Shape|Shape[];
  /** List of output shape tuples. */
  outputShapes: Shape|Shape[];
}

let _nextNodeID = 0;

/**
 * A `Node` describes the connectivity between two layers.
 *
 * Each time a layer is connected to some new input,
 * a node is added to `layer.inboundNodes`.
 *
 * Each time the output of a layer is used by another layer,
 * a node is added to `layer.outboundNodes`.
 *
 * `nodeIndices` and `tensorIndices` are basically fine-grained coordinates
 * describing the origin of the `inputTensors`, verifying the following:
 *
 * `inputTensors[i] ==
 * inboundLayers[i].inboundNodes[nodeIndices[i]].outputTensors[
 *   tensorIndices[i]]`
 *
 * A node from layer A to layer B is added to:
 *     A.outboundNodes
 *     B.inboundNodes
 */
export class Node {
  /**
   * The layer that takes `inputTensors` and turns them into `outputTensors`
   * (the node gets created when the `call` method of the layer is called).
   */
  outboundLayer: Layer;
  /**
   * A list of layers, the same length as `inputTensors`, the layers from where
   * `inputTensors` originate.
   */
  inboundLayers: Layer[];
  /**
   * A list of integers, the same length as `inboundLayers`. `nodeIndices[i]` is
   * the origin node of `inputTensors[i]` (necessary since each inbound layer
   * might have several nodes, e.g. if the layer is being shared with a
   * different data stream).
   */
  nodeIndices: number[];
  /**
   * A list of integers, the same length as `inboundLayers`. `tensorIndices[i]`
   * is the index of `inputTensors[i]` within the output of the inbound layer
   * (necessary since each inbound layer might have multiple tensor outputs,
   * with each one being independently manipulable).
   */
  tensorIndices: number[];
  /** List of input tensors. */
  inputTensors: SymbolicTensor[];
  /** List of output tensors. */
  outputTensors: SymbolicTensor[];
  /** List of input masks (a mask can be a tensor, or null). */
  inputMasks: Tensor[];
  /** List of output masks (a mask can be a tensor, or null). */
  outputMasks: Tensor[];
  /** List of input shape tuples. */
  inputShapes: Shape|Shape[];
  /** List of output shape tuples. */
  outputShapes: Shape|Shape[];

  readonly id: number;

  constructor(
      config: NodeConfig,
      // TODO(michaelterry): Define actual type for this.
      // tslint:disable-next-line:no-any
      public callArgs?: any) {
    this.id = _nextNodeID++;
    /*
      Layer instance (NOT a list).
      this is the layer that takes a list of input tensors
      and turns them into a list of output tensors.
      the current node will be added to
      the inboundNodes of outboundLayer.
    */
    this.outboundLayer = config.outboundLayer;

    /*
        The following 3 properties describe where
        the input tensors come from: which layers,
        and for each layer, which node and which
        tensor output of each node.
    */

    // List of layer instances.
    this.inboundLayers = config.inboundLayers;
    // List of integers, 1:1 mapping with inboundLayers.
    this.nodeIndices = config.nodeIndices;
    // List of integers, 1:1 mapping with inboundLayers.
    this.tensorIndices = config.tensorIndices;

    /*
        Following 2 properties:
        tensor inputs and outputs of outboundLayer.
    */

    // List of tensors. 1:1 mapping with inboundLayers.
    this.inputTensors = config.inputTensors;
    // List of tensors, created by outboundLayer.call().
    this.outputTensors = config.outputTensors;

    /*
        Following 2 properties: input and output masks.
        List of tensors, 1:1 mapping with inputTensor.
    */
    this.inputMasks = config.inputMasks;
    // List of tensors, created by outboundLayer.computeMask().
    this.outputMasks = config.outputMasks;

    // Following 2 properties: input and output shapes.

    // List of shape tuples, shapes of inputTensors.
    this.inputShapes = config.inputShapes;
    // List of shape tuples, shapes of outputTensors.
    this.outputShapes = config.outputShapes;


    // Add nodes to all layers involved.
    for (const layer of config.inboundLayers) {
      if (layer != null) {
        layer.outboundNodes.push(this);
      }
    }
    config.outboundLayer.inboundNodes.push(this);
  }

  getConfig(): serialization.ConfigDict {
    const inboundNames: string[] = [];
    for (const layer of this.inboundLayers) {
      if (layer != null) {
        inboundNames.push(layer.name);
      } else {
        inboundNames.push(null);
      }
    }
    return {
      outboundLayer: this.outboundLayer ? this.outboundLayer.name : null,
      inboundLayers: inboundNames,
      nodeIndices: this.nodeIndices,
      tensorIndices: this.tensorIndices
    };
  }
}

/** Constructor arguments for Layer. */
export interface LayerConfig {
  /**
   * If defined, will be used to create an input layer to insert before this
   * layer. If both `inputShape` and `batchInputShape` are defined,
   * `batchInputShape` will be used. This argument is only applicable to input
   * layers (the first layer of a model).
   */
  inputShape?: Shape;
  /**
   * If defined, will be used to create an input layer to insert before this
   * layer. If both `inputShape` and `batchInputShape` are defined,
   * `batchInputShape` will be used. This argument is only applicable to input
   * layers (the first layer of a model).
   */
  batchInputShape?: Shape;
  /**
   * If `inputShape` is specified and `batchInputShape` is *not* specifiedd,
   * `batchSize` is used to construct the `batchInputShape`: `[batchSize,
   * ...inputShape]`
   */
  batchSize?: number;
  /**
   * The data-type for this layer. Defaults to 'float32'.
   * This argument is only applicable to input layers (the first layer of a
   * model).
   */
  dtype?: DType;
  /** Name for this layer. */
  name?: string;
  /** Whether this layer is trainable. Defaults to true. */
  trainable?: boolean;
  /** Whether the weights of this layer are updatable by `fit`. */
  updatable?: boolean;
  /**
   * Initial weight values of the layer.
   */
  weights?: Tensor[];
  /** Legacy support. Do not use for new code. */
  inputDType?: DType;
}

// If necessary, add `output` arguments to the CallHook function.
// This is currently used for testing only, but may be used for debugger-related
// purposes in the future.
// tslint:disable-next-line:no-any
export type CallHook = (inputs: Tensor|Tensor[], kwargs: any) => void;

let _nextLayerID = 0;

/**
 * A layer is a grouping of operations and weights that can be composed to
 * create a `Model`.
 *
 * Layers are constructed by using the functions under the
 * [tf.layers](#Layers-Basic) namespace.
 */
@doc({heading: 'Layers', subheading: 'Classes', namespace: 'layers'})
export abstract class Layer extends serialization.Serializable {
  /** Name for this layer. Must be unique within a model. */
  name: string;
  /**
   * List of InputSpec class instances.
   *
   * Each entry describes one required input:
   * - ndim
   * - dtype
   * A layer with `n` input tensors must have an `inputSpec` of length `n`.
   */
  inputSpec: InputSpec[];
  supportsMasking: boolean;
  /** Whether the layer weights will be updated during training. */
  trainable: boolean;
  updatable: boolean;
  batchInputShape: Shape;
  dtype: DType;
  initialWeights: Tensor[];

  inboundNodes: Node[];
  outboundNodes: Node[];

  activityRegularizer: Regularizer;

  protected _trainableWeights: LayerVariable[];
  private _nonTrainableWeights: LayerVariable[];
  private _losses: RegularizerFn[];
  private _updates: TensorInterface[];
  private _built: boolean;
  private _callHook: CallHook = null;

  private _addedWeightNames: string[] = [];

  readonly id: number;

  // Porting Notes: PyKeras does not have this property in this base Layer
  //   class. Instead lets Layer subclass set it dynamically and checks the
  //   value with `hasattr`. In tfjs-layers, we let this be a member of this
  //   base class.
  protected _stateful = false;

  constructor(config: LayerConfig) {
    super();
    this.id = _nextLayerID++;

    this.activityRegularizer = null;

    this.inputSpec = null;
    this.supportsMasking = false;

    // These properties will be set upon call of this.build()
    this._trainableWeights = [];
    this._nonTrainableWeights = [];
    this._losses = [];
    this._updates = [];
    this._built = false;

    /*
      These lists will be filled via successive calls
      to this.addInboundNode().
     */
    this.inboundNodes = [];
    this.outboundNodes = [];

    let name = config.name;
    if (!name) {
      const prefix = this.getClassName();
      name = generic_utils.toSnakeCase(prefix) + '_' + K.getUid(prefix);
    }
    this.name = name;

    this.trainable = generic_utils.pyGetAttr(config, 'trainable', true);
    this.updatable = generic_utils.pyGetAttr(config, 'updatable', true);

    if (config.inputShape != null || config.batchInputShape != null) {
      /*
        In this case we will later create an input layer
        to insert before the current layer
       */
      let batchInputShape: Shape;
      if (config.batchInputShape != null) {
        batchInputShape = config.batchInputShape;
      } else if (config.inputShape != null) {
        let batchSize: number = null;
        if (config.batchSize != null) {
          batchSize = config.batchSize;
        }
        batchInputShape = [batchSize].concat(config.inputShape);
      }
      this.batchInputShape = batchInputShape;

      // Set dtype.
      let dtype = config.dtype;
      if (dtype == null) {
        dtype = config.inputDType;
      }
      if (dtype == null) {
        dtype = K.floatx();
      }
      this.dtype = dtype;
    }

    if (config.weights != null) {
      this.initialWeights = config.weights;
    } else {
      this.initialWeights = null;
    }
  }

  /**
   * Converts a layer and its index to a unique (immutable type) name.
   * This function is used internally with `this.containerNodes`.
   * @param layer The layer.
   * @param nodeIndex The layer's position (e.g. via enumerate) in a list of
   *   nodes.
   *
   * @returns The unique name.
   */
  protected static nodeKey(layer: Layer, nodeIndex: number) {
    return layer.name + '_ib-' + nodeIndex.toString();
  }

  /**
   * Returns this.inboundNode at index nodeIndex.
   *
   * Porting note: This is a replacement for _get_node_attribute_at_index()
   * @param nodeIndex
   * @param attrName The name of the attribute related to request for this node.
   */
  private getNodeAtIndex(nodeIndex: number, attrName: string): Node {
    if (this.inboundNodes.length === 0) {
      throw new RuntimeError(
          'The layer has never been called ' +
          `and thus has no defined ${attrName}.`);
    }
    if (this.inboundNodes.length <= nodeIndex) {
      throw new ValueError(
          `Asked to get ${attrName} at node ${nodeIndex}, ` +
          `but the layer has only ${this.inboundNodes.length} inbound nodes.`);
    }
    return this.inboundNodes[nodeIndex];
  }

  /**
   * Retrieves the input tensor(s) of a layer at a given node.
   *
   * @param nodeIndex Integer, index of the node from which to retrieve the
   *   attribute. E.g. `nodeIndex=0` will correspond to the first time the layer
   *   was called.
   *
   * @return A tensor (or list of tensors if the layer has multiple inputs).
   */
  getInputAt(nodeIndex: number): SymbolicTensor|SymbolicTensor[] {
    return generic_utils.singletonOrArray(
        this.getNodeAtIndex(nodeIndex, 'input').inputTensors);
  }

  /**
   * Retrieves the output tensor(s) of a layer at a given node.
   *
   * @param nodeIndex Integer, index of the node from which to retrieve the
   *   attribute. E.g. `nodeIndex=0` will correspond to the first time the layer
   *   was called.
   *
   * @return A tensor (or list of tensors if the layer has multiple outputs).
   */
  getOutputAt(nodeIndex: number): SymbolicTensor|SymbolicTensor[] {
    return generic_utils.singletonOrArray(
        this.getNodeAtIndex(nodeIndex, 'output').outputTensors);
  }

  // Properties

  /**
   * Retrieves the input tensor(s) of a layer.
   *
   * Only applicable if the layer has exactly one inbound node,
   * i.e. if it is connected to one incoming layer.
   *
   * @return Input tensor or list of input tensors.
   *
   * @exception AttributeError if the layer is connected to more than one
   *   incoming layers.
   */
  get input(): SymbolicTensor|SymbolicTensor[] {
    if (this.inboundNodes.length > 1) {
      throw new AttributeError(
          `Layer ${this.name}` +
          ' has multiple inbound nodes, ' +
          'hence the notion of "layer input" ' +
          'is ill-defined. ' +
          'Use `getInputAt(nodeIndex)` instead.');
    } else if (this.inboundNodes.length === 0) {
      throw new AttributeError(
          `Layer ${this.name}` +
          ' is not connected, no input to return.');
    }
    return generic_utils.singletonOrArray(
        this.getNodeAtIndex(0, 'input').inputTensors);
  }

  /**
   * Retrieves the output tensor(s) of a layer.
   *
   * Only applicable if the layer has exactly one inbound node,
   * i.e. if it is connected to one incoming layer.
   *
   * @return Output tensor or list of output tensors.
   *
   * @exception AttributeError if the layer is connected to more than one
   *   incoming layers.
   */
  get output(): SymbolicTensor|SymbolicTensor[] {
    if (this.inboundNodes.length === 0) {
      throw new AttributeError(
          `Layer ${this.name}` +
          ' has no inbound nodes.');
    }
    if (this.inboundNodes.length > 1) {
      throw new AttributeError(
          `Layer ${this.name}` +
          ' has multiple inbound nodes, ' +
          'hence the notion of "layer output" ' +
          'is ill-defined. ' +
          'Use `getOutputAt(nodeIndex)` instead.');
    }
    return generic_utils.singletonOrArray(
        this.getNodeAtIndex(0, 'output').outputTensors);
  }

  get losses(): RegularizerFn[] {
    return this._losses;
  }

  /**
   * Retrieves the Layer's current loss values.
   *
   * Used for regularizers during training.
   */
  calculateLosses(): Scalar[] {
    // Porting Node: This is an augmentation to Layer.loss in PyKeras.
    //   In PyKeras, Layer.loss returns symbolic tensors. Here a concrete
    //   Tensor (specifically Scalar) values are returned. This is due to the
    //   imperative backend.
    return this.losses.map(lossFn => lossFn());
  }

  get updates(): TensorInterface[] {
    return this._updates;
  }

  get built(): boolean {
    return this._built;
  }

  set built(built: boolean) {
    this._built = built;
  }

  get trainableWeights(): LayerVariable[] {
    if (this.trainable) {
      return this._trainableWeights;
    } else {
      return [];
    }
  }

  set trainableWeights(weights: LayerVariable[]) {
    this._trainableWeights = weights;
  }

  get nonTrainableWeights(): LayerVariable[] {
    if (!this.trainable) {
      return this._trainableWeights.concat(this._nonTrainableWeights);
    } else {
      return this._nonTrainableWeights;
    }
  }

  set nonTrainableWeights(weights: LayerVariable[]) {
    this._nonTrainableWeights = weights;
  }

  /**
   * The concatenation of the lists trainableWeights and nonTrainableWeights
   * (in this order).
   */
  get weights(): LayerVariable[] {
    return this.trainableWeights.concat(this.nonTrainableWeights);
  }

  get stateful(): boolean {
    return this._stateful;
  }

  /**
   * Checks compatibility between the layer and provided inputs.
   *
   * This checks that the tensor(s) `input`
   * verify the input assumptions of the layer
   * (if any). If not, exceptions are raised.
   *
   * @param inputs Input tensor or list of input tensors.
   *
   * @exception ValueError in case of mismatch between
   *   the provided inputs and the expectations of the layer.
   */
  protected assertInputCompatibility(inputs: Tensor|Tensor[]|SymbolicTensor|
                                     SymbolicTensor[]): void {
    inputs = generic_utils.toList(inputs);
    if (this.inputSpec == null || this.inputSpec.length === 0) {
      return;
    }
    const inputSpec = generic_utils.toList(this.inputSpec);
    if (inputs.length !== inputSpec.length) {
      throw new ValueError(
          `Layer ${this.name} expects ${inputSpec.length} inputs, ` +
          `but it received ${inputs.length} input tensors. ` +
          `Input received: ${inputs}`);
    }
    for (let inputIndex = 0; inputIndex < inputs.length; inputIndex++) {
      const x = inputs[inputIndex];
      const spec: InputSpec = inputSpec[inputIndex];
      if (spec == null) {
        continue;
      }

      // Check ndim.
      const ndim = K.ndim(x);
      if (spec.ndim != null) {
        if (ndim !== spec.ndim) {
          throw new ValueError(
              `Input ${inputIndex} is incompatible with layer ${this.name}: ` +
              `expected ndim=${spec.ndim}, found ndim=${ndim}`);
        }
      }
      if (spec.maxNDim != null) {
        if (ndim > spec.maxNDim) {
          throw new ValueError(
              `Input ${inputIndex} is incompatible with layer ${this.name}` +
              `: expected max_ndim=${spec.maxNDim}, found ndim=${ndim}`);
        }
      }
      if (spec.minNDim != null) {
        if (ndim < spec.minNDim) {
          throw new ValueError(
              `Input ${inputIndex} is incompatible with layer ${this.name}` +
              `: expected min_ndim=${spec.minNDim}, found ndim=${ndim}.`);
        }
      }

      // Check dtype.
      if (spec.dtype != null) {
        if (K.dtype(x) !== spec.dtype) {
          const xDType = K.dtype(x);
          throw new ValueError(
              `Input ${inputIndex} is incompatible with layer ${this.name} ` +
              `: expected dtype=${spec.dtype}, found dtype=${xDType}.`);
        }
      }

      // Check specific shape axes.
      if (spec.axes) {
        const xShape = K.intShape(x);
        for (const key in spec.axes) {
          const axis = Number(key);
          const value = spec.axes[key];
          // Perform Python-style slicing in case axis < 0;
          // TODO(cais): Use https://github.com/alvivi/typescript-underscore to
          // ensure type safety through Underscore calls.
          const xShapeAtAxis =
              axis >= 0 ? xShape[axis] : xShape[xShape.length + axis];
          if (value != null && [value, null].indexOf(xShapeAtAxis) === -1) {
            throw new ValueError(
                `Input ${inputIndex} is incompatible with layer ` +
                `${this.name}: expected axis ${axis} of input shape to ` +
                `have value ${value} but got shape ${xShape}.`);
          }
        }
      }

      // Check shape.
      if (spec.shape != null) {
        const xShape = K.intShape(x);
        for (let i = 0; i < spec.shape.length; ++i) {
          const specDim = spec.shape[i];
          const dim = xShape[i];
          if (specDim != null && dim != null) {
            if (specDim !== dim) {
              throw new ValueError(
                  `Input ${inputIndex} is incompatible with layer ` +
                  `${this.name}: expected shape=${spec.shape}, ` +
                  'found shape=${xShape}.');
            }
          }
        }
      }
    }
  }

  /**
   * This is where the layer's logic lives.
   *
   * @param inputs Input tensor, or list/tuple of input tensors.
   * @param kwargs Additional keyword arguments.
   *
   * @return A tensor or list/tuple of tensors.
   */
  // tslint:disable-next-line:no-any
  call(inputs: Tensor|Tensor[], kwargs: any): Tensor|Tensor[] {
    return inputs;
  }

  // tslint:disable-next-line:no-any
  protected invokeCallHook(inputs: Tensor|Tensor[], kwargs: any) {
    if (this._callHook != null) {
      this._callHook(inputs, kwargs);
    }
  }

  /**
   * Set call hook.
   * This is currently used for testing only.
   * @param callHook
   */
  setCallHook(callHook: CallHook) {
    this._callHook = callHook;
  }

  /**
   * Clear call hook.
   * This is currently used for testing only.
   */
  clearCallHook() {
    this._callHook = null;
  }

  /**
   * Builds or executes a `Layer's logic.
   *
   * When called with `Tensor`(s), execute the `Layer`s computation and
   * return Tensor(s). For example:
   *
   * ```js
   * const denseLayer = tf.layers.dense({
   *   units: 1,
   *   kernelInitializer: 'zeros',
   *   useBias: false
   * });
   *
   * // Invoke the layer's apply() method with a `Tensor` (with concrete
   * // numeric values).
   * const input = tf.ones([2, 2]);
   * const output = denseLayer.apply(input);
   *
   * // The output's value is expected to be [[0], [0]], due to the fact that
   * // the dense layer has a kernel initialized to all-zeros and does not have
   * // a bias.
   * output.print();
   * ```
   *
   * When called with `SymbolicTensor`(s), this will prepare the layer for
   * future execution.  This entails internal book-keeping on shapes of
   * expected Tensors, wiring layers together, and initializing weights.
   *
   * Calling `apply` with `SymbolicTensor`s are typically used during the
   * building of non-`Sequential` models. For example:
   *
   * ```js
   * const flattenLayer = tf.layers.flatten();
   * const denseLayer = tf.layers.dense({units: 1});
   *
   * // Use tf.layers.input() to obtain a SymbolicTensor as input to apply().
   * const input = tf.input({shape: [2, 2]});
   * const output1 = flattenLayer.apply(input);
   *
   * // output1.shape is [null, 4]. The first dimension is the undetermined
   * // batch size. The second dimension comes from flattening the [2, 2]
   * // shape.
   * console.log(JSON.stringify(output1.shape));
   *
   * // The output SymbolicTensor of the flatten layer can be used to call
   * // the apply() of the dense layer:
   * const output2 = denseLayer.apply(output1);
   *
   * // output2.shape is [null, 1]. The first dimension is the undetermined
   * // batch size. The second dimension matches the number of units of the
   * // dense layer.
   * console.log(JSON.stringify(output2.shape));
   *
   * // The input and output and be used to construct a model that consists
   * // of the flatten and dense layers.
   * const model = tf.model({inputs: input, outputs: output2});
   * ```
   *
   * @param inputs a `Tensor` or `SymbolicTensor` or an Array of them.
   * @param kwargs Additional keyword arguments to be passed to `call()`.
   *
   * @return Output of the layer's `call` method.
   *
   * @exception ValueError error in case the layer is missing shape information
   *   for its `build` call.
   */
  // Porting Note: This is a replacement for __call__() in Python.
  @doc({heading: 'Models', 'subheading': 'Classes'})
  apply(
      inputs: Tensor|Tensor[]|SymbolicTensor|SymbolicTensor[],
      // tslint:disable-next-line:no-any
      kwargs?: any): Tensor|Tensor[]|SymbolicTensor|SymbolicTensor[] {
    kwargs = kwargs || {};

    // Ensure inputs are all the same type.
    const inputsList = generic_utils.toList(inputs);

    let allAreSymbolic = true;
    for (const input of inputsList) {
      if (!(input instanceof SymbolicTensor)) {
        allAreSymbolic = false;
        break;
      }
    }
    let noneAreSymbolic = true;
    for (const input of inputsList) {
      if (input instanceof SymbolicTensor) {
        noneAreSymbolic = false;
        break;
      }
    }

    if (allAreSymbolic === noneAreSymbolic) {
      throw new ValueError(
          'Arguments to apply() must be all ' +
          'SymbolicTensors or all Tensors');
    }

    // TODO(michaelterry): nameScope() may not be necessary.
    return K.nameScope(this.name, () => {
      // Handle laying building (weight creating, input spec locking).
      if (!this.built) {
        /*
          Throw exceptions in case the input is not compatible
          with the inputSpec specified in the layer constructor.
         */
        this.assertInputCompatibility(inputs);

        // Collect input shapes to build layer.
        const inputShapes: Shape[] = [];
        for (const xElem of generic_utils.toList(inputs)) {
          inputShapes.push(K.intShape(xElem));
        }
        this.build(generic_utils.singletonOrArray(inputShapes));
        this.built = true;

        // Load weights that were specified at layer instantiation.
        if (this.initialWeights) {
          this.setWeights(this.initialWeights);
        }
      }

      /*
        Throw exceptions in case the input is not compatible
        with the inputSpec set at build time.
      */
      this.assertInputCompatibility(inputs);

      // Handle mask propagation.
      // TODO(michaelterry): Mask propagation not currently implemented.

      // Actually call the layer, collecting output(s), mask(s), and shape(s).
      if (noneAreSymbolic) {
        let output = this.call(inputs as Tensor | Tensor[], kwargs);
        // TODO(michaelterry): Compute the outputMask

        // If the layer returns tensors from its inputs, unmodified,
        // we copy them to avoid loss of tensor metadata.
        const outputList: Tensor[] = generic_utils.toList(output);
        const outputListCopy: Tensor[] = [];
        // TODO(michaelterry): This copying may not be necessary given our eager
        // backend.
        for (let x of outputList) {
          if (inputsList.indexOf(x) !== -1) {
            x = K.identity(x);
          }
          outputListCopy.push(x);
        }
        output = generic_utils.singletonOrArray(outputListCopy);

        if (this.activityRegularizer != null) {
          throw new NotImplementedError(
              'Layer invocation in the presence of activity ' +
              'regularizer(s) is not supported yet.');
        }

        // TODO(michaelterry): Call addInboundNode()?
        return output;
      } else {
        const inputShape = collectInputShape(inputs);
        const outputShape = this.computeOutputShape(inputShape);
        let output: SymbolicTensor|SymbolicTensor[];
        const outputDType = guessOutputDType(inputs);

        if (outputShape != null && outputShape.length > 0 &&
            Array.isArray(outputShape[0])) {
          // We have multiple output shapes. Create multiple output tensors.
          output = (outputShape as Shape[])
                       .map(
                           (shape, index) => new SymbolicTensor(
                               outputDType, shape, this,
                               generic_utils.toList(inputs), kwargs, this.name,
                               index));
        } else {
          output = new SymbolicTensor(
              outputDType, outputShape as Shape, this,
              generic_utils.toList(inputs), kwargs, this.name);
        }

        /*
          Add an inbound node to the layer, so that it keeps track
          of the call and of all new variables created during the call.
          This also updates the layer history of the output tensor(s).
          If the input tensor(s) had no previous history,
          this does nothing.
        */
        this.addInboundNode(
            inputs as SymbolicTensor | SymbolicTensor[], output, null, null,
            inputShape, outputShape, kwargs);

        if (this.activityRegularizer != null) {
          throw new NotImplementedError(
              'Layer invocation in the presence of activity ' +
              'regularizer(s) is not supported yet.');
        }

        return output;
      }
    });
  }

  /**
   * Creates the layer weights.
   *
   * Must be implemented on all layers that have weights.
   *
   * Called when apply() is called to construct the weights.
   *
   * @param inputShape A `Shape` or array of `Shape` (unused).
   */
  public build(inputShape: Shape|Shape[]): void {
    this.built = true;
  }

  /**
   * Returns the current values of the weights of the layer.
   *
   * @param trainableOnly Whether to get the values of only trainable weights.
   * @returns Weight values as an `Array` of `Tensor`s.
   */
  getWeights(trainableOnly = false): Tensor[] {
    return K.batchGetValue(
        trainableOnly ? this.trainableWeights : this.weights);
  }

  /**
   * Sets the weights of the layer, from Tensors.
   *
   * @param weights a list of Tensors. The number of arrays and their shape
   *   must match number of the dimensions of the weights of the layer (i.e.
   *   it should match the output of `getWeights`).
   *
   * @exception ValueError If the provided weights list does not match the
   *   layer's specifications.
   */
  setWeights(weights: Tensor[]): void {
    const params = this.weights;
    if (params.length !== weights.length) {
      // TODO(cais): Restore the following and use `providedWeights`, instead of
      // `weights` in the error message, once the deeplearn.js bug is fixed:
      // https://github.com/PAIR-code/deeplearnjs/issues/498
      // const providedWeights = JSON.stringify(weights).substr(0, 50);
      throw new ValueError(
          `You called setWeights(weights) on layer "${this.name}" ` +
          `with a weight list of length ${weights.length}, ` +
          `but the layer was expecting ${params.length} weights. ` +
          `Provided weights: ${weights}...`);
    }
    if (params.length === 0) {
      return;
    }
    const weightValueTuples: Array<[LayerVariable, Tensor]> = [];
    const paramValues = K.batchGetValue(params);
    for (let i = 0; i < paramValues.length; ++i) {
      const pv = paramValues[i];
      const p = params[i];
      const w = weights[i];
      if (!util.arraysEqual(pv.shape, w.shape)) {
        throw new ValueError(
            `Layer weight shape ${pv.shape} ` +
            `not compatible with provided weight shape ${w.shape}`);
      }
      weightValueTuples.push([p, w]);
    }
    K.batchSetValue(weightValueTuples);
  }

  /**
   * Adds a weight variable to the layer.
   *
   * @param name Name of the new weight variable.
   * @param shape The shape of the weight.
   * @param dtype The dtype of the weight.
   * @param initializer An initializer instance.
   * @param regularizer A regularizer instance.
   * @param trainable Whether the weight should be trained via backprop or not
   *   (assuming that the layer itself is also trainable).
   * @param constraint An optional trainable.
   * @return The created weight variable.
   */
  protected addWeight(
      name: string, shape: Shape, dtype?: DType, initializer?: Initializer,
      regularizer?: Regularizer, trainable?: boolean,
      constraint?: Constraint): LayerVariable {
    // Reject duplicate weight names.
    if (this._addedWeightNames.indexOf(name) !== -1) {
      throw new ValueError(
          `Duplicate weight name ${name} for layer ${this.name}`);
    }
    this._addedWeightNames.push(name);

    if (dtype == null) {
      dtype = K.floatx();
    }
    const weight = new LayerVariable(
        initializer.apply(shape, dtype), dtype, name, trainable, constraint);
    // Request backend not to dispose the weights of the model on scope() exit.
    if (regularizer != null) {
      this.addLoss(() => regularizer.apply(weight.read()));
    }
    if (trainable == null) {
      trainable = true;
    }
    if (trainable) {
      this._trainableWeights.push(weight);
    } else {
      this._nonTrainableWeights.push(weight);
    }
    return weight;
  }

  /**
   * Add losses to the layer.
   *
   * The loss may potentionally be conditional on some inputs tensors,
   * for instance activity losses are conditional on the layer's inputs.
   */
  addLoss(losses: RegularizerFn|RegularizerFn[]): void {
    if (losses == null || Array.isArray(losses) && losses.length === 0) {
      return;
    }
    // Update this.losses
    losses = generic_utils.toList(losses);
    if (this._losses !== undefined && this._losses !== null) {
      this.losses.push(...losses);
    }
  }

  /**
   * Computes the output shape of the layer.
   *
   * Assumes that the layer will be built to match that input shape provided.
   *
   * @param inputShape A shape (tuple of integers) or a list of shape tuples
   *   (one per output tensor of the layer). Shape tuples can include null for
   *   free dimensions, instead of an integer.
   */
  computeOutputShape(inputShape: Shape|Shape[]): Shape|Shape[] {
    return inputShape;
  }

  /**
   * Computes an output mask tensor.
   *
   * @param inputs Tensor or list of tensors.
   * @param mask Tensor or list of tensors.
   *
   * @return null or a tensor (or list of tensors, one per output tensor of the
   * layer).
   */
  computeMask(inputs: Tensor|Tensor[], mask?: Tensor|Tensor[]): Tensor
      |Tensor[] {
    if (!this.supportsMasking) {
      if (mask != null) {
        if (Array.isArray(mask)) {
          mask.forEach(maskElement => {
            if (maskElement != null) {
              throw new TypeError(
                  `Layer ${this.name} does not support masking,` +
                  'but was passed an inputMask.');
            }
          });
        } else {
          throw new TypeError(
              `Layer ${this.name} does not support masking,` +
              'but was passed an inputMask.');
        }
      }
      // masking not explicitly supported: return null as mask
      return null;
    }
    // if masking is explictly supported, by default
    // carry over the input mask
    return mask;
  }

  /**
   * Internal method to create an inbound node for the layer.
   *
   * @param inputTensors List of input tensors.
   * @param outputTensors List of output tensors.
   * @param inputMasks List of input masks (a mask can be a tensor, or null).
   * @param outputMasks List of output masks (a mask can be a tensor, or null).
   * @param inputShapes List of input shape tuples.
   * @param outputShapes List of output shape tuples.
   * @param kwargs Dictionary of keyword arguments that were passed to the
   *   `call` method of the layer at the call that created the node.
   */
  private addInboundNode(
      inputTensors: SymbolicTensor|SymbolicTensor[],
      outputTensors: SymbolicTensor|SymbolicTensor[],
      inputMasks: Tensor|Tensor[], outputMasks: Tensor|Tensor[],
      inputShapes: Shape|Shape[], outputShapes: Shape|Shape[],
      kwargs: {} = null): void {
    const inputTensorList: SymbolicTensor[] =
        generic_utils.toList(inputTensors);
    outputTensors = generic_utils.toList(outputTensors);
    inputMasks = generic_utils.toList(inputMasks);
    outputMasks = generic_utils.toList(outputMasks);
    inputShapes = generic_utils.normalizeShapeList(inputShapes);
    outputShapes = generic_utils.normalizeShapeList(outputShapes);

    // Collect input tensor(s) coordinates.
    const inboundLayers: Layer[] = [];
    const nodeIndices: number[] = [];
    const tensorIndices: number[] = [];
    for (const x of inputTensorList) {
      /*
       * TODO(michaelterry): Keras adds this value to tensors; it's not
       * clear whether we'll use this or not.
       */
      inboundLayers.push(x.sourceLayer);
      nodeIndices.push(x.nodeIndex);
      tensorIndices.push(x.tensorIndex);
    }

    // Create node, add it to inbound nodes.
    // (This call has side effects.)
    // tslint:disable-next-line:no-unused-expression
    new Node(
        {
          outboundLayer: this,
          inboundLayers,
          nodeIndices,
          tensorIndices,
          inputTensors: inputTensorList,
          outputTensors,
          inputMasks,
          outputMasks,
          inputShapes,
          outputShapes
        },
        kwargs);

    // Update tensor history
    for (let i = 0; i < outputTensors.length; i++) {
      // TODO(michaelterry: _uses_learning_phase not tracked.
      outputTensors[i].sourceLayer = this;
      outputTensors[i].nodeIndex = this.inboundNodes.length - 1;
      outputTensors[i].tensorIndex = i;
    }
  }

  /**
   * Returns the config of the layer.
   *
   * A layer config is a TS dictionary (serializable)
   * containing the configuration of a layer.
   * The same layer can be reinstantiated later
   * (without its trained weights) from this configuration.
   *
   * The config of a layer does not include connectivity
   * information, nor the layer class name.  These are handled
   * by 'Container' (one layer of abstraction above).
   *
   * Porting Note: The TS dictionary follows TS naming standrds for
   * keys, and uses tfjs-layers type-safe Enums.  Serialization methods
   * should use a helper function to convert to the pythonic storage
   * standard. (see serialization_utils.convertTsToPythonic)
   *
   * @returns TS dictionary of configuration.
   */
  getConfig(): serialization.ConfigDict {
    const config:
        serialization.ConfigDict = {name: this.name, trainable: this.trainable};
    if (this.batchInputShape != null) {
      config['batchInputShape'] = this.batchInputShape;
    }
    if (this.dtype != null) {
      config['dtype'] = this.dtype;
    }
    return config;
  }
}

/**
 * Collects the input shape(s) of a list of `Tensor`s or `SymbolicTensor`s.
 *
 * TODO(michaelterry): Update PyKeras docs (backport).
 *
 * @param inputTensors List of input tensors (or single input tensor).
 *
 * @return List of shape tuples (or single tuple), one tuple per input.
 */
function collectInputShape(inputTensors: SymbolicTensor|SymbolicTensor[]|Tensor|
                           Tensor[]): Shape|Shape[] {
  inputTensors =
      generic_utils.toList(inputTensors) as SymbolicTensor[] | Tensor[];
  const shapes: Shape[] = [];
  for (const x of inputTensors) {
    shapes.push(K.intShape(x));
  }
  return generic_utils.singletonOrArray(shapes);
}

/**
 * Guesses output dtype based on inputs.
 *
 * At present, just returns DType.float32 for any input.
 *
 * @param inputTensors List of input tensors (or single input tensor).
 *
 * @return The guessed DType. At present, always returns DType.float32.
 */
function guessOutputDType(inputTensors: SymbolicTensor|SymbolicTensor[]|Tensor|
                          Tensor[]): DType {
  return DType.float32;
}


/**
 * Constructor arguments for InputLayer.
 *
 * Note: You should provide only inputShape or batchInputShape (not both).
 * If only inputShape is provided, then the batchInputShape is determined by
 * the batchSize argument and the inputShape: [batchSize].concat(inputShape).
 */
export interface InputLayerConfig {
  /** Input shape, not including the batch axis. */
  inputShape?: Shape;
  /** Optional input batch size (integer or null). */
  batchSize?: number;
  /** Batch input shape, including the batch axis. */
  batchInputShape?: Shape;
  /** Datatype of the input.  */
  dtype?: DType;
  /**
   * Whether the placeholder created is meant to be sparse.
   */
  sparse?: boolean;  // TODO(michaelterry): Not clear whether we'll need this.

  /** Name of the layer. */
  name?: string;
}

/**
 * An input layer is an entry point into a `Model`.
 *
 * `InputLayer` is generated automatically for `Sequential` models by specifying
 * the `inputshape` or `batchInputShape` for the first layer.  It should not be
 * specified explicitly.
 *
 * ```js
 * // Define a model which simply adds two inputs.
 * const inputA = tf.input({shape: [3]});
 * const inputB = tf.input({shape: [3]});
 * const sum = tf.layers.add().apply([inputA, inputB]);
 * const model = tf.model({inputs: [inputA, inputB], outputs: sum});
 * const batchSize = 2;
 * model.predict([tf.ones([batchSize, 3]), tf.ones([batchSize, 3])]).print();
 * ```
 */
export class InputLayer extends Layer {
  static readonly className = 'InputLayer';
  sparse: boolean;
  constructor(config: InputLayerConfig) {
    super({
      dtype: config.dtype,
      name: config.name != null ? config.name : K.getUid('input').toString()
    });
    // Normalize config.batchSize and config.sparse
    if (config.batchSize == null) {
      config.batchSize = null;
    }
    if (config.sparse == null) {
      config.sparse = false;
    }

    this.trainable = false;
    this.built = true;
    this.sparse = config.sparse;

    if (config.inputShape != null && config.batchInputShape != null) {
      throw new ValueError(
          'Only provide the inputShape OR ' +
          'batchInputShape argument to inputLayer, not both at the same time.');
    }
    let batchInputShape = config.batchInputShape;
    if (batchInputShape == null) {
      if (config.inputShape == null) {
        throw new ValueError(
            'An InputLayer should be passed either a ' +
            '`batchInputShape` or an `inputShape`.');
      } else {
        batchInputShape = [config.batchSize].concat(config.inputShape);
      }
    } else {
      // TODO(michaelterry): Backport to PyKeras
      if (config.batchSize != null) {
        throw new ValueError(
            'Cannot specify batchSize if batchInputShape is' +
            'specified when creating an InputLayer.');
      }
    }

    const dtype = config.dtype || K.floatx();

    this.batchInputShape = batchInputShape;
    this.dtype = dtype;
    // TODO(michaelterry): Backport this to PyKeras?
    this.inputSpec = [{shape: batchInputShape}];

    const inputTensor = new SymbolicTensor(
        this.dtype, this.batchInputShape, this, [], {}, this.name);
    inputTensor.nodeIndex = 0;
    inputTensor.tensorIndex = 0;

    // Create an input node to add to this.outboundNode.
    // (This call has side effects.)
    // tslint:disable-next-line:no-unused-expression
    new Node({
      outboundLayer: this,
      inboundLayers: [],
      nodeIndices: [],
      tensorIndices: [],
      inputTensors: [inputTensor],
      outputTensors: [inputTensor],
      inputMasks: [null],
      outputMasks: [null],
      inputShapes: [batchInputShape],
      outputShapes: [batchInputShape]
    });
  }

  apply(
      inputs: Tensor|Tensor[]|SymbolicTensor|SymbolicTensor[],
      // tslint:disable-next-line:no-any
      kwargs?: any): Tensor|Tensor[]|SymbolicTensor {
    throw new ValueError(
        'Cannot pass any input to an ' +
        `InputLayer's apply() method. InputLayer name: ${this.name}`);
  }

  getConfig(): serialization.ConfigDict {
    return {
      batchInputShape: this.batchInputShape,
      dtype: this.dtype,
      sparse: this.sparse,
      name: this.name
    };
  }
}
serialization.SerializationMap.register(InputLayer);

/**
 * Config for the Input function.
 *
 * Note: You should provide only shape or batchShape (not both).
 * If only shape is provided, then the batchShape becomes
 * [null].concat(inputShape).
 */
export interface InputConfig {
  /**
   * A shape, not including the batch size. For instance, `shape=[32]`
   * indicates that the expected input will be batches of 32-dimensional
   * vectors.
   */
  shape?: Shape;
  /**
   * A shape tuple (integer), including the batch size. For instance,
   * `batchShape=[10, 32]` indicates that the expected input will be batches of
   * 10 32-dimensional vectors. `batchShape=[null, 32]` indicates batches of an
   * arbitrary number of 32-dimensional vectors.
   */
  batchShape?: Shape;
  /**
   * An optional name string for the layer. Should be unique in a model (do not
   * reuse the same name twice). It will be autogenerated if it isn't provided.
   */
  name?: string;
  dtype?: DType;
  /**
   * A boolean specifying whether the placeholder to be created is sparse.
   */
  sparse?: boolean;
}

/**
 * Used to instantiate an input to a model as a `SymbolicTensor`.
 *
 * Users should call the `input` factory function for
 * consistency with other generator functions.
 *
 * Example:
 *
 * ```js
 * // Defines a simple logistic regression model with 32 dimensional input
 * // and 3 dimensional output.
 * const x = tf.input({shape: [32]});
 * const y = tf.layers.dense({units: 3, activation: 'softmax'}).apply(x);
 * const model = tf.model({inputs: x, outputs: y});
 * model.predict(tf.ones([2, 32])).print();
 * ```
 *
 * Note: `input` is only necessary when using `model`. When using
 * `sequential`, specify `inputShape` for the first layer or use `inputLayer`
 * as the first layer.
 */
export function Input(config: InputConfig): SymbolicTensor {
  if (config.batchShape == null && config.shape == null) {
    throw new Error(
        'Please provide to Input either a `shape`' +
        ' or a `batchShape` argument. Note that ' +
        '`shape` does not include the batch ' +
        'dimension.');
  }
  if (config.batchShape != null && config.shape != null) {
    // TODO(michaelterry): Backport to PyKeras.
    throw new ValueError(
        'Please provide either a `shape` or `batchShape` ' +
        'argument to Input, but not both.');
  }
  let batchShape = config.batchShape;
  if (config.shape != null && batchShape == null) {
    batchShape = [null].concat(config.shape);
  }

  let dtype = config.dtype;
  if (dtype == null) {
    dtype = K.floatx();
  }

  const inputLayer = new InputLayer({
    batchInputShape: batchShape,
    name: config.name,
    dtype,
    sparse: config.sparse
  });

  const outputs = inputLayer.inboundNodes[0].outputTensors;
  return outputs[0];
}

/** Constructor config for Container. */
export interface ContainerConfig {
  inputs: SymbolicTensor|SymbolicTensor[];
  outputs: SymbolicTensor|SymbolicTensor[];
  name?: string;
}

/**
 * A Container is a directed acyclic graph of layers.
 *
 * It is the topological form of a "model". A Model
 * is simply a Container with added training routines.
 *
 */
export abstract class Container extends Layer {
  inputs: SymbolicTensor[];
  outputs: SymbolicTensor[];

  inputLayers: Layer[];
  inputLayersNodeIndices: number[];
  inputLayersTensorIndices: number[];

  outputLayers: Layer[];
  outputLayersNodeIndices: number[];
  outputLayersTensorIndices: number[];

  layers: Layer[];
  layersByDepth: {[depth: string]: Layer[]};
  nodesByDepth: {[depth: string]: Node[]};

  containerNodes = new Set<string>();

  // TODO(michaelterry): Add cache support
  // private outputMaskCache: any;
  // private outputTensorCache: any;
  // private outputShapeCache: any;

  inputNames: string[];
  outputNames: string[];
  feedInputShapes: Shape[];

  protected internalInputShapes: Shape[];
  protected internalOutputShapes: Shape[];
  // TODO(cais): Maybe 'feed' should not in the names of these variables,
  //   due to the fact that our backend is not symbolic.
  protected feedInputNames: string[];
  protected feedOutputNames: string[];

  constructor(config: ContainerConfig) {
    // No args passed to super's constructor.
    super({});
    this.name = config.name;
    if (this.name == null) {
      const prefix = this.getClassName().toLowerCase();
      this.name = K.getUid(prefix);
    }

    this.supportsMasking = false;
    this.trainable = true;
    this.updatable = true;

    // TODO(michaelterry): Initialize perInputLosses/Updates here.

    // Container-specific properties.
    if (Array.isArray(config.inputs)) {
      this.inputs = config.inputs.slice();
    } else {
      this.inputs = [config.inputs];
    }
    if (Array.isArray(config.outputs)) {
      this.outputs = config.outputs.slice();
    } else {
      this.outputs = [config.outputs];
    }

    // Check for redundancy in inputs.
    if (generic_utils.unique(this.inputs).length !== this.inputs.length) {
      throw new ValueError(
          'The list of inputs passed to the model is ' +
          'redundant. All inputs should only appear once. Found: ' +
          this.inputs.map(x => x.name));
    }

    // Check for redundancy in outputs.
    if (generic_utils.unique(this.outputs).length !== this.outputs.length) {
      console.warn(
          'The list of outputs passed to the model is redundant. ' +
          'All outputs should only appear once. Found: ' +
          this.outputs.map(x => x.name));
    }

    /*
      List of initial layers (1 to 1 mapping with this.inputs, hence the same
      layer might appear twice)
    */
    this.inputLayers = [];
    this.inputLayersNodeIndices = [];
    this.inputLayersTensorIndices = [];
    /*
      List of layers (1 to 1 mapping with this.outputs, hence the same layer
      might appear twice)
    */
    this.outputLayers = [];
    this.outputLayersNodeIndices = [];
    this.outputLayersTensorIndices = [];
    /*
      All layers in order of horizontal graph traversal. Entries are unique.
      Includes input and output layers.
    */
    this.layers = [];

    // TODO(michaelterry): Determine if caching still needed with eager
    // backend.
    /*
      This is for performance optimization when calling the Container on new
      inputs. Every time the Container is called on a set on input tensors,
      we compute the output tensors, output masks and output shapes in one pass,
      then cache them here. When one of these outputs is queried later,
      we retrieve it from there instead of recomputing it.
    */
    // this.outputTensorCache = {};
    // this.outputShapeCache = {};

    // Build this.outputLayers:
    for (const x of this.outputs) {
      const layer = x.sourceLayer;
      const nodeIndex = x.nodeIndex;
      const tensorIndex = x.tensorIndex;
      this.outputLayers.push(layer);
      this.outputLayersNodeIndices.push(nodeIndex);
      this.outputLayersTensorIndices.push(tensorIndex);
    }

    // TODO(michaelterry): Add output mask cache code.

    // Build this.inputLayers:
    for (const x of this.inputs) {
      const layer = x.sourceLayer;
      const nodeIndex = x.nodeIndex;
      const tensorIndex = x.tensorIndex;
      /*
        It's supposed to be an input layer, so only one node
        and one tensor output.
      */
      generic_utils.assert(nodeIndex === 0, 'input layer has >1 nodes');
      generic_utils.assert(tensorIndex === 0, 'input layer has >1 tensors');
      this.inputLayers.push(layer);
      this.inputLayersNodeIndices.push(nodeIndex);
      this.inputLayersTensorIndices.push(tensorIndex);
    }

    // Build this.inputNames and this.outputNames.
    this.inputNames = [];
    this.outputNames = [];
    this.feedInputShapes = [];
    this.feedInputNames = [];
    this.feedOutputNames = [];
    for (let i = 0; i < this.inputLayers.length; i++) {
      const layer = this.inputLayers[i];
      // Check that layer is an InputLayer.
      if (!(layer instanceof InputLayer)) {
        throw new TypeError(
            'Input layers to a Model must be InputLayer objects. ' +
            `Received inputs: ${config.inputs}. ` +
            `Input ${i} (0-based) originates ` +
            `from layer type ${layer.getClassName()}.`);
      }
      this.inputNames.push(layer.name);
      this.feedInputShapes.push(layer.batchInputShape);

      this.feedInputNames.push(layer.name);
    }
    for (const layer of this.outputLayers) {
      this.outputNames.push(layer.name);
    }

    this.internalInputShapes = this.inputs.map(x => x.shape);
    this.internalOutputShapes = this.outputs.map(x => x.shape);

    /*
      Container_nodes: set of nodes included in the graph (not all nodes
      included in the layers are relevant to the current graph).
    */
    // ids of all nodes relevant to the Container:
    const nodesDepths: {[nodeID: string]: number} = {};
    // To recover nodes from their ID.
    const nodeIDToNode: {[nodeID: string]: Node} = {};
    const layersDepths: {[layerID: string]: number} = {};
    // To layers from their ID.
    const layerIDToLayer: {[layerID: string]: Layer} = {};
    const layerIndices: {[layerID: string]: number} = {};
    const nodesInDecreasingDepth: Node[] = [];

    /**
     * Builds a map of the graph of layers.
     *
     * This recursively updates the map `layerIndices`,
     * the list `nodesInDecreasingDepth` and the set `containerNodes`.
     *
     * @param tensor Some tensor in a graph.
     * @param finishedNodes Set of nodes whose subgraphs have been traversed
     *         completely. Useful to prevent duplicated work.
     * @param nodesInProgress Set of nodes that are currently active on the
     *         recursion stack. Useful to detect cycles.
     * @param layer Layer from which `tensor` comes from. If not provided,
     *   will be obtained from tensor.sourceLayer.
     * @param nodeIndex Node index from which `tensor` comes from.
     * @param tensorIndex TensorIndex from which `tensor` comes from.
     *
     * @exception RuntimeError if a cycle is detected.
     */
    const buildMapOfGraph =
        (tensor: SymbolicTensor, finishedNodes: Node[], nodesInProgress: Node[],
         layer?: Layer, nodeIndex?: number, tensorIndex?: number) => {
          if (layer == null || nodeIndex == null || tensorIndex == null) {
            layer = tensor.sourceLayer;
            nodeIndex = tensor.nodeIndex;
            tensorIndex = tensor.tensorIndex;
          }
          const node = layer.inboundNodes[nodeIndex];

          // Prevent cycles.
          if (nodesInProgress.indexOf(node) !== -1) {
            throw new RuntimeError(
                `The tensor ${tensor.name} at layer "${layer.name}" ` +
                'is part of a cycle.');
          }

          // Don't repeat work for shared subgraphs
          if (finishedNodes.indexOf(node) !== -1) {
            return;
          }

          // Update containerNodes.
          this.containerNodes.add(Container.nodeKey(layer, nodeIndex));

          // Store the traversal order for layer sorting.
          if (!(layer.id in layerIndices)) {
            layerIndices[layer.id] = Object.keys(layerIndices).length;
          }

          if (nodesInProgress.indexOf(node) === -1) {
            nodesInProgress.push(node);
          }

          // Propagate to all previous tensors connected to this node.
          const numInboundLayers = node.inboundLayers.length;
          for (let i = 0; i < numInboundLayers; i++) {
            const x = node.inputTensors[i];
            const layer = node.inboundLayers[i];
            const nodeIndex = node.nodeIndices[i];
            const tensorIndex = node.tensorIndices[i];
            buildMapOfGraph(
                x, finishedNodes, nodesInProgress, layer, nodeIndex,
                tensorIndex);
          }
          finishedNodes.push(node);
          while (nodesInProgress.indexOf(node) >= 0) {
            nodesInProgress.splice(nodesInProgress.indexOf(node), 1);
          }
          nodesInDecreasingDepth.push(node);
        };

    const finishedNodes: Node[] = [];
    const nodesInProgress: Node[] = [];
    for (const x of this.outputs) {
      buildMapOfGraph(x, finishedNodes, nodesInProgress);
    }

    const reversedNodesInDecreasingDepth =
        nodesInDecreasingDepth.slice().reverse();
    for (const node of reversedNodesInDecreasingDepth) {
      nodeIDToNode[node.id] = node;
      // If the depth is not set, the node has no outbound nodes (depth 0).
      if (!(node.id in nodesDepths)) {
        nodesDepths[node.id] = 0;
      }
      let depth = nodesDepths[node.id];

      // Update the depth of the corresponding layer
      const previousDepth =
          (layersDepths[node.outboundLayer.id] == null ?
               0 :
               layersDepths[node.outboundLayer.id]);

      /*
        If we've seen this layer before at a higher depth, we should use that
        depth instead of the node depth.  This is necessary for shared layers
        that have inputs at different depth levels in the graph.
      */
      depth = Math.max(depth, previousDepth);
      layersDepths[node.outboundLayer.id] = depth;
      layerIDToLayer[node.outboundLayer.id] = node.outboundLayer;
      nodesDepths[node.id] = depth;

      // Update the depth of inbound nodes.
      for (let i = 0; i < node.inboundLayers.length; i++) {
        const inboundLayer = node.inboundLayers[i];
        const nodeIndex = node.nodeIndices[i];
        const inboundNode = inboundLayer.inboundNodes[nodeIndex];
        const previousDepth =
            (nodesDepths[inboundNode.id] == null ? 0 :
                                                   nodesDepths[inboundNode.id]);
        nodesDepths[inboundNode.id] = Math.max(depth + 1, previousDepth);
        nodeIDToNode[inboundNode.id] = inboundNode;
      }
    }

    // Build a dict {depth: list of nodes with this depth}
    const nodesByDepth: {[depth: string]: Node[]} = {};
    for (const nodeID in nodesDepths) {
      const depth = nodesDepths[nodeID];
      if (!(depth in nodesByDepth)) {
        nodesByDepth[depth] = [];
      }
      nodesByDepth[depth].push(nodeIDToNode[nodeID]);
    }

    // Build a dict {depth: list of layers with this depth}
    const layersByDepth: {[depth: string]: Layer[]} = {};
    for (const layerID in layersDepths) {
      const depth = layersDepths[layerID];
      if (!(depth in layersByDepth)) {
        layersByDepth[depth] = [];
      }
      layersByDepth[depth].push(layerIDToLayer[layerID]);
    }

    // Get sorted list of layer depths.
    let depthKeys = Object.keys(layersByDepth)
                        .map(x => parseInt(x, 10))
                        .sort(generic_utils.reverseNumberCompare);

    // Set this.layers and this.layersByDepth.
    this.layers = [];
    for (const depth of depthKeys) {
      const layersForDepth = layersByDepth[depth];
      // Container.layers needs to have a deterministic order:
      // here we order them by traversal order.
      layersForDepth.sort((a, b) => {
        const aIndex = layerIndices[a.id];
        const bIndex = layerIndices[b.id];
        if (aIndex < bIndex) {
          return -1;
        }
        if (aIndex > bIndex) {
          return 1;
        }
        return 0;
      });
      for (const layer of layersForDepth) {
        this.layers.push(layer);
      }
    }
    this.layersByDepth = layersByDepth;

    // Get sorted list of node depths;
    depthKeys = Object.keys(nodesByDepth)
                    .map(x => parseInt(x, 10))
                    .sort(generic_utils.reverseNumberCompare);

    // Check that all tensors required are computable.
    // computable_tensors: all tensors in the graph
    // that can be computed from the inputs provided.
    const computableTensors = this.inputs.slice();

    // To provide a better error msg.
    const layersWithCompleteInput: string[] = [];
    for (const depth of depthKeys) {
      for (const node of nodesByDepth[depth]) {
        const layer = node.outboundLayer;
        if (layer != null) {
          for (const x of node.inputTensors) {
            if (computableTensors.indexOf(x) === -1) {
              throw new RuntimeError(
                  `Graph disconnected: cannot obtain value for tensor ${x}` +
                  ` at layer "${layer.name}". ` +
                  'The following previous layers were accessed without ' +
                  `issue: ${layersWithCompleteInput}`);
            }
          }
          for (const x of node.outputTensors) {
            computableTensors.push(x);
          }
          layersWithCompleteInput.push(layer.name);
        }
      }
    }

    // Set this.containerNodes and this.nodesByDepth.
    this.nodesByDepth = nodesByDepth;

    // Ensure name unicity, which will be crucial for serialization
    // (since serialized nodes refer to layers by their name).
    const allNames = this.layers.map(x => x.name);
    for (const name of allNames) {
      const numOccurrences = allNames.filter(x => x === name).length;
      if (numOccurrences !== 1) {
        throw new RuntimeError(
            `The name "${name}" is used ${numOccurrences} times ` +
            'in the model. All layer names should be unique. Layer names: ' +
            JSON.stringify(allNames));
      }
    }

    // Layer parameters.
    // The new container starts with a single inbound node
    // for its inputs, and no outbound nodes.
    // Will be appended to by future calls to apply().
    this.outboundNodes = [];
    // Will be appended to below, and by future calls to apply().
    this.inboundNodes = [];

    // Create the node linking internal inputs to internal outputs.
    // (This call has side effects.)
    // tslint:disable-next-line:no-unused-expression
    new Node({
      outboundLayer: this,
      inboundLayers: [],
      nodeIndices: [],
      tensorIndices: [],
      inputTensors: this.inputs,
      outputTensors: this.outputs,
      inputMasks: this.inputs.map(x => null),
      outputMasks: this.outputs.map(x => null),
      inputShapes: this.inputs.map(x => x.shape),
      outputShapes: this.outputs.map(x => x.shape)
    });
    this.built = true;
  }

  get trainableWeights(): LayerVariable[] {
    // Porting Note: This check below is to prevent errors where the
    //   _trainableWeights inherited from the parent class (Layer) gets
    //   inadvertently used.
    if (this._trainableWeights.length > 0) {
      throw new ValueError(
          'Container instance unexpectedly contains _trainableWeights.' +
          'The trainable weights of a Container are a union of the ' +
          'trainable weights of its consituent Layers. Its own ' +
          '_trainableWeights must remain an empty Array.');
    }

    if (!this.trainable) {
      return [];
    }
    let weights: LayerVariable[] = [];
    for (const layer of this.layers) {
      weights = weights.concat(layer.trainableWeights);
    }
    return weights;
  }

  get nonTrainableWeights(): LayerVariable[] {
    const weights: LayerVariable[] = [];
    for (const layer of this.layers) {
      weights.push(...layer.nonTrainableWeights);
    }
    if (!this.trainable) {
      const trainableWeights: LayerVariable[] = [];
      for (const layer of this.layers) {
        trainableWeights.push(...layer.trainableWeights);
      }
      return trainableWeights.concat(weights);
    }
    return weights;
  }

  get weights(): LayerVariable[] {
    return this.trainableWeights.concat(this.nonTrainableWeights);
  }

  /**
   * Loads all layer weights from a JSON object.
   *
   * Porting Note: HDF5 weight files cannot be directly loaded in JavaScript /
   *   TypeScript. The utility script at `scripts/pykeras.py` offers means
   *   to convert them into JSON strings compatible with this method.
   * Porting Note: TensorFlow.js Layers supports only loading by name currently.
   *
   * @param weightsJSON A JSON mapping weight names to weight values as nested
   *   arrays of numbers, or a `NamedTensorMap`, i.e., a JSON mapping weight
   *   names to `Tensor` objects.
   * @param skipMismatch Whether to skip loading of layers where there is a
   *   mismatch in the number of weights, or a mismatch in the shape of the
   *   weight (only valid when `by_name`=True).
   * @param isNamedTensorMap Whether the 1st argument (`weightsJSON`) is a
   *   `NamedTensorMap`.
   */
  loadWeights(
      weightsJSON: JsonDict|NamedTensorMap, skipMismatch = false,
      isNamedTensorMap = false) {
    // TODO(cais): Maybe the JsonDict support should be removed after serving
    //   weights from XHR is working. If so, the `loadWeightsFromJson` flag
    //   should be removed as well. (b/74015805)
    // TODO(cais): See if we can use smarter type resolution to avoid sending
    //   the type info as a separate arg (isNamedTensormap).
    if (isNamedTensorMap) {
      loadWeightsFromNamedTensorMap(weightsJSON as NamedTensorMap, this.layers);
    } else {
      loadWeightsFromJson(weightsJSON as JsonDict, this.layers, skipMismatch);
    }
  }

  /**
   * Util shared between different serialization methods.
   * @returns Model config with Keras version information added.
   */
  private updatedConfig(): serialization.ConfigDict {
    const theConfig = this.getConfig();
    const modelConfig: serialization.ConfigDict = {
      className: this.getClassName(),
      config: theConfig,
      kerasVersion: `tfjs-layers ${layersVersion}`,
      // TODO(nielsene): Replace something like K.backend() once
      // possible.
      backend: 'TensorFlow.js'
    };
    return modelConfig;
  }

  /**
   * Returns a JSON string containing the network configuration.
   *
   * To load a network from a JSON save file, use
   * models.modelFromJSON(jsonString);
   * @param extraJsonArgs Unused in tfjs-layers, maintained for PyKeras
   * @param returnString Whether the return value should be stringified
   *    (default: `true`).
   * @returns a JSON string if `returnString` (default), or a JSON object if
   *   `!returnString`.
   */
  // tslint:disable-next-line:no-any
  toJSON(unused?: any, returnString = true): string|JsonDict {
    const modelConfig = convertTsToPythonic(this.updatedConfig()) as JsonDict;
    return returnString ? JSON.stringify(modelConfig) : modelConfig;
  }

  /**
   * Call the model on new inputs.
   *
   * In this case `call` just reapplies all ops in the graph to the new inputs
   * (e.g. build a new computational graph from the provided inputs).
   *
   * @param inputs A tensor or list of tensors.
   * @param mask A mask or list of masks. A mask can be either a tensor or null
   *   (no mask).
   *
   * @return A tensor if there is a single output, or a list of tensors if there
   *   are more than one outputs.
   */
  // tslint:disable-next-line:no-any
  call(inputs: Tensor|Tensor[], kwargs: any): Tensor|Tensor[] {
    inputs = generic_utils.toList(inputs);
    let masks: Tensor[];

    if ('mask' in kwargs) {
      masks = generic_utils.toList(kwargs['mask']);
    } else {
      masks = generic_utils.pyListRepeat(null, inputs.length);
    }
    // TODO(michaelterry): Add support for caching.
    return this.runInternalGraph(inputs, masks)[0];
  }

  /**
   * Computes an output mask tensor.
   *
   * @param inputs Tensor or list of tensors.
   * @param mask Tensor or list of tensors.
   *
   * @return null or a tensor (or list of tensors, one per output tensor of the
   * layer).
   */
  computeMask(inputs: Tensor|Tensor[], mask?: Tensor|Tensor[]): Tensor
      |Tensor[] {
    inputs = generic_utils.toList(inputs);
    let masks: Tensor[];
    if (mask == null) {
      masks = generic_utils.pyListRepeat(null, inputs.length);
    } else {
      masks = generic_utils.toList(mask);
    }
    // TODO(michaelterry): Add support for mask caching.
    return this.runInternalGraph(inputs, masks)[1];
  }

  /**
   * Computes the output shape of the layer.
   *
   * Assumes that the layer will be built to match that input shape provided.
   *
   * @param inputShape A shape (tuple of integers) or a list of shape tuples
   *   (one per output tensor of the layer). Shape tuples can include null for
   *   free dimensions, instead of an integer.
   */
  computeOutputShape(inputShape: Shape|Shape[]): Shape|Shape[] {
    const inputShapes = generic_utils.normalizeShapeList(inputShape);
    if (inputShapes.length !== this.inputLayers.length) {
      throw new ValueError(
          `Invalid inputShape argument ${inputShape}: ` +
          `model has ${this.inputLayers.length} tensor inputs.`);
    }

    // TODO(michaelterry): Add caching
    const layersToOutputShapes: {[shapeKey: string]: Shape} = {};
    for (let i = 0; i < inputShapes.length; i++) {
      const layer = this.inputLayers[i];
      const inputShape = inputShapes[i];
      // It's an input layer: computeOutputShape is identity,
      // and there is only one node and one tensor output.
      const shapeKey = layer.name + '_0_0';
      layersToOutputShapes[shapeKey] = inputShape;
    }

    const depthKeys = Object.keys(this.nodesByDepth)
                          .map(x => parseInt(x, 10))
                          .sort(generic_utils.reverseNumberCompare);
    // Iterate over nodes, by depth level.
    if (depthKeys.length > 1) {
      for (const depth of depthKeys) {
        const nodes = this.nodesByDepth[depth];
        for (const node of nodes) {
          // This is always a single layer, never a list.
          const layer = node.outboundLayer;
          if (this.inputLayers.map(x => x.id).indexOf(layer.id) !== -1) {
            // We've already covered the input layers a few lines above.
            continue;
          }
          // Potentially redundant list, same size of node.inputTensors.
          const inputShapes: Shape[] = [];
          for (let j = 0; j < node.inboundLayers.length; j++) {
            const inboundLayer = node.inboundLayers[j];
            const nodeIndex = node.nodeIndices[j];
            const tensorIndex = node.tensorIndices[j];
            const shapeKey = `${inboundLayer.name}_${nodeIndex}_${tensorIndex}`;
            const inputShape = layersToOutputShapes[shapeKey];
            inputShapes.push(inputShape);
          }

          const outputShape = layer.computeOutputShape(
              generic_utils.singletonOrArray(inputShapes));

          const outputShapes = generic_utils.normalizeShapeList(outputShape);
          const nodeIndex = layer.inboundNodes.indexOf(node);
          for (let j = 0; j < outputShapes.length; j++) {
            const shapeKey = `${layer.name}_${nodeIndex}_${j}`;
            layersToOutputShapes[shapeKey] = outputShapes[j];
          }
        }
      }
    }

    // Read final output shapes from layersToOutputShapes.
    const outputShapes: Shape[] = [];
    const outputShapeKeys: string[] = [];
    for (let i = 0; i < this.outputLayers.length; i++) {
      const layer = this.outputLayers[i];
      const nodeIndex = this.outputLayersNodeIndices[i];
      const tensorIndex = this.outputLayersTensorIndices[i];
      const shapeKey = `${layer.name}_${nodeIndex}_${tensorIndex}`;
      outputShapeKeys.push(shapeKey);
    }

    for (let i = 0; i < outputShapeKeys.length; i++) {
      const key = outputShapeKeys[i];
      generic_utils.assert(key in layersToOutputShapes);
      outputShapes.push(layersToOutputShapes[key]);
    }

    // TODO(michaelterry): Update cache
    return generic_utils.singletonOrArray(outputShapes);
  }

  /**
   * Computes output tensors for new inputs.
   *
   * Note:
   *   - Expects `inputs` to be a list (potentially with 1 element).
   *
   * @param inputs List of tensors
   * @param masks List of masks (tensors or null).
   * @return Three lists: outputTensors, outputMasks, outputShapes
   */
  protected runInternalGraph(inputs: Tensor[], masks?: Tensor[]):
      [Tensor[], Tensor[], Shape[]] {
    if (masks == null) {
      masks = generic_utils.pyListRepeat(null, inputs.length);
    }

    // Dictionary mapping reference tensors to tuples
    // (computed tensor, compute mask)
    // we assume a 1:1 mapping from tensor to mask
    // TODO: raise exception when a `.computeMask()` call
    // does not return a list the same size as `call`
    const tensorMap: {[tensorID: string]: [Tensor, Tensor]} = {};
    for (let i = 0; i < this.inputs.length; ++i) {
      const x = this.inputs[i];
      const y = inputs[i];
      const mask = masks[i];
      tensorMap[x.id] = [y, mask];
    }

    const depthKeys = Object.keys(this.nodesByDepth)
                          .map(x => parseInt(x, 10))
                          .sort(generic_utils.reverseNumberCompare);
    for (const depth of depthKeys) {
      const nodes = this.nodesByDepth[depth];
      for (const node of nodes) {
        // This is always a single layer, never a list.
        const layer = node.outboundLayer;
        const referenceInputTensors = node.inputTensors;
        const referenceOutputTensors = node.outputTensors;

        // If all previous input tensors are available in tensorMap,
        // then call node.inboundLayer on them.
        // List of tuples [input, mask]:
        const computedData = new Array<[Tensor, Tensor]>();
        for (const x of referenceInputTensors) {
          if (x.id in tensorMap) {
            computedData.push(tensorMap[x.id]);
          }
        }
        if (computedData.length === referenceInputTensors.length) {
          // TODO(michaelterry): Add K.name_scope here, if we need it.
          // tslint:disable-next-line:no-any
          let kwargs: any = {};
          let computedTensors: Tensor[];
          let computedMasks: Tensor[];
          let outputTensors: Tensor[];
          let outputMasks: Tensor[];
          // call layer
          if (node.callArgs != null) {
            kwargs = node.callArgs;
          }
          if (computedData.length === 1) {
            const [computedTensor, computedMask] = computedData[0];
            if (kwargs.mask == null) {
              kwargs['mask'] = computedMask;
            }
            outputTensors =
                generic_utils.toList(layer.call(computedTensor, kwargs));
            outputMasks = generic_utils.toList(
                layer.computeMask(computedTensor, computedMask));
            computedTensors = [computedTensor];
            computedMasks = [computedMask];
          } else {
            computedTensors = computedData.map(x => x[0]);
            computedMasks = computedData.map(x => x[1]);
            if (kwargs.mask == null) {
              kwargs['mask'] = computedMasks;
            }
            outputTensors =
                generic_utils.toList(layer.call(computedTensors, kwargs));
            outputMasks = generic_utils.toList(
                layer.computeMask(computedTensors, computedMasks));
          }

          if (layer.activityRegularizer) {
            throw new NotImplementedError(
                'Model invocation with concrete Tensor value(s) in the ' +
                'presence of activity regularizer(s) is not supported yet.');
          }
          // TODO(michaelterry): Add model updates and losses

          // Update tensor map.
          for (let i = 0; i < referenceOutputTensors.length; ++i) {
            const x = referenceOutputTensors[i];
            const y = outputTensors[i];
            const mask = outputMasks[i];
            tensorMap[x.id] = [y, mask];
          }
        }
      }
    }

    const outputTensors: Tensor[] = [];
    const outputMasks: Tensor[] = [];
    const outputShapes: Shape[] = [];
    for (const x of this.outputs) {
      generic_utils.assert(
          x.id in tensorMap, `Could not compute output ${x.name} : ${x.id}`);
      const [tensor, mask] = tensorMap[x.id];
      outputShapes.push(tensor.shape);
      outputTensors.push(tensor);
      outputMasks.push(mask);
    }

    // TODO(michaelterry): Add support for caches.
    return [outputTensors, outputMasks, outputShapes];
  }

  /**
   * Builds a map of internal node keys to node ordering.
   * Used in serializaion a node orderings may change as unused nodes are
   * dropped. Porting Note:  This helper method was pulled out of getConfig to
   * improve readability.
   * @param layers An array of Layers in the model.
   * @returns Map of Node Keys to index order within the layer.
   */
  private buildNodeConversionMap(layers: Layer[]): {[nodeKey: string]: number} {
    const nodeConversionMap: {[nodeKey: string]: number} = {};
    let keptNodes: number;
    for (const layer of this.layers) {
      keptNodes = layer instanceof Container ? 1 : 0;
      for (let originalNodeIndex = 0;
           originalNodeIndex < layer.inboundNodes.length; originalNodeIndex++) {
        const nodeKey = Container.nodeKey(layer, originalNodeIndex);
        if (nodeKey in this.containerNodes) {
          // i.e. we mark it to be saved
          nodeConversionMap[nodeKey] = keptNodes;
          keptNodes += 1;
        }
      }
    }
    return nodeConversionMap;
  }

  /**
   * Retrieves a layer based on either its name (unique) or index.
   *
   * Indices are based on order of horizontal graph traversal (bottom-up).
   *
   * If both `name` and `index` are specified, `index` takes precedence.
   *
   * @param name Name of layer.
   * @param index Index of layer.
   * @returns A Layer instance.
   * @throws ValueError: In case of invalid layer name or index.
   */
  @doc({
    heading: 'Layers',
    subheading: 'Classes',
    namespace: 'layers',
    subclasses: ['Model']
  })
  getLayer(name?: string, index?: number): Layer {
    if (index != null) {
      if (this.layers.length <= index) {
        throw new ValueError(
            `Was asked to retrieve layer at index ${index}, but model only ` +
            `has ${this.layers.length} layer(s).`);
      } else {
        return this.layers[index];
      }
    } else {
      if (name == null) {
        throw new ValueError('Provide either a layer name or layer index');
      }
    }

    for (const layer of this.layers) {
      if (layer.name === name) {
        return layer;
      }
    }
    throw new ValueError(`No such layer: ${name}`);
  }

  /**
   * Retrieves the Container's current loss values.
   *
   * Used for regularizers during training.
   */
  calculateLosses(): Scalar[] {
    // Porting Node: This is an augmentation to Container.loss in PyKeras.
    //   In PyKeras, Container.loss returns symbolic tensors. Here a concrete
    //   Tensor (specifically Scalar) values are returned. This is due to the
    //   imperative backend.
    return tidy(() => {
      const losses: Scalar[] = [];
      for (const layer of this.layers) {
        for (let nodeIndex = 0; nodeIndex < layer.inboundNodes.length;
             ++nodeIndex) {
          const nodeKey = Container.nodeKey(layer, nodeIndex);
          if (this.containerNodes.has(nodeKey)) {
            losses.push(...layer.calculateLosses());
          }
        }
      }
      // TODO(cais): Add any unconditional model-level losses?
      return losses;
    });
  }

  getConfig(): serialization.ConfigDict {
    const config: serialization.ConfigDict = {name: this.name};

    // Build a map from layer unique name (self._node_key)
    // to the index of the nodes that are saved in the config.
    // Only nodes in container_nodes are saved.
    const nodeConversionMap: {[nodeKey: string]: number} =
        this.buildNodeConversionMap(this.layers);

    // Serialize and save the layers in layerConfigs
    const layerConfigs = [];
    for (const layer of this.layers) {
      const layerClassName = layer.getClassName();
      const layerConfig = layer.getConfig();
      const filteredInboundNodes = [];
      for (let originalNodeIndex = 0;
           originalNodeIndex < layer.inboundNodes.length; originalNodeIndex++) {
        const node = layer.inboundNodes[originalNodeIndex];
        const nodeKey = Container.nodeKey(layer, originalNodeIndex);
        let kwargs = {};
        if (this.containerNodes.has(nodeKey)) {
          // The node is relevant to the model:
          // add to filteredInboundNodes.
          if (node.callArgs) {
            const testString = JSON.stringify(node.callArgs);
            if (testString.indexOf('undefined') === -1) {
              kwargs = node.callArgs;
            } else {
              console.warn(
                  `Layer ${layer.name} was passed ` +
                  `non-serializable keyword arguments: ` +
                  `${node.callArgs}. They will not be included ` +
                  `in the serialized model (and thus will be ` +
                  `missing at deserialization time).`);
              kwargs = {};
            }
          }
          if (node.inboundLayers.length > 0) {
            const nodeData = [];
            for (let i = 0; i < node.inboundLayers.length; i++) {
              const inboundLayer = node.inboundLayers[i];
              const nodeIndex = node.nodeIndices[i];
              const tensorIndex = node.tensorIndices[i];
              const nodeKey = Container.nodeKey(inboundLayer, nodeIndex);
              let newNodeIndex = nodeConversionMap[nodeKey];
              if (newNodeIndex === null || newNodeIndex === undefined) {
                newNodeIndex = 0;
              }
              nodeData.push(
                  [inboundLayer.name, newNodeIndex, tensorIndex, kwargs]);
            }
            filteredInboundNodes.push(nodeData);
          }
        }
      }
      layerConfigs.push({
        name: layer.name,
        className: layerClassName,
        config: layerConfig,
        inboundNodes: filteredInboundNodes
      });
    }
    config['layers'] = layerConfigs;
    // Gather info about inputs and outputs
    const modelInputs = [];
    for (let i = 0; i < this.inputLayers.length; i++) {
      const layer = this.inputLayers[i];
      const nodeIndex = this.inputLayersNodeIndices[i];

      const nodeKey = Container.nodeKey(layer, nodeIndex);
      if (!this.containerNodes.has(nodeKey)) {
        continue;
      }
      let newNodeIndex = nodeConversionMap[nodeKey];
      if (newNodeIndex === null || newNodeIndex === undefined) {
        newNodeIndex = 0;
      }
      const tensorIndex = this.inputLayersTensorIndices[i];
      modelInputs.push([layer.name, newNodeIndex, tensorIndex]);
    }
    config['inputLayers'] = modelInputs;

    const modelOutputs = [];
    for (let i = 0; i < this.outputLayers.length; i++) {
      const layer = this.outputLayers[i];
      const nodeIndex = this.outputLayersNodeIndices[i];

      const nodeKey = Container.nodeKey(layer, nodeIndex);
      if (!this.containerNodes.has(nodeKey)) {
        continue;
      }
      let newNodeIndex = nodeConversionMap[nodeKey];
      if (newNodeIndex === null || newNodeIndex === undefined) {
        newNodeIndex = 0;
      }
      const tensorIndex = this.outputLayersTensorIndices[i];
      modelOutputs.push([layer.name, newNodeIndex, tensorIndex]);
    }
    config['outputLayers'] = modelOutputs;
    return config;
  }

  /**
   * Instantiates a Model from its config (output of `get_config()`).
   * @param cls: the class to create
   * @param config: Model config dictionary.
   * @returns A model instance.
   * @throws ValueError: In case of improperly formatted config dict.
   */
  static fromConfig<T extends serialization.Serializable>(
      cls: serialization.SerializableConstructor<T>,
      config: serialization.ConfigDict): T {
    // Layer instances created during
    // the graph reconstruction process
    const createdLayers: {[layerName: string]: Layer} = {};

    // Dictionary mapping layer instances to
    // node data that specifies a layer call.
    // It acts as a queue that maintains any unprocessed
    // layer call until it becomes possible to process it
    // (i.e. until the input tensors to the call all exist).
    const unprocessedNodes:
        {[layer: string]: serialization.ConfigDict[][]} = {};
    function addUnprocessedNode(
        layer: Layer, nodeData: serialization.ConfigDict[]) {
      if (!(layer.name in unprocessedNodes)) {
        unprocessedNodes[layer.name] = [nodeData];
      } else {
        unprocessedNodes[layer.name].push(nodeData);
      }
    }

    function processNode(layer: Layer, nodeData: serialization.ConfigDict[]) {
      const inputTensors: SymbolicTensor[] = [];
      let kwargs;
      for (const inputData of nodeData) {
        const inboundLayerName = inputData[0] as string;
        const inboundNodeIndex = inputData[1] as number;
        const inboundTensorIndex = inputData[2] as number;
        if (inputData.length === 3) {
          kwargs = {};
        } else if (inputData.length === 4) {
          kwargs = inputData[3] as serialization.ConfigDict;
        } else {
          throw new ValueError(`Improperly formatted model config for layer ${
              JSON.stringify(layer)}: ${JSON.stringify(inputData)}`);
        }
        if (!(inboundLayerName in createdLayers)) {
          addUnprocessedNode(layer, nodeData);
          return;
        }
        const inboundLayer = createdLayers[inboundLayerName];
        if (inboundLayer.inboundNodes.length <= inboundNodeIndex) {
          addUnprocessedNode(layer, nodeData);
          return;
        }
        const inboundNode = inboundLayer.inboundNodes[inboundNodeIndex];
        inputTensors.push(inboundNode.outputTensors[inboundTensorIndex]);
      }
      // Call layer on its inputs, thus creating the node
      // and building the layer if needed.
      // Note: This has Eager vs Graph Implications.
      if (inputTensors.length > 0) {
        layer.apply(
            generic_utils.singletonOrArray(inputTensors),
            kwargs);  // was ** kwargs
      }
    }

    /**
     * Deserialize a layer, then call it on appropriate inputs.
     * @param layerData: layer config dict.
     * @throws ValueError: In case of improperly formatted `layer_data`
     * dict.
     */
    function processLayer(layerData: serialization.ConfigDict|null) {
      const layerName = layerData.name as string;
      // Instantiate layer.
      const layer = deserializeLayer(
                        layerData,
                        config.customObjects != null ?
                            config.customObjects as serialization.ConfigDict :
                            {}) as Layer;
      createdLayers[layerName] = layer;
      // Gather layer inputs.
      const inboundNodesData =
          layerData.inboundNodes as serialization.ConfigDict[];
      for (const nodeData of inboundNodesData) {
        if (!(nodeData instanceof Array)) {
          throw new ValueError(
              `Corrupted configuration, expected array for nodeData: ${
                  nodeData}`);
        }
        // We don't process nodes (i.e. make layer calls)
        // on the fly because the inbound node may not yet exist,
        // in case of layer shared at different topological depths
        // (e.g.a model such as A(B(A(B(x)))))
        addUnprocessedNode(layer, nodeData);
      }
    }

    // First, we create all layers and enqueue nodes to be processed
    const name = config.name;
    const layersFromConfig = config.layers as serialization.ConfigDict[];
    for (const layerData of layersFromConfig) {
      processLayer(layerData);
    }

    // Then we process nodes in order of layer depth.
    // Nodes that cannot yet be processed(if the inbound node
    // does not yet exist) are re - enqueued, and the process
    // is repeated until all nodes are processed.
    while (!generic_utils.isObjectEmpty(unprocessedNodes)) {
      for (const layerData of layersFromConfig) {
        const layer = createdLayers[layerData.name as string];
        if (layer.name in unprocessedNodes) {
          for (const nodeData of unprocessedNodes[layer.name]) {
            processNode(layer, nodeData);
          }
          delete unprocessedNodes[layer.name];
        }
      }
    }
    const inputTensors: SymbolicTensor[] = [];
    const outputTensors: SymbolicTensor[] = [];
    const inputLayersFromConfig =
        config.inputLayers as serialization.ConfigDict[];
    for (const layerData of inputLayersFromConfig) {
      const layerName = layerData[0] as string;
      const nodeIndex = layerData[1] as number;
      const tensorIndex = layerData[2] as number;
      generic_utils.assert(layerName in createdLayers);
      const layer = createdLayers[layerName];
      const layerOutputTensors = layer.inboundNodes[nodeIndex].outputTensors;
      inputTensors.push(layerOutputTensors[tensorIndex]);
    }
    const outputLayersFromConfig =
        config.outputLayers as serialization.ConfigDict[];
    for (const layerData of outputLayersFromConfig) {
      const layerName = layerData[0] as string;
      const nodeIndex = layerData[1] as number;
      const tensorIndex = layerData[2] as number;
      generic_utils.assert(layerName in createdLayers);
      const layer = createdLayers[layerName];
      const layerOutputTensors = layer.inboundNodes[nodeIndex].outputTensors;
      outputTensors.push(layerOutputTensors[tensorIndex]);
    }
    return new cls({inputs: inputTensors, outputs: outputTensors, name});
  }

  /**
   * Determine whether the container is stateful.
   *
   * Porting Note: this is the equivalent of the stateful @property of
   *   the Container class in PyKeras.
   */
  get stateful(): boolean {
    // Porting Note: This check is to prevent inadvertent setting of the
    //   _stateful property of the Container instance.
    if (this._stateful) {
      throw new ValueError(
          'Container instance unexpectedly has _stateful = true. The ' +
          'statefulness of a Container is determined by the Layers it ' +
          'contains. Its _stateful property must remain the default false.');
    }
    for (const layer of this.layers) {
      if (layer.stateful) {
        return true;
      }
    }
    return false;
  }
}

/**
 * Returns the list of input tensors necessary to compute `tensor`.
 *
 * Output will always be a list of tensors (potentially with 1 element).
 *
 * @param tensor The tensor to start from.
 * @param layer Origin layer of the tensor.
 * @param nodeIndex Origin node index of the tensor.
 *
 * @return Array of input tensors.
 */
export function getSourceInputs(
    tensor: SymbolicTensor, layer?: Layer,
    nodeIndex?: number): SymbolicTensor[] {
  if (layer == null || (nodeIndex != null && nodeIndex > 0)) {
    layer = tensor.sourceLayer;
    nodeIndex = tensor.nodeIndex;
  }
  if (layer.inboundNodes.length === 0) {
    return [tensor];
  } else {
    const node = layer.inboundNodes[nodeIndex];
    if (node.inboundLayers.length === 0) {
      return node.inputTensors;
    } else {
      const sourceTensors: SymbolicTensor[] = [];
      for (let i = 0; i < node.inboundLayers.length; i++) {
        const x = node.inputTensors[i];
        const layer = node.inboundLayers[i];
        const nodeIndex = node.nodeIndices[i];
        const previousSources = getSourceInputs(x, layer, nodeIndex);
        // Avoid input redundancy.
        for (const x of previousSources) {
          if (sourceTensors.indexOf(x) === -1) {
            sourceTensors.push(x);
          }
        }
      }
      return sourceTensors;
    }
  }
}

/**
 * Create an Tensor from info about dtype, shape and values.
 * @param dtype DType string.
 * @param shape Shape.
 * @param value Values of the array, as a scalar or nested Array of proper
 *   shape.
 * @returns An Tensor instance.
 */
// tslint:disable-next-line:no-any
function loadTensor(dtype: string, shape: Shape, value: any): Tensor {
  const dataType = generic_utils.stringToDType(dtype);
  return Tensor.make(
      shape, {values: shape.length === 0 ? value : util.flatten(value)},
      dataType);
}

/**
 * Converts layers weights to a format suitable for TensorFlow.js Layers.
 *
 * Porting Note: The function `preprocess_weights_for_loading()` in PyKeras
 * performs conversion from Keras 1 to Keras 2. But in TypeScript, we
 * require Keras version to be 2. Thus this conversion is not applicable. We
 * simply check the Keras version and pass the weights through.
 *
 * @param layer Layer instance.
 * @param weights Input weights.
 * @param originalKerasVersion Keras version for the weights.
 * @param originalBackend Keras backend the weights were trained with.
 * @returns Output weights as Tensors.
 */
function preprocessWeightsForLoading(
    layer: Layer, weights: LayerVariable[], originalKerasVersion?: string,
    originalBackend?: string): LayerVariable[] {
  if (!originalKerasVersion.startsWith('2.')) {
    throw new ValueError(
        'Unsupported Keras version in weights being loaded: ' +
        originalKerasVersion);
  }
  return weights;
}

/**
 * Load weights from a named tensor map.
 *
 * Porting Note: This is ported from the Python function
 *   load_weights_from_hdf5_group_by_name()
 *
 * @param weights The named tensor map mapping names of weights to weight
 *   values.
 * @param layers An array of target layers.
 */
export function loadWeightsFromNamedTensorMap(
    weights: NamedTensorMap, layers: Layer[]): void {
  // Make a dictionary mapping weight name to weight.
  const nameToWeight: {[name: string]: LayerVariable} = {};
  let totalWeightsCount = 0;
  for (const layer of layers) {
    for (const weight of layer.weights) {
      if (nameToWeight[weight.originalName] != null) {
        throw new ValueError(`Duplicate weight name: ${weight.originalName}`);
      }
      nameToWeight[weight.originalName] = weight;
      totalWeightsCount++;
    }
  }

  const weightValueTuples: Array<[LayerVariable, Tensor]> = [];
  for (const name in weights) {
    weightValueTuples.push([nameToWeight[name], weights[name]]);
    delete nameToWeight[name];
  }

  // Check that all weights are set.
  const unsetNames: string[] = [];
  for (const name in nameToWeight) {
    unsetNames.push(name);
  }
  if (unsetNames.length > 0) {
    throw new ValueError(
        `${unsetNames.length} of ${totalWeightsCount} weights are not set: ` +
        `${unsetNames}`);
  }

  K.batchSetValue(weightValueTuples);
}


// TODO(cais): Maybe remove the following (b/74015805).
/**
 * Load weights from a weights JSON object to an array of layers.
 *
 * Porting Note: This is ported from the Python function
 *   load_weights_from_hdf5_group_by_name()
 *
 * @param weightsJSON. The input JSON object represent the weights from a
 *   trained Keras model. See scripts/pykeras.py for more details.
 * @param layers An array of target layers.
 * @param skipMismatch Whether to skip loading of layers where there is a
 *   mismatch in the number of weights, or a mismatch in the shape of the
 *   weights.
 */
export function loadWeightsFromJson(
    weightsJSON: JsonDict, layers: Layer[], skipMismatch = false): void {
  const originalKerasVersion = weightsJSON['keras_version'] as string;
  const originalBackend = weightsJSON['backend'] as string;
  const layerNames = layers.map(layer => layer.name);

  // Reverse index of layer name to list of layers with name.
  const index: {[layerName: string]: Layer[]} = {};
  for (const layer of layers) {
    if (layer.name != null) {
      if (index[layer.name] == null) {
        index[layer.name] = [];
      }
      index[layer.name].push(layer);
    }
  }

  // tslint:disable-next-line:no-any
  const nameToWeights = weightsJSON['weights'] as {[name: string]: any};
  const weightValueTuples: Array<[LayerVariable, Tensor]> = [];
  for (let k = 0; k < layerNames.length; ++k) {
    const name = layerNames[k];
    let layerWeights = nameToWeights[name];
    if (layerWeights == null) {
      layerWeights = [];
    }

    let weightValues: LayerVariable[] = [];
    for (let n = 0; n < layerWeights.length; ++n) {
      // tslint:disable:no-any
      const weightEntry =
          layerWeights[n] as {[key: string]: string | Shape | any};
      // tslint:enable
      weightValues.push(new LayerVariable(loadTensor(
          weightEntry['dtype'], weightEntry['shape'] as Shape,
          weightEntry['value'])));
    }
    for (const layer of index[name]) {
      const symbolicWeights = layer.weights;
      weightValues = preprocessWeightsForLoading(
          layer, weightValues, originalKerasVersion, originalBackend);
      if (weightValues.length !== symbolicWeights.length) {
        if (skipMismatch) {
          console.warn(
              `Skipping loading of weights of layer ${layer.name} ` +
              `due to mismatch in number of weights: (${weightValues.length} ` +
              `vs ${symbolicWeights.length}).`);
        } else {
          throw new ValueError(
              `Layer #${k} (named "${layer.name}") expects ` +
              `${symbolicWeights.length} weight(s), but the saved weights ` +
              `have ${weightValues.length} element(s).`);
        }
      }

      // Set values.
      for (let i = 0; i < weightValues.length; ++i) {
        if (skipMismatch) {
          if (!util.arraysEqual(
                  symbolicWeights[i].shape, weightValues[i].shape)) {
            console.warn(
                `Skipping loading of weights for layer ${layer.name} due ` +
                `to mismatch in shape (${symbolicWeights[i].shape} vs ` +
                `${weightValues[i].shape})`);
            continue;
          }
        }
        weightValueTuples.push([symbolicWeights[i], weightValues[i].read()]);
      }
    }
  }
  K.batchSetValue(weightValueTuples);
}
