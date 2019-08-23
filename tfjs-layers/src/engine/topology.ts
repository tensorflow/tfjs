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

import {DataType, Scalar, serialization, Tensor, tidy, util} from '@tensorflow/tfjs-core';

import {getNextUniqueTensorId, getUid} from '../backend/state';
import {getScopedTensorName, getUniqueTensorName, nameScope} from '../common';
import {Constraint} from '../constraints';
import {AttributeError, NotImplementedError, RuntimeError, ValueError} from '../errors';
import {getInitializer, Initializer} from '../initializers';
import {Shape} from '../keras_format/common';
import {Regularizer} from '../regularizers';
import {Kwargs, RegularizerFn} from '../types';
import * as generic_utils from '../utils/generic_utils';
import * as types_utils from '../utils/types_utils';
import * as variable_utils from '../utils/variable_utils';
import {batchGetValue, batchSetValue, LayerVariable} from '../variables';

// TODO(michaelterry): This is a stub until it's defined.
export type Op = (x: LayerVariable) => LayerVariable;

/**
 * Constructor arguments for InputSpec.
 */
export interface InputSpecArgs {
  /** Expected datatype of the input. */
  dtype?: DataType;
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
  dtype?: DataType;
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

  constructor(args: InputSpecArgs) {
    this.dtype = args.dtype;
    this.shape = args.shape;
    /*
      TODO(michaelterry): Could throw error if ndim and shape are both defined
        (then backport).
    */
    if (args.shape != null) {
      this.ndim = args.shape.length;
    } else {
      this.ndim = args.ndim;
    }
    this.maxNDim = args.maxNDim;
    this.minNDim = args.minNDim;
    this.axes = args.axes || {};
  }
}

/**
 * `tf.SymbolicTensor` is a placeholder for a Tensor without any concrete value.
 *
 * They are most often encountered when building a graph of `Layer`s for a
 * a `tf.LayersModel` and the input data's shape, but not values are known.
 */
/** @doc {heading: 'Models', 'subheading': 'Classes'} */
export class SymbolicTensor {
  /* A unique ID for the tensor to be able to differentiate tensors. */
  readonly id: number;
  // The fully scoped name of this Variable, including a unique suffix if needed
  readonly name: string;
  // The originally requested fully scoped name of this Variable, not including
  // any unique suffix.  This may be needed when restoring weights because this
  // original name is used as a key.
  readonly originalName?: string;
  /**
   * Rank/dimensionality of the tensor.
   */
  readonly rank: number;
  /**
   * Replacement for _keras_history.
   */
  nodeIndex: number;
  /**
   * Replacement for _keras_history.
   */
  tensorIndex: number;

  /**
   *
   * @param dtype
   * @param shape
   * @param sourceLayer The Layer that produced this symbolic tensor.
   * @param inputs The inputs passed to sourceLayer's __call__() method.
   * @param nodeIndex
   * @param tensorIndex
   * @param callArgs The keyword arguments passed to the __call__() method.
   * @param name
   * @param outputTensorIndex The index of this tensor in the list of outputs
   *   returned by apply().
   */
  constructor(
      readonly dtype: DataType, readonly shape: Shape,
      public sourceLayer: Layer, readonly inputs: SymbolicTensor[],
      readonly callArgs: Kwargs, name?: string,
      readonly outputTensorIndex?: number) {
    this.id = getNextUniqueTensorId();
    if (name != null) {
      this.originalName = getScopedTensorName(name);
      this.name = getUniqueTensorName(this.originalName);
    }
    this.rank = shape.length;
  }
}

/**
 * Constructor arguments for Node.
 */
export interface NodeArgs {
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

/**
 * The type of the return value of Layer.dispose() and Container.dispose().
 */
export interface DisposeResult {
  /**
   * Reference count after the dispose call.
   */
  refCountAfterDispose: number;

  /**
   * Number of variables dispose in this dispose call.
   */
  numDisposedVariables: number;
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
      args: NodeArgs,
      // TODO(michaelterry): Define actual type for this.
      public callArgs?: Kwargs) {
    this.id = _nextNodeID++;
    /*
      Layer instance (NOT a list).
      this is the layer that takes a list of input tensors
      and turns them into a list of output tensors.
      the current node will be added to
      the inboundNodes of outboundLayer.
    */
    this.outboundLayer = args.outboundLayer;

    /*
        The following 3 properties describe where
        the input tensors come from: which layers,
        and for each layer, which node and which
        tensor output of each node.
    */

    // List of layer instances.
    this.inboundLayers = args.inboundLayers;
    // List of integers, 1:1 mapping with inboundLayers.
    this.nodeIndices = args.nodeIndices;
    // List of integers, 1:1 mapping with inboundLayers.
    this.tensorIndices = args.tensorIndices;

    /*
        Following 2 properties:
        tensor inputs and outputs of outboundLayer.
    */

    // List of tensors. 1:1 mapping with inboundLayers.
    this.inputTensors = args.inputTensors;
    // List of tensors, created by outboundLayer.call().
    this.outputTensors = args.outputTensors;

    /*
        Following 2 properties: input and output masks.
        List of tensors, 1:1 mapping with inputTensor.
    */
    this.inputMasks = args.inputMasks;
    // List of tensors, created by outboundLayer.computeMask().
    this.outputMasks = args.outputMasks;

    // Following 2 properties: input and output shapes.

    // List of shape tuples, shapes of inputTensors.
    this.inputShapes = args.inputShapes;
    // List of shape tuples, shapes of outputTensors.
    this.outputShapes = args.outputShapes;

    // Add nodes to all layers involved.
    for (const layer of args.inboundLayers) {
      if (layer != null) {
        layer.outboundNodes.push(this);
      }
    }
    args.outboundLayer.inboundNodes.push(this);
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
export declare interface LayerArgs {
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
   * If `inputShape` is specified and `batchInputShape` is *not* specified,
   * `batchSize` is used to construct the `batchInputShape`: `[batchSize,
   * ...inputShape]`
   */
  batchSize?: number;
  /**
   * The data-type for this layer. Defaults to 'float32'.
   * This argument is only applicable to input layers (the first layer of a
   * model).
   */
  dtype?: DataType;
  /** Name for this layer. */
  name?: string;
  /**
   * Whether the weights of this layer are updatable by `fit`.
   * Defaults to true.
   */
  trainable?: boolean;
  /**
   * Initial weight values of the layer.
   */
  weights?: Tensor[];
  /** Legacy support. Do not use for new code. */
  inputDType?: DataType;
}

// If necessary, add `output` arguments to the CallHook function.
// This is currently used for testing only, but may be used for debugger-related
// purposes in the future.
export type CallHook = (inputs: Tensor|Tensor[], kwargs: Kwargs) => void;

let _nextLayerID = 0;

/**
 * A layer is a grouping of operations and weights that can be composed to
 * create a `tf.LayersModel`.
 *
 * Layers are constructed by using the functions under the
 * [tf.layers](#Layers-Basic) namespace.
 */
/** @doc {heading: 'Layers', subheading: 'Classes', namespace: 'layers'} */
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
  protected trainable_: boolean;
  batchInputShape: Shape;
  dtype: DataType;
  initialWeights: Tensor[];

  inboundNodes: Node[];
  outboundNodes: Node[];

  activityRegularizer: Regularizer;

  protected _trainableWeights: LayerVariable[];
  private _nonTrainableWeights: LayerVariable[];
  private _losses: RegularizerFn[];
  // TODO(cais): _updates is currently unused.
  private _updates: Tensor[];
  private _built: boolean;
  private _callHook: CallHook = null;

  private _addedWeightNames: string[] = [];

  readonly id: number;

  // Porting Notes: PyKeras does not have this property in this base Layer
  //   class. Instead lets Layer subclass set it dynamically and checks the
  //   value with `hasattr`. In tfjs-layers, we let this be a member of this
  //   base class.
  protected _stateful = false;

  protected _refCount: number|null;

  // A flag for whether fast (i.e., all-zero) weight initialization is to
  // be used during `build()` call. This speeds up weight initialization
  // by saving unnecessary calls to expensive initializers in cases where
  // the initialized values will be overwritten by loaded weight values
  // during model loading.
  private fastWeightInitDuringBuild: boolean;

  constructor(args: LayerArgs) {
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

    let name = args.name;
    if (!name) {
      const prefix = this.getClassName();
      name = generic_utils.toSnakeCase(prefix) + '_' + getUid(prefix);
    }
    this.name = name;

    this.trainable_ = args.trainable == null ? true : args.trainable;

    if (args.inputShape != null || args.batchInputShape != null) {
      /*
        In this case we will later create an input layer
        to insert before the current layer
       */
      let batchInputShape: Shape;
      if (args.batchInputShape != null) {
        batchInputShape = args.batchInputShape;
      } else if (args.inputShape != null) {
        let batchSize: number = null;
        if (args.batchSize != null) {
          batchSize = args.batchSize;
        }
        batchInputShape = [batchSize].concat(args.inputShape);
      }
      this.batchInputShape = batchInputShape;

      // Set dtype.
      let dtype = args.dtype;
      if (dtype == null) {
        dtype = args.inputDType;
      }
      if (dtype == null) {
        dtype = 'float32';
      }
      this.dtype = dtype;
    }

    if (args.weights != null) {
      this.initialWeights = args.weights;
    } else {
      this.initialWeights = null;
    }

    // The value of `_refCount` is initialized to null. When the layer is used
    // in a symbolic way for the first time, it will be set to 1.
    this._refCount = null;

    this.fastWeightInitDuringBuild = false;
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

  get updates(): Tensor[] {
    return this._updates;
  }

  get built(): boolean {
    return this._built;
  }

  set built(built: boolean) {
    this._built = built;
  }

  get trainable(): boolean {
    return this.trainable_;
  }

  set trainable(trainable: boolean) {
    this._trainableWeights.forEach(w => w.trainable = trainable);
    this.trainable_ = trainable;
  }

  get trainableWeights(): LayerVariable[] {
    if (this.trainable_) {
      return this._trainableWeights.filter(w => w.trainable);
    } else {
      return [];
    }
  }

  set trainableWeights(weights: LayerVariable[]) {
    this._trainableWeights = weights;
  }

  get nonTrainableWeights(): LayerVariable[] {
    if (this.trainable) {
      return this._trainableWeights.filter(w => !w.trainable)
          .concat(this._nonTrainableWeights);
    } else {
      return this._trainableWeights.concat(this._nonTrainableWeights);
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
   * Reset the states of the layer.
   *
   * This method of the base Layer class is essentially a no-op.
   * Subclasses that are stateful (e.g., stateful RNNs) should override this
   * method.
   */
  resetStates(): void {
    if (!this.stateful) {
      throw new Error(
          'Cannot call the resetStates() method of a non-stateful Layer ' +
          'object.');
    }
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
      const ndim = x.rank;
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
        if (x.dtype !== spec.dtype) {
          throw new ValueError(
              `Input ${inputIndex} is incompatible with layer ${this.name} ` +
              `: expected dtype=${spec.dtype}, found dtype=${x.dtype}.`);
        }
      }

      // Check specific shape axes.
      if (spec.axes) {
        const xShape = x.shape;
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
        for (let i = 0; i < spec.shape.length; ++i) {
          const specDim = spec.shape[i];
          const dim = x.shape[i];
          if (specDim != null && dim != null) {
            if (specDim !== dim) {
              throw new ValueError(
                  `Input ${inputIndex} is incompatible with layer ` +
                  `${this.name}: expected shape=${spec.shape}, ` +
                  `found shape=${x.shape}.`);
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
  call(inputs: Tensor|Tensor[], kwargs: Kwargs): Tensor|Tensor[] {
    return inputs;
  }

  protected invokeCallHook(inputs: Tensor|Tensor[], kwargs: Kwargs) {
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
   * When called with `tf.Tensor`(s), execute the `Layer`s computation and
   * return Tensor(s). For example:
   *
   * ```js
   * const denseLayer = tf.layers.dense({
   *   units: 1,
   *   kernelInitializer: 'zeros',
   *   useBias: false
   * });
   *
   * // Invoke the layer's apply() method with a `tf.Tensor` (with concrete
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
   * When called with `tf.SymbolicTensor`(s), this will prepare the layer for
   * future execution.  This entails internal book-keeping on shapes of
   * expected Tensors, wiring layers together, and initializing weights.
   *
   * Calling `apply` with `tf.SymbolicTensor`s are typically used during the
   * building of non-`tf.Sequential` models. For example:
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
   * @param inputs a `tf.Tensor` or `tf.SymbolicTensor` or an Array of them.
   * @param kwargs Additional keyword arguments to be passed to `call()`.
   *
   * @return Output of the layer's `call` method.
   *
   * @exception ValueError error in case the layer is missing shape information
   *   for its `build` call.
   */
  // Porting Note: This is a replacement for __call__() in Python.
  /** @doc {heading: 'Models', 'subheading': 'Classes'} */
  apply(
      inputs: Tensor|Tensor[]|SymbolicTensor|SymbolicTensor[],
      kwargs?: Kwargs): Tensor|Tensor[]|SymbolicTensor|SymbolicTensor[] {
    kwargs = kwargs || {};

    this.assertNotDisposed();

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
    return nameScope(this.name, () => {
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
          inputShapes.push(xElem.shape);
        }
        this.build(generic_utils.singletonOrArray(inputShapes));
        this.built = true;

        // Load weights that were specified at layer instantiation.
        if (this.initialWeights) {
          this.setWeights(this.initialWeights);
        }

        if (this._refCount === null && noneAreSymbolic) {
          // The first use of this layer is a non-symbolic call, set ref count
          // to 1 so the Layer can be properly disposed if its dispose() method
          // is called.
          this._refCount = 1;
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
            x = x.clone();
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
        this.warnOnIncompatibleInputShape(
            Array.isArray(inputs) ? inputShape[0] as Shape :
                                    inputShape as Shape);

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
        this._refCount++;

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
   * Check compatibility between input shape and this layer's batchInputShape.
   *
   * Print warning if any incompatibility is found.
   *
   * @param inputShape Input shape to be checked.
   */
  protected warnOnIncompatibleInputShape(inputShape: Shape) {
    if (this.batchInputShape == null) {
      return;
    } else if (inputShape.length !== this.batchInputShape.length) {
      console.warn(
          `The rank of the input tensor provided (shape: ` +
          `${JSON.stringify(inputShape)}) does not match that of the ` +
          `batchInputShape (${JSON.stringify(this.batchInputShape)}) ` +
          `of the layer ${this.name}`);
    } else {
      let dimMismatch = false;
      this.batchInputShape.forEach((dimension, i) => {
        if (dimension != null && inputShape[i] != null &&
            inputShape[i] !== dimension) {
          dimMismatch = true;
        }
      });
      if (dimMismatch) {
        console.warn(
            `The shape of the input tensor ` +
            `(${JSON.stringify(inputShape)}) does not ` +
            `match the expectation of layer ${this.name}: ` +
            `${JSON.stringify(this.batchInputShape)}`);
      }
    }
  }

  /**
   * Retrieves the output shape(s) of a layer.
   *
   * Only applicable if the layer has only one inbound node, or if all inbound
   * nodes have the same output shape.
   *
   * @returns Output shape or shapes.
   * @throws AttributeError: if the layer is connected to more than one incoming
   *   nodes.
   */
  /** @doc {heading: 'Models', 'subheading': 'Classes'} */
  get outputShape(): Shape|Shape[] {
    if (this.inboundNodes == null || this.inboundNodes.length === 0) {
      throw new AttributeError(
          `The layer ${this.name} has never been called and thus has no ` +
          `defined output shape.`);
    }
    const allOutputShapes: string[] = [];
    for (const node of this.inboundNodes) {
      const shapeString = JSON.stringify(node.outputShapes);
      if (allOutputShapes.indexOf(shapeString) === -1) {
        allOutputShapes.push(shapeString);
      }
    }
    if (allOutputShapes.length === 1) {
      const outputShapes = this.inboundNodes[0].outputShapes;
      if (Array.isArray(outputShapes) && Array.isArray(outputShapes[0]) &&
          outputShapes.length === 1) {
        return (outputShapes as Shape[])[0];
      } else {
        return outputShapes;
      }

    } else {
      throw new AttributeError(
          `The layer ${this.name} has multiple inbound nodes with different ` +
          `output shapes. Hence the notion of "outut shape" is ill-defined ` +
          `for the layer.`);
      // TODO(cais): Implement getOutputShapeAt().
    }
  }

  /**
   * Counts the total number of numbers (e.g., float32, int32) in the
   * weights.
   *
   * @returns An integer count.
   * @throws RuntimeError: If the layer is not built yet (in which case its
   *   weights are not defined yet.)
   */
  /** @doc {heading: 'Models', 'subheading': 'Classes'} */
  countParams(): number {
    if (!this.built) {
      throw new RuntimeError(
          `You tried to call countParams() on ${this.name}, ` +
          `but the layer is not built yet. Build it first by calling ` +
          `build(batchInputShape).`);
    }
    return variable_utils.countParamsInWeights(this.weights);
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
  /** @doc {heading: 'Models', 'subheading': 'Classes'} */
  build(inputShape: Shape|Shape[]) {
    this.built = true;
  }

  /**
   * Returns the current values of the weights of the layer.
   *
   * @param trainableOnly Whether to get the values of only trainable weights.
   * @returns Weight values as an `Array` of `tf.Tensor`s.
   */
  /** @doc {heading: 'Models', 'subheading': 'Classes'} */
  getWeights(trainableOnly = false): Tensor[] {
    return batchGetValue(trainableOnly ? this.trainableWeights : this.weights);
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
  /** @doc {heading: 'Models', 'subheading': 'Classes'} */
  setWeights(weights: Tensor[]): void {
    tidy(() => {
      const params = this.weights;
      if (params.length !== weights.length) {
        // TODO(cais): Restore the following and use `providedWeights`, instead
        // of `weights` in the error message, once the deeplearn.js bug is
        // fixed: https://github.com/PAIR-code/deeplearnjs/issues/498 const
        // providedWeights = JSON.stringify(weights).substr(0, 50);
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
      const paramValues = batchGetValue(params);
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
      batchSetValue(weightValueTuples);
    });
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
  /** @doc {heading: 'Models', 'subheading': 'Classes'} */
  protected addWeight(
      name: string, shape: Shape, dtype?: DataType, initializer?: Initializer,
      regularizer?: Regularizer, trainable?: boolean,
      constraint?: Constraint): LayerVariable {
    // Reject duplicate weight names.
    if (this._addedWeightNames.indexOf(name) !== -1) {
      throw new ValueError(
          `Duplicate weight name ${name} for layer ${this.name}`);
    }
    this._addedWeightNames.push(name);

    if (dtype == null) {
      dtype = 'float32';
    }

    if (this.fastWeightInitDuringBuild) {
      initializer = getInitializer('zeros');
    }
    const initValue = initializer.apply(shape, dtype);
    const weight =
        new LayerVariable(initValue, dtype, name, trainable, constraint);
    initValue.dispose();
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
   * Set the fast-weight-initialization flag.
   *
   * In cases where the initialized weight values will be immediately
   * overwritten by loaded weight values during model loading, setting
   * the flag to `true` saves unnecessary calls to potentially expensive
   * initializers and speeds up the loading process.
   *
   * @param value Target value of the flag.
   */
  setFastWeightInitDuringBuild(value: boolean) {
    this.fastWeightInitDuringBuild = value;
  }

  /**
   * Add losses to the layer.
   *
   * The loss may potentionally be conditional on some inputs tensors,
   * for instance activity losses are conditional on the layer's inputs.
   */
  /** @doc {heading: 'Models', 'subheading': 'Classes'} */
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
  /** @doc {heading: 'Models', 'subheading': 'Classes'} */
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
                  `Layer ${this.name} does not support masking, ` +
                  'but was passed an inputMask.');
            }
          });
        } else {
          throw new TypeError(
              `Layer ${this.name} does not support masking, ` +
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
    inputShapes = types_utils.normalizeShapeList(inputShapes);
    outputShapes = types_utils.normalizeShapeList(outputShapes);

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
  /** @doc {heading: 'Models', 'subheading': 'Classes'} */
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

  /**
   * Dispose the weight variables that this Layer instance holds.
   *
   * @returns {number} Number of disposed variables.
   */
  protected disposeWeights(): number {
    this.weights.forEach(weight => weight.dispose());
    return this.weights.length;
  }

  protected assertNotDisposed() {
    if (this._refCount === 0) {
      throw new Error(`Layer '${this.name}' is already disposed.`);
    }
  }

  /**
   * Attempt to dispose layer's weights.
   *
   * This method decrease the reference count of the Layer object by 1.
   *
   * A Layer is reference-counted. Its reference count is incremented by 1
   * the first item its `apply()` method is called and when it becomes a part
   * of a new `Node` (through calling the `apply()`) method on a
   * `tf.SymbolicTensor`).
   *
   * If the reference count of a Layer becomes 0, all the weights will be
   * disposed and the underlying memory (e.g., the textures allocated in WebGL)
   * will be freed.
   *
   * Note: If the reference count is greater than 0 after the decrement, the
   * weights of the Layer will *not* be disposed.
   *
   * After a Layer is disposed, it cannot be used in calls such as `apply()`,
   * `getWeights()` or `setWeights()` anymore.
   *
   * @returns A DisposeResult Object with the following fields:
   *   - refCountAfterDispose: The reference count of the Container after this
   *     `dispose()` call.
   *   - numDisposedVariables: Number of `tf.Variable`s (i.e., weights) disposed
   *     during this `dispose()` call.
   * @throws {Error} If the layer is not built yet, or if the layer has already
   *   been disposed.
   */
  /** @doc {heading: 'Models', 'subheading': 'Classes'} */
  dispose(): DisposeResult {
    if (!this.built) {
      throw new Error(
          `Cannot dispose Layer ${this.name} because it has not been ` +
          `built yet.`);
    }

    if (this._refCount === null) {
      throw new Error(
          `Cannot dispose Layer ${this.name} because it has not been used ` +
          `yet.`);
    }

    this.assertNotDisposed();

    let numDisposedVariables = 0;
    if (--this._refCount === 0) {
      numDisposedVariables = this.disposeWeights();
    }

    return {refCountAfterDispose: this._refCount, numDisposedVariables};
  }
}

/**
 * Collects the input shape(s) of a list of `tf.Tensor`s or
 * `tf.SymbolicTensor`s.
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
    shapes.push(x.shape);
  }
  return generic_utils.singletonOrArray(shapes);
}

/**
 * Guesses output dtype based on inputs.
 *
 * At present, just returns 'float32' for any input.
 *
 * @param inputTensors List of input tensors (or single input tensor).
 *
 * @return The guessed DType. At present, always returns 'float32'.
 */
function guessOutputDType(inputTensors: SymbolicTensor|SymbolicTensor[]|Tensor|
                          Tensor[]): DataType {
  return 'float32';
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
