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

import {NamedTensorMap, Scalar, serialization, Tensor, tidy} from '@tensorflow/tfjs-core';

import {getUid} from '../backend/state';
import {NotImplementedError, RuntimeError, ValueError} from '../errors';
import {Shape} from '../keras_format/common';
import {TensorKeyWithArgsArray} from '../keras_format/node_config';
import {PyJsonDict} from '../keras_format/types';
import {deserialize as deserializeLayer} from '../layers/serialization';
import {Kwargs} from '../types';
import * as generic_utils from '../utils/generic_utils';
import {convertTsToPythonic} from '../utils/serialization_utils';
import * as types_utils from '../utils/types_utils';
import {batchSetValue, LayerVariable} from '../variables';
import {version as layersVersion} from '../version';

import {execute, FeedDict} from './executor';
import {InputLayer} from './input_layer';
import {DisposeResult, Layer, Node, SymbolicTensor} from './topology';

/** Constructor config for Container. */
export interface ContainerArgs {
  inputs: SymbolicTensor|SymbolicTensor[];
  outputs: SymbolicTensor|SymbolicTensor[];
  name?: string;
}

/**
 * A Container is a directed acyclic graph of layers.
 *
 * It is the topological form of a "model". A LayersModel
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

  constructor(args: ContainerArgs) {
    // No args passed to super's constructor.
    super({});
    this.name = args.name;
    if (this.name == null) {
      const prefix = this.getClassName().toLowerCase();
      this.name = getUid(prefix);
    }

    this.supportsMasking = false;
    this.trainable_ = true;
    this.updatable = true;

    // TODO(michaelterry): Initialize perInputLosses/Updates here.

    // Container-specific properties.
    if (Array.isArray(args.inputs)) {
      this.inputs = args.inputs.slice();
    } else {
      this.inputs = [args.inputs];
    }
    if (Array.isArray(args.outputs)) {
      this.outputs = args.outputs.slice();
    } else {
      this.outputs = [args.outputs];
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
      this.outputLayers.push(layer as Layer);
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
      this.inputLayers.push(layer as Layer);
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
            'Input layers to a LayersModel must be InputLayer objects. ' +
            `Received inputs: ${args.inputs}. ` +
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
            layer = tensor.sourceLayer as Layer;
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
    this._refCount = 1;  // The ref count of a container always start at 1.
  }

  protected assertNotDisposed() {
    if (this._refCount === 0) {
      throw new Error(`Container '${this.name}' is already disposed.`);
    }
  }

  /**
   * Attempt to dispose a LayersModel's weights.
   *
   * This method decrease the reference count of the LayersModel object by 1.
   *
   * A LayersModel is reference-counted. Its reference count is incremented by 1
   * when it is first constructed and when it is used as a Layer of another
   * LayersModel.
   *
   * If the reference count of a LayersModel becomes 0, the `dispose` method of
   * all its constituent `Layer`s will be called.
   *
   * Note: If the reference count is greater than 0 after the decrement, the
   * `dispose` method of its constituent `Layer`s will *not* be called.
   *
   * After a LayersModel is disposed, it cannot be used in calls such as
   * 'predict`, `evaluate` or `fit` anymore.
   *
   * @returns A DisposeResult Object with the following fields:
   *   - refCountAfterDispose: The reference count of the LayersModel after this
   *     `dispose()` call.
   *   - numDisposedVariables: Number of `tf.Variable`s (i.e., weights) disposed
   *     during this `dispose()` call.
   * @throws {Error} If the layer is not built yet, or if the LayersModel has
   *   already been disposed.
   */
  dispose(): DisposeResult {
    this.assertNotDisposed();
    const result:
        DisposeResult = {refCountAfterDispose: null, numDisposedVariables: 0};
    if (--this._refCount === 0) {
      for (const layer of this.layers) {
        result.numDisposedVariables += layer.dispose().numDisposedVariables;
      }
    }
    result.refCountAfterDispose = this._refCount;
    return result;
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
   * @param weights A JSON mapping weight names to weight values as nested
   *   arrays of numbers, or a `NamedTensorMap`, i.e., a JSON mapping weight
   *   names to `tf.Tensor` objects.
   * @param strict Require that the provided weights exactly match those
   *   required by the container.  Default: `true`.  Passing `false` means that
   *   extra weights and missing weights will be silently ignored.
   */
  loadWeights(weights: NamedTensorMap, strict = true) {
    const nameToWeight: {[name: string]: LayerVariable} = {};
    let totalWeightsCount = 0;
    for (const layer of this.layers) {
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
      if (nameToWeight[name] != null) {
        weightValueTuples.push([nameToWeight[name], weights[name]]);
      } else if (strict) {
        throw new ValueError(
            `Provided weight data has no target variable: ${name}`);
      }
      delete nameToWeight[name];
    }

    if (strict) {
      // Check that all weights are set.
      const unsetNames: string[] = [];
      for (const name in nameToWeight) {
        unsetNames.push(name);
      }
      if (unsetNames.length > 0) {
        throw new ValueError(
            `${unsetNames.length} of ${
                totalWeightsCount} weights are not set: ` +
            `${unsetNames}`);
      }
    }

    batchSetValue(weightValueTuples);
  }

  /**
   * Util shared between different serialization methods.
   * @returns LayersModel config with Keras version information added.
   */
  private updatedConfig(): serialization.ConfigDict {
    const theConfig = this.getConfig();
    const modelConfig: serialization.ConfigDict = {};
    modelConfig['className'] = this.getClassName();
    modelConfig['config'] = theConfig;
    modelConfig['kerasVersion'] = `tfjs-layers ${layersVersion}`;
    // TODO(nielsene): Replace something like K.backend() once
    // possible.
    modelConfig['backend'] = 'TensorFlow.js';
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
  toJSON(unused?: any, returnString = true): string|PyJsonDict {
    const modelConfig = convertTsToPythonic(this.updatedConfig()) as PyJsonDict;
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
  call(inputs: Tensor|Tensor[], kwargs: Kwargs): Tensor|Tensor[] {
    return tidy(() => {
      inputs = generic_utils.toList(inputs);
      const feedDict = new FeedDict();
      for (let i = 0; i < this.inputs.length; ++i) {
        feedDict.add(this.inputs[i], inputs[i]);
      }
      return execute(this.outputs, feedDict, kwargs) as Tensor | Tensor[];
    });
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
    return tidy(() => {
      inputs = generic_utils.toList(inputs);
      let masks: Tensor[];
      if (mask == null) {
        masks = generic_utils.pyListRepeat(null, inputs.length);
      } else {
        masks = generic_utils.toList(mask);
      }
      // TODO(michaelterry): Add support for mask caching.
      return this.runInternalGraph(inputs, masks)[1];
    });
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
    const inputShapes = types_utils.normalizeShapeList(inputShape);
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

          const outputShapes = types_utils.normalizeShapeList(outputShape);
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
          let kwargs: Kwargs = {};
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
            if (kwargs['mask'] == null) {
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
            if (kwargs['mask'] == null) {
              kwargs['mask'] = computedMasks;
            }
            outputTensors =
                generic_utils.toList(layer.call(computedTensors, kwargs));
            outputMasks = generic_utils.toList(
                layer.computeMask(computedTensors, computedMasks));
          }

          if (layer.activityRegularizer) {
            throw new NotImplementedError(
                'LayersModel invocation with concrete Tensor value(s) in the ' +
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
        if (this.containerNodes.has(nodeKey)) {
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
  /**
   * @doc {
   *    heading: 'Layers',
   *    subheading: 'Classes',
   *    namespace: 'layers',
   *    subclasses: ['LayersModel']
   * }
   */
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
            try {
              JSON.stringify(node.callArgs);
              kwargs = node.callArgs;
            } catch (err) {
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
              if (newNodeIndex == null) {
                newNodeIndex = 0;
              }
              nodeData.push(
                  [inboundLayer.name, newNodeIndex, tensorIndex, kwargs]);
            }
            filteredInboundNodes.push(nodeData);
          }
        }
      }
      const dict: serialization.ConfigDict = {};
      dict['name'] = layer.name;
      dict['className'] = layerClassName;
      dict['config'] = layerConfig;
      dict['inboundNodes'] = filteredInboundNodes;
      layerConfigs.push(dict);
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
   * Instantiates a LayersModel from its config (output of `get_config()`).
   * @param cls the class to create
   * @param config LayersModel config dictionary.
   * @param customObjects An optional dictionary of custom objects.
   * @param fastWeightInit Optional flag to use fast weight initialization
   *   during deserialization. This is applicable to cases in which
   *   the initialization will be immediately overwritten by loaded weight
   *   values. Default: `false`.
   * @returns A LayersModel instance.
   * @throws ValueError: In case of improperly formatted config dict.
   */
  /** @nocollapse */
  static fromConfig<T extends serialization.Serializable>(
      cls: serialization.SerializableConstructor<T>,
      config: serialization.ConfigDict,
      customObjects = {} as serialization.ConfigDict,
      fastWeightInit = false): T {
    // Layer instances created during
    // the graph reconstruction process
    const createdLayers: {[layerName: string]: Layer} = {};

    // Dictionary mapping layer instances to
    // node data that specifies a layer call.
    // It acts as a queue that maintains any unprocessed
    // layer call until it becomes possible to process it
    // (i.e. until the input tensors to the call all exist).
    const unprocessedNodes: {[layer: string]: TensorKeyWithArgsArray[][]} = {};
    function addUnprocessedNode(
        layer: Layer, nodeData: TensorKeyWithArgsArray[]) {
      if (!(layer.name in unprocessedNodes)) {
        unprocessedNodes[layer.name] = [nodeData];
      } else {
        unprocessedNodes[layer.name].push(nodeData);
      }
    }

    function processNode(layer: Layer, nodeData: TensorKeyWithArgsArray[]) {
      const inputTensors: SymbolicTensor[] = [];
      let kwargs;
      for (const inputData of nodeData) {
        const inboundLayerName = inputData[0];
        const inboundNodeIndex = inputData[1];
        const inboundTensorIndex = inputData[2];

        kwargs = inputData[3] == null ?
            {} :
            inputData[3] as serialization.ConfigDict;
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
      const layerName = layerData['name'] as string;
      // Instantiate layer.
      const layer =
          deserializeLayer(
              layerData,
              config['customObjects'] != null ?
                  config['customObjects'] as serialization.ConfigDict :
                  {}) as Layer;
      layer.setFastWeightInitDuringBuild(fastWeightInit);
      createdLayers[layerName] = layer;
      // Gather layer inputs.
      const inboundNodesData =
          layerData['inboundNodes'] as TensorKeyWithArgsArray[][];
      inboundNodesData.forEach(nodeData => {
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
      });
    }

    // First, we create all layers and enqueue nodes to be processed.
    const name = config['name'];
    const layersFromConfig = config['layers'] as serialization.ConfigDict[];
    for (const layerData of layersFromConfig) {
      processLayer(layerData);
    }

    // Then we process nodes in order of layer depth.
    // Nodes that cannot yet be processed(if the inbound node
    // does not yet exist) are re - enqueued, and the process
    // is repeated until all nodes are processed.
    while (!generic_utils.isObjectEmpty(unprocessedNodes)) {
      for (const layerData of layersFromConfig) {
        const layer = createdLayers[layerData['name'] as string];
        if (layer.name in unprocessedNodes) {
          const currentUnprocessedNodesForLayer = unprocessedNodes[layer.name];
          delete unprocessedNodes[layer.name];
          for (const nodeData of currentUnprocessedNodesForLayer) {
            processNode(layer, nodeData);
          }
        }
      }
    }

    const inputTensors: SymbolicTensor[] = [];
    const outputTensors: SymbolicTensor[] = [];
    const inputLayersFromConfig =
        config['inputLayers'] as serialization.ConfigDict[];
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
        config['outputLayers'] as serialization.ConfigDict[];
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

  /**
   * Reset the state of all stateful constituent layers (if any).
   *
   * Examples of stateful layers include RNN layers whose `stateful` property
   * is set as `true`.
   */
  resetStates() {
    tidy(() => {
      this.layers.forEach(layer => {
        // tslint:disable:no-any
        if (layer.stateful) {
          layer.resetStates();
        }
        // tslint:enable:no-any
      });
    });
  }
}
