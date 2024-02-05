/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import {DataType, env, keep, NamedTensorMap, Tensor, tidy, util} from '@tensorflow/tfjs-core';

import {ISignatureDef} from '../data/compiled_api';
import {NamedTensorsMap, TensorArrayMap, TensorInfo, TensorListMap} from '../data/types';
import {getNodeNameAndIndex, getParamValue, getTensor, getTensorsForCurrentContext, parseNodeName} from '../operations/executors/utils';
import {executeOp} from '../operations/operation_executor';
import {Graph, Node} from '../operations/types';

import {ExecutionContext, ExecutionContextInfo} from './execution_context';
import {getExecutionSubgraph, getNodeLiveUntilMap, getNodesInTopologicalOrder, isControlFlow} from './model_analysis';
import {ResourceManager} from './resource_manager';
import {FunctionExecutor} from './types';

interface NodeWithContexts {
  contexts: ExecutionContextInfo[];
  node: Node;
}

export class GraphExecutor implements FunctionExecutor {
  private compiledMap = new Map<string, ReturnType<typeof this.compile>>();
  private parseNodeNameCache = new Map<string, [string, number, string?]>();
  private _weightMap: NamedTensorsMap = {};
  private _weightIds: number[];
  private _signature: ISignatureDef;
  private _inputs: Node[];
  private _outputs: Node[];
  private _initNodes: Node[];  // Internal init nodes to start initialization.
  private SEPARATOR = ',';
  private _functions: {[key: string]: Graph} = {};
  private _functionExecutorMap: {[key: string]: FunctionExecutor} = {};
  private _resourceManager: ResourceManager;
  private clonedTensorsMap: NamedTensorsMap;
  private keepIntermediateTensors = false;

  get weightIds(): number[] {
    return this.parent ? this.parent.weightIds : this._weightIds;
  }

  get functionExecutorMap(): {[key: string]: FunctionExecutor} {
    return this.parent ? this.parent.functionExecutorMap :
                         this._functionExecutorMap;
  }

  get weightMap(): NamedTensorsMap {
    return this.parent ? this.parent.weightMap : this._weightMap;
  }

  set weightMap(weightMap: NamedTensorsMap) {
    const weightIds = Object.keys(weightMap).map(
        key => weightMap[key].map(tensor => tensor.id));
    this._weightIds = [].concat(...weightIds);
    this._weightMap = weightMap;
  }

  /**
   * Set `ResourceManager` shared by executors of a model.
   * @param resourceManager: `ResourceManager` of the `GraphModel`.
   */
  set resourceManager(resourceManager: ResourceManager) {
    this._resourceManager = resourceManager;
  }

  get inputs(): TensorInfo[] {
    return this._inputs.map(node => {
      return {
        name: node.name,
        shape: node.attrParams['shape'] ?
            node.attrParams['shape'].value as number[] :
            undefined,
        dtype: node.attrParams['dtype'] ?
            node.attrParams['dtype'].value as DataType :
            undefined
      };
    });
  }

  get outputs(): TensorInfo[] {
    return this._outputs.map(node => {
      return {
        name: node.name,
        shape: node.attrParams['shape'] ?
            node.attrParams['shape'].value as number[] :
            undefined,
        dtype: node.attrParams['dtype'] ?
            node.attrParams['dtype'].value as DataType :
            undefined
      };
    });
  }

  get inputNodes(): string[] {
    return this._inputs.map(node => node.signatureKey || node.name);
  }

  get outputNodes(): string[] {
    return this._outputs.map((node) => {
      const name = node.signatureKey || node.name;
      return node.defaultOutput ? (`${name}:${node.defaultOutput}`) : name;
    });
  }

  get functions(): {[key: string]: ISignatureDef} {
    return Object.keys(this._functions).reduce((map, key) => {
      map[key] = this._functions[key].signature;
      return map;
    }, {} as {[key: string]: ISignatureDef});
  }

  /**
   *
   * @param graph Graph the model or function graph to be executed.
   * @param parent When building function exector you need to set the parent
   * executor. Since the weights and function executor maps are set at parant
   * level, that function executor can access the function maps and weight maps
   * through the parent.
   */
  constructor(private graph: Graph, private parent?: GraphExecutor) {
    this._outputs = graph.outputs;
    this._inputs = graph.inputs;
    this._initNodes = graph.initNodes;
    this._signature = graph.signature;
    this._functions = graph.functions;
    // create sub-graph executors
    if (graph.functions != null) {
      Object.keys(graph.functions).forEach(name => {
        this._functionExecutorMap[name] =
            new GraphExecutor(graph.functions[name], this);
      });
    }
  }

  private getCompilationKey(inputs: Node[], outputs: Node[]): string {
    const sortedInputs = inputs.map(node => node.name).sort();
    const sortedOutputs = outputs.map(node => node.name).sort();
    return sortedInputs.join(this.SEPARATOR) + '--' +
        sortedOutputs.join(this.SEPARATOR);
  }

  /**
   * Compiles the inference graph and returns the minimal set of nodes that are
   * required for execution, in the correct execution order.
   * @returns {Object} compilation The compile result.
   * @returns {Node[]} compilation.orderedNodes Nodes in the correct execution
   *     order.
   * @returns {Map<string, Node[]>} compilation.nodeLiveUntilMap A map from node
   *     to disposable nodes after its execution. That is, for a node `x`,
   *     `nodeLiveUntilMap[x]` indicates all nodes whose intermediate
   *     tensors should be disposed after `x` is executed.
   */
  private compile(inputs: NamedTensorMap, outputs: Node[]):
      {orderedNodes: Node[], nodeLiveUntilMap: Map<string, Node[]>} {
    const executionInfo =
        getExecutionSubgraph(inputs, outputs, this.weightMap, this._initNodes);
    const {missingInputs, dynamicNode, syncInputs} = executionInfo;
    if (dynamicNode != null) {
      throw new Error(
          `This execution contains the node '${dynamicNode.name}', which has ` +
          `the dynamic op '${dynamicNode.op}'. Please use ` +
          `model.executeAsync() instead. Alternatively, to avoid the ` +
          `dynamic ops, specify the inputs [${syncInputs}]`);
    }

    if (missingInputs.length > 0) {
      const outNames = outputs.map(n => n.name);
      const inNames = Object.keys(inputs);
      throw new Error(
          `Cannot compute the outputs [${outNames}] from the provided inputs ` +
          `[${inNames}]. Missing the following inputs: [${missingInputs}]`);
    }

    const orderedNodes = getNodesInTopologicalOrder(this.graph, executionInfo);
    const nodeLiveUntilMap = getNodeLiveUntilMap(orderedNodes);
    return {orderedNodes, nodeLiveUntilMap};
  }

  private cloneAndKeepTensor(tensor: Tensor) {
    if (tensor == null) {
      return null;
    }
    const clone = tensor.clone();
    // Keep the clone because`model.execute()` may be called within
    // a `tidy()`, but the user may inspect these tensors after the
    // tidy.
    keep(clone);
    return clone;
  }

  private cloneTensorList(tensors: Tensor[]) {
    if (!tensors) {
      return null;
    }
    const clonedTensor = tensors.map(tensor => {
      return this.cloneAndKeepTensor(tensor);
    });
    return clonedTensor;
  }

  private cloneTensorMap(tensorsMap: NamedTensorsMap): NamedTensorsMap {
    return Object.fromEntries(
        Object.entries(tensorsMap).map(([name, tensorsList]) => {
          return [name, this.cloneTensorList(tensorsList)];
        }));
  }

  /**
   * Executes the inference for given input tensors.
   * @param inputs Tensor map for the model inputs, keyed by the input node
   * names.
   * @param outputs Optional. output node name from the Tensorflow model, if
   * no outputs are specified, the default outputs of the model would be used.
   * You can inspect intermediate nodes of the model by adding them to the
   * outputs array.
   */
  execute(inputs: NamedTensorMap, outputs?: string[]): Tensor[] {
    // Dispose any tensors from a prior run to avoid leaking them.
    this.disposeIntermediateTensors();
    inputs = this.mapInputs(inputs);
    const names = Object.keys(inputs).sort();
    this.checkInputs(inputs);
    this.checkInputShapeAndType(inputs);
    outputs = this.mapOutputs(outputs);
    this.checkOutputs(outputs);
    const inputNodes =
        names.map(name => this.graph.nodes[parseNodeName(name)[0]]);
    const outputNodeNames = outputs.map(name => parseNodeName(name)[0]);
    const outputNodeNameSet = new Set(outputNodeNames);
    let outputNodes = outputNodeNames.map(name => this.graph.nodes[name]);
    // If no outputs are specified, then use the default outputs of the model.
    if (outputNodes.length === 0) {
      outputNodes = this._outputs;
    }

    const compilationKey = this.getCompilationKey(inputNodes, outputNodes);

    // Do nothing if the compiled graph cache contains the input.
    let compilation = this.compiledMap.get(compilationKey);
    if (compilation == null) {
      compilation = this.compile(inputs, outputNodes);
      this.compiledMap.set(compilationKey, compilation);
    }

    // Keep tensors if KEEP_INTERMEDIATE_TENSORS is on.
    try {
      this.keepIntermediateTensors = env().getBool('KEEP_INTERMEDIATE_TENSORS');
    } catch (e) {
      this.keepIntermediateTensors = false;
      console.warn(e.message);
    }
    const tensorArrayMap: TensorArrayMap = {};
    const tensorListMap: TensorListMap = {};

    return tidy(() => {
      const context = new ExecutionContext(
          this.weightMap, tensorArrayMap, tensorListMap,
          this.functionExecutorMap, this.parseNodeNameCache);
      const tensorsMap: NamedTensorsMap = {...this.weightMap};
      if (this.keepIntermediateTensors) {
        this.clonedTensorsMap = this.cloneTensorMap(this.weightMap);
      }

      Object.keys(inputs).forEach(name => {
        const [nodeName, index] = parseNodeName(name, context);
        const tensors: Tensor[] = [];
        tensors[index] = inputs[name];
        tensorsMap[nodeName] = tensors;
        if (this.keepIntermediateTensors) {
          this.clonedTensorsMap[nodeName] = this.cloneTensorList(tensors);
        }
      });

      const tensorsToKeep = this.getFrozenTensorIds(tensorsMap);
      const {orderedNodes, nodeLiveUntilMap} = compilation;
      for (const node of orderedNodes) {
        if (tensorsMap[node.name]) {
          continue;
        }
        const tensors =
            executeOp(node, tensorsMap, context, this._resourceManager) as
            Tensor[];
        if (util.isPromise(tensors)) {
          throw new Error(
              `The execution of the op '${node.op}' returned a promise. ` +
              `Please use model.executeAsync() instead.`);
        }
        tensorsMap[node.name] = tensors;
        if (this.keepIntermediateTensors) {
          this.clonedTensorsMap[node.name] = this.cloneTensorList(tensors);
        }
        this.checkTensorForDisposalWithNodeLiveUntilInfo(
            node, tensorsMap, context, tensorsToKeep, outputNodeNameSet,
            nodeLiveUntilMap.get(node.name));
      }

      // dispose the context for the root executor
      if (this.parent == null) {
        context.dispose(tensorsToKeep);
      }

      return outputs.map(name => getTensor(name, tensorsMap, context));
    });
  }

  private getFrozenTensorIds(tensorMap: NamedTensorsMap): Set<number> {
    const ids = [].concat.apply(
        [],
        Object.keys(tensorMap)
            .map(key => tensorMap[key])
            .map(tensors => tensors.map(tensor => tensor.id)));
    return new Set(ids);
  }

  private checkTensorForDisposal(
      nodeName: string, node: Node, tensorMap: NamedTensorsMap,
      context: ExecutionContext, tensorsToKeep: Set<number>,
      outputNodeNameSet: Set<string>,
      intermediateTensorConsumerCount: {[key: string]: number}) {
    // Skip output nodes and any control flow nodes, since its dependency is
    // tricky to track correctly.
    if (isControlFlow(node) || outputNodeNameSet.has(nodeName)) {
      return;
    }

    for (const tensor of tensorMap[nodeName]) {
      if (tensor == null) {
        continue;
      }
      intermediateTensorConsumerCount[tensor.id] =
          (intermediateTensorConsumerCount[tensor.id] || 0) +
          node.children.length;
    }

    for (const input of node.inputs) {
      // Skip any control flow nodes, since its dependency is tricky to track
      // correctly.
      if (isControlFlow(input)) {
        continue;
      }

      const tensors =
          getTensorsForCurrentContext(input.name, tensorMap, context);
      if (tensors == null) {
        continue;
      }

      for (const tensor of tensors) {
        if (!tensor || tensor.kept || tensorsToKeep.has(tensor.id)) {
          continue;
        }

        // Only intermediate nodes' tensors have counts set, not marked as
        // kept, and not in `tensorsToKeep`.
        // Input and weight nodes' tensors should exist in `tensorsToKeep`.
        // Output and control flow nodes' tensors should never have count set.
        const count = intermediateTensorConsumerCount[tensor.id];
        if (count === 1) {
          tensor.dispose();
          delete intermediateTensorConsumerCount[tensor.id];
        } else if (count != null) {
          intermediateTensorConsumerCount[tensor.id]--;
        }
      }
    }
  }

  private checkTensorForDisposalWithNodeLiveUntilInfo(
      node: Node, tensorMap: NamedTensorsMap, context: ExecutionContext,
      tensorsToKeep: Set<number>, outputNodeNameSet: Set<string>,
      liveUntilNodes?: Node[]) {
    function isNonDisposableNode(node: Node) {
      // Skip output nodes and any control flow nodes, since its dependency is
      // tricky to track correctly.
      return isControlFlow(node) || outputNodeNameSet.has(node.name);
    }

    if (isControlFlow(node) || liveUntilNodes == null) {
      return;
    }

    for (const nodeToDispose of liveUntilNodes) {
      if (isNonDisposableNode(nodeToDispose)) {
        continue;
      }
      const tensors = getTensorsForCurrentContext(
          nodeToDispose.name, tensorMap, context);
      for (const tensor of tensors) {
        if (!tensor || tensor.kept || tensorsToKeep.has(tensor.id)) {
          continue;
        }
        tensor.dispose();
      }
    }
  }

  /**
   * Executes the inference for given input tensors in Async fashion.
   * @param inputs Tensor map for the model inputs, keyed by the input node
   * names.
   * @param outputs output node name from the Tensorflow model, if no outputs
   * are specified, the default outputs of the model would be used. You can
   * inspect intermediate nodes of the model by adding them to the outputs
   * array.
   */
  async executeAsync(inputs: NamedTensorMap, outputs?: string[]):
      Promise<Tensor[]> {
    return this._executeAsync(inputs, outputs);
  }

  disposeIntermediateTensors() {
    if (!this.clonedTensorsMap) {
      return;
    }
    Object.values(this.clonedTensorsMap).forEach(tensorsList => {
      for (const tensor of tensorsList) {
        if (tensor && !tensor.isDisposed) {
          tensor.dispose();
        }
      }
    });

    this.clonedTensorsMap = null;
  }

  getIntermediateTensors(): NamedTensorsMap {
    return this.clonedTensorsMap;
  }

  /**
   * Executes the inference for given input tensors in Async fashion.
   * @param inputs Tensor map for the model inputs, keyed by the input node
   * names.
   * @param outputs Optional. output node name from the Tensorflow model,
   * if no outputs are specified, the default outputs of the model would be
   * used. You can inspect intermediate nodes of the model by adding them to
   * the outputs array.
   * @param isFunctionExecution Optional. Flag for executing a function.
   * @param tensorArrayMap Optional, global TensorArray map by id. Used for
   * function execution.
   * @param tensorArrayMap Optional global TensorList map by id. Used for
   * function execution.
   */
  private async _executeAsync(
      inputs: NamedTensorMap, outputs?: string[], isFunctionExecution = false,
      tensorArrayMap: TensorArrayMap = {},
      tensorListMap: TensorListMap = {}): Promise<Tensor[]> {
    // Dispose any tensors from a prior run to avoid leaking them.
    this.disposeIntermediateTensors();
    if (!isFunctionExecution) {
      inputs = this.mapInputs(inputs);
      this.checkInputs(inputs);
      this.checkInputShapeAndType(inputs);
      outputs = this.mapOutputs(outputs);
      this.checkOutputs(outputs);
    }

    // Keep tensors if KEEP_INTERMEDIATE_TENSORS is on.
    try {
      this.keepIntermediateTensors = env().getBool('KEEP_INTERMEDIATE_TENSORS');
    } catch (e) {
      this.keepIntermediateTensors = false;
      console.warn(e.message);
    }

    const context = new ExecutionContext(
        this.weightMap, tensorArrayMap, tensorListMap, this.functionExecutorMap,
        this.parseNodeNameCache);

    if (this.keepIntermediateTensors) {
      this.clonedTensorsMap = this.cloneTensorMap(this.weightMap);
    }

    // Graph with control flow op requires runtime evaluation of the execution
    // order, while without control flow the execution order is pre-determined
    // in the compile method.
    const tensorsMap = await this.executeWithControlFlow(
        inputs, context, outputs, isFunctionExecution);
    const results = outputs.map(name => getTensor(name, tensorsMap, context));

    // dispose all the intermediate tensors
    const outputIds = results.map(t => t.id);
    const inputIds = Object.keys(inputs).map(name => inputs[name].id);
    const keepIds =
        new Set<number>([...outputIds, ...inputIds, ...this.weightIds]);

    Object.values(tensorsMap).forEach(tensorsList => {
      tensorsList.forEach(tensor => {
        if (tensor && !tensor.isDisposed && !keepIds.has(tensor.id)) {
          tensor.dispose();
        }
      });
    });

    // dispose the context for the root executor
    if (this.parent == null) {
      context.dispose(keepIds);
    }

    return results;
  }

  async executeFunctionAsync(
      inputs: Tensor[], tensorArrayMap: TensorArrayMap,
      tensorListMap: TensorListMap): Promise<Tensor[]> {
    const mappedInputs = inputs.reduce((map, tensor, index) => {
      map[this.inputs[index].name] = tensor;
      return map;
    }, {} as NamedTensorMap);

    return this._executeAsync(
        mappedInputs, this.outputNodes, true, tensorArrayMap, tensorListMap);
  }

  /**
   * When there are control flow nodes in the graph, the graph execution use
   * ExecutionContext to keep track of the frames and loop iterators.
   * @param inputs placeholder tensors for the graph.
   * @param context the execution context object for current execution.
   * @param outputNames Optional. output node name from the Tensorflow model,
   * if no outputs are specified, the default outputs of the model would be
   * used. You can inspect intermediate nodes of the model by adding them to
   * the outputs array.
   * @param isFunctionExecution Flag for executing a function.
   */
  private async executeWithControlFlow(
      inputs: NamedTensorMap, context: ExecutionContext, outputNames?: string[],
      isFunctionExecution?: boolean): Promise<NamedTensorsMap> {
    const names = Object.keys(inputs);
    const inputNodes =
        names.map(name => this.graph.nodes[parseNodeName(name)[0]]);
    const outputNodeNames = outputNames.map(name => parseNodeName(name)[0]);
    const outputNodeNameSet = new Set(outputNodeNames);
    let outputNodes = outputNodeNames.map(name => this.graph.nodes[name]);

    // If no outputs are specified, then use the default outputs of the model.
    if (outputNodes.length === 0) {
      outputNodes = this._outputs;
    }

    const {usedNodes, missingInputs, dynamicNode, syncInputs} =
        getExecutionSubgraph(
            inputs, outputNodes, this.weightMap, this._initNodes);

    // First nodes to execute include inputNodes, weights, and initNodes.
    const stack: NodeWithContexts[] = [
      ...inputNodes, ...this.graph.weights, ...(this._initNodes || [])
    ].map(node => {
      return {node, contexts: context.currentContext};
    });
    const tensorsMap: NamedTensorsMap = {...this.weightMap};
    Object.keys(inputs).forEach(name => {
      const [nodeName, index] = parseNodeName(name);
      const tensors: Tensor[] = [];
      tensors[index] = inputs[name];
      tensorsMap[nodeName] = tensors;
    });
    const intermediateTensorConsumerCount: {[key: number]: number} = {};
    const tensorsToKeep = this.getFrozenTensorIds(tensorsMap);
    const added: {[key: string]: boolean} = {};
    while (stack.length > 0) {
      const promises = this.processStack(
          inputNodes, stack, context, tensorsMap, added, tensorsToKeep,
          outputNodeNameSet, intermediateTensorConsumerCount, usedNodes);
      await Promise.all(promises);
    }
    if (dynamicNode == null && !isFunctionExecution) {
      console.warn(
          `This model execution did not contain any nodes with control flow ` +
          `or dynamic output shapes. You can use model.execute() instead.`);
    }
    const missingOutputs =
        outputNodes
            .filter(
                node => !isControlFlow(node) &&
                    !getTensor(node.name, tensorsMap, context))
            .map(node => node.name);
    if (missingOutputs.length > 0) {
      let alternativeMsg = '';
      if (dynamicNode != null) {
        alternativeMsg =
            `Alternatively, to avoid the dynamic ops, use model.execute() ` +
            `and specify the inputs [${syncInputs}]`;
      }
      throw new Error(
          `Cannot compute the outputs [${missingOutputs}] from the provided ` +
          `inputs [${names}]. Consider providing the following inputs: ` +
          `[${missingInputs}]. ${alternativeMsg}`);
    }
    return tensorsMap;
  }

  private processStack(
      inputNodes: Node[], stack: NodeWithContexts[], context: ExecutionContext,
      tensorMap: NamedTensorsMap, added: {[key: string]: boolean},
      tensorsToKeep: Set<number>, outputNodeNameSet: Set<string>,
      intermediateTensorConsumerCount: {[key: number]: number},
      usedNodes: Set<string>) {
    const promises: Array<Promise<Tensor[]>> = [];
    while (stack.length > 0) {
      const item = stack.pop();
      context.currentContext = item.contexts;
      let nodeName = '';
      // The tensor of the Enter op with isConstant set should be set
      // in the parent scope, so it will be available as constant for the
      // whole loop.
      if (item.node.op === 'Enter' &&
          getParamValue('isConstant', item.node, tensorMap, context)) {
        [nodeName] = getNodeNameAndIndex(item.node.name, context);
      }

      // only process nodes that are not in the tensorMap yet, this include
      // inputNodes and internal initNodes.
      if (tensorMap[item.node.name] == null) {
        const tensors =
            executeOp(item.node, tensorMap, context, this._resourceManager);
        if (!nodeName) {
          [nodeName] = getNodeNameAndIndex(item.node.name, context);
        }
        const currentContext = context.currentContext;
        if (util.isPromise(tensors)) {
          promises.push(tensors.then(t => {
            tensorMap[nodeName] = t;
            if (this.keepIntermediateTensors) {
              this.clonedTensorsMap[nodeName] = this.cloneTensorList(t);
            }
            context.currentContext = currentContext;
            this.checkTensorForDisposal(
                nodeName, item.node, tensorMap, context, tensorsToKeep,
                outputNodeNameSet, intermediateTensorConsumerCount);
            this.processChildNodes(
                item.node, stack, context, tensorMap, added, usedNodes);
            return t;
          }));
        } else {
          tensorMap[nodeName] = tensors;
          if (this.keepIntermediateTensors) {
            this.clonedTensorsMap[nodeName] = this.cloneTensorList(tensors);
          }
          this.checkTensorForDisposal(
              nodeName, item.node, tensorMap, context, tensorsToKeep,
              outputNodeNameSet, intermediateTensorConsumerCount);
          this.processChildNodes(
              item.node, stack, context, tensorMap, added, usedNodes);
        }
      } else {
        this.processChildNodes(
            item.node, stack, context, tensorMap, added, usedNodes);
      }
    }
    return promises;
  }

  private processChildNodes(
      node: Node, stack: NodeWithContexts[], context: ExecutionContext,
      tensorMap: NamedTensorsMap, added: {[key: string]: boolean},
      usedNodes: Set<string>) {
    node.children.forEach((childNode) => {
      const [nodeName, ] = getNodeNameAndIndex(childNode.name, context);
      if (added[nodeName] || !usedNodes.has(childNode.name)) {
        return;
      }
      // Merge op can be pushed if any of its inputs has value.
      if (childNode.op === 'Merge') {
        if (childNode.inputNames.some(name => {
              return !!getTensor(name, tensorMap, context);
            })) {
          added[nodeName] = true;
          stack.push({contexts: context.currentContext, node: childNode});
        }
      } else  // Otherwise all inputs must to have value.
          if (childNode.inputNames.every(name => {
                return !!getTensor(name, tensorMap, context);
              })) {
        added[nodeName] = true;
        stack.push({contexts: context.currentContext, node: childNode});
      }
    });
  }

  /**
   * Releases the memory used by the weight tensors.
   */
  dispose() {
    Object.keys(this.weightMap)
        .forEach(
            key => this.weightMap[key].forEach(tensor => tensor.dispose()));
  }

  private checkInputShapeAndType(inputs: NamedTensorMap) {
    Object.keys(inputs).forEach(name => {
      const input = inputs[name];
      const [nodeName, ] = parseNodeName(name);
      const node = this.graph.nodes[nodeName];
      if (node.attrParams['shape'] && node.attrParams['shape'].value) {
        const shape = node.attrParams['shape'].value as number[];
        const match = shape.length === input.shape.length &&
            input.shape.every(
                (dim, index) => shape[index] === -1 || shape[index] === dim);
        util.assert(
            match,
            () => `The shape of dict['${node.name}'] provided in ` +
                `model.execute(dict) must be [${shape}], but was ` +
                `[${input.shape}]`);
      }
      if (node.attrParams['dtype'] && node.attrParams['dtype'].value) {
        util.assert(
            input.dtype === node.attrParams['dtype'].value as string,
            () => `The dtype of dict['${node.name}'] provided in ` +
                `model.execute(dict) must be ` +
                `${node.attrParams['dtype'].value}, but was ${input.dtype}`);
      }
    });
  }

  private mapInputs(inputs: NamedTensorMap) {
    const result: NamedTensorMap = {};
    for (const inputName in inputs) {
      const tensor = this._signature ?.inputs ?.[inputName];
      if (tensor != null) {
        result[tensor.name] = inputs[inputName];
      } else {
        result[inputName] = inputs[inputName];
      }
    }
    return result;
  }

  private checkInputs(inputs: NamedTensorMap) {
    const notInGraph = Object.keys(inputs).filter(name => {
      const [nodeName] = parseNodeName(name);
      return this.graph.nodes[nodeName] == null;
    });
    if (notInGraph.length > 0) {
      throw new Error(
          `The dict provided in model.execute(dict) has ` +
          `keys: [${notInGraph}] that are not part of graph`);
    }
  }

  private mapOutputs(outputs: string[]) {
    return outputs.map(name => {
      const tensor = this._signature ?.outputs ?.[name];
      if (tensor != null) {
        return tensor.name;
      }
      return name;
    }, {});
  }

  private checkOutputs(outputs: string[]): void {
    outputs.forEach(name => {
      const [normalizedName] = parseNodeName(name);
      if (!this.graph.nodes[normalizedName]) {
        throw new Error(`The output '${name}' is not found in the graph`);
      }
    });
  }
}
