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

import {DataType, NamedTensorMap, Tensor, tidy, util} from '@tensorflow/tfjs-core';

import {NamedTensorsMap, TensorArrayMap, TensorInfo} from '../data/types';
import {getNodeNameAndIndex, getParamValue, getTensor, getTensorsForCurrentContenxt, parseNodeName} from '../operations/executors/utils';
import {executeOp} from '../operations/operation_executor';
import {Graph, Node} from '../operations/types';

import {ExecutionContext, ExecutionContextInfo} from './execution_context';
import {getExecutionSubgraph, getNodesInTopologicalOrder, isControlFlow} from './model_analysis';
import {rewritePrelu} from './model_rewrite';

interface NodeWithContexts {
  contexts: ExecutionContextInfo[];
  node: Node;
}

export class GraphExecutor {
  private compiledMap: Map<string, Node[]> = new Map();
  private _weightMap: NamedTensorsMap = {};
  private weightIds: number[];
  private placeholders: Node[];
  private _outputs: Node[];
  private SEPERATOR = ',';
  get weightMap(): NamedTensorsMap {
    return this._weightMap;
  }
  set weightMap(weightMap: NamedTensorsMap) {
    const weightIds = Object.keys(weightMap).map(
        key => weightMap[key].map(tensor => tensor.id));
    this.weightIds = [].concat.apply([], weightIds);
    this._weightMap = weightMap;
  }

  get inputs(): TensorInfo[] {
    return this.placeholders.map(node => {
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
    return this.placeholders.map(node => node.name);
  }

  get outputNodes(): string[] {
    return this.outputs.map(node => node.name);
  }

  constructor(private graph: Graph) {
    this.placeholders = graph.placeholders;
    this._outputs = graph.outputs;
  }

  private getCompilationKey(inputs: Node[], outputs: Node[]): string {
    const sortedInputs = inputs.map(node => node.name).sort();
    const sortedOutputs = outputs.map(node => node.name).sort();
    return sortedInputs.join(this.SEPERATOR) + '--' +
        sortedOutputs.join(this.SEPERATOR);
  }

  /**
   * Compiles the inference graph and returns the minimal set of nodes that are
   * required for execution, in the correct execution order.
   */
  private compile(inputs: NamedTensorMap, outputs: Node[]): Node[] {
    const executionInfo = getExecutionSubgraph(inputs, outputs, this.weightMap);
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

    return getNodesInTopologicalOrder(
        this.graph, this.weightMap, executionInfo);
  }

  fusePrelu() {
    rewritePrelu(this.graph, this.weightMap);
  }
  /**
   * Executes the inference for given input tensors.
   * @param inputs Tensor map for the model inputs, keyed by the input node
   * names.
   * @param outputs output node name from the Tensorflow model, if no outputs
   * are specified, the default outputs of the model would be used. You can
   * inspect intermediate nodes of the model by adding them to the outputs
   * array.
   */
  execute(inputs: NamedTensorMap, outputs: string[]): Tensor[] {
    const names = Object.keys(inputs).sort();
    this.checkInputs(inputs);
    this.checkInputShapeAndType(inputs);
    this.checkOutputs(outputs);
    const inputNodes = names.map(name => this.graph.nodes[name]);
    const outputNodes =
        outputs.map(name => this.graph.nodes[parseNodeName(name)[0]]);
    const compilationKey = this.getCompilationKey(inputNodes, outputNodes);
    // Do nothing if the compiled graph cache contains the input.
    let orderedNodes = this.compiledMap.get(compilationKey);
    if (orderedNodes == null) {
      orderedNodes = this.compile(inputs, outputNodes);
      this.compiledMap.set(compilationKey, orderedNodes);
    }
    const tensorArrayMap: TensorArrayMap = {};
    return tidy(() => {
      const context = new ExecutionContext(this._weightMap, tensorArrayMap);
      const tensorsMap: NamedTensorsMap = {...this.weightMap};
      Object.keys(inputs).forEach(name => {
        tensorsMap[name] = [inputs[name]];
      });
      const tensorsToKeep = this.getFrozenTensorIds(tensorsMap);
      const intermediateTensorConsumerCount: {[key: number]: number} = {};
      for (let i = 0; i < orderedNodes.length; i++) {
        const node = orderedNodes[i];
        if (!tensorsMap[node.name]) {
          const tensors = executeOp(node, tensorsMap, context) as Tensor[];
          if (tensors instanceof Promise) {
            throw new Error(
                `The execution of the op '${node.op}' returned a promise. ` +
                `Please use model.executeAsync() instead.`);
          }
          tensorsMap[node.name] = tensors;
          this.checkTensorForDisposal(
              node.name, node, tensorsMap, context, tensorsToKeep, outputs,
              intermediateTensorConsumerCount);
        }
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
      outputNames: string[],
      intermediateTensorConsumerCount: {[key: string]: number}) {
    // Skip output nodes and any control flow nodes, since its dependency is
    // tricky to track correctly.
    if (node.category === 'control' || outputNames.indexOf(nodeName) !== -1) {
      return;
    }

    tensorMap[nodeName].forEach(tensor => {
      if (tensor != null) {
        intermediateTensorConsumerCount[tensor.id] =
            (intermediateTensorConsumerCount[tensor.id] || 0) +
            node.children.length;
      }
    });
    node.inputs.forEach(input => {
      // Skip any control flow nodes, since its dependency is tricky to track
      // correctly.
      if (input.category !== 'control') {
        const tensors =
            getTensorsForCurrentContenxt(input.name, tensorMap, context);
        if (tensors != null) {
          tensors.forEach(tensor => {
            if (tensor && !tensorsToKeep.has(tensor.id)) {
              const count = intermediateTensorConsumerCount[tensor.id];
              if (count === 1) {
                tensor.dispose();
                delete intermediateTensorConsumerCount[tensor.id];
              } else if (count != null) {
                // only intermediate nodes has count set, inputs and weights are
                // not.
                intermediateTensorConsumerCount[tensor.id]--;
              }
            }
          });
        }
      }
    });
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
  async executeAsync(inputs: NamedTensorMap, outputs: string[]):
      Promise<Tensor[]> {
    this.checkInputs(inputs);
    this.checkInputShapeAndType(inputs);
    this.checkOutputs(outputs);
    const tensorArrayMap: TensorArrayMap = {};
    const context = new ExecutionContext(this._weightMap, tensorArrayMap);
    // Graph with control flow op requires runtime evaluation of the execution
    // order, while without control flow the execution order is pre-determined
    // in the compile method.
    const tensorMap =
        await this.executeWithControlFlow(inputs, context, outputs);
    const results = outputs.map(name => getTensor(name, tensorMap, context));

    // dispose all the intermediate tensors
    const outputIds = new Set<number>(results.map(t => t.id));
    const inputIds =
        new Set<number>(Object.keys(inputs).map(name => inputs[name].id));
    Object.keys(tensorMap).forEach(key => {
      const tensorArray = tensorMap[key];
      tensorArray.forEach(tensor => {
        if (tensor && !tensor.isDisposed && !outputIds.has(tensor.id) &&
            !inputIds.has(tensor.id) &&
            this.weightIds.indexOf(tensor.id) === -1) {
          tensor.dispose();
        }
      });
    });
    return results;
  }

  /**
   * When there are control flow nodes in the graph, the graph execution use
   * ExecutionContext to keep track of the frames and loop iterators.
   * @param inputs placeholder tensors for the graph.
   * @param context the execution context object for current execution.
   */
  private async executeWithControlFlow(
      inputs: NamedTensorMap, context: ExecutionContext,
      outputNames: string[]): Promise<NamedTensorsMap> {
    const names = Object.keys(inputs);
    const inputNodes = names.map(name => this.graph.nodes[name]);
    const outputNodes =
        outputNames.map(name => this.graph.nodes[parseNodeName(name)[0]]);
    const {usedNodes, missingInputs, dynamicNode, syncInputs} =
        getExecutionSubgraph(inputs, outputNodes, this.weightMap);

    const stack: NodeWithContexts[] =
        [...inputNodes, ...this.graph.weights].map(node => {
          return {node, contexts: context.currentContext};
        });
    const tensorsMap: NamedTensorsMap = {...this.weightMap};
    Object.keys(inputs).forEach(name => {
      tensorsMap[name] = [inputs[name]];
    });
    const intermediateTensorConsumerCount: {[key: number]: number} = {};
    const tensorsToKeep = this.getFrozenTensorIds(tensorsMap);
    const added: {[key: string]: boolean} = {};
    while (stack.length > 0) {
      const promises = this.processStack(
          inputNodes, stack, context, tensorsMap, added, tensorsToKeep,
          outputNames, intermediateTensorConsumerCount, usedNodes);
      await Promise.all(promises);
    }
    if (dynamicNode == null) {
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
      tensorsToKeep: Set<number>, outputNames: string[],
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

      // only process nodes that are not provided as input nodes.
      if (inputNodes.indexOf(item.node) === -1) {
        const tensors = executeOp(item.node, tensorMap, context);
        if (!nodeName) {
          [nodeName] = getNodeNameAndIndex(item.node.name, context);
        }
        const currentContext = context.currentContext;
        if (tensors instanceof Promise) {
          promises.push(tensors.then(t => {
            tensorMap[nodeName] = t;
            context.currentContext = currentContext;
            this.checkTensorForDisposal(
                nodeName, item.node, tensorMap, context, tensorsToKeep,
                outputNames, intermediateTensorConsumerCount);
            this.processChildNodes(
                item.node, stack, context, tensorMap, added, usedNodes);
            return t;
          }));
        } else {
          tensorMap[nodeName] = tensors;
          this.checkTensorForDisposal(
              nodeName, item.node, tensorMap, context, tensorsToKeep,
              outputNames, intermediateTensorConsumerCount);
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
      const node = this.graph.nodes[name];
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

  private checkInputs(inputs: NamedTensorMap) {
    const notInGraph =
        Object.keys(inputs).filter(name => !this.graph.nodes[name]);
    if (notInGraph.length > 0) {
      throw new Error(
          `The dict provided in model.execute(dict) has ` +
          `keys: [${notInGraph}] that are not part of graph`);
    }
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
