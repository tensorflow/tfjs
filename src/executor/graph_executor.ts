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

import {DataType, Tensor, tidy, util} from '@tensorflow/tfjs-core';

// tslint:disable-next-line:max-line-length
import {NamedTensorMap, NamedTensorsMap, TensorArrayMap, TensorInfo} from '../data/types';
// tslint:disable-next-line:max-line-length
import {getNodeNameAndIndex, getParamValue, getTensor, getTensorsForCurrentContenxt, parseNodeName} from '../operations/executors/utils';
import {executeOp} from '../operations/operation_executor';
import {Graph, Node} from '../operations/types';

import {ExecutionContext, ExecutionContextInfo} from './execution_context';

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
    this.compile();
  }

  get isControlFlowModel(): boolean {
    return this.graph.withControlFlow;
  }

  get isDynamicShapeModel(): boolean {
    return this.graph.withDynamicShape;
  }

  /**
   * Compiles the inference graph to generate the topology order of op nodes,
   * cache the result for inference execution.
   */
  private compile(startNodes?: Node[]) {
    // Do not compile for graph with control flow, since the execution order
    // requires runtime evaluation of the output tensors.
    if (this.graph.withControlFlow || this.graph.withDynamicShape) {
      return;
    }
    const compiledOrder = [];
    const inputs = startNodes || this.graph.placeholders;
    const sortedNodeNames = inputs.map(node => node.name).sort();
    const nameKey = sortedNodeNames.join(this.SEPERATOR);

    // do nothing is the compiled graph cache contains the input.
    if (this.compiledMap.get(nameKey)) {
      return;
    }

    const stack = [...inputs, ...this.graph.weights];
    const visited: {[key: string]: boolean} = {};
    while (stack.length > 0) {
      const node = stack.pop();
      visited[node.name] = true;
      compiledOrder.push(node);
      node.children.forEach((childNode) => {
        if (!visited[childNode.name] && childNode.inputNames.every(name => {
              const [nodeName, ] = getNodeNameAndIndex(name);
              return visited[nodeName];
            })) {
          stack.push(childNode);
        }
      });
    }
    this.compiledMap.set(nameKey, compiledOrder);
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
  execute(
      inputs: NamedTensorsMap, strictInputCheck = true,
      outputs?: string|string[]): NamedTensorMap {
    const names = Object.keys(inputs).sort();
    this.checkInput(inputs, strictInputCheck);
    this.checkInputShapeAndType(inputs, strictInputCheck);

    this.compile(names.map(name => this.graph.nodes[name]));
    const outputNames = this.calculateOutputs(outputs);
    this.checkOutput(
        this.compiledMap.get(names.join(this.SEPERATOR)), outputNames);

    const tensorArrayMap: TensorArrayMap = {};
    const result = tidy(() => {
      const context = new ExecutionContext(this._weightMap, tensorArrayMap);
      const tensorMap = {...this.weightMap, ...inputs};
      const tensorsToKeep = this.getFrozenTensorIds(tensorMap);
      const intermediateTensorConsumerCount: {[key: number]: number} = {};

      const compiledNodes = this.compiledMap.get(names.join(this.SEPERATOR));
      for (let i = 0; i < compiledNodes.length; i++) {
        const node = compiledNodes[i];
        if (!tensorMap[node.name]) {
          tensorMap[node.name] =
              executeOp(node, tensorMap, context) as Tensor[];
          this.checkTensorForDisposal(
              node.name, node, tensorMap, context, tensorsToKeep, outputNames,
              intermediateTensorConsumerCount);
        }
        // stop the execution if all outputs are found.
        if (outputNames.every(name => !!tensorMap[name])) {
          break;
        }
      }
      return this.findOutputs(tensorMap, context, outputNames);
    });
    return result;
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
  async executeAsync(inputs: NamedTensorsMap, outputs?: string|string[]):
      Promise<NamedTensorMap> {
    this.checkInput(inputs, false);
    this.checkInputShapeAndType(inputs, false);
    const tensorArrayMap: TensorArrayMap = {};
    const context = new ExecutionContext(this._weightMap, tensorArrayMap);
    const outputNames = this.calculateOutputs(outputs);
    // Graph with control flow op requires runtime evaluation of the execution
    // order, while without control flow the execution order is pre-determined
    // in the compile method.
    const tensors =
        await this.executeWithControlFlow(inputs, context, outputNames);
    const results = this.findOutputs(tensors, context, outputs);

    // dispose all the intermediate tensors
    const outputIds = Object.keys(results).map(key => results[key].id);
    const inputIdArray =
        Object.keys(inputs).map(key => inputs[key].map(input => input.id));
    const inputIds = [].concat.apply([], inputIdArray);
    Object.keys(tensors).forEach(key => {
      const tensorArray = tensors[key];
      tensorArray.forEach(tensor => {
        if (tensor && !tensor.isDisposed &&
            outputIds.indexOf(tensor.id) === -1 &&
            inputIds.indexOf(tensor.id) === -1 &&
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
      inputs: NamedTensorsMap, context: ExecutionContext,
      outputNames: string[]): Promise<NamedTensorsMap> {
    const names = Object.keys(inputs);
    const inputNodes = names.map(name => this.graph.nodes[name]);
    const stack: NodeWithContexts[] =
        [...inputNodes, ...this.graph.weights].map(node => {
          return {node, contexts: context.currentContext};
        });
    const tensorMap = {...this.weightMap, ...inputs};
    const intermediateTensorConsumerCount: {[key: number]: number} = {};
    const tensorsToKeep = this.getFrozenTensorIds(tensorMap);
    const added: {[key: string]: boolean} = {};
    while (stack.length > 0) {
      const promises = this.processStack(
          inputNodes, stack, context, tensorMap, added, tensorsToKeep,
          outputNames, intermediateTensorConsumerCount);
      await Promise.all(promises);
    }
    return tensorMap;
  }

  private processStack(
      inputNodes: Node[], stack: NodeWithContexts[], context: ExecutionContext,
      tensorMap: NamedTensorsMap, added: {[key: string]: boolean},
      tensorsToKeep: Set<number>, outputNames: string[],
      intermediateTensorConsumerCount: {[key: number]: number}) {
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
            this.processChildNodes(item.node, stack, context, tensorMap, added);
            return t;
          }));
        } else {
          tensorMap[nodeName] = tensors;
          this.checkTensorForDisposal(
              nodeName, item.node, tensorMap, context, tensorsToKeep,
              outputNames, intermediateTensorConsumerCount);
          this.processChildNodes(item.node, stack, context, tensorMap, added);
        }
      } else {
        this.processChildNodes(item.node, stack, context, tensorMap, added);
      }
    }
    return promises;
  }

  private processChildNodes(
      node: Node, stack: NodeWithContexts[], context: ExecutionContext,
      tensorMap: NamedTensorsMap, added: {[key: string]: boolean}) {
    node.children.forEach((childNode) => {
      const [nodeName, ] = getNodeNameAndIndex(childNode.name, context);
      if (!added[nodeName]) {
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
      }
    });
  }

  private calculateOutputs(outputs?: string|string[]): string[] {
    if (outputs && !(outputs instanceof Array)) {
      outputs = [outputs];
    }
    return (outputs || this.graph.outputs.map(node => node.name)) as string[];
  }

  private findOutputs(
      tensorMap: NamedTensorsMap, context: ExecutionContext,
      outputs?: string|string[]): NamedTensorMap {
    const requestedOutputs = this.calculateOutputs(outputs);
    return requestedOutputs.reduce<NamedTensorMap>((map, name) => {
      map[name] = getTensor(name, tensorMap, context);
      return map;
    }, {});
  }
  /**
   * Releases the memory used by the weight tensors.
   */
  dispose() {
    Object.keys(this.weightMap)
        .forEach(
            key => this.weightMap[key].forEach(tensor => tensor.dispose()));
  }

  private checkInputShapeAndType(
      inputs: NamedTensorsMap, strictInputCheck = true) {
    this.placeholders.forEach(node => {
      const inputTensors = inputs[node.name];
      // do nothing if not strict input check and input tensors is not for
      // the placeholders.
      if (!strictInputCheck && !inputTensors) {
        return;
      }

      const input = inputTensors[0];
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

  private checkInput(inputs: NamedTensorsMap, strictInputCheck = true) {
    const inputKeys = Object.keys(inputs);
    const missing: string[] = [];
    const extra: string[] = [];

    this.inputNodes.forEach(name => {
      if (inputKeys.indexOf(name) === -1) missing.push(name);
    });

    inputKeys.forEach(name => {
      if (this.inputNodes.indexOf(name) === -1) extra.push(name);
    });

    const notInGraph = extra.filter(name => !this.graph.nodes[name]);

    if (missing.length > 0 && strictInputCheck) {
      throw new Error(
          `The dict provided in model.execute(dict) has the keys ` +
          `[${inputKeys}], but is missing the required keys: [${missing}].`);
    }

    if (extra.length > 0 && strictInputCheck) {
      throw new Error(
          `The dict provided in model.execute(dict) has ` +
          `unused keys: [${extra}]. Please provide only the following keys: ` +
          `[${this.inputNodes}].`);
    }

    if (notInGraph.length > 0) {
      throw new Error(
          `The dict provided in model.execute(dict) has ` +
          `keys: [${notInGraph}] not part of model graph.`);
    }
  }

  private checkOutput(compiledNodes: Node[], outputs: string[]) {
    const compiledNodeNames = compiledNodes.map(node => node.name);
    const extra: string[] = [];
    outputs.forEach(name => {
      const [nodeName] = parseNodeName(name);
      if (compiledNodeNames.indexOf(nodeName) === -1) extra.push(nodeName);
    });

    if (extra.length > 0) {
      throw new Error(
          `The following outputs are not generated by the execution: ` +
          `[${extra}].`);
    }
  }
}
