/**
 * @license
 * Copyright 2017 Google Inc. All Rights Reserved.
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

import {tidy} from '@tensorflow/tfjs-core';

import {NamedTensorMap, NamedTensorsMap} from '../data/index';
import {getTensor} from '../operations/executors/utils';
import * as operations from '../operations/index';

export class GraphExecutor {
  private compiledOrder: operations.Node[] = [];
  private _weightMap: NamedTensorsMap = {};
  private placeholders: string[];
  private outputs: string[];
  get weightMap(): NamedTensorsMap {
    return this._weightMap;
  }
  set weightMap(weightMap: NamedTensorsMap) {
    this._weightMap = weightMap;
  }

  get inputNodes(): string[] {
    return this.placeholders;
  }

  get outputNodes(): string[] {
    return this.outputs;
  }

  constructor(private graph: operations.Graph) {
    this.placeholders = graph.placeholders.map(node => node.name);
    this.outputs = graph.outputs.map(node => node.name);
    this.compile();
  }

  /**
   * Compiles the inference graph to generate the topology order of op nodes,
   * cache the result for inference execution.
   */
  private compile() {
    const stack = [...this.graph.inputs];
    const visited: {[key: string]: boolean} = {};
    while (stack.length > 0) {
      const node = stack.pop();
      visited[node.name] = true;
      this.compiledOrder.push(node);
      node.children.forEach((childNode) => {
        if (childNode.inputNames.every(name => visited[name])) {
          stack.push(childNode);
        }
      });
    }
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

  execute(inputs: NamedTensorsMap, outputs?: string|string[]): NamedTensorMap {
    this.checkInput(inputs);
    const result = tidy(() => {
      const tensors =
          this.compiledOrder.reduce<NamedTensorsMap>((map, node) => {
            map[node.name] = operations.executeOp(node, map);
            return map;
          }, {...this.weightMap, ...inputs});
      if (outputs && !(outputs instanceof Array)) {
        outputs = [outputs];
      }
      const requestedOutputs =
          (outputs || this.graph.outputs.map(node => node.name)) as string[];

      return requestedOutputs.reduce<NamedTensorMap>((map, name) => {
        map[name] = getTensor(name, tensors);
        return map;
      }, {});
    });
    return result;
  }

  /**
   * Releases the memory used by the weight tensors.
   */
  dispose() {
    Object.keys(this.weightMap)
        .forEach(
            key => this.weightMap[key].forEach(tensor => tensor.dispose()));
  }

  private checkInput(inputs: NamedTensorsMap) {
    const inputKeys = Object.keys(inputs);
    const missing: string[] = [];
    const extra: string[] = [];

    this.placeholders.forEach(name => {
      if (inputKeys.indexOf(name) === -1) missing.push(name);
    });

    inputKeys.forEach(name => {
      if (this.placeholders.indexOf(name) === -1) extra.push(name);
    });

    if (missing.length > 0) {
      throw new Error(`Missing input placeholders: ${missing}`);
    }

    if (extra.length > 0) {
      throw new Error(`Extra input tensors: ${extra}`);
    }
  }
}
