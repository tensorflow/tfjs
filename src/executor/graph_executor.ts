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

import {tidy} from 'deeplearn';

import {NamedTensorMap} from '../data/index';
import * as operations from '../operations/index';

export class GraphExecutor {
  private compiledOrder: operations.Node[] = [];
  private _weightMap: NamedTensorMap = {};
  get weightMap(): NamedTensorMap {
    return this._weightMap;
  }
  set weightMap(weightMap: NamedTensorMap) {
    this._weightMap = weightMap;
  }

  constructor(private graph: operations.Graph) {
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
   */
  execute(inputs: NamedTensorMap): NamedTensorMap {
    const outputs = tidy(() => {
      const tensors = this.compiledOrder.reduce<NamedTensorMap>((map, node) => {
        map[node.name] = operations.executeOp(node, map);
        return map;
      }, {...this.weightMap, ...inputs});

      return this.graph.outputs.reduce<NamedTensorMap>((map, node) => {
        map[node.name] = tensors[node.name];
        return map;
      }, {});
    });
    return outputs;
  }

  /**
   * Releases the memory used by the weight tensors.
   */
  dispose() {
    Object.keys(this.weightMap).forEach(key => this.weightMap[key].dispose());
  }
}
