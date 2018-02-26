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

import {Tensor} from 'deeplearn';
import {precompile} from 'handlebars/handlebars.runtime';

import {tensorflow} from '../data/index';
import {TensorMap} from '../data/types';
import {Graph, Node} from '../operations/index';
import { executeOp } from '../operations/operation_executor';

export class GraphExecutor {
  private compiledOrder: Node[] = [];
  private _weightMap: TensorMap;
  get weightMap(): TensorMap {
    return this._weightMap;
  }
  set weightMap(weightMap: TensorMap) {
    this._weightMap = weightMap;
  }

  constructor(private graph: Graph) {
    this.precompile();
  }

  private precompile() {
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

  execute(inputs: TensorMap): TensorMap {
    const tensors = this.compiledOrder.reduce<TensorMap>((map, node) => {
      map[node.name] = this.executeOp(node, map);
      return map;
    }, {});

    return this.graph.outputs.reduce<TensorMap>((map, node) => {
      map[node.name] = tensors[node.name];
      return map;
    }, {});
  }

  executeOp(node: Node, tensorMap: TensorMap): Tensor {
    return executeOp(node, tensorMap);
  }
}
