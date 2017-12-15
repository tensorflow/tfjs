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

import {NDArray, Scalar} from '../ndarray';

import {MathBackend} from './backend';
import {KernelNode, TapeNode} from './tape_types';
import * as tape_util from './tape_util';

export class Tape {
  private evaluatedTapeNodes: TapeNode[] = [];

  private outputNodeMap: {[id: number]: TapeNode} = {};

  constructor(private backend: MathBackend) {}

  addEvaluatedKernelNode(node: KernelNode) {
    this.outputNodeMap[node.output.id] = node;
    this.evaluatedTapeNodes.push(node);
  }

  gradientWrt(y: Scalar, xs: NDArray[]): NDArray[] {
    if (this.outputNodeMap[y.id] == null) {
      throw new Error(`Cannot compute gradient: y is not part of this tape.`);
    }

    // Filter out the nodes that don't connect x => y.
    const filteredNodes =
        tape_util.getFilteredNodesXToY(this.evaluatedTapeNodes, xs, y);

    // Seed the gradient of dy to be 1.
    const arrayAccumulatedGradientMap: {[ndarrayId: number]: NDArray} = {};
    arrayAccumulatedGradientMap[y.id] = Scalar.new(1);

    // Backprop gradients through the filtered nodes.
    tape_util.backpropagateGradients(
        this.backend, arrayAccumulatedGradientMap, filteredNodes);

    const gradients: NDArray[] = [];
    for (let i = 0; i < xs.length; i++) {
      gradients.push(arrayAccumulatedGradientMap[xs[i].id]);
    }
    return gradients;
  }
}
