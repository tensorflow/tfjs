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

import {KernelNode} from '../../tape_types';
import {Tensor, Tensor1D} from '../../tensor';
import {Rank} from '../../types';

// Values
export interface TopKValuesNode<R extends Rank, T extends Tensor<R> = Tensor<R>>
    extends KernelNode {
  inputAndArgs: {inputs: {x: T;}; args: {k: number};};
  output: Tensor1D;
  gradient: (dy: Tensor1D, y: Tensor1D) => {
    x: () => T;
  };
}

// Indices
export interface TopKIndicesNode extends KernelNode {
  inputAndArgs: {inputs: {x: Tensor;}; args: {k: number};};
  output: Tensor1D;
  gradient: (dy: Tensor1D, y: Tensor1D) => {
    x: () => Tensor;
  };
}
