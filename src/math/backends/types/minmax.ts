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

import {NDArray} from '../../ndarray';
import {KernelNode} from '../tape_types';

// Reduction min.
export interface MinNode extends KernelNode {
  inputAndArgs: {inputs: {x: NDArray;}; args: {axes: number[]}};
  output: NDArray;
  gradient: (dy: NDArray, y: NDArray) => {
    x: () => NDArray;
  };
}

// Element-wise min.
export interface MinimumNode extends KernelNode {
  inputAndArgs: {inputs: {a: NDArray, b: NDArray};};
  output: NDArray;
  gradient: (dy: NDArray, y: NDArray) => {
    a: () => NDArray, b: () => NDArray
  };
}

// Reduction Max
export interface MaxNode extends KernelNode {
  inputAndArgs: {inputs: {x: NDArray;}; args: {axes: number[]}};
  output: NDArray;
  gradient: (dy: NDArray, y: NDArray) => {
    x: () => NDArray;
  };
}

// Element-wise max.
export interface MaximumNode extends KernelNode {
  inputAndArgs: {inputs: {a: NDArray, b: NDArray};};
  output: NDArray;
  gradient: (dy: NDArray, y: NDArray) => {
    a: () => NDArray, b: () => NDArray
  };
}
