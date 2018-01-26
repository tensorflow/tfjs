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

import {Array1D, NDArray} from '../../ndarray';
import {Rank} from '../../types';
import {KernelNode} from '../tape_types';

// Values
export interface TopKValuesNode<
    R extends Rank, T extends NDArray<R> = NDArray<R>> extends KernelNode {
  inputAndArgs: {inputs: {x: T;}; args: {k: number};};
  output: Array1D;
  gradient: (dy: Array1D, y: Array1D) => {
    x: () => T;
  };
}

// Indices
export interface TopKIndicesNode extends KernelNode {
  inputAndArgs: {inputs: {x: NDArray;}; args: {k: number};};
  output: Array1D;
  gradient: (dy: Array1D, y: Array1D) => {
    x: () => NDArray;
  };
}
