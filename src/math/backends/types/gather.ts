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

export interface GatherNode<R extends Rank, T extends NDArray<R> = NDArray<R>>
    extends KernelNode {
  inputAndArgs: {inputs: {x: T; indices: Array1D;}; args: {axis: number};};
  output: T;
  gradient: (dy: NDArray<R>, y: T) => {
    x: () => NDArray<R>;
    indices: () => Array1D;
  };
}
