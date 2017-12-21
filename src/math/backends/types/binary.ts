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

import {NamedArrayMap} from '../../../util';
import {NDArray} from '../../ndarray';
// tslint:disable-next-line:max-line-length
import {KernelInputConfig, KernelNode, TapeNodeInputGradientArrays} from '../tape_types';

export interface BinaryNode extends KernelNode {
  inputAndArgs: BinaryInputConfig;
  output: NDArray;
  gradient: (dy: NDArray, y: NDArray) => BinaryInputGradientArrays;
}

export interface BinaryInputConfig extends KernelInputConfig {
  inputs: BinaryInputArrays;
}

export interface BinaryInputArrays extends NamedArrayMap {
  a: NDArray;
  b: NDArray;
}

export interface BinaryInputGradientArrays extends TapeNodeInputGradientArrays {
  a: () => NDArray;
  b: () => NDArray;
}
