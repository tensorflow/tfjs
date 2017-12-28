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
import {DataType, NDArray} from '../../ndarray';
// tslint:disable-next-line:max-line-length
import {KernelInputConfig, KernelNode, TapeNodeInputGradientArrays} from '../tape_types';

// Min
export interface MinNode<G extends DataType> extends KernelNode {
  inputAndArgs: MinInputConfig<G>;
  output: NDArray<G>;
  gradient: (dy: NDArray<G>, y: NDArray<G>) => MinGradientInputArrays<G>;
}

export interface MinInputConfig<G extends DataType> extends KernelInputConfig {
  inputs: MinInputArrays<G>;
}

export interface MinInputArrays<G extends DataType> extends NamedArrayMap {
  x: NDArray<G>;
}

export interface MinGradientInputArrays<G extends DataType> extends
    TapeNodeInputGradientArrays {
  x: () => NDArray<G>;
}

// Max
export interface MaxNode<G extends DataType> extends KernelNode {
  inputAndArgs: MaxInputConfig<G>;
  output: NDArray<G>;
  gradient: (dy: NDArray<G>, y: NDArray<G>) => MaxGradientInputArrays<G>;
}

export interface MaxInputConfig<G extends DataType> extends KernelInputConfig {
  inputs: MaxInputArrays<G>;
}

export interface MaxInputArrays<G extends DataType> extends NamedArrayMap {
  x: NDArray<G>;
}

export interface MaxGradientInputArrays<G extends DataType> extends
    TapeNodeInputGradientArrays {
  x: () => NDArray<G>;
}
