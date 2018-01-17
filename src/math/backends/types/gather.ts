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
import {NDArray, Array1D} from '../../ndarray';
// tslint:disable-next-line:max-line-length
import {KernelInputConfig, KernelNode, TapeNodeInputGradientArrays} from '../tape_types';

export interface GatherNode<T extends NDArray> extends KernelNode {
  inputAndArgs: GatherInputConfig<T>;
  output: T;
  gradient: (dy: T, y: T) => GatherGradientInputArrays<T>;
}

export interface GatherInputConfig<T extends NDArray> extends 
    KernelInputConfig {
  inputs: GatherInputArrays<T>;
  args: {axis: number};
}

export interface GatherInputArrays<T extends NDArray> extends NamedArrayMap {
  x: T;
  indices: Array1D<'int32'>;
}

export interface GatherGradientInputArrays<T extends NDArray> extends 
    TapeNodeInputGradientArrays {
  x: () => T;
  indices: () => Array1D;
}
