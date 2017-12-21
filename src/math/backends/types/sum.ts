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
import {DataTypes, NDArray} from '../../ndarray';
import {SumTypes} from '../../types';
// tslint:disable-next-line:max-line-length
import {KernelInputConfig, KernelNode, TapeNodeInputGradientArrays} from '../tape_types';

export interface SumNode<T extends keyof DataTypes> extends KernelNode {
  inputAndArgs: SumInputConfig<T>;
  output: NDArray<SumTypes[T]>;
  gradient:
      (dy: NDArray<SumTypes[T]>,
       y: NDArray<SumTypes[T]>) => SumGradientInputArrays<T>;
}

export interface SumInputConfig<T extends keyof DataTypes> extends
    KernelInputConfig {
  inputs: SumInputArrays<T>;
  args: {axes: number[];};
}

export interface SumInputArrays<T extends keyof DataTypes> extends
    NamedArrayMap {
  x: NDArray<T>;
}

export interface SumGradientInputArrays<T extends keyof DataTypes> extends
    TapeNodeInputGradientArrays {
  x: () => NDArray<T>;
}
