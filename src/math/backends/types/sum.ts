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
import {SumTypes} from '../../types';
// tslint:disable-next-line:max-line-length
import {KernelInputConfig, KernelNode, TapeNodeInputGradientArrays} from '../tape_types';

export interface SumNode<D extends DataType> extends KernelNode {
  inputAndArgs: SumInputConfig<D>;
  output: NDArray<SumTypes[D]>;
  gradient:
      (dy: NDArray<SumTypes[D]>,
       y: NDArray<SumTypes[D]>) => SumGradientInputArrays<D>;
}

export interface SumInputConfig<D extends DataType> extends KernelInputConfig {
  inputs: SumInputArrays<D>;
  args: {axes: number[];};
}

export interface SumInputArrays<D extends DataType> extends NamedArrayMap {
  x: NDArray<D>;
}

export interface SumGradientInputArrays<D extends DataType> extends
    TapeNodeInputGradientArrays {
  x: () => NDArray<D>;
}
