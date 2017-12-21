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

export interface ArgMaxNode extends KernelNode {
  inputAndArgs: ArgMaxInputConfig;
  output: NDArray<'int32'>;
  gradient:
      (dy: NDArray<'int32'>, y: NDArray<'int32'>) => ArgMaxGradientInputArrays;
}

export interface ArgMaxInputConfig extends KernelInputConfig {
  inputs: ArgMaxInputArrays;
  args: {axes: number[];};
}

export interface ArgMaxInputArrays extends NamedArrayMap {
  x: NDArray;
}

export interface ArgMaxGradientInputArrays extends TapeNodeInputGradientArrays {
  x: () => NDArray;
}

export interface ArgMinNode extends KernelNode {
  inputAndArgs: ArgMinInputConfig;
  output: NDArray<'int32'>;
  gradient:
      (dy: NDArray<'int32'>, y: NDArray<'int32'>) => ArgMinGradientInputArrays;
}

export interface ArgMinInputConfig extends KernelInputConfig {
  inputs: ArgMinInputArrays;
  args: {axes: number[];};
}

export interface ArgMinInputArrays extends NamedArrayMap {
  x: NDArray;
}

export interface ArgMinGradientInputArrays extends TapeNodeInputGradientArrays {
  x: () => NDArray;
}
