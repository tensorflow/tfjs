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
import {Array4D} from '../../ndarray';
// tslint:disable-next-line:max-line-length
import {KernelInputConfig, KernelNode, TapeNodeInputGradientArrays} from '../tape_types';

// 4D
export interface Reverse4DNode extends KernelNode {
  inputAndArgs: Reverse4DInputConfig;
  output: Array4D;
  gradient: (dy: Array4D, y: Array4D) => Reverse4DGradientInputArrays;
}

export interface Reverse4DInputConfig extends KernelInputConfig {
  inputs: Reverse4DInputArrays;
  args: {axis: number[];};
}

export interface Reverse4DInputArrays extends NamedArrayMap {
  x: Array4D;
}

export interface Reverse4DGradientInputArrays extends
    TapeNodeInputGradientArrays {
  x: () => Array4D;
}
