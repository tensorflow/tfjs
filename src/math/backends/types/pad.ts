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
import {Array1D, Array2D} from '../../ndarray';
// tslint:disable-next-line:max-line-length
import {KernelInputConfig, KernelNode, TapeNodeInputGradientArrays} from '../tape_types';

// Pad1D
export interface Pad1DNode extends KernelNode {
  inputAndArgs: Pad1DInputConfig;
  output: Array1D;
  gradient: (dy: Array1D, y: Array1D) => Pad1DGradientInputArrays;
}

export interface Pad1DInputConfig extends KernelInputConfig {
  inputs: Pad1DInputArrays;
  args: {paddings: [number, number], constantValue: number};
}

export interface Pad1DInputArrays extends NamedArrayMap { x: Array1D; }

export interface Pad1DGradientInputArrays extends TapeNodeInputGradientArrays {
  x: () => Array1D;
}

// Pad2D
export interface Pad2DNode extends KernelNode {
  inputAndArgs: Pad2DInputConfig;
  output: Array2D;
  gradient: (dy: Array2D, y: Array2D) => Pad2DGradientInputArrays;
}

export interface Pad2DInputConfig extends KernelInputConfig {
  inputs: Pad2DInputArrays;
  args: {paddings: [[number, number], [number, number]], constantValue: number};
}

export interface Pad2DInputArrays extends NamedArrayMap { x: Array2D; }

export interface Pad2DGradientInputArrays extends TapeNodeInputGradientArrays {
  x: () => Array2D;
}
