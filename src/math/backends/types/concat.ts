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
import {Array1D, Array2D, Array3D, Array4D} from '../../ndarray';
// tslint:disable-next-line:max-line-length
import {KernelInputConfig, KernelNode, TapeNodeInputGradientArrays} from '../tape_types';

// 1D
export interface Concat1DNode extends KernelNode {
  inputAndArgs: Concat1DInputConfig;
  output: Array1D;
  gradient: (dy: Array1D, y: Array1D) => Concat1DGradientArrays;
}

export interface Concat1DInputConfig extends KernelInputConfig {
  inputs: Concat1DInputArrays;
}

export interface Concat1DInputArrays extends NamedArrayMap {
  a: Array1D;
  b: Array1D;
}

export interface Concat1DGradientArrays extends TapeNodeInputGradientArrays {
  a: () => Array1D;
  b: () => Array1D;
}

// 2D
export interface Concat2DNode extends KernelNode {
  inputAndArgs: Concat2DInputConfig;
  output: Array2D;
  gradient: (dy: Array2D, y: Array2D) => Concat2DGradientArrays;
}

export interface Concat2DInputConfig extends KernelInputConfig {
  inputs: Concat2DInputArrays;
  args: {axis: number};
}

export interface Concat2DInputArrays extends NamedArrayMap {
  a: Array2D;
  b: Array2D;
}

export interface Concat2DGradientArrays extends TapeNodeInputGradientArrays {
  a: () => Array2D;
  b: () => Array2D;
}

// 3D
export interface Concat3DNode extends KernelNode {
  inputAndArgs: Concat3DInputConfig;
  output: Array3D;
  gradient: (dy: Array3D, y: Array3D) => Concat3DGradientArrays;
}

export interface Concat3DInputConfig extends KernelInputConfig {
  inputs: Concat3DInputArrays;
  args: {axis: number};
}

export interface Concat3DInputArrays extends NamedArrayMap {
  a: Array3D;
  b: Array3D;
}

export interface Concat3DGradientArrays extends TapeNodeInputGradientArrays {
  a: () => Array3D;
  b: () => Array3D;
}

// 4D
export interface Concat4DNode extends KernelNode {
  inputAndArgs: Concat4DInputConfig;
  output: Array4D;
  gradient: (dy: Array4D, y: Array4D) => Concat4DGradientArrays;
}

export interface Concat4DInputConfig extends KernelInputConfig {
  inputs: Concat4DInputArrays;
  args: {axis: number};
}

export interface Concat4DInputArrays extends NamedArrayMap {
  a: Array4D;
  b: Array4D;
}

export interface Concat4DGradientArrays extends TapeNodeInputGradientArrays {
  a: () => Array4D;
  b: () => Array4D;
}
