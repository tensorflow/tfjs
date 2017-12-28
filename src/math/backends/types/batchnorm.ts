
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

// 4D
export interface BatchNorm4DNode extends KernelNode {
  inputAndArgs: BatchNorm4DInputConfig;
  output: Array4D;
  gradient: (dy: Array4D, y: Array4D) => BatchNorm4DGradientInputArrays;
}

export interface BatchNorm4DInputConfig extends KernelInputConfig {
  inputs: BatchNorm4DInputArrays;
  args: {varianceEpsilon: number};
}

export interface BatchNorm4DInputArrays extends NamedArrayMap {
  x: Array4D;
  mean: Array4D|Array1D;
  variance: Array4D|Array1D;
  scale?: Array4D|Array1D;
  offset?: Array4D|Array1D;
}

export interface BatchNorm4DGradientInputArrays extends
    TapeNodeInputGradientArrays {
  x: () => Array4D;
  mean: () => Array4D | Array1D;
  variance: () => Array4D | Array1D;
  scale?: () => Array4D | Array1D;
  offset?: () => Array4D | Array1D;
}

// 3D
export interface BatchNorm3DNode extends KernelNode {
  inputAndArgs: BatchNorm3DInputConfig;
  output: Array3D;
  gradient: (dy: Array3D, y: Array3D) => BatchNorm3DGradientInputArrays;
}

export interface BatchNorm3DInputConfig extends KernelInputConfig {
  inputs: BatchNorm3DInputArrays;
  args: {varianceEpsilon: number};
}

export interface BatchNorm3DInputArrays extends NamedArrayMap {
  x: Array3D;
  mean: Array3D|Array1D;
  variance: Array3D|Array1D;
  scale?: Array3D|Array1D;
  offset?: Array3D|Array1D;
}

export interface BatchNorm3DGradientInputArrays extends
    TapeNodeInputGradientArrays {
  x: () => Array3D;
  mean: () => Array3D | Array1D;
  variance: () => Array3D | Array1D;
  scale?: () => Array3D | Array1D;
  offset?: () => Array3D | Array1D;
}

// 2D
export interface BatchNorm2DNode extends KernelNode {
  inputAndArgs: BatchNorm2DInputConfig;
  output: Array2D;
  gradient: (dy: Array2D, y: Array2D) => BatchNorm2DGradientInputArrays;
}

export interface BatchNorm2DInputConfig extends KernelInputConfig {
  inputs: BatchNorm2DInputArrays;
  args: {varianceEpsilon: number};
}

export interface BatchNorm2DInputArrays extends NamedArrayMap {
  x: Array2D;
  mean: Array2D|Array1D;
  variance: Array2D|Array1D;
  scale?: Array2D|Array1D;
  offset?: Array2D|Array1D;
}

export interface BatchNorm2DGradientInputArrays extends
    TapeNodeInputGradientArrays {
  x: () => Array2D;
  mean: () => Array2D | Array1D;
  variance: () => Array2D | Array1D;
  scale?: () => Array2D | Array1D;
  offset?: () => Array2D | Array1D;
}
