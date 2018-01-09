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
import {Conv2DInfo} from '../../conv_util';
import {Array1D, Array4D} from '../../ndarray';
// tslint:disable-next-line:max-line-length
import {KernelInputConfig, KernelNode, TapeNodeInputGradientArrays} from '../tape_types';

// Conv2D
export interface Conv2DNode extends KernelNode {
  inputAndArgs: Conv2DInputConfig;
  output: Array4D;
  gradient: (dy: Array4D, y: Array4D) => Conv2DGradientInputArrays;
}

export interface Conv2DInputConfig extends KernelInputConfig {
  inputs: Conv2DInputArrays;
  args: {convInfo: Conv2DInfo;};
}

export interface Conv2DInputArrays extends NamedArrayMap {
  x: Array4D;
  filter: Array4D;
  bias?: Array1D;
}

export interface Conv2DGradientInputArrays extends TapeNodeInputGradientArrays {
  x: () => Array4D;
  filter: () => Array4D;
  bias?: () => Array1D;
}

// Conv2DDerInput
export interface Conv2DDerInputNode extends KernelNode {
  inputAndArgs: Conv2DDerInputInputConfig;
  output: Array4D;
  gradient: (dy: Array4D, y: Array4D) => Conv2DDerInputGradientInputArrays;
}

export interface Conv2DDerInputInputConfig extends KernelInputConfig {
  inputs: Conv2DDerInputInputArrays;
  args: {convInfo: Conv2DInfo;};
}

export interface Conv2DDerInputInputArrays extends NamedArrayMap {
  dy: Array4D;
  filter: Array4D;
}

export interface Conv2DDerInputGradientInputArrays extends
    TapeNodeInputGradientArrays {
  dy: () => Array4D;
  filter: () => Array4D;
}

// Conv2DDerFilter
export interface Conv2DDerFilterNode extends KernelNode {
  inputAndArgs: Conv2DDerFilterInputConfig;
  output: Array4D;
  gradient: (dy: Array4D, y: Array4D) => Conv2DDerFilterGradientInputArrays;
}

export interface Conv2DDerFilterInputConfig extends KernelInputConfig {
  inputs: Conv2DDerFilterInputArrays;
  args: {convInfo: Conv2DInfo;};
}

export interface Conv2DDerFilterInputArrays extends NamedArrayMap {
  x: Array4D;
  dy: Array4D;
}

export interface Conv2DDerFilterGradientInputArrays extends
    TapeNodeInputGradientArrays {
  x: () => Array4D;
  dy: () => Array4D;
}

// Conv2DDerBias
export interface Conv2DDerBiasNode extends KernelNode {
  inputAndArgs: Conv2DDerBiasInputConfig;
  output: Array1D;
  gradient: (dy: Array1D, y: Array1D) => Conv2DDerBiasGradientInputArrays;
}

export interface Conv2DDerBiasInputConfig extends KernelInputConfig {
  inputs: Conv2DDerBiasInputArrays;
}

export interface Conv2DDerBiasInputArrays extends NamedArrayMap { dy: Array4D; }

export interface Conv2DDerBiasGradientInputArrays extends
    TapeNodeInputGradientArrays {
  dy: () => Array4D;
}

// DepthwiseConv2D
export interface DepthwiseConv2DNode extends KernelNode {
  inputAndArgs: DepthwiseConv2DInputConfig;
  output: Array4D;
  gradient: (dy: Array4D, y: Array4D) => DepthwiseConv2DGradientInputArrays;
}

export interface DepthwiseConv2DInputConfig extends KernelInputConfig {
  inputs: DepthwiseConv2DInputArrays;
  args: {convInfo: Conv2DInfo;};
}

export interface DepthwiseConv2DInputArrays extends NamedArrayMap {
  x: Array4D;
  filter: Array4D;
}

export interface DepthwiseConv2DGradientInputArrays extends
    TapeNodeInputGradientArrays {
  x: () => Array4D;
  filter: () => Array4D;
}
