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

import {Conv2DInfo} from '../../ops/conv_util';
import {KernelNode} from '../../tape_types';
import {Tensor1D, Tensor4D} from '../../tensor';

export interface Conv2DNode extends KernelNode {
  inputAndArgs: {
    inputs: {x: Tensor4D; filter: Tensor4D; bias?: Tensor1D;};
    args: {convInfo: Conv2DInfo;};
  };
  output: Tensor4D;
  gradient: (dy: Tensor4D, y: Tensor4D) => {
    x: () => Tensor4D;
    filter: () => Tensor4D;
    bias?: () => Tensor1D;
  };
}

export interface Conv2DDerInputNode extends KernelNode {
  inputAndArgs: {
    inputs: {dy: Tensor4D; filter: Tensor4D;}; args: {convInfo: Conv2DInfo;};
  };
  output: Tensor4D;
  gradient: (dy: Tensor4D, y: Tensor4D) => {
    dy: () => Tensor4D;
    filter: () => Tensor4D;
  };
}

export interface Conv2DDerFilterNode extends KernelNode {
  inputAndArgs:
      {inputs: {x: Tensor4D; dy: Tensor4D;}; args: {convInfo: Conv2DInfo;};};
  output: Tensor4D;
  gradient: (dy: Tensor4D, y: Tensor4D) => {
    x: () => Tensor4D;
    dy: () => Tensor4D;
  };
}

export interface Conv2DDerBiasNode extends KernelNode {
  inputAndArgs: {inputs: {dy: Tensor4D;};};
  output: Tensor1D;
  gradient: (dy: Tensor1D, y: Tensor1D) => {
    dy: () => Tensor4D;
  };
}

export interface DepthwiseConv2DNode extends KernelNode {
  inputAndArgs: {
    inputs: {x: Tensor4D; filter: Tensor4D;}; args: {convInfo: Conv2DInfo;};
  };
  output: Tensor4D;
  gradient: (dy: Tensor4D, y: Tensor4D) => {
    x: () => Tensor4D;
    filter: () => Tensor4D;
  };
}
