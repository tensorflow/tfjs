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

import {KernelNode} from '../../tape_types';
import {Tensor1D, Tensor2D, Tensor3D, Tensor4D} from '../../tensor';

export interface Slice1DNode extends KernelNode {
  inputAndArgs: {inputs: {x: Tensor1D;}; args: {begin: number; size: number;};};
  output: Tensor1D;
  gradient: (dy: Tensor1D, y: Tensor1D) => {
    x: () => Tensor1D;
  };
}

export interface Slice2DNode extends KernelNode {
  inputAndArgs: {
    inputs: {x: Tensor2D;};
    args: {begin: [number, number]; size: [number, number];};
  };
  output: Tensor2D;
  gradient: (dy: Tensor2D, y: Tensor2D) => {
    x: () => Tensor2D;
  };
}

export interface Slice3DNode extends KernelNode {
  inputAndArgs: {
    inputs: {x: Tensor3D;};
    args: {begin: [number, number, number]; size: [number, number, number];};
  };
  output: Tensor3D;
  gradient: (dy: Tensor3D, y: Tensor3D) => {
    x: () => Tensor3D;
  };
}

export interface Slice4DNode extends KernelNode {
  inputAndArgs: {
    inputs: {x: Tensor4D;}; args: {
      begin: [number, number, number, number];
      size: [number, number, number, number];
    };
  };
  output: Tensor4D;
  gradient: (dy: Tensor4D, y: Tensor4D) => {
    x: () => Tensor4D;
  };
}
