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

import {Array1D, Array2D, Array3D, Array4D} from '../../ndarray';
import {KernelNode} from '../tape_types';

export interface Concat1DNode extends KernelNode {
  inputAndArgs: {inputs: {a: Array1D; b: Array1D;};};
  output: Array1D;
  gradient: (dy: Array1D, y: Array1D) => {
    a: () => Array1D;
    b: () => Array1D;
  };
}

export interface Concat2DNode extends KernelNode {
  inputAndArgs: {inputs: {a: Array2D; b: Array2D;}; args: {axis: number};};
  output: Array2D;
  gradient: (dy: Array2D, y: Array2D) => {
    a: () => Array2D;
    b: () => Array2D;
  };
}

export interface Concat3DNode extends KernelNode {
  inputAndArgs: {inputs: {a: Array3D; b: Array3D;}; args: {axis: number};};
  output: Array3D;
  gradient: (dy: Array3D, y: Array3D) => {
    a: () => Array3D;
    b: () => Array3D;
  };
}

export interface Concat4DNode extends KernelNode {
  inputAndArgs: {inputs: {a: Array4D; b: Array4D;}; args: {axis: number};};
  output: Array4D;
  gradient: (dy: Array4D, y: Array4D) => {
    a: () => Array4D;
    b: () => Array4D;
  };
}
