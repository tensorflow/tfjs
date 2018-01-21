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

import {Array1D, Array2D, Array3D, Array4D, DataType} from '../../ndarray';
import {KernelNode} from '../tape_types';

export interface Slice1DNode<D extends DataType> extends KernelNode {
  inputAndArgs:
      {inputs: {x: Array1D<D>;}; args: {begin: number; size: number;};};
  output: Array1D<D>;
  gradient: (dy: Array1D<'float32'>, y: Array1D<D>) => {
    x: () => Array1D<'float32'>;
  };
}

export interface Slice2DNode<D extends DataType> extends KernelNode {
  inputAndArgs: {
    inputs: {x: Array2D<D>;};
    args: {begin: [number, number]; size: [number, number];};
  };
  output: Array2D<D>;
  gradient: (dy: Array2D<'float32'>, y: Array2D<D>) => {
    x: () => Array2D<'float32'>;
  };
}

export interface Slice3DNode<D extends DataType> extends KernelNode {
  inputAndArgs: {
    inputs: {x: Array3D<D>;};
    args: {begin: [number, number, number]; size: [number, number, number];};
  };
  output: Array3D<D>;
  gradient: (dy: Array3D<'float32'>, y: Array3D<D>) => {
    x: () => Array3D<'float32'>;
  };
}

export interface Slice4DNode<D extends DataType> extends KernelNode {
  inputAndArgs: {
    inputs: {x: Array4D<D>;}; args: {
      begin: [number, number, number, number];
      size: [number, number, number, number];
    };
  };
  output: Array4D<D>;
  gradient: (dy: Array4D<'float32'>, y: Array4D<D>) => {
    x: () => Array4D<'float32'>;
  };
}
