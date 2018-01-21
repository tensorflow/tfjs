
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

export interface BatchNorm4DNode extends KernelNode {
  inputAndArgs: {
    inputs: {
      x: Array4D; mean: Array4D | Array1D; variance: Array4D | Array1D;
      scale?: Array4D | Array1D;
      offset?: Array4D | Array1D;
    };
    args: {varianceEpsilon: number};
  };
  output: Array4D;
  gradient: (dy: Array4D<'float32'>, y: Array4D) => {
    x: () => Array4D<'float32'>;
    mean: () => Array4D<'float32'>| Array1D<'float32'>;
    variance: () => Array4D<'float32'>| Array1D<'float32'>;
    scale?: () => Array4D<'float32'>| Array1D<'float32'>;
    offset?: () => Array4D<'float32'>| Array1D<'float32'>;
  };
}

export interface BatchNorm3DNode extends KernelNode {
  inputAndArgs: {
    inputs: {
      x: Array3D; mean: Array3D | Array1D; variance: Array3D | Array1D;
      scale?: Array3D | Array1D;
      offset?: Array3D | Array1D;
    };
    args: {varianceEpsilon: number};
  };
  output: Array3D;
  gradient: (dy: Array3D<'float32'>, y: Array3D) => {
    x: () => Array3D<'float32'>;
    mean: () => Array3D<'float32'>| Array1D<'float32'>;
    variance: () => Array3D<'float32'>| Array1D<'float32'>;
    scale?: () => Array3D<'float32'>| Array1D<'float32'>;
    offset?: () => Array3D<'float32'>| Array1D<'float32'>;
  };
}

export interface BatchNorm2DNode extends KernelNode {
  inputAndArgs: {
    inputs: {
      x: Array2D; mean: Array2D | Array1D; variance: Array2D | Array1D;
      scale?: Array2D | Array1D;
      offset?: Array2D | Array1D;
    };
    args: {varianceEpsilon: number};
  };
  output: Array2D;
  gradient: (dy: Array2D<'float32'>, y: Array2D) => {
    x: () => Array2D<'float32'>;
    mean: () => Array2D<'float32'>| Array1D<'float32'>;
    variance: () => Array2D<'float32'>| Array1D<'float32'>;
    scale?: () => Array2D<'float32'>| Array1D<'float32'>;
    offset?: () => Array2D<'float32'>| Array1D<'float32'>;
  };
}
