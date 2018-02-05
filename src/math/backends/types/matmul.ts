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

import {Tensor2D} from '../../tensor';
import {KernelNode} from '../tape_types';

export interface MatMulNode extends KernelNode {
  inputAndArgs: {
    inputs: {a: Tensor2D; b: Tensor2D;};
    args: {aOrientation: MatrixOrientation; bOrientation: MatrixOrientation};
  };
  output: Tensor2D;
  gradient: (dy: Tensor2D, y: Tensor2D) => {
    a: () => Tensor2D;
    b: () => Tensor2D;
  };
}

export enum MatrixOrientation {
  REGULAR,
  TRANSPOSED
}
