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
import {Tensor} from '../../tensor';
import {DataType} from '../../types';

// Equal/NotEqual/Less/LessEqual/Greater/GreaterEqual
export interface EqualNode extends KernelNode {
  inputAndArgs: {inputs: {a: Tensor; b: Tensor;};};
  output: Tensor;
  gradient: (dy: Tensor, y: Tensor) => {
    a: () => Tensor;
    b: () => Tensor;
  };
}

// LogicalAnd/LogicalOr/LogicalXor
export interface LogicalNode extends KernelNode {
  inputAndArgs: {inputs: {a: Tensor; b: Tensor;};};
  output: Tensor;
  gradient: (dy: Tensor, y: Tensor) => {
    a: () => Tensor;
    b: () => Tensor;
  };
}

// Where
export interface WhereNode extends KernelNode {
  inputAndArgs: {
    inputs: {condition: Tensor; a: Tensor; b: Tensor;};
    args: {dtype: DataType};
  };
  output: Tensor;
  gradient: (dy: Tensor, y: Tensor) => {
    condition: () => Tensor;
    a: () => Tensor;
    b: () => Tensor;
  };
}
