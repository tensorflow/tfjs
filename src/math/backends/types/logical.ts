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

import {DataType, NDArray} from '../../ndarray';
import {KernelNode} from '../tape_types';

// Equal/NotEqual/Less/LessEqual/Greater/GreaterEqual
export interface EqualNode extends KernelNode {
  inputAndArgs: {inputs: {a: NDArray; b: NDArray;};};
  output: NDArray<'bool'>;
  gradient: (dy: NDArray<'float32'>, y: NDArray<'bool'>) => {
    a: () => NDArray<'float32'>;
    b: () => NDArray<'float32'>;
  };
}

// LogicalAnd/LogicalOr
export interface LogicalNode extends KernelNode {
  inputAndArgs: {inputs: {a: NDArray; b: NDArray;};};
  output: NDArray<'bool'>;
  gradient: (dy: NDArray<'float32'>, y: NDArray<'bool'>) => {
    a: () => NDArray<'float32'>;
    b: () => NDArray<'float32'>;
  };
}

// Where
export interface WhereNode extends KernelNode {
  inputAndArgs: {
    inputs: {condition: NDArray; a: NDArray; b: NDArray;};
    args: {dtype: DataType};
  };
  output: NDArray;
  gradient: (dy: NDArray<'float32'>, y: NDArray) => {
    condition: () => NDArray<'float32'>;
    a: () => NDArray<'float32'>;
    b: () => NDArray<'float32'>;
  };
}
