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

import {Array4D} from '../../ndarray';
import {KernelNode} from '../tape_types';

export interface Reverse4DNode extends KernelNode {
  inputAndArgs: {inputs: {x: Array4D;}; args: {axis: number[];};};
  output: Array4D;
  gradient: (dy: Array4D<'float32'>, y: Array4D) => {
    x: () => Array4D<'float32'>;
  };
}
