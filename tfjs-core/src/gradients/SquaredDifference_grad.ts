/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
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

import {SquaredDifference} from '../kernel_names';
import {GradConfig} from '../kernel_registry';
import {mul, sub} from '../ops/binary_ops';
import {scalar} from '../ops/tensor_ops';
import {Tensor} from '../tensor';

export const squaredDifferenceGradConfig: GradConfig = {
  kernelName: SquaredDifference,
  inputsToSave: ['a', 'b'],
  gradFunc: (dy: Tensor, saved: Tensor[]) => {
    const [a, b] = saved;
    const two = scalar(2);
    const derA = () => mul(dy, mul(two, sub(a, b)));
    const derB = () => mul(dy, mul(two, sub(b, a)));
    return {a: derA, b: derB};
  }
};
