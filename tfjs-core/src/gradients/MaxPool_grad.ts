/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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

import {MaxPool, MaxPoolAttrs} from '../kernel_names';
import {GradConfig, NamedAttrMap} from '../kernel_registry';
import {maxPoolBackprop} from '../ops/max_pool_backprop';
import {Tensor, Tensor4D} from '../tensor';

export const maxPoolGradConfig: GradConfig = {
  kernelName: MaxPool,
  inputsToSave: ['x'],
  outputsToSave: [true],
  gradFunc: (dy: Tensor, saved: Tensor[], attrs: NamedAttrMap) => {
    const [x, y] = saved as [Tensor4D, Tensor4D];
    const {filterSize, strides, pad} = attrs as {} as MaxPoolAttrs;

    return {
      x: () => maxPoolBackprop(dy as Tensor4D, x, y, filterSize, strides, pad)
    };
  }
};
