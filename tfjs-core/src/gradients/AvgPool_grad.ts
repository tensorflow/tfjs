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

import {AvgPool, AvgPoolAttrs} from '../kernel_names';
import {GradConfig, NamedAttrMap} from '../kernel_registry';
import {avgPoolBackprop} from '../ops/avg_pool_backprop';
import {Tensor, Tensor4D} from '../tensor';

export const avgPoolGradConfig: GradConfig = {
  kernelName: AvgPool,
  inputsToSave: ['x'],
  gradFunc: (dy: Tensor, saved: Tensor[], attrs: NamedAttrMap) => {
    const [x] = saved as [Tensor4D];
    const {filterSize, strides, pad} = attrs as {} as AvgPoolAttrs;
    return {
      x: () => avgPoolBackprop(dy as Tensor4D, x, filterSize, strides, pad)
    };
  }
};
