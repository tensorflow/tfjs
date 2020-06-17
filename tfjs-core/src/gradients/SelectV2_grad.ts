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

import {SelectV2} from '../kernel_names';
import {GradConfig} from '../kernel_registry';
import {cast} from '../ops/array_ops';
import {logicalNot} from '../ops/logical_not';
import {mul} from '../ops/mul';
import {zerosLike} from '../ops/tensor_ops';
import {Tensor} from '../tensor';

export const selectV2PoolGradConfig: GradConfig = {
  kernelName: SelectV2,
  inputsToSave: ['condition'],
  gradFunc: (dy: Tensor, saved: Tensor[]) => {
    const [condition] = saved;
    return {
      // TODO(julianoks): Return null for condition gradient
      // when backprop supports it.
      condition: () => cast(zerosLike(condition), 'float32'),
      t: () => mul(dy, cast(condition, dy.dtype)),
      e: () => mul(dy, cast(logicalNot(condition), dy.dtype))
    };
  }
};
