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

import {ClipByValue, ClipByValueAttrs} from '../kernel_names';
import {GradConfig, NamedAttrMap} from '../kernel_registry';
import {greaterEqual} from '../ops/greater_equal';
import {lessEqual} from '../ops/less_equal';
import {logicalAnd} from '../ops/logical_and';
import {where} from '../ops/where';
import {zerosLike} from '../ops/zeros_like';
import {Tensor} from '../tensor';

export const clipByValueGradConfig: GradConfig = {
  kernelName: ClipByValue,
  inputsToSave: ['x'],
  gradFunc: (dy: Tensor, saved: Tensor[], attrs: NamedAttrMap) => {
    const [x] = saved;
    const {clipValueMin, clipValueMax} = attrs as {} as ClipByValueAttrs;
    return {
      x: () => where(
          logicalAnd(greaterEqual(x, clipValueMin), lessEqual(x, clipValueMax)),
          dy, zerosLike(dy)),
    };
  }
};
