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

import {MirrorPad, MirrorPadAttrs} from '../kernel_names';
import {GradConfig, NamedAttrMap} from '../kernel_registry';
import {slice} from '../ops/slice';
import {Tensor} from '../tensor';

export const mirrorPadGradConfig: GradConfig = {
  kernelName: MirrorPad,
  inputsToSave: ['x'],
  gradFunc: (dy: Tensor, saved: Tensor[], attrs: NamedAttrMap) => {
    // Pad introduces values around the original tensor, so the gradient
    // slices the original shape out of the gradient.
    const x = saved[0];
    const {paddings} = attrs as unknown as MirrorPadAttrs;
    const begin = paddings.map(p => p[0]);
    return {x: () => slice(dy, begin, x.shape)};
  }
};
