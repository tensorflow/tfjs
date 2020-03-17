/**
 * @license
 * Copyright 2019 Google Inc. All Rights Reserved.
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

import {BroadcastTo, BroadCastToAttrs} from '../kernel_names';
import {GradConfig, NamedAttrMap} from '../kernel_registry';
import {Tensor} from '../tensor';

export const broadcastToGradConfig: GradConfig = {
  kernelName: BroadcastTo,
  gradFunc: (dy: Tensor, saved: Tensor[], attrs: NamedAttrMap) => {
    const broadCastToAttrs: BroadCastToAttrs =
        attrs as unknown as BroadCastToAttrs;
    const axes =
        broadCastToAttrs.reps.map((n, i) => n > 1 ? i : -1).filter(i => i >= 0);
    const keepDims = true;
    return {x: () => dy.sum(axes, keepDims)};
  }
};
