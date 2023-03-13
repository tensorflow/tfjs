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

import {BroadcastTo, BroadCastToAttrs} from '../kernel_names';
import {GradConfig, NamedAttrMap} from '../kernel_registry';
import {sum} from '../ops/sum';
import {Tensor} from '../tensor';

export const broadcastToGradConfig: GradConfig = {
  kernelName: BroadcastTo,
  gradFunc: (dy: Tensor, saved: Tensor[], attrs: NamedAttrMap) => {
    const broadCastToAttrs: BroadCastToAttrs =
        attrs as unknown as BroadCastToAttrs;

    const inputShape = broadCastToAttrs.inputShape;
    const outputShape = broadCastToAttrs.shape;

    const reps: number[] = Array.from(outputShape);
    for (let i = inputShape.length - 1; i >= 0; i--) {
      if (inputShape[i] === outputShape[i]) {
        reps[i] = 1;
      } else if (inputShape[i] !== 1) {
        throw new Error(`broadcastTo(): [${
            inputShape}] cannot be broadcast to [${outputShape}].`);
      }
    }
    const axes: number[] = [];
    for (let i = 0; i < reps.length; i++) {
      if (reps[i] > 1) {
        axes.push(i);
      }
    }

    return {x: () => sum(dy, axes, true /* keepDims */)};
  }
};
