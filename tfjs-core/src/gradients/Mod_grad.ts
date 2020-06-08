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

import {Mod} from '../kernel_names';
import {GradConfig} from '../kernel_registry';
import {assertAndGetBroadcastShape, getReductionAxes} from '../ops/broadcast_util';
import {Tensor} from '../tensor';

export const modGradConfig: GradConfig = {
  kernelName: Mod,
  inputsToSave: ['a', 'b'],
  gradFunc: (dy: Tensor, saved: Tensor[]) => {
    const [a, b] = saved;
    const outShape = assertAndGetBroadcastShape(a.shape, b.shape);

    const derA = () => {
      const reduceAxes = getReductionAxes(a.shape, outShape);
      if (reduceAxes.length > 0) {
        return dy.sum(reduceAxes).reshape(a.shape);
      }
      return dy;
    };
    const derB = () => {
      const res = dy.mul(a.div(b).floor().neg());
      const reduceAxes = getReductionAxes(b.shape, outShape);
      if (reduceAxes.length > 0) {
        return res.sum(reduceAxes).reshape(b.shape);
      }
      return res;
    };
    return {a: derA, b: derB};
  }
};
