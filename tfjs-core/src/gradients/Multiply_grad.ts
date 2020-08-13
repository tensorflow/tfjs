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

import {Multiply} from '../kernel_names';
import {GradConfig} from '../kernel_registry';
import {assertAndGetBroadcastShape, getReductionAxes} from '../ops/broadcast_util';
import {cast} from '../ops/cast';
import {mul} from '../ops/mul';
import {reshape} from '../ops/reshape';
import {sum} from '../ops/sum';
import {Tensor} from '../tensor';

export const multiplyGradConfig: GradConfig = {
  kernelName: Multiply,
  inputsToSave: ['a', 'b'],
  gradFunc: (dy: Tensor, saved: Tensor[]) => {
    const [a, b] = saved;
    const outShape = assertAndGetBroadcastShape(a.shape, b.shape);

    const derA = () => {
      const res = mul(dy, cast(b, 'float32'));
      const reduceAxes = getReductionAxes(a.shape, outShape);
      if (reduceAxes.length > 0) {
        return reshape(sum(res, reduceAxes), a.shape);
      }
      return res;
    };
    const derB = () => {
      const res = mul(dy, cast(a, 'float32'));
      const reduceAxes = getReductionAxes(b.shape, outShape);
      if (reduceAxes.length > 0) {
        return reshape(sum(res, reduceAxes), b.shape);
      }
      return res;
    };
    return {a: derA, b: derB};
  }
};
