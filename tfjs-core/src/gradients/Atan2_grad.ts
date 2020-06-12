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

import {Atan2} from '../kernel_names';
import {GradConfig} from '../kernel_registry';
import {add} from '../ops/add';
import {reshape} from '../ops/array_ops';
import {assertAndGetBroadcastShape, getReductionAxes} from '../ops/broadcast_util';
import {div} from '../ops/div';
import {mul} from '../ops/mul';
import {sum} from '../ops/reduction_ops';
import {square} from '../ops/square';
import {neg} from '../ops/unary_ops';
import {Tensor} from '../tensor';

export const atan2GradConfig: GradConfig = {
  kernelName: Atan2,
  inputsToSave: ['a', 'b'],
  gradFunc: (dy: Tensor, saved: Tensor[]) => {
    const [a, b] = saved;
    const outShape = assertAndGetBroadcastShape(a.shape, b.shape);

    const derA = () => {
      const d = add(square(a), square(b));
      let res = mul(dy, div(b, d));
      const reduceAxes = getReductionAxes(a.shape, outShape);
      if (reduceAxes.length > 0) {
        res = sum(res, reduceAxes);
      }
      return reshape(res, a.shape);
    };
    const derB = () => {
      const d = add(square(a), square(b));
      let res = neg(mul(dy, div(a, d)));
      const reduceAxes = getReductionAxes(b.shape, outShape);
      if (reduceAxes.length > 0) {
        res = sum(res, reduceAxes);
      }
      return reshape(res, b.shape);
    };
    return {a: derA, b: derB};
  }
};
