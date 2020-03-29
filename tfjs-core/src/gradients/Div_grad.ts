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

import {Div} from '../kernel_names';
import {GradConfig} from '../kernel_registry';
import * as broadcast_util from '../ops/broadcast_util';
import {div} from '../ops/div';
import {sum} from '../ops/reduction_ops';
import {square} from '../ops/square';
import {neg} from '../ops/unary_ops';
import {Tensor} from '../tensor';

export const divGradConfig: GradConfig = {
  kernelName: Div,
  inputsToSave: ['a', 'b'],
  gradFunc: (dy: Tensor, saved: Tensor[]) => {
    const [a, b] = saved;
    const outShape =
        broadcast_util.assertAndGetBroadcastShape(a.shape, b.shape);
    const derA = () => {
      const res = div(dy, b.toFloat());
      const reduceAxes = broadcast_util.getReductionAxes(a.shape, outShape);
      if (reduceAxes.length > 0) {
        return sum(res, reduceAxes).reshape(a.shape);
      }
      return res;
    };
    const derB = () => {
      let res = dy.mul(a.toFloat());
      const reduceAxes = broadcast_util.getReductionAxes(b.shape, outShape);
      if (reduceAxes.length > 0) {
        res = sum(res, reduceAxes).reshape(b.shape);
      }
      const tmp = square(b);
      return neg(div(res, tmp.toFloat()));
    };
    return {a: derA, b: derB};
  }
};
