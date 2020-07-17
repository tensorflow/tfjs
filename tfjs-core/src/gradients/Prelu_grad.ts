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
import {Prelu} from '../kernel_names';
import {GradConfig} from '../kernel_registry';
import {getReductionAxes} from '../ops/broadcast_util';
import {greater} from '../ops/greater';
import {mul} from '../ops/mul';
import {reshape} from '../ops/reshape';
import {sum} from '../ops/sum';
import {where} from '../ops/where';
import {zerosLike} from '../ops/zeros_like';
import {Tensor} from '../tensor';

export const preluGradConfig: GradConfig = {
  kernelName: Prelu,
  inputsToSave: ['x', 'alpha'],
  gradFunc: (dy: Tensor, saved: Tensor[]) => {
    const [x, alpha] = saved;
    const mask = greater(x, 0);

    return {
      x: () => where(mask, dy, mul(dy, alpha)),
      alpha: () => {
        let res = where(mask, zerosLike(dy), mul(dy, x));
        const reduceAxes = getReductionAxes(alpha.shape, dy.shape);
        if (reduceAxes.length > 0) {
          res = sum(res, reduceAxes);
        }
        return reshape(res, alpha.shape);
      }
    };
  }
};
