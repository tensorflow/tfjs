/**
 * @license
 * Copyright 2018 Google Inc. All Rights Reserved.
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

import {doc} from '../doc';
import {Scalar, Tensor} from '../tensor';
import * as util from '../util';
import {ArrayOps} from './array_ops';
import {BinaryOps} from './binary_ops';
import {operation} from './operation';

export class MovingAverageOps {
  /**
   * Compute the moving average of a variable.
   *
   * Without zeroDebias, the moving average operation is defined by:
   *   `v += delta`
   * where
   *   `delta = (1 - decay) * (x - v)`
   *
   * With zeroDebias (default), the `delta` term is scaled to debias the
   * effect of the (assumed) zero-initialization of `v`.
   *   `delta /= (1 - decay ^ step)`
   *
   * For more details on the zero-debiasing algorithm, see:
   *   https://arxiv.org/abs/1412.6980
   *
   * Note that this function is completely stateless and does not keep track of
   * step count. The step count needs to be maintained by the caller and passed
   * in as `step`.
   *
   * @param v The current moving average value.
   * @param x New input value, must have the same shape and dtype as `v`.
   * @param decay The decay factor. Typical values are 0.95 and 0.99.
   * @param step Step count.
   * @param zeroDebias: Whether zeroDebias is to be performed (default: `true`).
   * @returns The new moving average value.
   */
  @doc({heading: 'Operations', subheading: 'Moving Average'})
  @operation
  static movingAverage<T extends Tensor>(
      v: T, x: T, decay: number|Scalar, step?: number|Scalar,
      zeroDebias = true): T {
    util.assertTypesMatch(v, x);
    util.assert(
        util.arraysEqual(v.shape, x.shape), 'Shape mismatch in v and x');

    const one = ArrayOps.scalar(1);
    decay = typeof decay === 'number' ? ArrayOps.scalar(decay) : decay;
    const oneMinusDecay = one.sub(decay);

    let update = x.sub(v).mul(oneMinusDecay);
    if (zeroDebias) {
      util.assert(
          step != null, 'When using zeroDebias: true, step is required.');
      step = typeof step === 'number' ? ArrayOps.scalar(step) : step;
      update = update.div(one.sub(BinaryOps.pow(decay, step)));
    }
    return v.add(update);
  }
}
