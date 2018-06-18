/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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
import {Tensor} from '../tensor';
import * as util from '../util';

import {operation} from './operation';

export class SigmoidCrossEntropyOps {
  /**
   * Computes sigmoid cross entropy given logits.
   *
   * @param labels A Tensor of the same type and shape as logits.
   * @param logits A Tensor of type float32 or float64.
   */
  @doc({heading: 'Operations', subheading: 'Cross Entropy'})
  @operation
  static sigmoidCrossEntropyWithLogits<T extends Tensor, O extends Tensor>(
      labels: T, logits: T): O {
    util.assertArgumentsAreTensors(
        {labels, logits}, 'sigmoidCrossEntropyWithLogits');
    util.assertShapesMatch(
        labels.shape, logits.shape, 'Error in sigmoidCrossEntropyWithLogits: ');

    /**
     * Implementation Details:
     *
     * For brevity, let `x = logits`, `z = labels`.  The logistic loss is
     *     z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
     *   = z * -log(1 / (1 + exp(-x))) + (1 - z) * -log(exp(-x) / (1 + exp(-x)))
     *   = z * log(1 + exp(-x)) + (1 - z) * (-log(exp(-x)) + log(1 + exp(-x)))
     *   = z * log(1 + exp(-x)) + (1 - z) * (x + log(1 + exp(-x))
     *   = (1 - z) * x + log(1 + exp(-x))
     *   = x - x * z + log(1 + exp(-x))
     *
     *   For x < 0, to avoid overflow in exp(-x), we reformulate the above
     *     x - x * z + log(1 + exp(-x))
     *   = log(exp(x)) - x * z + log(1 + exp(-x))
     *   = - x * z + log(1 + exp(x))
     *
     * Hence, to ensure stability and avoid overflow, the implementation uses
     * this equivalent formulation:
     *     max(x, 0) - x * z + log(1 + exp(-abs(x)))
     */
    const maxOutput = logits.relu();
    const outputXTarget = logits.mul(labels);
    const sigmoidOutput = logits.abs().neg().exp().log1p();

    return maxOutput.sub(outputXTarget).add(sigmoidOutput);
  }
}
