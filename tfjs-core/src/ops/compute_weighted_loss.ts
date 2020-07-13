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
import {Tensor} from '../tensor';
import {convertToTensor} from '../tensor_util_env';
import {TensorLike} from '../types';

import {cast} from './cast';
import {div} from './div';
import {Reduction} from './loss_ops_utils';
import {mean} from './mean';
import {mul} from './mul';
import {notEqual} from './not_equal';
import {ones} from './ones';
import {op} from './operation';
import {scalar} from './scalar';
import {sum} from './sum';

/**
 * Computes the weighted loss between two tensors.
 *
 * @param losses Tensor of shape `[batch_size, d1, ... dN]`.
 * @param weights Tensor whose rank is either 0, or the same rank as
 *    `losses`, and must be broadcastable to `losses` (i.e., all
 *    dimensions must be either `1`, or the same as the corresponding
 *    `losses` dimension).
 */
/** @doc {heading: 'Training', subheading: 'Losses', namespace: 'losses'} */
function computeWeightedLoss_<T extends Tensor, O extends Tensor>(
    losses: T|TensorLike, weights?: Tensor|TensorLike,
    reduction = Reduction.SUM_BY_NONZERO_WEIGHTS): O {
  const $losses = convertToTensor(losses, 'losses', 'computeWeightedLoss');
  let $weights: Tensor = null;
  if (weights != null) {
    $weights = convertToTensor(weights, 'weights', 'computeWeightedLoss');
  }

  const weightedLoss = ($weights == null) ? $losses : mul($losses, $weights);

  if (reduction === Reduction.NONE) {
    return weightedLoss as O;
  }
  if (reduction === Reduction.SUM) {
    return sum(weightedLoss);
  }
  if (reduction === Reduction.MEAN) {
    if ($weights == null) {
      return mean(weightedLoss);
    } else {
      const broadcastFactor = $losses.size / $weights.size;
      const result = div(sum(weightedLoss), sum($weights));
      return broadcastFactor > 1 ? div(result, scalar(broadcastFactor)) :
                                   result as O;
    }
  }
  if (reduction === Reduction.SUM_BY_NONZERO_WEIGHTS) {
    if ($weights == null) {
      return div(sum(weightedLoss), scalar($losses.size));
    } else {
      const broadcastedWeights = mul($weights, ones($losses.shape));

      const numNonZeros =
          cast(sum(notEqual(broadcastedWeights, scalar(0))), 'float32');
      return div(sum(weightedLoss), numNonZeros);
    }
  }

  throw Error(`Unknown reduction: ${reduction}`);
}
export const computeWeightedLoss = op({computeWeightedLoss_});
