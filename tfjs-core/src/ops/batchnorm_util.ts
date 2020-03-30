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
import {deprecationWarn} from '../globals';
import {Tensor, Tensor4D} from '../tensor';
import {Rank} from '../types';

export function warnDeprecation(): void {
  deprecationWarn(
      'tf.batchNormalization() is going away. ' +
      'Use tf.batchNorm() instead, and note the positional argument change ' +
      'of scale, offset, and varianceEpsilon');
}

export function xAs4D<R extends Rank>(x: Tensor<R>) {
  let x4D: Tensor4D;
  if (x.rank === 0 || x.rank === 1) {
    x4D = x.as4D(1, 1, 1, x.size);
  } else if (x.rank === 2) {
    x4D = x.as4D(1, 1, x.shape[0], x.shape[1]);
  } else if (x.rank === 3) {
    x4D = x.as4D(1, x.shape[0], x.shape[1], x.shape[2]);
  } else {
    x4D = x as Tensor4D;
  }

  return x4D;
}
