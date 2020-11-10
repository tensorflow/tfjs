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

import {Tensor1D} from '../tensor';
import {convertToTensor} from '../tensor_util_env';
import {TensorLike} from '../types';
import * as util from '../util';
import {op} from './operation';
import {reverse} from './reverse';

/**
 * Reverses a `tf.Tensor1D`.
 *
 * @param x The input tensor.
 */
function reverse1d_(x: Tensor1D|TensorLike): Tensor1D {
  const $x = convertToTensor(x, 'x', 'reverse');
  util.assert(
      $x.rank === 1,
      () => `Error in reverse1D: x must be rank 1 but got rank ${$x.rank}.`);
  return reverse($x, 0);
}

export const reverse1d = op({reverse1d_});
