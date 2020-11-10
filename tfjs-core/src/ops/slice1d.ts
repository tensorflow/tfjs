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

import {Tensor1D} from '../tensor';
import {convertToTensor} from '../tensor_util_env';
import {TensorLike} from '../types';
import * as util from '../util';

import {op} from './operation';
import {slice} from './slice';

/**
 * Extracts a 1D slice from 1D array starting at coordinates `begin` and is
 * of length `size`. See `slice` for details.
 */
function slice1d_(
    x: Tensor1D|TensorLike, begin: number, size: number): Tensor1D {
  const $x = convertToTensor(x, 'x', 'slice1d');
  util.assert(
      $x.rank === 1,
      () =>
          `slice1d expects a rank-1 tensor, but got a rank-${$x.rank} tensor`);
  return slice($x, [begin], [size]);
}
export const slice1d = op({slice1d_});
