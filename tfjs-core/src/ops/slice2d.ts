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

import {Tensor2D} from '../tensor';
import {convertToTensor} from '../tensor_util_env';
import {TensorLike} from '../types';
import * as util from '../util';

import {op} from './operation';
import {slice} from './slice';

/**
 * Extracts a 2D slice from a 2D array starting at coordinates `begin` and
 * is of size `size`. See `slice` for details.
 */
function slice2d_(
    x: Tensor2D|TensorLike, begin: [number, number],
    size: [number, number]): Tensor2D {
  const $x = convertToTensor(x, 'x', 'slice2d');
  util.assert(
      $x.rank === 2,
      () =>
          `slice2d expects a rank-2 tensor, but got a rank-${$x.rank} tensor`);
  return slice($x, begin, size);
}
export const slice2d = op({slice2d_});
