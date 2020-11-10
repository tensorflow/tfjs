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

import {Tensor3D} from '../tensor';
import {convertToTensor} from '../tensor_util_env';
import {TensorLike} from '../types';
import * as util from '../util';

import {op} from './operation';
import {slice} from './slice';

/**
 * Extracts a 3D slice from a 3D array starting at coordinates `begin` and
 * is of size `size`. See `slice` for details.
 */
function slice3d_(
    x: Tensor3D|TensorLike, begin: [number, number, number],
    size: [number, number, number]): Tensor3D {
  const $x = convertToTensor(x, 'x', 'slice3d');
  util.assert(
      $x.rank === 3,
      () =>
          `slice3d expects a rank-3 tensor, but got a rank-${$x.rank} tensor`);
  return slice($x, begin, size);
}
export const slice3d = op({slice3d_});
