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
import {Tensor4D} from '../tensor';
import {TensorLike} from '../types';
import {assert} from '../util';
import {op} from './operation';
import {pad} from './pad';

/**
 * Pads a `tf.Tensor4D` with a given value and paddings. See `pad` for details.
 */
function pad4d_(
    x: Tensor4D|TensorLike,
    paddings:
        [
          [number, number], [number, number], [number, number], [number, number]
        ],
    constantValue = 0): Tensor4D {
  assert(
      paddings.length === 4 && paddings[0].length === 2 &&
          paddings[1].length === 2 && paddings[2].length === 2 &&
          paddings[3].length === 2,
      () => 'Invalid number of paddings. Must be length of 2 each.');
  return pad(x, paddings, constantValue);
}

export const pad4d = op({pad4d_});
