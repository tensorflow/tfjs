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
import {Tensor2D} from '../tensor';
import {TensorLike} from '../types';
import {assert} from '../util';
import {op} from './operation';
import {pad} from './pad';

/**
 * Pads a `tf.Tensor2D` with a given value and paddings. See `pad` for details.
 */
function pad2d_(
    x: Tensor2D|TensorLike, paddings: [[number, number], [number, number]],
    constantValue = 0): Tensor2D {
  assert(
      paddings.length === 2 && paddings[0].length === 2 &&
          paddings[1].length === 2,
      () => 'Invalid number of paddings. Must be length of 2 each.');
  return pad(x, paddings, constantValue);
}

export const pad2d = op({pad2d_});
