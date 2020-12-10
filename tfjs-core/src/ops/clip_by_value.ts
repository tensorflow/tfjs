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
import {ENGINE} from '../engine';
import {ClipByValue, ClipByValueAttrs, ClipByValueInputs} from '../kernel_names';
import {NamedAttrMap} from '../kernel_registry';
import {Tensor} from '../tensor';
import {NamedTensorMap} from '../tensor_types';
import {convertToTensor} from '../tensor_util_env';
import {TensorLike} from '../types';
import * as util from '../util';

import {op} from './operation';

/**
 * Clips values element-wise. `max(min(x, clipValueMax), clipValueMin)`
 *
 * ```js
 * const x = tf.tensor1d([-1, 2, -3, 4]);
 *
 * x.clipByValue(-2, 3).print();  // or tf.clipByValue(x, -2, 3)
 * ```
 * @param x The input tensor.
 * @param clipValueMin Lower-bound of range to be clipped to.
 * @param clipValueMax Upper-bound of range to be clipped to.
 *
 * @doc {heading: 'Operations', subheading: 'Basic math'}
 */
function clipByValue_<T extends Tensor>(
    x: T|TensorLike, clipValueMin: number, clipValueMax: number): T {
  const $x = convertToTensor(x, 'x', 'clipByValue');
  util.assert(
      (clipValueMin <= clipValueMax),
      () => `Error in clip: min (${clipValueMin}) must be ` +
          `less than or equal to max (${clipValueMax}).`);

  const inputs: ClipByValueInputs = {x: $x};
  const attrs: ClipByValueAttrs = {clipValueMin, clipValueMax};

  return ENGINE.runKernel(
      ClipByValue, inputs as {} as NamedTensorMap, attrs as {} as NamedAttrMap);
}

export const clipByValue = op({clipByValue_});
