/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
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

import {op} from '../ops/operation';
import {Tensor, Tensor1D} from '../tensor';

import {concat} from './concat';
import {fill} from './fill';
import {reshape} from './reshape';
import {slice} from './slice';
import {tensor2d} from './tensor2d';

/**
 * Expands input into frames of frameLength.
 * Slides a window size with frameStep.
 *
 * ```js
 * tf.signal.frame([1, 2, 3], 2, 1).print();
 * ```
 * @param signal The input tensor to be expanded
 * @param frameLength Length of each frame
 * @param frameStep The frame hop size in samples.
 * @param padEnd Whether to pad the end of signal with padValue.
 * @param padValue An number to use where the input signal does
 *     not exist when padEnd is True.
 */
/**
 * @doc {heading: 'Operations', subheading: 'Signal', namespace: 'signal'}
 */
function frame_(
    signal: Tensor1D, frameLength: number, frameStep: number, padEnd = false,
    padValue = 0): Tensor {
  let start = 0;
  const output: Tensor[] = [];
  while (start + frameLength <= signal.size) {
    output.push(slice(signal, start, frameLength));
    start += frameStep;
  }

  if (padEnd) {
    while (start < signal.size) {
      const padLen = (start + frameLength) - signal.size;
      const pad = concat([
        slice(signal, start, frameLength - padLen), fill([padLen], padValue)
      ]);
      output.push(pad);
      start += frameStep;
    }
  }

  if (output.length === 0) {
    return tensor2d([], [0, frameLength]);
  }

  return reshape(concat(output), [output.length, frameLength]);
}
export const frame = op({frame_});
