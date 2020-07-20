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

import {ENGINE} from '../engine';
import {RotateWithOffset, RotateWithOffsetAttrs, RotateWithOffsetInputs} from '../kernel_names';
import {NamedAttrMap} from '../kernel_registry';
import {Tensor4D} from '../tensor';
import {NamedTensorMap} from '../tensor_types';
import {convertToTensor} from '../tensor_util_env';
import {TensorLike} from '../types';
import * as util from '../util';

import {op} from './operation';

/**
 * Rotates the input image tensor counter-clockwise with an optional offset
 * center of rotation. Currently available in the CPU, WebGL, and WASM backends.
 *
 * @param image 4d tensor of shape `[batch, imageHeight, imageWidth, depth]`.
 * @param radians The amount of rotation.
 * @param fillValue The value to fill in the empty space leftover
 *     after rotation. Can be either a single grayscale value (0-255), or an
 *     array of three numbers `[red, green, blue]` specifying the red, green,
 *     and blue channels. Defaults to `0` (black).
 * @param center The center of rotation. Can be either a single value (0-1), or
 *     an array of two numbers `[centerX, centerY]`. Defaults to `0.5` (rotates
 *     the image around its center).
 */
/** @doc {heading: 'Operations', subheading: 'Images', namespace: 'image'} */
function rotateWithOffset_(
    image: Tensor4D|TensorLike, radians: number,
    fillValue: number|[number, number, number] = 0,
    center: number|[number, number] = 0.5): Tensor4D {
  const $image = convertToTensor(image, 'image', 'rotateWithOffset', 'float32');

  util.assert(
      $image.rank === 4,
      () => 'Error in rotateWithOffset: image must be rank 4,' +
          `but got rank ${$image.rank}.`);

  const inputs: RotateWithOffsetInputs = {image: $image};
  const attrs: RotateWithOffsetAttrs = {radians, fillValue, center};
  const res = ENGINE.runKernel(
      RotateWithOffset, inputs as {} as NamedTensorMap,
      attrs as {} as NamedAttrMap);
  return res as Tensor4D;
}

export const rotateWithOffset = op({rotateWithOffset_});
