/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
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
import {ENGINE} from '../../engine';
import {Transform, TransformAttrs, TransformInputs} from '../../kernel_names';
import {NamedAttrMap} from '../../kernel_registry';
import {Tensor2D, Tensor4D} from '../../tensor';
import {NamedTensorMap} from '../../tensor_types';
import {convertToTensor} from '../../tensor_util_env';
import {TensorLike} from '../../types';
import * as util from '../../util';

import {op} from '../operation';

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
 *
 * @doc {heading: 'Operations', subheading: 'Images', namespace: 'image'}
 */
function transform_(
    image: Tensor4D|TensorLike, transforms: Tensor2D|TensorLike,
    interpolation: 'nearest'|'bilinear' = 'nearest',
    fillMode: 'constant'|'reflect'|'wrap'|'nearest' = 'constant', fillValue = 0,
    outputShape?: [number, number]): Tensor4D {
  const $image = convertToTensor(image, 'image', 'transform', 'float32');
  const $transforms =
      convertToTensor(transforms, 'transforms', 'transform', 'float32');

  util.assert(
      $image.rank === 4,
      () => 'Error in transform: image must be rank 4,' +
          `but got rank ${$image.rank}.`);

  util.assert(
      $transforms.rank === 2 &&
          ($transforms.shape[0] === $image.shape[0] ||
           $transforms.shape[0] === 1) &&
          $transforms.shape[1] === 8,
      () => `Error in transform: Input transform should be batch x 8 or 1 x 8`);

  util.assert(
      outputShape == null || outputShape.length === 2,
      () =>
          'Error in transform: outputShape must be [height, width] or null, ' +
          `but got ${outputShape}.`);

  const inputs: TransformInputs = {image: $image, transforms: $transforms};
  const attrs:
      TransformAttrs = {interpolation, fillMode, fillValue, outputShape};

  return ENGINE.runKernel(
             Transform, inputs as {} as NamedTensorMap,
             attrs as {} as NamedAttrMap) as Tensor4D;
}

export const transform = op({transform_});
