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
 * Applies the given transform(s) to the image(s).
 *
 * @param image 4d tensor of shape `[batch, imageHeight, imageWidth, depth]`.
 * @param transforms Projective transform matrix/matrices. A tensor1d of length
 *     8 or tensor of size N x 8. If one row of transforms is [a0, a1, a2, b0
 *     b1, b2, c0, c1], then it maps the output point (x, y) to a transformed
 *     input point (x', y') = ((a0 x + a1 y + a2) / k, (b0 x + b1 y + b2) / k),
 *     where k = c0 x + c1 y + 1. The transforms are inverted compared to the
 *     transform mapping input points to output points.
 * @param interpolation Interpolation mode.
 *     Supported values: 'nearest', 'bilinear'. Default to 'nearest'.
 * @param fillMode Points outside the boundaries of the input are filled
 *     according to the given mode, one of 'constant', 'reflect', 'wrap',
 *     'nearest'. Default to 'constant'.
 *     'reflect': (d c b a | a b c d | d c b a ) The input is extended by
 *     reflecting about the edge of the last pixel.
 *     'constant': (k k k k | a b c d | k k k k) The input is extended by
 *     filling all values beyond the edge with the same constant value k.
 *     'wrap': (a b c d | a b c d | a b c d) The input is extended by
 *     wrapping around to the opposite edge.
 *     'nearest': (a a a a | a b c d | d d d d) The input is extended by
 *     the nearest pixel.
 * @param fillValue A float represents the value to be filled outside the
 *     boundaries when fillMode is 'constant'.
 * @param Output dimension after the transform, [height, width]. If undefined,
 *     output is the same size as input image.
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
      Transform, inputs as {} as NamedTensorMap, attrs as {} as NamedAttrMap);
}

export const transform = op({transform_});
