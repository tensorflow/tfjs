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

import {Tensor2D, Tensor3D, Tensor4D, Tensor5D, Tensor6D} from '../../tensor';
import {convertToTensor} from '../../tensor_util_env';
import {TensorLike} from '../../types';
import * as util from '../../util';

import {op} from '../operation';
import {tile} from '../tile';

/**
 * Converts images from grayscale to RGB format.
 *
 * @param image A grayscale tensor to convert. The `image`'s last dimension must
 *     be size 1 with at least a two-dimensional shape.
 *
 * @doc {heading: 'Operations', subheading: 'Images', namespace: 'image'}
 */
function grayscaleToRGB_<T extends Tensor2D|Tensor3D|Tensor4D|Tensor5D|
                         Tensor6D>(image: T|TensorLike): T {
  const $image = convertToTensor(image, 'image', 'grayscaleToRGB');

  const lastDimsIdx = $image.rank - 1;
  const lastDims = $image.shape[lastDimsIdx];

  util.assert(
      $image.rank >= 2,
      () => 'Error in grayscaleToRGB: images must be at least rank 2, ' +
          `but got rank ${$image.rank}.`);

  util.assert(
      lastDims === 1,
      () => 'Error in grayscaleToRGB: last dimension of a grayscale image ' +
          `should be size 1, but got size ${lastDims}.`);

  const reps = new Array($image.rank);

  reps.fill(1, 0, lastDimsIdx);
  reps[lastDimsIdx] = 3;

  return tile($image, reps);
}

export const grayscaleToRGB = op({grayscaleToRGB_});
