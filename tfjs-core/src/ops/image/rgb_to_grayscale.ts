/**
 * @license
 * Copyright 2023 Google LLC.
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
import {cast} from '../cast';
import {einsum} from '../einsum';
import {expandDims} from '../expand_dims';
import {op} from '../operation';
import {tensor1d} from '../tensor1d';

/**
 * Converts images from RGB format to grayscale.
 *
 * @param image A RGB tensor to convert. The `image`'s last dimension must
 *     be size 3 with at least a two-dimensional shape.
 *
 * @doc {heading: 'Operations', subheading: 'Images', namespace: 'image'}
 */
function rgbToGrayscale_<T extends Tensor2D|Tensor3D|Tensor4D|Tensor5D|
                         Tensor6D>(image: T|TensorLike): T {
  const $image = convertToTensor(image, 'image', 'RGBToGrayscale');

  const lastDimsIdx = $image.rank - 1;
  const lastDims = $image.shape[lastDimsIdx];

  util.assert(
      $image.rank >= 2,
      () => 'Error in RGBToGrayscale: images must be at least rank 2, ' +
          `but got rank ${$image.rank}.`);

  util.assert(
      lastDims === 3,
      () => 'Error in RGBToGrayscale: last dimension of an RGB image ' +
          `should be size 3, but got size ${lastDims}.`);

  // Remember original dtype so we can convert back if needed
  const origDtype = $image.dtype;
  const fltImage = cast($image, 'float32');

  const rgbWeights = tensor1d([0.2989, 0.5870, 0.1140]);

  let grayFloat;
  switch ($image.rank) {
    case 2:
      grayFloat = einsum('ij,j->i', fltImage, rgbWeights);
      break;
    case 3:
      grayFloat = einsum('ijk,k->ij', fltImage, rgbWeights);
      break;
    case 4:
      grayFloat = einsum('ijkl,l->ijk', fltImage, rgbWeights);
      break;
    case 5:
      grayFloat = einsum('ijklm,m->ijkl', fltImage, rgbWeights);
      break;
    case 6:
      grayFloat = einsum('ijklmn,n->ijklm', fltImage, rgbWeights);
      break;
    default:
      throw new Error('Not a valid tensor rank.');
  }
  grayFloat = expandDims(grayFloat, -1);

  return cast(grayFloat, origDtype) as T;
}

export const rgbToGrayscale = /* @__PURE__ */ op({rgbToGrayscale_});
