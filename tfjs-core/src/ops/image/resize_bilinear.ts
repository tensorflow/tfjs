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

import {ENGINE, ForwardFunc} from '../../engine';
import {ResizeBilinear, ResizeBilinearAttrs, ResizeBilinearInputs} from '../../kernel_names';
import {NamedAttrMap} from '../../kernel_registry';
import {Tensor3D, Tensor4D} from '../../tensor';
import {NamedTensorMap} from '../../tensor_types';
import {convertToTensor} from '../../tensor_util_env';
import {TensorLike} from '../../types';
import * as util from '../../util';

import {op} from '../operation';
import {reshape} from '../reshape';

/**
 * Bilinear resize a batch of 3D images to a new shape.
 *
 * @param images The images, of rank 4 or rank 3, of shape
 *     `[batch, height, width, inChannels]`. If rank 3, batch of 1 is assumed.
 * @param size The new shape `[newHeight, newWidth]` to resize the
 *     images to. Each channel is resized individually.
 * @param alignCorners Defaults to False. If true, rescale
 *     input by `(new_height - 1) / (height - 1)`, which exactly aligns the 4
 *     corners of images and resized images. If false, rescale by
 *     `new_height / height`. Treat similarly the width dimension.
 *
 * @doc {heading: 'Operations', subheading: 'Images', namespace: 'image'}
 */
function resizeBilinear_<T extends Tensor3D|Tensor4D>(
    images: T|TensorLike, size: [number, number], alignCorners = false): T {
  const $images = convertToTensor(images, 'images', 'resizeBilinear');

  util.assert(
      $images.rank === 3 || $images.rank === 4,
      () => `Error in resizeBilinear: x must be rank 3 or 4, but got ` +
          `rank ${$images.rank}.`);
  util.assert(
      size.length === 2,
      () => `Error in resizeBilinear: new shape must 2D, but got shape ` +
          `${size}.`);

  let batchImages = $images as Tensor4D;
  let reshapedTo4D = false;
  if ($images.rank === 3) {
    reshapedTo4D = true;
    batchImages = reshape(
        $images, [1, $images.shape[0], $images.shape[1], $images.shape[2]]);
  }

  const [newHeight, newWidth] = size;
  const forward: ForwardFunc<Tensor4D> = (backend, save) => {
    save([batchImages]);
    return backend.resizeBilinear(
        batchImages, newHeight, newWidth, alignCorners);
  };

  const inputs: ResizeBilinearInputs = {images: batchImages};
  const attrs: ResizeBilinearAttrs = {alignCorners, size};

  const res = ENGINE.runKernelFunc(
      forward, inputs as {} as NamedTensorMap, null /* gradient */,
      ResizeBilinear, attrs as {} as NamedAttrMap);

  if (reshapedTo4D) {
    return reshape(res, [res.shape[1], res.shape[2], res.shape[3]]) as T;
  }
  return res as T;
}

export const resizeBilinear = op({resizeBilinear_});
