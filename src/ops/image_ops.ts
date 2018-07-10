/**
 * @license
 * Copyright 2018 Google Inc. All Rights Reserved.
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

import {doc} from '../doc';
import {ForwardFunc} from '../engine';
import {ENV} from '../environment';
import {Tensor, Tensor3D, Tensor4D} from '../tensor';
import {convertToTensor} from '../tensor_util';
import {TensorLike} from '../types';
import * as util from '../util';
import {op} from './operation';

class ImageOps {
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
   */
  @doc({heading: 'Operations', subheading: 'Images', namespace: 'image'})
  static resizeBilinear<T extends Tensor3D|Tensor4D>(
      images: T|TensorLike, size: [number, number], alignCorners = false): T {
    const $images = convertToTensor(images, 'images', 'resizeBilinear');
    util.assert(
        $images.rank === 3 || $images.rank === 4,
        `Error in resizeBilinear: x must be rank 3 or 4, but got ` +
            `rank ${$images.rank}.`);
    util.assert(
        size.length === 2,
        `Error in resizeBilinear: new shape must 2D, but got shape ` +
            `${size}.`);

    let batchImages = $images as Tensor4D;
    let reshapedTo4D = false;
    if ($images.rank === 3) {
      reshapedTo4D = true;
      batchImages =
          $images.as4D(1, $images.shape[0], $images.shape[1], $images.shape[2]);
    }

    const [newHeight, newWidth] = size;
    const forward: ForwardFunc<Tensor4D> = (backend, save) =>
        backend.resizeBilinear(batchImages, newHeight, newWidth, alignCorners);

    const backward = (dy: Tensor4D, saved: Tensor[]) => {
      return {
        batchImages: () => ENV.engine.runKernel(
            backend =>
                backend.resizeBilinearBackprop(dy, batchImages, alignCorners),
            {})
      };
    };

    const res = ENV.engine.runKernel(forward, {batchImages}, backward);
    if (reshapedTo4D) {
      return res.as3D(res.shape[1], res.shape[2], res.shape[3]) as T;
    }
    return res as T;
  }

  /**
   * NearestNeighbor resize a batch of 3D images to a new shape.
   *
   * @param images The images, of rank 4 or rank 3, of shape
   *     `[batch, height, width, inChannels]`. If rank 3, batch of 1 is assumed.
   * @param size The new shape `[newHeight, newWidth]` to resize the
   *     images to. Each channel is resized individually.
   * @param alignCorners Defaults to False. If true, rescale
   *     input by `(new_height - 1) / (height - 1)`, which exactly aligns the 4
   *     corners of images and resized images. If false, rescale by
   *     `new_height / height`. Treat similarly the width dimension.
   */
  @doc({heading: 'Operations', subheading: 'Images', namespace: 'image'})
  static resizeNearestNeighbor<T extends Tensor3D|Tensor4D>(
      images: T|TensorLike, size: [number, number], alignCorners = false): T {
    const $images = convertToTensor(images, 'images', 'resizeNearestNeighbor');
    util.assert(
        $images.rank === 3 || $images.rank === 4,
        `Error in resizeNearestNeighbor: x must be rank 3 or 4, but got ` +
            `rank ${$images.rank}.`);
    util.assert(
        size.length === 2,
        `Error in resizeNearestNeighbor: new shape must 2D, but got shape ` +
            `${size}.`);
    util.assert(
        $images.dtype === 'float32' || $images.dtype === 'int32',
        '`images` must have `int32` or `float32` as dtype');

    let batchImages = $images as Tensor4D;
    let reshapedTo4D = false;
    if ($images.rank === 3) {
      reshapedTo4D = true;
      batchImages =
          $images.as4D(1, $images.shape[0], $images.shape[1], $images.shape[2]);
    }
    const [newHeight, newWidth] = size;

    const forward: ForwardFunc<Tensor4D> = (backend, save) =>
        backend.resizeNearestNeighbor(
            batchImages, newHeight, newWidth, alignCorners);

    const backward = (dy: Tensor4D, saved: Tensor[]) => {
      return {
        batchImages: () => ENV.engine.runKernel(
            backend => backend.resizeNearestNeighborBackprop(
                dy, batchImages, alignCorners),
            {})
      };
    };

    const res = ENV.engine.runKernel(forward, {batchImages}, backward);

    if (reshapedTo4D) {
      return res.as3D(res.shape[1], res.shape[2], res.shape[3]) as T;
    }
    return res as T;
  }
}

export const resizeBilinear = op(ImageOps.resizeBilinear);
export const resizeNearestNeighbor = op(ImageOps.resizeNearestNeighbor);
