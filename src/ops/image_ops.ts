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
import {ENV} from '../environment';
import {Tensor3D, Tensor4D} from '../tensor';
import * as util from '../util';
import {operation} from './operation';

export class ImageOps {
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
  @operation
  static resizeBilinear<T extends Tensor3D|Tensor4D>(
      images: T, size: [number, number], alignCorners = false): T {
    util.assert(
        images.rank === 3 || images.rank === 4,
        `Error in resizeBilinear: x must be rank 3 or 4, but got ` +
            `rank ${images.rank}.`);
    util.assert(
        size.length === 2,
        `Error in resizeBilinear: new shape must 2D, but got shape ` +
            `${size}.`);
    let batchImages = images as Tensor4D;
    let reshapedTo4D = false;
    if (images.rank === 3) {
      reshapedTo4D = true;
      batchImages =
          images.as4D(1, images.shape[0], images.shape[1], images.shape[2]);
    }
    const [newHeight, newWidth] = size;
    const res = ENV.engine.runKernel(
        backend => backend.resizeBilinear(
            batchImages, newHeight, newWidth, alignCorners));
    if (reshapedTo4D) {
      return res.as3D(res.shape[1], res.shape[2], res.shape[3]) as T;
    }
    return res as T;
  }
}
