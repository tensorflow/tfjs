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

import {ForwardFunc} from '../engine';
import {ENV} from '../environment';
import {Tensor, Tensor1D, Tensor2D, Tensor3D, Tensor4D} from '../tensor';
import {convertToTensor} from '../tensor_util';
import {TensorLike} from '../types';
import * as util from '../util';
import {op} from './operation';
import {tensor1d} from './tensor_ops';

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
/** @doc {heading: 'Operations', subheading: 'Images', namespace: 'image'} */
function resizeBilinear_<T extends Tensor3D|Tensor4D>(
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
/** @doc {heading: 'Operations', subheading: 'Images', namespace: 'image'} */
function resizeNearestNeighbor_<T extends Tensor3D|Tensor4D>(
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

/**
 * Performs non maximum suppression of bounding boxes based on
 * iou (intersection over union)
 *
 * @param boxes tensor with bounding boxes, of rank 2,
 *     of shape `[numBoxes, 4]`
 * @param scores tensor containing the scores box scores, of rank 1,
 *     of shape `[numBoxes]`
 * @param maxOutputSize tensor, of rank 1, of type int32, containing the max
 *     number of returned indices
 * @param iouThreshold max iou threshold value, must have value between [0, 1]
 * @param scoreThreshold min score threshold value
 */
/** @doc {heading: 'Operations', subheading: 'Images', namespace: 'image'} */
function nonMaxSuppression_(
    boxes: Tensor2D, scores: Tensor1D, maxOutputSize: Tensor1D,
    iouThreshold: number, scoreThreshold: number): Tensor1D {
  util.assert(
      0 <= iouThreshold && iouThreshold <= 1, 'iouThreshold must be in [0, 1]');
  util.assert(boxes.shape.length === 2, 'boxes must be 2-D');
  util.assert(boxes.shape[1] === 4, 'boxes must have 4 columns');

  const numBoxes = boxes.shape[0];
  util.assert(scores.shape.length === 1, 'scores must be 1-D');
  util.assert(
      scores.shape[0] === numBoxes,
      'scores has incompatible shape, expected [numBoxes]');

  const outputSize =
      Math.min(...Array.from(maxOutputSize.dataSync()), numBoxes);

  const candidates = Array.from(scores.dataSync())
                         .map((score, boxIndex) => ({score, boxIndex}))
                         .filter(c => c.score > scoreThreshold)
                         .sort((c1, c2) => c2.score - c1.score);

  const suppressFunc = (x: number) => x <= iouThreshold ? 1 : 0;

  const selected: number[] = [];

  candidates.forEach(c => {
    if (selected.length >= outputSize) {
      return;
    }
    const originalScore = c.score;

    for (let j = selected.length - 1; j >= 0; --j) {
      const iou = IOU(boxes, c.boxIndex, selected[j]);
      if (iou === 0.0) {
        continue;
      }
      c.score *= suppressFunc(iou);
      if (c.score <= scoreThreshold) {
        break;
      }
    }

    if (originalScore === c.score) {
      selected.push(c.boxIndex);
    }
  });

  return tensor1d(selected, 'int32');
}

function IOU(boxes: Tensor2D, i: number, j: number) {
  const yminI = Math.min(boxes.get(i, 0), boxes.get(i, 2));
  const xminI = Math.min(boxes.get(i, 1), boxes.get(i, 3));
  const ymaxI = Math.max(boxes.get(i, 0), boxes.get(i, 2));
  const xmaxI = Math.max(boxes.get(i, 1), boxes.get(i, 3));
  const yminJ = Math.min(boxes.get(j, 0), boxes.get(j, 2));
  const xminJ = Math.min(boxes.get(j, 1), boxes.get(j, 3));
  const ymaxJ = Math.max(boxes.get(j, 0), boxes.get(j, 2));
  const xmaxJ = Math.max(boxes.get(j, 1), boxes.get(j, 3));
  const areaI = (ymaxI - yminI) * (xmaxI - xminI);
  const areaJ = (ymaxJ - yminJ) * (xmaxJ - xminJ);
  if (areaI <= 0 || areaJ <= 0) {
    return 0.0;
  }
  const intersectionYmin = Math.max(yminI, yminJ);
  const intersectionXmin = Math.max(xminI, xminJ);
  const intersectionYmax = Math.min(ymaxI, ymaxJ);
  const intersectionXmax = Math.min(xmaxI, xmaxJ);
  const intersectionArea = Math.max(intersectionYmax - intersectionYmin, 0.0) *
      Math.max(intersectionXmax - intersectionXmin, 0.0);
  return intersectionArea / (areaI + areaJ - intersectionArea);
}

export const resizeBilinear = op({resizeBilinear_});
export const resizeNearestNeighbor = op({resizeNearestNeighbor_});
export const nonMaxSuppression = op({nonMaxSuppression_});
