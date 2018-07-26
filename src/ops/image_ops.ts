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
import {nonMaxSuppressionImpl} from '../kernels/non_max_suppression_impl';
import {Tensor, Tensor1D, Tensor2D, Tensor3D, Tensor4D} from '../tensor';
import {convertToTensor} from '../tensor_util_env';
import {TensorLike} from '../types';
import * as util from '../util';
import {op} from './operation';

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
 * @param boxes a 2d tensor of shape `[numBoxes, 4]`. Each entry is
 *     `[y1, x1, y2, x2]`, where `(y1, x1)` and `(y2, x2)` are the corners of
 *     the bounding box.
 * @param scores a 1d tensor providing the box scores of shape `[numBoxes]`.
 * @param maxOutputSize The maximum number of boxes to be selected.
 * @param iouThreshold A float representing the threshold for deciding whether
 *     boxes overlap too much with respect to IOU. Must be betwen [0, 1].
 *     Defaults to 0.5 (50% box overlap).
 * @param scoreThreshold A threshold for deciding when to remove boxes based
 *     on score. Defaults to -inf, which means any score is accepted.
 * @return A 1D tensor with the selected box indices.
 */
/** @doc {heading: 'Operations', subheading: 'Images', namespace: 'image'} */
function nonMaxSuppression_(
    boxes: Tensor2D|TensorLike, scores: Tensor1D|TensorLike,
    maxOutputSize: number, iouThreshold = 0.5,
    scoreThreshold = Number.NEGATIVE_INFINITY): Tensor1D {
  const $boxes = convertToTensor(boxes, 'boxes', 'nonMaxSuppression');
  const $scores = convertToTensor(scores, 'scores', 'nonMaxSuppression');

  const inputs = nonMaxSuppSanityCheck(
      $boxes, $scores, maxOutputSize, iouThreshold, scoreThreshold);
  maxOutputSize = inputs.maxOutputSize;
  iouThreshold = inputs.iouThreshold;
  scoreThreshold = inputs.scoreThreshold;

  return ENV.engine.runKernel(
      b => b.nonMaxSuppression(
          $boxes, $scores, maxOutputSize, iouThreshold, scoreThreshold),
      {$boxes});
}

/** This is the async version of `nonMaxSuppression` */
async function nonMaxSuppressionAsync_(
    boxes: Tensor2D|TensorLike, scores: Tensor1D|TensorLike,
    maxOutputSize: number, iouThreshold = 0.5,
    scoreThreshold = Number.NEGATIVE_INFINITY): Promise<Tensor1D> {
  const $boxes = convertToTensor(boxes, 'boxes', 'nonMaxSuppressionAsync');
  const $scores = convertToTensor(scores, 'scores', 'nonMaxSuppressionAsync');

  const inputs = nonMaxSuppSanityCheck(
      $boxes, $scores, maxOutputSize, iouThreshold, scoreThreshold);
  maxOutputSize = inputs.maxOutputSize;
  iouThreshold = inputs.iouThreshold;
  scoreThreshold = inputs.scoreThreshold;

  const boxesVals = await $boxes.data();
  const scoresVals = await $scores.data();
  const res = nonMaxSuppressionImpl(
      boxesVals, scoresVals, maxOutputSize, iouThreshold, scoreThreshold);
  if ($boxes !== boxes) {
    $boxes.dispose();
  }
  if ($scores !== scores) {
    $scores.dispose();
  }
  return res;
}

function nonMaxSuppSanityCheck(
    boxes: Tensor2D, scores: Tensor1D, maxOutputSize: number,
    iouThreshold: number, scoreThreshold: number):
    {maxOutputSize: number, iouThreshold: number, scoreThreshold: number} {
  if (iouThreshold == null) {
    iouThreshold = 0.5;
  }
  if (scoreThreshold == null) {
    scoreThreshold = Number.NEGATIVE_INFINITY;
  }
  const numBoxes = boxes.shape[0];
  maxOutputSize = Math.min(maxOutputSize, numBoxes);

  util.assert(
      0 <= iouThreshold && iouThreshold <= 1,
      `iouThreshold must be in [0, 1], but was '${iouThreshold}'`);
  util.assert(
      boxes.rank === 2,
      `boxes must be a 2D tensor, but was of rank '${boxes.rank}'`);
  util.assert(
      boxes.shape[1] === 4,
      `boxes must have 4 columns, but 2nd dimension was ${boxes.shape[1]}`);
  util.assert(scores.rank === 1, 'scores must be a 1D tensor');
  util.assert(
      scores.shape[0] === numBoxes,
      `scores has incompatible shape with boxes. Expected ${numBoxes}, ` +
          `but was ${scores.shape[0]}`);
  return {maxOutputSize, iouThreshold, scoreThreshold};
}

export const resizeBilinear = op({resizeBilinear_});
export const resizeNearestNeighbor = op({resizeNearestNeighbor_});
export const nonMaxSuppression = op({nonMaxSuppression_});
export const nonMaxSuppressionAsync = nonMaxSuppressionAsync_;
