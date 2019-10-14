/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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
import * as tf from '../index';
import {ALL_ENVS, BROWSER_ENVS, describeWithFlags} from '../jasmine_util';
import {expectArraysClose, expectArraysEqual} from '../test_util';

describeWithFlags('nonMaxSuppression', ALL_ENVS, () => {
  it('select from three clusters', async () => {
    const boxes = tf.tensor2d(
        [
          0, 0,  1, 1,  0, 0.1,  1, 1.1,  0, -0.1, 1, 0.9,
          0, 10, 1, 11, 0, 10.1, 1, 11.1, 0, 100,  1, 101
        ],
        [6, 4]);
    const scores = tf.tensor1d([0.9, 0.75, 0.6, 0.95, 0.5, 0.3]);
    const maxOutputSize = 3;
    const iouThreshold = 0.5;
    const scoreThreshold = 0;
    const indices = tf.image.nonMaxSuppression(
        boxes, scores, maxOutputSize, iouThreshold, scoreThreshold);

    expect(indices.shape).toEqual([3]);
    expectArraysEqual(await indices.data(), [3, 0, 5]);
  });

  it('select from three clusters flipped coordinates', async () => {
    const boxes = tf.tensor2d(
        [
          1, 1,  0, 0,  0, 0.1,  1, 1.1,  0, .9,  1, -0.1,
          0, 10, 1, 11, 1, 10.1, 0, 11.1, 1, 101, 0, 100
        ],
        [6, 4]);
    const scores = tf.tensor1d([0.9, 0.75, 0.6, 0.95, 0.5, 0.3]);
    const maxOutputSize = 3;
    const iouThreshold = 0.5;
    const scoreThreshold = 0;
    const indices = tf.image.nonMaxSuppression(
        boxes, scores, maxOutputSize, iouThreshold, scoreThreshold);

    expect(indices.shape).toEqual([3]);
    expectArraysEqual(await indices.data(), [3, 0, 5]);
  });

  it('select at most two boxes from three clusters', async () => {
    const boxes = tf.tensor2d(
        [
          0, 0,  1, 1,  0, 0.1,  1, 1.1,  0, -0.1, 1, 0.9,
          0, 10, 1, 11, 0, 10.1, 1, 11.1, 0, 100,  1, 101
        ],
        [6, 4]);
    const scores = tf.tensor1d([0.9, 0.75, 0.6, 0.95, 0.5, 0.3]);
    const maxOutputSize = 2;
    const iouThreshold = 0.5;
    const scoreThreshold = 0;
    const indices = tf.image.nonMaxSuppression(
        boxes, scores, maxOutputSize, iouThreshold, scoreThreshold);

    expect(indices.shape).toEqual([2]);
    expectArraysEqual(await indices.data(), [3, 0]);
  });

  it('select at most thirty boxes from three clusters', async () => {
    const boxes = tf.tensor2d(
        [
          0, 0,  1, 1,  0, 0.1,  1, 1.1,  0, -0.1, 1, 0.9,
          0, 10, 1, 11, 0, 10.1, 1, 11.1, 0, 100,  1, 101
        ],
        [6, 4]);
    const scores = tf.tensor1d([0.9, 0.75, 0.6, 0.95, 0.5, 0.3]);
    const maxOutputSize = 30;
    const iouThreshold = 0.5;
    const scoreThreshold = 0;
    const indices = tf.image.nonMaxSuppression(
        boxes, scores, maxOutputSize, iouThreshold, scoreThreshold);

    expect(indices.shape).toEqual([3]);
    expectArraysEqual(await indices.data(), [3, 0, 5]);
  });

  it('select single box', async () => {
    const boxes = tf.tensor2d([0, 0, 1, 1], [1, 4]);
    const scores = tf.tensor1d([0.9]);
    const maxOutputSize = 3;
    const iouThreshold = 0.5;
    const scoreThreshold = 0;
    const indices = tf.image.nonMaxSuppression(
        boxes, scores, maxOutputSize, iouThreshold, scoreThreshold);

    expect(indices.shape).toEqual([1]);
    expectArraysEqual(await indices.data(), [0]);
  });

  it('select from ten identical boxes', async () => {
    const boxes = tf.tensor2d([0, 0, 1, 1], [1, 4]);
    const scores = tf.tensor1d([0.9]);
    const maxOutputSize = 3;
    const iouThreshold = 0.5;
    const scoreThreshold = 0;
    const indices = tf.image.nonMaxSuppression(
        boxes, scores, maxOutputSize, iouThreshold, scoreThreshold);

    expect(indices.shape).toEqual([1]);
    expectArraysEqual(await indices.data(), [0]);
  });

  it('select from ten identical boxes', async () => {
    const numBoxes = 10;
    const corners = new Array(numBoxes)
                        .fill(0)
                        .map(_ => [0, 0, 1, 1])
                        .reduce((arr, curr) => arr.concat(curr));
    const boxes = tf.tensor2d(corners, [numBoxes, 4]);
    const scores = tf.tensor1d(Array(numBoxes).fill(0.9));
    const maxOutputSize = 3;
    const iouThreshold = 0.5;
    const scoreThreshold = 0;
    const indices = tf.image.nonMaxSuppression(
        boxes, scores, maxOutputSize, iouThreshold, scoreThreshold);

    expect(indices.shape).toEqual([1]);
    expectArraysEqual(await indices.data(), [0]);
  });

  it('inconsistent box and score shapes', () => {
    const boxes = tf.tensor2d(
        [
          0, 0,  1, 1,  0, 0.1,  1, 1.1,  0, -0.1, 1, 0.9,
          0, 10, 1, 11, 0, 10.1, 1, 11.1, 0, 100,  1, 101
        ],
        [6, 4]);
    const scores = tf.tensor1d([0.9, 0.75, 0.6, 0.95, 0.5]);
    const maxOutputSize = 30;
    const iouThreshold = 0.5;
    const scoreThreshold = 0;
    expect(
        () => tf.image.nonMaxSuppression(
            boxes, scores, maxOutputSize, iouThreshold, scoreThreshold))
        .toThrowError(/scores has incompatible shape with boxes/);
  });

  it('invalid iou threshold', () => {
    const boxes = tf.tensor2d([0, 0, 1, 1], [1, 4]);
    const scores = tf.tensor1d([0.9]);
    const maxOutputSize = 3;
    const iouThreshold = 1.2;
    const scoreThreshold = 0;
    expect(
        () => tf.image.nonMaxSuppression(
            boxes, scores, maxOutputSize, iouThreshold, scoreThreshold))
        .toThrowError(/iouThreshold must be in \[0, 1\]/);
  });

  it('empty input', async () => {
    const boxes = tf.tensor2d([], [0, 4]);
    const scores = tf.tensor1d([]);
    const maxOutputSize = 3;
    const iouThreshold = 0.5;
    const scoreThreshold = 0;
    const indices = tf.image.nonMaxSuppression(
        boxes, scores, maxOutputSize, iouThreshold, scoreThreshold);

    expect(indices.shape).toEqual([0]);
    expectArraysEqual(await indices.data(), []);
  });

  it('accepts a tensor-like object', async () => {
    const boxes = [[0, 0, 1, 1], [0, 1, 1, 2]];
    const scores = [1, 2];
    const indices = tf.image.nonMaxSuppression(boxes, scores, 10);
    expect(indices.shape).toEqual([2]);
    expect(indices.dtype).toEqual('int32');
    expectArraysEqual(await indices.data(), [1, 0]);
  });
});

describeWithFlags('nonMaxSuppressionAsync', ALL_ENVS, () => {
  it('select from three clusters', async () => {
    const boxes = tf.tensor2d(
        [
          0, 0,  1, 1,  0, 0.1,  1, 1.1,  0, -0.1, 1, 0.9,
          0, 10, 1, 11, 0, 10.1, 1, 11.1, 0, 100,  1, 101
        ],
        [6, 4]);
    const scores = tf.tensor1d([0.9, 0.75, 0.6, 0.95, 0.5, 0.3]);
    const maxOutputSize = 3;
    const iouThreshold = 0.5;
    const scoreThreshold = 0;
    const indices = await tf.image.nonMaxSuppressionAsync(
        boxes, scores, maxOutputSize, iouThreshold, scoreThreshold);

    expect(indices.shape).toEqual([3]);
    expectArraysEqual(await indices.data(), [3, 0, 5]);
  });

  it('accepts a tensor-like object', async () => {
    const boxes = [[0, 0, 1, 1], [0, 1, 1, 2]];
    const scores = [1, 2];
    const indices = await tf.image.nonMaxSuppressionAsync(boxes, scores, 10);
    expect(indices.shape).toEqual([2]);
    expect(indices.dtype).toEqual('int32');
    expectArraysEqual(await indices.data(), [1, 0]);
  });
});

describeWithFlags('rotate', BROWSER_ENVS, () => {
  // tslint:disable:max-line-length
  const imageBase64String =
      'data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAASABIAAD/4QCMRXhpZgAATU0AKgAAAAgABQESAAMAAAABAAEAAAEaAAUAAAABAAAASgEbAAUAAAABAAAAUgEoAAMAAAABAAIAAIdpAAQAAAABAAAAWgAAAAAAAABIAAAAAQAAAEgAAAABAAOgAQADAAAAAQABAACgAgAEAAAAAQAAAAigAwAEAAAAAQAAAAgAAAAA/+0AOFBob3Rvc2hvcCAzLjAAOEJJTQQEAAAAAAAAOEJJTQQlAAAAAAAQ1B2M2Y8AsgTpgAmY7PhCfv/AABEIAAgACAMBIgACEQEDEQH/xAAfAAABBQEBAQEBAQAAAAAAAAAAAQIDBAUGBwgJCgv/xAC1EAACAQMDAgQDBQUEBAAAAX0BAgMABBEFEiExQQYTUWEHInEUMoGRoQgjQrHBFVLR8CQzYnKCCQoWFxgZGiUmJygpKjQ1Njc4OTpDREVGR0hJSlNUVVZXWFlaY2RlZmdoaWpzdHV2d3h5eoOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4eLj5OXm5+jp6vHy8/T19vf4+fr/xAAfAQADAQEBAQEBAQEBAAAAAAAAAQIDBAUGBwgJCgv/xAC1EQACAQIEBAMEBwUEBAABAncAAQIDEQQFITEGEkFRB2FxEyIygQgUQpGhscEJIzNS8BVictEKFiQ04SXxFxgZGiYnKCkqNTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqCg4SFhoeIiYqSk5SVlpeYmZqio6Slpqeoqaqys7S1tre4ubrCw8TFxsfIycrS09TV1tfY2dri4+Tl5ufo6ery8/T19vf4+fr/2wBDAAkGBw0HCA0HBw0HBwcHBw0HBwcHDQ8IDQcNFREWFhURExMYHSggGBolGxUTITEhMSk3Ojo6Fx8zODMtNygtLiv/2wBDAQoKCg0NDRUNDRUrGRUZKysrKy0rKy0rKysrKy0rKysrKystKzctKysrKy0rKysrLSsrKysrKzcrLSsrKy0rKyv/3QAEAAH/2gAMAwEAAhEDEQA/AOin1Kxs7JgEVSsZAPU5xWF/wkdp6L+QqlrX/Hm/0P8AKuSqsRk+FlUcmnd+Z4WBzzGulrO+vY//2Q==';
  const size = 8;

  it('basic', async () => {
    const img = new Image();
    img.src = imageBase64String;

    await new Promise(resolve => {
      img.onload = () => resolve(img);
    });

    img.width = size;
    img.height = size;

    const pixels = await tf.browser.fromPixels(img, 4);

    const rotatedPixels =
        tf.image.rotate(pixels.toFloat().expandDims(0), 90 * Math.PI / 180)
            .toInt();
    const rotatedPixelsData = await rotatedPixels.data();

    const expected = [
      0,   0,   0,   0,   0,   193, 228, 255, 18,  200, 224, 255, 55,  207, 212,
      255, 108, 214, 202, 255, 163, 208, 187, 255, 179, 176, 159, 255, 168, 129,
      130, 255, 0,   0,   0,   0,   0,   192, 221, 255, 19,  204, 222, 255, 62,
      217, 213, 255, 119, 226, 206, 255, 179, 226, 194, 255, 199, 198, 168, 255,
      186, 152, 140, 255, 0,   0,   0,   0,   0,   189, 209, 255, 19,  203, 211,
      255, 64,  219, 201, 255, 121, 231, 192, 255, 184, 234, 181, 255, 211, 216,
      162, 255, 202, 174, 137, 255, 0,   0,   0,   0,   0,   194, 204, 255, 30,
      209, 205, 255, 76,  225, 193, 255, 130, 235, 179, 255, 191, 239, 165, 255,
      225, 228, 151, 255, 220, 190, 128, 255, 0,   0,   0,   0,   4,   182, 186,
      255, 35,  200, 186, 255, 87,  220, 175, 255, 141, 232, 162, 255, 201, 236,
      142, 255, 235, 227, 128, 255, 230, 193, 104, 255, 0,   0,   0,   0,   3,
      158, 162, 255, 37,  177, 164, 255, 94,  206, 156, 255, 155, 226, 146, 255,
      213, 230, 126, 255, 243, 220, 106, 255, 241, 188, 82,  255, 0,   0,   0,
      0,   6,   133, 140, 255, 39,  150, 141, 255, 98,  181, 135, 255, 162, 206,
      127, 255, 218, 210, 103, 255, 247, 201, 81,  255, 250, 177, 64,  255, 0,
      0,   0,   0,   0,   102, 113, 255, 15,  115, 105, 255, 71,  143, 97,  255,
      135, 166, 86,  255, 191, 170, 61,  255, 222, 164, 41,  255, 233, 148, 31,
      255
    ];

    expectArraysClose(expected, rotatedPixelsData, 10);
  });

  it('offset center of rotation', async () => {
    const img = new Image();
    img.src = imageBase64String;

    await new Promise(resolve => {
      img.onload = () => resolve(img);
    });

    img.width = size;
    img.height = size;

    const pixels = await tf.browser.fromPixels(img, 4);

    const rotatedPixels = tf.image
                              .rotate(
                                  pixels.toFloat().expandDims(0),
                                  45 * Math.PI / 180, 0, [0.25, 0.75])
                              .toInt();
    const rotatedPixelsData = await rotatedPixels.data();

    const expected = [
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   156, 100, 111, 255, 0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   168, 129, 130, 255, 171, 120, 117, 255, 171, 120, 117, 255,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   179, 176, 159, 255, 199, 198, 168, 255, 186, 152, 140, 255, 183, 138,
      109, 255, 200, 155, 98,  255, 0,   0,   0,   0,   0,   0,   0,   0,   108,
      214, 202, 255, 179, 226, 194, 255, 199, 198, 168, 255, 211, 216, 162, 255,
      220, 190, 128, 255, 200, 155, 98,  255, 0,   0,   0,   0,   55,  207, 212,
      255, 62,  217, 213, 255, 119, 226, 206, 255, 184, 234, 181, 255, 225, 228,
      151, 255, 225, 228, 151, 255, 230, 193, 104, 255, 0,   193, 228, 255, 19,
      204, 222, 255, 62,  217, 213, 255, 64,  219, 201, 255, 130, 235, 179, 255,
      191, 239, 165, 255, 235, 227, 128, 255, 243, 220, 106, 255, 0,   192, 221,
      255, 0,   192, 221, 255, 19,  203, 211, 255, 76,  225, 193, 255, 76,  225,
      193, 255, 141, 232, 162, 255, 213, 230, 126, 255, 247, 201, 81,  255, 0,
      0,   0,   0,   0,   189, 209, 255, 0,   194, 204, 255, 30,  209, 205, 255,
      87,  220, 175, 255, 94,  206, 156, 255, 162, 206, 127, 255, 218, 210, 103,
      255
    ];

    expectArraysClose(expected, rotatedPixelsData, 10);
  });

  it('offset center of rotation with white fill', async () => {
    const img = new Image();
    img.src = imageBase64String;

    await new Promise(resolve => {
      img.onload = () => resolve(img);
    });

    img.width = size;
    img.height = size;

    const pixels = await tf.browser.fromPixels(img, 4);

    const rotatedPixels = tf.image
                              .rotate(
                                  pixels.toFloat().expandDims(0),
                                  45 * Math.PI / 180, 255, [0.25, 0.75])
                              .toInt();
    const rotatedPixelsData = await rotatedPixels.data();

    const expected = [
      255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
      255, 255, 255, 255, 255, 156, 100, 111, 255, 255, 255, 255, 255, 255, 255,
      255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
      255, 255, 255, 168, 129, 130, 255, 171, 120, 117, 255, 171, 120, 117, 255,
      255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
      255, 179, 176, 159, 255, 199, 198, 168, 255, 186, 152, 140, 255, 183, 138,
      109, 255, 200, 155, 98,  255, 255, 255, 255, 255, 255, 255, 255, 255, 108,
      214, 202, 255, 179, 226, 194, 255, 199, 198, 168, 255, 211, 216, 162, 255,
      220, 190, 128, 255, 200, 155, 98,  255, 255, 255, 255, 255, 55,  207, 212,
      255, 62,  217, 213, 255, 119, 226, 206, 255, 184, 234, 181, 255, 225, 228,
      151, 255, 225, 228, 151, 255, 230, 193, 104, 255, 0,   193, 228, 255, 19,
      204, 222, 255, 62,  217, 213, 255, 64,  219, 201, 255, 130, 235, 179, 255,
      191, 239, 165, 255, 235, 227, 128, 255, 243, 220, 106, 255, 0,   192, 221,
      255, 0,   192, 221, 255, 19,  203, 211, 255, 76,  225, 193, 255, 76,  225,
      193, 255, 141, 232, 162, 255, 213, 230, 126, 255, 247, 201, 81,  255, 255,
      255, 255, 255, 0,   189, 209, 255, 0,   194, 204, 255, 30,  209, 205, 255,
      87,  220, 175, 255, 94,  206, 156, 255, 162, 206, 127, 255, 218, 210, 103,
      255
    ];

    expectArraysClose(expected, rotatedPixelsData, 10);
  });
});

describeWithFlags('cropAndResize', ALL_ENVS, () => {
  it('1x1-bilinear', async () => {
    const image: tf.Tensor4D = tf.tensor4d([1, 2, 3, 4], [1, 2, 2, 1]);
    const boxes: tf.Tensor2D = tf.tensor2d([0, 0, 1, 1], [1, 4]);

    const boxInd: tf.Tensor1D = tf.tensor1d([0], 'int32');
    const output =
        tf.image.cropAndResize(image, boxes, boxInd, [1, 1], 'bilinear', 0);
    expect(output.shape).toEqual([1, 1, 1, 1]);
    expectArraysClose(await output.data(), [2.5]);
  });
  it('1x1-nearest', async () => {
    const image: tf.Tensor4D = tf.tensor4d([1, 2, 3, 4], [1, 2, 2, 1]);
    const boxes: tf.Tensor2D = tf.tensor2d([0, 0, 1, 1], [1, 4]);
    const boxInd: tf.Tensor1D = tf.tensor1d([0], 'int32');
    const output =
        tf.image.cropAndResize(image, boxes, boxInd, [1, 1], 'nearest', 0);
    expect(output.shape).toEqual([1, 1, 1, 1]);
    expectArraysClose(await output.data(), [4.0]);
  });
  it('1x1Flipped-bilinear', async () => {
    const image: tf.Tensor4D = tf.tensor4d([1, 2, 3, 4], [1, 2, 2, 1]);
    const boxes: tf.Tensor2D = tf.tensor2d([1, 1, 0, 0], [1, 4]);
    const boxInd: tf.Tensor1D = tf.tensor1d([0], 'int32');
    const output =
        tf.image.cropAndResize(image, boxes, boxInd, [1, 1], 'bilinear', 0);
    expect(output.shape).toEqual([1, 1, 1, 1]);
    expectArraysClose(await output.data(), [2.5]);
  });
  it('1x1Flipped-nearest', async () => {
    const image: tf.Tensor4D = tf.tensor4d([1, 2, 3, 4], [1, 2, 2, 1]);
    const boxes: tf.Tensor2D = tf.tensor2d([1, 1, 0, 0], [1, 4]);
    const boxInd: tf.Tensor1D = tf.tensor1d([0], 'int32');
    const output =
        tf.image.cropAndResize(image, boxes, boxInd, [1, 1], 'nearest', 0);
    expect(output.shape).toEqual([1, 1, 1, 1]);
    expectArraysClose(await output.data(), [4.0]);
  });
  it('3x3-bilinear', async () => {
    const image: tf.Tensor4D = tf.tensor4d([1, 2, 3, 4], [1, 2, 2, 1]);
    const boxes: tf.Tensor2D = tf.tensor2d([0, 0, 1, 1], [1, 4]);
    const boxInd: tf.Tensor1D = tf.tensor1d([0], 'int32');
    const output =
        tf.image.cropAndResize(image, boxes, boxInd, [3, 3], 'bilinear', 0);
    expect(output.shape).toEqual([1, 3, 3, 1]);
    expectArraysClose(await output.data(), [1, 1.5, 2, 2, 2.5, 3, 3, 3.5, 4]);
  });
  it('3x3-nearest', async () => {
    const image: tf.Tensor4D = tf.tensor4d([1, 2, 3, 4], [1, 2, 2, 1]);
    const boxes: tf.Tensor2D = tf.tensor2d([0, 0, 1, 1], [1, 4]);
    const boxInd: tf.Tensor1D = tf.tensor1d([0], 'int32');
    const output =
        tf.image.cropAndResize(image, boxes, boxInd, [3, 3], 'nearest', 0);
    expect(output.shape).toEqual([1, 3, 3, 1]);
    expectArraysClose(await output.data(), [1, 2, 2, 3, 4, 4, 3, 4, 4]);
  });
  it('3x3Flipped-bilinear', async () => {
    const image: tf.Tensor4D = tf.tensor4d([1, 2, 3, 4], [1, 2, 2, 1]);
    const boxes: tf.Tensor2D = tf.tensor2d([1, 1, 0, 0], [1, 4]);
    const boxInd: tf.Tensor1D = tf.tensor1d([0], 'int32');
    const output =
        tf.image.cropAndResize(image, boxes, boxInd, [3, 3], 'bilinear', 0);
    expect(output.shape).toEqual([1, 3, 3, 1]);
    expectArraysClose(await output.data(), [4, 3.5, 3, 3, 2.5, 2, 2, 1.5, 1]);
  });
  it('3x3Flipped-nearest', async () => {
    const image: tf.Tensor4D = tf.tensor4d([1, 2, 3, 4], [1, 2, 2, 1]);
    const boxes: tf.Tensor2D = tf.tensor2d([1, 1, 0, 0], [1, 4]);
    const boxInd: tf.Tensor1D = tf.tensor1d([0], 'int32');
    const output =
        tf.image.cropAndResize(image, boxes, boxInd, [3, 3], 'nearest', 0);
    expect(output.shape).toEqual([1, 3, 3, 1]);
    expectArraysClose(await output.data(), [4, 4, 3, 4, 4, 3, 2, 2, 1]);
  });
  it('3x3to2x2-bilinear', async () => {
    const image: tf.Tensor4D =
        tf.tensor4d([1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 3, 3, 1]);
    const boxes: tf.Tensor2D =
        tf.tensor2d([0, 0, 1, 1, 0, 0, 0.5, 0.5], [2, 4]);
    const boxInd: tf.Tensor1D = tf.tensor1d([0, 0], 'int32');
    const output =
        tf.image.cropAndResize(image, boxes, boxInd, [2, 2], 'bilinear', 0);
    expect(output.shape).toEqual([2, 2, 2, 1]);
    expectArraysClose(await output.data(), [1, 3, 7, 9, 1, 2, 4, 5]);
  });
  it('3x3to2x2-nearest', async () => {
    const image: tf.Tensor4D =
        tf.tensor4d([1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 3, 3, 1]);
    const boxes: tf.Tensor2D =
        tf.tensor2d([0, 0, 1, 1, 0, 0, 0.5, 0.5], [2, 4]);
    const boxInd: tf.Tensor1D = tf.tensor1d([0, 0], 'int32');
    const output =
        tf.image.cropAndResize(image, boxes, boxInd, [2, 2], 'nearest', 0);
    expect(output.shape).toEqual([2, 2, 2, 1]);
    expectArraysClose(await output.data(), [1, 3, 7, 9, 1, 2, 4, 5]);
  });
  it('3x3to2x2Flipped-bilinear', async () => {
    const image: tf.Tensor4D =
        tf.tensor4d([1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 3, 3, 1]);
    const boxes: tf.Tensor2D =
        tf.tensor2d([1, 1, 0, 0, 0.5, 0.5, 0, 0], [2, 4]);
    const boxInd: tf.Tensor1D = tf.tensor1d([0, 0], 'int32');
    const output =
        tf.image.cropAndResize(image, boxes, boxInd, [2, 2], 'bilinear', 0);
    expect(output.shape).toEqual([2, 2, 2, 1]);
    expectArraysClose(await output.data(), [9, 7, 3, 1, 5, 4, 2, 1]);
  });
  it('3x3to2x2Flipped-nearest', async () => {
    const image: tf.Tensor4D =
        tf.tensor4d([1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 3, 3, 1]);
    const boxes: tf.Tensor2D =
        tf.tensor2d([1, 1, 0, 0, 0.5, 0.5, 0, 0], [2, 4]);
    const boxInd: tf.Tensor1D = tf.tensor1d([0, 0], 'int32');
    const output =
        tf.image.cropAndResize(image, boxes, boxInd, [2, 2], 'nearest', 0);
    expect(output.shape).toEqual([2, 2, 2, 1]);
    expectArraysClose(await output.data(), [9, 7, 3, 1, 5, 4, 2, 1]);
  });
  it('3x3-BoxisRectangular', async () => {
    const image: tf.Tensor4D = tf.tensor4d([1, 2, 3, 4], [1, 2, 2, 1]);
    const boxes: tf.Tensor2D = tf.tensor2d([0, 0, 1, 1.5], [1, 4]);
    const boxInd: tf.Tensor1D = tf.tensor1d([0], 'int32');
    const output =
        tf.image.cropAndResize(image, boxes, boxInd, [3, 3], 'bilinear', 0);
    expect(output.shape).toEqual([1, 3, 3, 1]);
    expectArraysClose(
        await output.data(), [1, 1.75, 0, 2, 2.75, 0, 3, 3.75, 0]);
  });
  it('3x3-BoxisRectangular-nearest', async () => {
    const image: tf.Tensor4D = tf.tensor4d([1, 2, 3, 4], [1, 2, 2, 1]);
    const boxes: tf.Tensor2D = tf.tensor2d([0, 0, 1, 1.5], [1, 4]);
    const boxInd: tf.Tensor1D = tf.tensor1d([0], 'int32');
    const output =
        tf.image.cropAndResize(image, boxes, boxInd, [3, 3], 'nearest', 0);
    expect(output.shape).toEqual([1, 3, 3, 1]);
    expectArraysClose(await output.data(), [1, 2, 0, 3, 4, 0, 3, 4, 0]);
  });
  it('2x2to3x3-Extrapolated', async () => {
    const val = -1;
    const image: tf.Tensor4D = tf.tensor4d([1, 2, 3, 4], [1, 2, 2, 1]);
    const boxes: tf.Tensor2D = tf.tensor2d([-1, -1, 1, 1], [1, 4]);
    const boxInd: tf.Tensor1D = tf.tensor1d([0], 'int32');
    const output =
        tf.image.cropAndResize(image, boxes, boxInd, [3, 3], 'bilinear', val);
    expect(output.shape).toEqual([1, 3, 3, 1]);
    expectArraysClose(
        await output.data(), [val, val, val, val, 1, 2, val, 3, 4]);
  });
  it('2x2to3x3-Extrapolated-Float', async () => {
    const val = -1.5;
    const image: tf.Tensor4D = tf.tensor4d([1, 2, 3, 4], [1, 2, 2, 1]);
    const boxes: tf.Tensor2D = tf.tensor2d([-1, -1, 1, 1], [1, 4]);
    const boxInd: tf.Tensor1D = tf.tensor1d([0], 'int32');
    const output =
        tf.image.cropAndResize(image, boxes, boxInd, [3, 3], 'bilinear', val);
    expect(output.shape).toEqual([1, 3, 3, 1]);
    expectArraysClose(
        await output.data(), [val, val, val, val, 1, 2, val, 3, 4]);
  });
  it('2x2to3x3-NoCrop', async () => {
    const val = -1.0;
    const image: tf.Tensor4D = tf.tensor4d([1, 2, 3, 4], [1, 2, 2, 1]);
    const boxes: tf.Tensor2D = tf.tensor2d([], [0, 4]);
    const boxInd: tf.Tensor1D = tf.tensor1d([], 'int32');
    const output =
        tf.image.cropAndResize(image, boxes, boxInd, [3, 3], 'bilinear', val);
    expect(output.shape).toEqual([0, 3, 3, 1]);
    expectArraysClose(await output.data(), []);
  });
  it('MultipleBoxes-DifferentBoxes', async () => {
    const image: tf.Tensor4D =
        tf.tensor4d([1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2, 1]);
    const boxes: tf.Tensor2D =
        tf.tensor2d([0, 0, 1, 1.5, 0, 0, 1.5, 1], [2, 4]);
    const boxInd: tf.Tensor1D = tf.tensor1d([0, 1], 'int32');
    const output =
        tf.image.cropAndResize(image, boxes, boxInd, [3, 3], 'bilinear', 0);
    expect(output.shape).toEqual([2, 3, 3, 1]);
    expectArraysClose(
        await output.data(),
        [1, 1.75, 0, 2, 2.75, 0, 3, 3.75, 0, 5, 5.5, 6, 6.5, 7, 7.5, 0, 0, 0]);
  });
  it('MultipleBoxes-DifferentBoxes-Nearest', async () => {
    const image: tf.Tensor4D =
        tf.tensor4d([1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2, 1]);
    const boxes: tf.Tensor2D = tf.tensor2d([0, 0, 1, 1.5, 0, 0, 2, 1], [2, 4]);
    const boxInd: tf.Tensor1D = tf.tensor1d([0, 1], 'int32');
    const output =
        tf.image.cropAndResize(image, boxes, boxInd, [3, 3], 'nearest', 0);
    expect(output.shape).toEqual([2, 3, 3, 1]);
    expectArraysClose(
        await output.data(),
        [1, 2, 0, 3, 4, 0, 3, 4, 0, 5, 6, 6, 7, 8, 8, 0, 0, 0]);
  });
});
