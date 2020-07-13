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
import * as tf from '../index';
import {BROWSER_ENVS, describeWithFlags} from '../jasmine_util';
import {expectArraysClose} from '../test_util';

describeWithFlags('rotateWithOffset', BROWSER_ENVS, () => {
  // tslint:disable:max-line-length
  const imageBase64String =
      'data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAASABIAAD/4QCMRXhpZgAATU0AKgAAAAgABQESAAMAAAABAAEAAAEaAAUAAAABAAAASgEbAAUAAAABAAAAUgEoAAMAAAABAAIAAIdpAAQAAAABAAAAWgAAAAAAAABIAAAAAQAAAEgAAAABAAOgAQADAAAAAQABAACgAgAEAAAAAQAAAAigAwAEAAAAAQAAAAgAAAAA/+0AOFBob3Rvc2hvcCAzLjAAOEJJTQQEAAAAAAAAOEJJTQQlAAAAAAAQ1B2M2Y8AsgTpgAmY7PhCfv/AABEIAAgACAMBIgACEQEDEQH/xAAfAAABBQEBAQEBAQAAAAAAAAAAAQIDBAUGBwgJCgv/xAC1EAACAQMDAgQDBQUEBAAAAX0BAgMABBEFEiExQQYTUWEHInEUMoGRoQgjQrHBFVLR8CQzYnKCCQoWFxgZGiUmJygpKjQ1Njc4OTpDREVGR0hJSlNUVVZXWFlaY2RlZmdoaWpzdHV2d3h5eoOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4eLj5OXm5+jp6vHy8/T19vf4+fr/xAAfAQADAQEBAQEBAQEBAAAAAAAAAQIDBAUGBwgJCgv/xAC1EQACAQIEBAMEBwUEBAABAncAAQIDEQQFITEGEkFRB2FxEyIygQgUQpGhscEJIzNS8BVictEKFiQ04SXxFxgZGiYnKCkqNTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqCg4SFhoeIiYqSk5SVlpeYmZqio6Slpqeoqaqys7S1tre4ubrCw8TFxsfIycrS09TV1tfY2dri4+Tl5ufo6ery8/T19vf4+fr/2wBDAAkGBw0HCA0HBw0HBwcHBw0HBwcHDQ8IDQcNFREWFhURExMYHSggGBolGxUTITEhMSk3Ojo6Fx8zODMtNygtLiv/2wBDAQoKCg0NDRUNDRUrGRUZKysrKy0rKy0rKysrKy0rKysrKystKzctKysrKy0rKysrLSsrKysrKzcrLSsrKy0rKyv/3QAEAAH/2gAMAwEAAhEDEQA/AOin1Kxs7JgEVSsZAPU5xWF/wkdp6L+QqlrX/Hm/0P8AKuSqsRk+FlUcmnd+Z4WBzzGulrO+vY//2Q==';
  const size = 8;

  it('should rotate 90 degrees', async () => {
    const img = new Image();
    img.src = imageBase64String;

    await new Promise(resolve => {
      img.onload = () => resolve(img);
    });

    img.width = size;
    img.height = size;

    const pixels = await tf.browser.fromPixels(img, 4);
    const pixels4D: tf.Tensor4D = pixels.toFloat().expandDims(0);

    const rotatedPixels =
        tf.image.rotateWithOffset(pixels4D, 90 * Math.PI / 180).toInt();
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

  it('should rotate negative 90 degrees', async () => {
    const img = new Image();
    img.src = imageBase64String;

    await new Promise(resolve => {
      img.onload = () => resolve(img);
    });

    img.width = size;
    img.height = size;

    const pixels = await tf.browser.fromPixels(img, 4);
    const pixels4D: tf.Tensor4D = pixels.toFloat().expandDims(0);

    const rotatedPixels =
        tf.image.rotateWithOffset(pixels4D, -90 * Math.PI / 180).toInt();
    const rotatedPixelsData = await rotatedPixels.data();

    const expected = [
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   230, 133, 18,  255, 233, 148, 31,  255, 222, 164, 41,  255, 191,
      170, 61,  255, 135, 166, 86,  255, 71,  143, 97,  255, 15,  115, 105, 255,
      0,   102, 113, 255, 241, 153, 43,  255, 250, 177, 64,  255, 247, 201, 81,
      255, 218, 210, 103, 255, 162, 206, 127, 255, 98,  181, 135, 255, 39,  150,
      141, 255, 6,   133, 140, 255, 224, 156, 55,  255, 241, 188, 82,  255, 243,
      220, 106, 255, 213, 230, 126, 255, 155, 226, 146, 255, 94,  206, 156, 255,
      37,  177, 164, 255, 3,   158, 162, 255, 212, 157, 75,  255, 230, 193, 104,
      255, 235, 227, 128, 255, 201, 236, 142, 255, 141, 232, 162, 255, 87,  220,
      175, 255, 35,  200, 186, 255, 4,   182, 186, 255, 200, 155, 98,  255, 220,
      190, 128, 255, 225, 228, 151, 255, 191, 239, 165, 255, 130, 235, 179, 255,
      76,  225, 193, 255, 30,  209, 205, 255, 0,   194, 204, 255, 183, 138, 109,
      255, 202, 174, 137, 255, 211, 216, 162, 255, 184, 234, 181, 255, 121, 231,
      192, 255, 64,  219, 201, 255, 19,  203, 211, 255, 0,   189, 209, 255, 171,
      120, 117, 255, 186, 152, 140, 255, 199, 198, 168, 255, 179, 226, 194, 255,
      119, 226, 206, 255, 62,  217, 213, 255, 19,  204, 222, 255, 0,   192, 221,
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
    const pixels4D: tf.Tensor4D = pixels.toFloat().expandDims(0);

    const rotatedPixels =
        tf.image.rotateWithOffset(pixels4D, 45 * Math.PI / 180, 0, [0.25, 0.75])
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
    const pixels4D: tf.Tensor4D = pixels.toFloat().expandDims(0);

    const rotatedPixels =
        tf.image
            .rotateWithOffset(pixels4D, 45 * Math.PI / 180, 255, [0.25, 0.75])
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
