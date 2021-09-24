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

import * as tf from '../../index';
import {ALL_ENVS, describeWithFlags} from '../../jasmine_util';
import {Tensor2D} from '../../tensor';
import {expectArraysClose} from '../../test_util';

describeWithFlags('grayscaleToRGB', ALL_ENVS, () => {
  it('should convert (1,1,3,1) images into (1,1,3,3)', async () => {
    const grayscale = tf.tensor4d([1.0, 2.0, 3.0], [1, 1, 3, 1]);

    const rgb = tf.image.grayscaleToRGB(grayscale);
    const rgbData = await rgb.data();

    const expected = [1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0];

    expect(rgb.shape).toEqual([1, 1, 3, 3]);
    expectArraysClose(rgbData, expected);
  });

  it('should convert (1,2,1) images into (1,2,3)', async () => {
    const grayscale = tf.tensor3d([1.6, 2.4], [1, 2, 1]);

    const rgb = tf.image.grayscaleToRGB(grayscale);
    const rgbData = await rgb.data();

    const expected = [1.6, 1.6, 1.6, 2.4, 2.4, 2.4];

    expect(rgb.shape).toEqual([1, 2, 3]);
    expectArraysClose(rgbData, expected);
  });

  it('should convert (2,1) images into (2,3)', async () => {
    const grayscale = tf.tensor2d([16, 24], [2, 1]);

    const rgb = tf.image.grayscaleToRGB(grayscale);
    const rgbData = await rgb.data();

    const expected = [16, 16, 16, 24, 24, 24];

    expect(rgb.shape).toEqual([2, 3]);
    expectArraysClose(rgbData, expected);
  });

  it('should convert [[[191], [3]]] array into (1,2,3) images', async () => {
    const grayscale = [[[191], [3]]];

    const rgb = tf.image.grayscaleToRGB(grayscale);
    const rgbData = await rgb.data();

    const expected = [191, 191, 191, 3, 3, 3];

    expect(rgb.shape).toEqual([1, 2, 3]);
    expectArraysClose(rgbData, expected);
  });

  it('should throw an error because of input last dim is not 1', () => {
    const grayscale = tf.tensor4d([1.0, 1.0, 2.0, 2.0, 3.0, 3.0], [1, 1, 3, 2]);

    expect(() => tf.image.grayscaleToRGB(grayscale))
        .toThrowError(/last dimension of a grayscale image should be size 1/);
  });

  it('should throw an error because of image\'s rank is less than 2', () => {
    const grayscale = tf.tensor1d([1, 2, 3]) as {} as Tensor2D;

    expect(() => tf.image.grayscaleToRGB(grayscale))
        .toThrowError(/images must be at least rank 2/);
  });
});
