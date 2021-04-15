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
import {expectArraysClose} from '../../test_util';

describeWithFlags('image.transform', ALL_ENVS, () => {
  it('extreme projective transform.', async () => {
    const images = tf.tensor4d(
        [1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1], [1, 4, 4, 1]);
    const transform = tf.tensor2d([1, 0, 0, 0, 1, 0, -1, 0], [1, 8]);
    const transformedImages = tf.image.transform(images, transform).toInt();
    const transformedImagesData = await transformedImages.data();

    const expected = [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0];

    expectArraysClose(expected, transformedImagesData);
  });

  it('static output shape.', async () => {
    const images = tf.tensor4d([1, 2, 3, 4], [1, 2, 2, 1]);
    const transform = tf.randomUniform([1, 8], -1, 1);
    const transformedImages = tf.image.transform(
        images, transform as tf.Tensor2D, 'nearest', 'constant', 0, [3, 5]);

    expectArraysClose(transformedImages.shape, [1, 3, 5, 1]);
  });

  it('fill=constant, interpolation=nearest.', async () => {
    const images = tf.tensor4d(
        [1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0], [1, 4, 4, 1]);
    const transform = tf.tensor2d([0, 0.5, 1, -1, 2, 3, 0, 0], [1, 8]);
    const transformedImages = tf.image.transform(images, transform);
    const transformedImagesData = await transformedImages.data();

    const expected = [1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0];

    expectArraysClose(expected, transformedImagesData);
  });

  it('fill=constant, interpolation=bilinear.', async () => {
    const images = tf.tensor4d(
        [1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0], [1, 4, 4, 1]);
    const transform = tf.tensor2d([0, 0.5, 1, -1, 2, 3, 0, 0], [1, 8]);
    const transformedImages = tf.image.transform(images, transform, 'bilinear');
    const transformedImagesData = await transformedImages.data();

    const expected = [1, 0, 1, 1, 0, 0, 0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0];

    expectArraysClose(expected, transformedImagesData);
  });

  it('fill=reflect, interpolation=bilinear.', async () => {
    const images = tf.tensor4d(
        [1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0], [1, 4, 4, 1]);
    const transform = tf.tensor2d([0, 0.5, 1, -1, 2, 3, 0, 0], [1, 8]);
    const transformedImages =
        tf.image.transform(images, transform, 'bilinear', 'reflect');
    const transformedImagesData = await transformedImages.data();

    const expected =
        [1, 0, 1, 1, 0.5, 0.5, 0.5, 0.5, 1, 0, 1, 0, 0, 0.5, 0.5, 0];

    expectArraysClose(expected, transformedImagesData);
  });

  it('fill=wrap, interpolation=bilinear.', async () => {
    const images = tf.tensor4d(
        [1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0], [1, 4, 4, 1]);
    const transform = tf.tensor2d([0, 0.5, 1, -1, 2, 3, 0, 0], [1, 8]);
    const transformedImages =
        tf.image.transform(images, transform, 'bilinear', 'wrap');
    const transformedImagesData = await transformedImages.data();

    const expected =
        [1, 0, 1, 1, 0.5, 1, 0.5, 0.5, 1, 1, 0, 1, 0.5, 0.5, 0.5, 0.5];

    expectArraysClose(expected, transformedImagesData);
  });

  it('fill=nearest, interpolation=bilinear.', async () => {
    const images = tf.tensor4d(
        [1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0], [1, 4, 4, 1]);
    const transform = tf.tensor2d([0, 0.5, 1, -1, 2, 3, 0, 0], [1, 8]);
    const transformedImages =
        tf.image.transform(images, transform, 'bilinear', 'nearest');
    const transformedImagesData = await transformedImages.data();

    const expected = [1, 0, 1, 1, 0.5, 0.5, 0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0];

    expectArraysClose(expected, transformedImagesData);
  });
});
