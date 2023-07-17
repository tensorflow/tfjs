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
import * as tf from '../../index';
import {ALL_ENVS, describeWithFlags} from '../../jasmine_util';
import {expectArraysClose} from '../../test_util';
import {Tensor2D} from '../../tensor';

describeWithFlags('RGBToGrayscale', ALL_ENVS, () => {
  it('should return int32 dtype tensor for int32 dtype input', async () => {
    const rgb = tf.tensor2d([16,24,56,1,2,9], [2,3], 'int32');
    const grayscale = tf.image.rgbToGrayscale(rgb);

    const expected = [[25],[2]];
    const grayscaleData = await grayscale.data();

    expect(grayscale.shape).toEqual([2,1]);
    expectArraysClose(grayscaleData, expected);
  });

  it('basic 3 rank array conversion', async () => {
    const rgb = [[[1.0, 2.0, 3.0]]];
    const grayscale = tf.image.rgbToGrayscale(rgb);

    const expected = [[[1.8149]]];
    const grayscaleData = await grayscale.data();

    expect(grayscale.shape).toEqual([1, 1, 1]);
    expectArraysClose(grayscaleData, expected);
  });

  it('basic 4 rank array conversion', async () => {
    const rgb = [[[[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]]];
    const grayscale = tf.image.rgbToGrayscale(rgb);

    const expected = [[[[1.8149],[2.8148]]]];
    const grayscaleData = await grayscale.data();

    expect(grayscale.shape).toEqual([1, 1, 2, 1]);
    expectArraysClose(grayscaleData, expected);
  });

  it('should throw an error because of input last dim is not 3', () => {
    const grayscale = tf.tensor4d([1.0, 1.0, 2.0, 2.0, 3.0, 3.0], [1, 1, 3, 2]);

    expect(() => tf.image.rgbToGrayscale(grayscale))
        .toThrowError(/last dimension of an RGB image should be size 3/);
  });

  it('should throw an error because of image\'s rank is less than 2', () => {
    const grayscale = tf.tensor1d([1, 2, 3]) as unknown as Tensor2D;

    expect(() => tf.image.rgbToGrayscale(grayscale))
        .toThrowError(/images must be at least rank 2/);
  });
});
