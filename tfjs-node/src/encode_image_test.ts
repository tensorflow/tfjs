/**
 * @license
 * Copyright 2019 Google Inc. All Rights Reserved.
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
import * as tf from '.';
import {getImageType, ImageType} from './decode_image';

describe('encode images', () => {
  it('encodeJpeg', async () => {
    const imageTensor = tf.tensor3d(new Uint8Array(
      [239, 100, 0, 46, 48, 47, 92, 49, 0, 194, 98, 47]), [2, 2, 3]);
    const jpegEncodedData = await tf.node.encodeJpeg(imageTensor);
    imageTensor.dispose();
    expect(getImageType(jpegEncodedData)).toEqual(ImageType.JPEG);
  });

  it('encodeJpeg grayscale', async () => {
    const imageTensor = tf.tensor3d(new Uint8Array(
      [239, 0, 47, 0]), [2, 2, 1]);
    const jpegEncodedData = await tf.node.encodeJpeg(imageTensor, 'grayscale');
    imageTensor.dispose();
    expect(getImageType(jpegEncodedData)).toEqual(ImageType.JPEG);
  });

  it('encodeJpeg with parameters', async () => {
    const imageTensor = tf.tensor3d(new Uint8Array(
      [239, 100, 0, 46, 48, 47, 92, 49, 0, 194, 98, 47]), [2, 2, 3]);
    const format = 'rgb';
    const quality = 50;
    const progressive = true;
    const optimizeSize = true;
    const chromaDownsampling = false;
    const densityUnit = 'cm';
    const xDensity = 500;
    const yDensity = 500;

    const jpegEncodedData = await tf.node.encodeJpeg(imageTensor, format,
      quality, progressive, optimizeSize, chromaDownsampling, densityUnit,
      xDensity, yDensity);
    imageTensor.dispose();
    expect(getImageType(jpegEncodedData)).toEqual(ImageType.JPEG);
  });

  it('encodePng', async () => {
    const imageTensor = tf.tensor3d(new Uint8Array(
      [239, 100, 0, 46, 48, 47, 92, 49, 0, 194, 98, 47]), [2, 2, 3]);
    const pngEncodedData = await tf.node.encodePng(imageTensor);
    imageTensor.dispose();
    expect(getImageType(pngEncodedData)).toEqual(ImageType.PNG);
  });

  it('encodePng grayscale', async () => {
    const imageTensor = tf.tensor3d(new Uint8Array(
      [239, 0, 47, 0]), [2, 2, 1]);
    const pngEncodedData = await tf.node.encodePng(imageTensor);
    imageTensor.dispose();
    expect(getImageType(pngEncodedData)).toEqual(ImageType.PNG);
  });

  it('encodePng with parameters', async () => {
    const imageTensor = tf.tensor3d(new Uint8Array(
      [239, 100, 0, 46, 48, 47, 92, 49, 0, 194, 98, 47]), [2, 2, 3]);
    const compression = 4;

    const pngEncodedData = await tf.node.encodePng(imageTensor, compression);
    imageTensor.dispose();
    expect(getImageType(pngEncodedData)).toEqual(ImageType.PNG);
  });
});
