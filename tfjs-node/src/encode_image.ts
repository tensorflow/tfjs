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

import {Tensor, Tensor3D, Rank} from '@tensorflow/tfjs-core';
import {ensureTensorflowBackend, nodeBackend} from './ops/op_utils';

export async function encodeJpeg(
  image: Tensor3D, format: '' | 'grayscale' | 'rgb' = '', quality = 95,
  progressive = false, optimizeSize = false,
  chromaDownsampling = true, densityUnit: 'in' | 'cm' = 'in',
  xDensity = 300, yDensity = 300, xmpMetadata = ''
  ): Promise<Uint8Array> {
  ensureTensorflowBackend();

  const backendEncodeImage = (imageData: Uint8Array) =>
    nodeBackend().encodeJpeg(
      imageData, image.shape, format, quality, progressive, optimizeSize,
      chromaDownsampling, densityUnit, xDensity, yDensity, xmpMetadata);

    return encodeImage(image, backendEncodeImage);
}

export async function encodePng(
  image: Tensor3D, compression = 1
  ): Promise<Uint8Array> {
  ensureTensorflowBackend();

  const backendEncodeImage = (imageData: Uint8Array) => nodeBackend().encodePng(
    imageData, image.shape, compression);
  return encodeImage(image, backendEncodeImage);
}

async function encodeImage(
  image: Tensor3D, backendEncodeImage: (imageData: Uint8Array) => Tensor<Rank>
  ): Promise<Uint8Array> {
  const encodedDataTensor = backendEncodeImage(new Uint8Array(
    await image.data()));

  const encodedPngData = (
    // tslint:disable-next-line:no-any
    await encodedDataTensor.data())[0] as any as Uint8Array;
  encodedDataTensor.dispose();
  return encodedPngData;
}
