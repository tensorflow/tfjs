/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
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

import * as tf from '@tensorflow/tfjs';
import * as ImageManipulator from 'expo-image-manipulator';
import * as jpeg from 'jpeg-js';

export function toDataUri(base64: string): string {
  return `data:image/jpeg;base64,${base64}`;
}

export async function resizeImage(
    imageUrl: string, width: number): Promise<ImageManipulator.ImageResult> {
  const actions = [{
    resize: {
      width,
    },
  }];
  const saveOptions = {
    compress: 0.75,
    format: ImageManipulator.SaveFormat.JPEG,
    base64: true,
  };
  const res =
      await ImageManipulator.manipulateAsync(imageUrl, actions, saveOptions);
  return res;
}

export async function base64ImageToTensor(base64: string):
    Promise<tf.Tensor3D> {
  const rawImageData = tf.util.encodeString(base64, 'base64');
  const TO_UINT8ARRAY = true;
  const {width, height, data} = jpeg.decode(rawImageData, TO_UINT8ARRAY);
  // Drop the alpha channel info
  const buffer = new Uint8Array(width * height * 3);
  let offset = 0;  // offset into original data
  for (let i = 0; i < buffer.length; i += 3) {
    buffer[i] = data[offset];
    buffer[i + 1] = data[offset + 1];
    buffer[i + 2] = data[offset + 2];

    offset += 4;
  }
  return tf.tensor3d(buffer, [height, width, 3]);
}

export async function tensorToImageUrl(imageTensor: tf.Tensor3D):
    Promise<string> {
  const [height, width] = imageTensor.shape;
  const buffer = await imageTensor.toInt().data();
  const frameData = new Uint8Array(width * height * 4);

  let offset = 0;
  for (let i = 0; i < frameData.length; i += 4) {
    frameData[i] = buffer[offset];
    frameData[i + 1] = buffer[offset + 1];
    frameData[i + 2] = buffer[offset + 2];
    frameData[i + 3] = 0xFF;

    offset += 3;
  }

  const rawImageData = {
    data: frameData,
    width,
    height,
  };
  const jpegImageData = jpeg.encode(rawImageData, 75);
  const base64Encoding = tf.util.decodeString(jpegImageData.data, 'base64');
  return base64Encoding;
}
