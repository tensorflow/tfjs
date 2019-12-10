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

/**
 * Custom ops for the tfjs-react-native platform
 */

import * as tf from '@tensorflow/tfjs-core';
import {FromCameraAttrs} from './kernels/from_camera';

function fromCamera_(
    texture: WebGLTexture, width: number, height: number, numChannels = 3) {
  tf.util.assert(
      numChannels > 0 && numChannels < 5,
      () => `numChannels must be between 1 and 4. Got '${numChannels}'`);

  tf.util.assert(
      //@ts-ignore
      texture instanceof WebGLTexture,
      () => 'texture must be a WebGLTexture object');

  const fromCameraAttrs:
      FromCameraAttrs = {width, height, texture, numChannels};
  return tf.engine().runKernel('FromCamera', {}, fromCameraAttrs as {});
}

export const fromCamera = tf.op({fromCamera_});
