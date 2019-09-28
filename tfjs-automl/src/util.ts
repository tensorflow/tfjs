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

import {browser, Tensor, Tensor3D, util} from '@tensorflow/tfjs-core';
import {ImageInput} from './types';

export function imageToTensor(img: ImageInput): Tensor3D {
  return img instanceof Tensor ? img : browser.fromPixels(img);
}

/** Loads and parses the dictionary. */
export async function loadDictionary(modelUrl: string): Promise<string[]> {
  const lastIndexOfSlash = modelUrl.lastIndexOf('/');
  const prefixUrl =
      lastIndexOfSlash >= 0 ? modelUrl.slice(0, lastIndexOfSlash + 1) : '';
  const dictUrl = `${prefixUrl}dict.txt`;
  const response = await util.fetch(dictUrl);
  const text = await response.text();
  return text.trim().split('\n');
}
