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

import {arrayBufferToBase64String, base64StringToArrayBuffer, urlSafeBase64, urlUnsafeBase64} from '../io/io_utils';
import {StringTensor} from '../tensor';
import {decodeString} from '../util';
import {ENGINE} from '../engine';

/** Shared implementation of the encodeBase64 kernel across WebGL and CPU. */
export function encodeBase64Impl<T extends StringTensor>(
    values: Uint8Array[], shape: number[], pad = false): T {
  const resultValues = new Array(values.length);

  for (let i = 0; i < values.length; ++i) {
    const bStr = arrayBufferToBase64String(values[i].buffer);
    const bStrUrl = urlSafeBase64(bStr);

    if (pad) {
      resultValues[i] = bStrUrl;
    } else {
      // Remove padding
      resultValues[i] = bStrUrl.replace(/=/g, '');
    }
  }

  return ENGINE.makeTensor(resultValues, shape, 'string') as T;
}

/** Shared implementation of the decodeBase64 kernel across WebGL and CPU. */
export function decodeBase64Impl<T extends StringTensor>(
    values: Uint8Array[], shape: number[]): T {
  const resultValues = new Array(values.length);

  for (let i = 0; i < values.length; ++i) {
    // Undo URL safe and decode from Base64 to ArrayBuffer
    const bStrUrl = decodeString(values[i]);
    const bStr = urlUnsafeBase64(bStrUrl);
    const aBuff = base64StringToArrayBuffer(bStr);

    resultValues[i] = decodeString(new Uint8Array(aBuff));
  }

  return ENGINE.makeTensor(resultValues, shape, 'string') as T;
}
