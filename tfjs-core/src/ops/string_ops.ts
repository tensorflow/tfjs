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
import {ENGINE} from '../engine';
import {StringTensor, Tensor} from '../tensor';
import {convertToTensor} from '../tensor_util_env';

import {op} from './operation';

/**
 * Encodes the values of a `tf.Tensor` (of dtype `string`) to Base64.
 *
 * Given a String tensor, returns a new tensor with the values encoded into
 * web-safe base64 format.
 *
 * Web-safe means that the encoder uses `-` and `_` instead of `+` and `/`:
 *
 * en.wikipedia.org/wiki/Base64
 *
 * ```js
 * const x = tf.tensor1d(['Hello world!'], 'string');
 *
 * x.encodeBase64().print();
 * ```
 * @param str The input `tf.Tensor` of dtype `string` to encode.
 * @param pad Whether to add padding (`=`) to the end of the encoded string.
 */
/** @doc {heading: 'Operations', subheading: 'String'} */
function encodeBase64_<T extends StringTensor>(
    str: StringTensor|Tensor, pad = false): T {
  const $str = convertToTensor(str, 'str', 'encodeBase64', 'string');

  const backwardsFunc = (dy: T) => ({$str: () => decodeBase64(dy)});

  return ENGINE.runKernelFunc(
      backend => backend.encodeBase64($str, pad), {$str}, backwardsFunc);
}

/**
 * Decodes the values of a `tf.Tensor` (of dtype `string`) from Base64.
 *
 * Given a String tensor of Base64 encoded values, returns a new tensor with the
 * decoded values.
 *
 * en.wikipedia.org/wiki/Base64
 *
 * ```js
 * const y = tf.scalar('SGVsbG8gd29ybGQh', 'string');
 *
 * y.decodeBase64().print();
 * ```
 * @param str The input `tf.Tensor` of dtype `string` to decode.
 */
/** @doc {heading: 'Operations', subheading: 'String'} */
function decodeBase64_<T extends StringTensor>(str: StringTensor|Tensor): T {
  const $str = convertToTensor(str, 'str', 'decodeBase64', 'string');

  const backwardsFunc = (dy: T) => ({$str: () => encodeBase64(dy)});

  return ENGINE.runKernelFunc(
      backend => backend.decodeBase64($str), {$str}, backwardsFunc);
}

export const encodeBase64 = op({encodeBase64_});
export const decodeBase64 = op({decodeBase64_});
