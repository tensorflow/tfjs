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

import {ENGINE} from '../../engine';
import {StringToHashBucketFast, StringToHashBucketFastAttrs, StringToHashBucketFastInputs} from '../../kernel_names';
import {Tensor} from '../../tensor';
import {convertToTensor} from '../../tensor_util_env';
import {TensorLike} from '../../types';
import {op} from '../operation';

/**
 * Converts each string in the input Tensor to its hash mod by a number of
 * buckets.
 *
 * The hash function is deterministic on the content of the string within the
 * process and will never change. However, it is not suitable for cryptography.
 * This function may be used when CPU time is scarce and inputs are trusted or
 * unimportant. There is a risk of adversaries constructing inputs that all hash
 * to the same bucket.
 *
 * ```js
 * const result = tf.string.stringToHashBucketFast(
 *   ['Hello', 'TensorFlow', '2.x'], 3);
 * result.print(); // [0, 2, 2]
 * ```
 * @param input: The strings to assign a hash bucket.
 * @param numBuckets: The number of buckets.
 * @return A Tensor of the same shape as the input tensor.
 *
 * @doc {heading: 'Operations', subheading: 'String'}
 */
function stringToHashBucketFast_(
    input: Tensor|TensorLike, numBuckets: number): Tensor {
  const $input =
      convertToTensor(input, 'input', 'stringToHashBucketFast', 'string');
  const attrs: StringToHashBucketFastAttrs = {numBuckets};

  if (numBuckets <= 0) {
    throw new Error(`Number of buckets must be at least 1`);
  }

  const inputs: StringToHashBucketFastInputs = {input: $input};
  return ENGINE.runKernel(StringToHashBucketFast, inputs as {}, attrs as {});
}

export const stringToHashBucketFast = op({stringToHashBucketFast_});
