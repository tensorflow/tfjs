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
import {StringSplit, StringSplitAttrs, StringSplitInputs} from '../../kernel_names';
import {Scalar, Tensor, Tensor1D} from '../../tensor';
import {NamedTensorMap} from '../../tensor_types';
import {convertToTensor} from '../../tensor_util_env';
import {ScalarLike, TensorLike} from '../../types';
import {op} from '../operation';

/**
 * Split elements of `input` based on `delimiter` into a SparseTensor .
 *
 * Let N be the size of source (typically N will be the batch size). Split each
 * element of `input` based on `delimiter` and return a SparseTensor containing
 * the splitted tokens. Empty tokens are ignored if `skipEmpty` is set to True.
 *
 * `delimiter` can be empty, or a string of split characters. If `delimiter` is
 * an empty string, each element of `input` is split into individual
 * character strings. Otherwise every character of `delimiter` is a potential
 * split point.
 *
 * ```js
 * const result = tf.string.stringSplit(['hello world',  'a b c'], ' ');
 * result['indices'].print(); // [[0, 0], [0, 1], [1, 0], [1, 1], [1, 2]]
 * result['values'].print(); // ['hello', 'world', 'a', 'b', 'c']
 * result['shape'].print(); // [2, 3]
 * ```
 * @param input: 1-D. Strings to split.
 * @param delimiter: 0-D. Delimiter characters, or empty string.
 * @param skipEmpty: Optional. If true, skip the empty strings from the result.
 *     Defaults to true.
 * @return A map with the following properties:
 *     - indices: A dense matrix of int32 representing the indices of the sparse
 *       tensor.
 *     - values: A vector of strings corresponding to the splited values.
 *     - shape: a length-2 vector of int32 representing the shape of the sparse
 * tensor, where the first value is N and the second value is the maximum number
 * of tokens in a single input entry.
 *
 * @doc {heading: 'Operations', subheading: 'String'}
 */
function stringSplit_(
    input: Tensor1D|TensorLike, delimiter: Scalar|ScalarLike,
    skipEmpty = true): NamedTensorMap {
  const $input = convertToTensor(input, 'input', 'stringSplit', 'string');
  const $delimiter =
      convertToTensor(delimiter, 'delimiter', 'stringSplit', 'string');

  if ($input.rank !== 1) {
    throw new Error(
        `Input should be Tensor1D but received shape ${$input.shape}`);
  }
  if ($delimiter.rank !== 0) {
    throw new Error(
        `Delimiter should be a scalar but received shape ${$delimiter.shape}`);
  }

  const attrs: StringSplitAttrs = {skipEmpty};
  const inputs: StringSplitInputs = {input: $input, delimiter: $delimiter};
  const result: Tensor[] =
      ENGINE.runKernel(StringSplit, inputs as {}, attrs as {});
  return {indices: result[0], values: result[1], shape: result[2]};
}

export const stringSplit = op({stringSplit_});
