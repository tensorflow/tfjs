/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
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

import {ENGINE, ForwardFunc} from '../engine';
import {LogSoftmax, LogSoftmaxAttrs, LogSoftmaxInputs} from '../kernel_names';
import {NamedAttrMap} from '../kernel_registry';
import {Tensor} from '../tensor';
import {NamedTensorMap} from '../tensor_types';
import {convertToTensor} from '../tensor_util_env';
import {TensorLike} from '../types';

import {cast} from './cast';
import {exp} from './exp';
import {log} from './log';
import {max} from './max';
import {op} from './operation';
import {sub} from './sub';
import {sum} from './sum';

/**
 * Computes the log softmax.
 *
 * ```js
 * const a = tf.tensor1d([1, 2, 3]);
 *
 * a.logSoftmax().print();  // or tf.logSoftmax(a)
 * ```
 *
 * ```js
 * const a = tf.tensor2d([2, 4, 6, 1, 2, 3], [2, 3]);
 *
 * a.logSoftmax().print();  // or tf.logSoftmax(a)
 * ```
 *
 * @param logits The logits array.
 * @param axis The dimension softmax would be performed on. Defaults to `-1`
 *     which indicates the last dimension.
 *
 * @doc {heading: 'Operations', subheading: 'Normalization'}
 */
function logSoftmax_<T extends Tensor>(logits: T|TensorLike, axis = -1): T {
  const $logits = convertToTensor(logits, 'logits', 'logSoftmax');

  if (axis === -1) {
    axis = $logits.rank - 1;
  }
  if (axis !== $logits.rank - 1) {
    throw Error(
        'Log Softmax along a non-last dimension is not yet supported. ' +
        `Logits was rank ${$logits.rank} and axis was ${axis}`);
  }

  const forward: ForwardFunc<Tensor> = (backend, save) => {
    const keepDims = true;
    const xMax = max(logits, axis, true);
    const shifted = sub(logits, xMax);
    const value =
        sub(cast(shifted, 'float32'), log(sum(exp(shifted), axis, keepDims)));
    save([value]);
    return value;
  };

  const inputs: LogSoftmaxInputs = {logits: $logits};
  const attrs: LogSoftmaxAttrs = {axis};

  return ENGINE.runKernelFunc(
             forward, inputs as {} as NamedTensorMap, null /* grad */,
             LogSoftmax, attrs as {} as NamedAttrMap) as T;
}

export const logSoftmax = op({logSoftmax_});
