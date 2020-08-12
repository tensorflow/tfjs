/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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
import {Relu, ReluInputs} from '../kernel_names';
import {Tensor} from '../tensor';
import {NamedTensorMap} from '../tensor_types';
import {convertToTensor} from '../tensor_util_env';
import {TensorLike} from '../types';

import {op} from './operation';
import { cast } from './cast';

/**
 * Computes rectified linear element-wise: `max(x, 0)`.
 *
 * ```js
 * const x = tf.tensor1d([-1, 2, -3, 4]);
 *
 * x.relu().print();  // or tf.relu(x)
 * ```
 * @param x The input tensor. If the dtype is `bool`, the output dtype will be
 *     `int32'.
 */
/** @doc {heading: 'Operations', subheading: 'Basic math'} */
function relu_<T extends Tensor>(x: T|TensorLike): T {
  const $x = convertToTensor(x, 'x', 'relu');

  const forward: ForwardFunc<Tensor> = (backend, save) => {
    save([$x]);

    if ($x.dtype === 'bool') {
      return cast($x, 'int32');
    }

    return backend.relu($x);
  };

  const inputs: ReluInputs = {x: $x};

  return ENGINE.runKernelFunc(
             forward, inputs as {} as NamedTensorMap, null /* grad */, Relu) as
      T;
}

export const relu = op({relu_});
