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
import {Prelu, PreluInputs} from '../kernel_names';
import {Tensor} from '../tensor';
import {NamedTensorMap} from '../tensor_types';
import {convertToTensor} from '../tensor_util_env';
import {TensorLike} from '../types';

import {op} from './operation';

/**
 * Computes leaky rectified linear element-wise with parametric alphas.
 *
 * `x < 0 ? alpha * x : f(x) = x`
 *
 * ```js
 * const x = tf.tensor1d([-1, 2, -3, 4]);
 * const alpha = tf.scalar(0.1);
 *
 * x.prelu(alpha).print();  // or tf.prelu(x, alpha)
 * ```
 * @param x The input tensor.
 * @param alpha Scaling factor for negative values.
 *
 * @doc {heading: 'Operations', subheading: 'Basic math'}
 */
function prelu_<T extends Tensor>(x: T|TensorLike, alpha: T|TensorLike): T {
  const $x = convertToTensor(x, 'x', 'prelu');
  const $alpha = convertToTensor(alpha, 'alpha', 'prelu');

  const forward: ForwardFunc<Tensor> = (backend, save) => {
    const res = backend.prelu($x, $alpha);
    save([$x, $alpha]);
    return res;
  };

  const inputs: PreluInputs = {x: $x, alpha: $alpha};
  return ENGINE.runKernelFunc(
             forward, inputs as {} as NamedTensorMap, null /* grad */, Prelu) as
      T;
}

export const prelu = op({prelu_});
