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
import {Elu, EluInputs} from '../kernel_names';
import {Tensor} from '../tensor';
import {NamedTensorMap} from '../tensor_types';
import {convertToTensor} from '../tensor_util_env';
import {TensorLike} from '../types';

import {op} from './operation';

/**
 * Computes exponential linear element-wise: `x > 0 ? e ^ x - 1 : 0`.
 *
 * ```js
 * const x = tf.tensor1d([-1, 1, -3, 2]);
 *
 * x.elu().print();  // or tf.elu(x)
 * ```
 * @param x The input tensor.
 */
/** @doc {heading: 'Operations', subheading: 'Basic math'} */
function elu_<T extends Tensor>(x: T|TensorLike): T {
  const $x = convertToTensor(x, 'x', 'elu');

  const forward: ForwardFunc<Tensor> = (backend, save) => {
    const y = backend.elu($x);
    save([y]);
    return y;
  };

  const inputs: EluInputs = {x: $x};

  return ENGINE.runKernelFunc(
             forward, inputs as {} as NamedTensorMap, null /* grad */, Elu) as
      T;
}

export const elu = op({elu_});
