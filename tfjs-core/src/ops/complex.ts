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
import {Complex, ComplexInputs} from '../kernel_names';
import {Tensor} from '../tensor';
import {NamedTensorMap} from '../tensor_types';
import {convertToTensor} from '../tensor_util_env';
import {TensorLike} from '../types';
import * as util from '../util';

import {op} from './operation';

/**
 * Converts two real numbers to a complex number.
 *
 * Given a tensor `real` representing the real part of a complex number, and a
 * tensor `imag` representing the imaginary part of a complex number, this
 * operation returns complex numbers elementwise of the form [r0, i0, r1, i1],
 * where r represents the real part and i represents the imag part.
 *
 * The input tensors real and imag must have the same shape.
 *
 * ```js
 * const real = tf.tensor1d([2.25, 3.25]);
 * const imag = tf.tensor1d([4.75, 5.75]);
 * const complex = tf.complex(real, imag);
 *
 * complex.print();
 * ```
 */
/** @doc {heading: 'Tensors', subheading: 'Creation'} */
function complex_<T extends Tensor>(real: T|TensorLike, imag: T|TensorLike): T {
  const $real = convertToTensor(real, 'real', 'complex');
  const $imag = convertToTensor(imag, 'imag', 'complex');
  util.assertShapesMatch(
      $real.shape, $imag.shape,
      `real and imag shapes, ${$real.shape} and ${$imag.shape}, ` +
          `must match in call to tf.complex().`);

  const forward: ForwardFunc<Tensor> = (backend) => {
    return backend.complex($real, $imag);
  };
  const inputs: ComplexInputs = {real: $real, imag: $imag};
  return ENGINE.runKernelFunc(
             forward, inputs as {} as NamedTensorMap, null /* gradient */,
             Complex) as T;
}

export const complex = op({complex_});
