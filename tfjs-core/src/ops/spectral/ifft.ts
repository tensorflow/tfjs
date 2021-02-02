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

import {ENGINE} from '../../engine';
import {IFFT, IFFTInputs} from '../../kernel_names';
import {Tensor} from '../../tensor';
import {NamedTensorMap} from '../../tensor_types';
import {assert} from '../../util';
import {op} from '../operation';

/**
 * Inverse fast Fourier transform.
 *
 * Computes the inverse 1-dimensional discrete Fourier transform over the
 * inner-most dimension of input.
 *
 * ```js
 * const real = tf.tensor1d([1, 2, 3]);
 * const imag = tf.tensor1d([1, 2, 3]);
 * const x = tf.complex(real, imag);
 *
 * x.ifft().print();  // tf.spectral.ifft(x).print();
 * ```
 * @param input The complex input to compute an ifft over.
 *
 * @doc {heading: 'Operations', subheading: 'Spectral', namespace: 'spectral'}
 */
function ifft_(input: Tensor): Tensor {
  assert(
      input.dtype === 'complex64',
      () => `The dtype for tf.spectral.ifft() must be complex64 ` +
          `but got ${input.dtype}.`);

  const inputs: IFFTInputs = {input};

  return ENGINE.runKernel(IFFT, inputs as {} as NamedTensorMap);
}

export const ifft = op({ifft_});
