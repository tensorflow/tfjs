/**
 * @license
 * Copyright 2018 Google Inc. All Rights Reserved.
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

import {ENV} from '../environment';
import {op} from '../ops/operation';
import {Tensor1D} from '../tensor';
import {assert} from '../util';

/**
 * Compute the 1-dimentional discrete fourier transform
 * The input is expected to be the 1D tensor with dtype complex64.
 *
 * ```js
 * const real = tf.tensor1d([1, 2, 3]);
 * const imag = tf.tensor1d([1, 2, 3]);
 * const x = tf.complex(real, imag);
 *
 * x.fft().print();  // tf.spectral.fft(x).print();
 * ```
 * @param input The complex input to compute an fft over.
 */
/**
 * @doc {heading: 'Operations', subheading: 'Spectral', namespace: 'spectral'}
 */
function fft_(input: Tensor1D): Tensor1D {
  assert(input.dtype === 'complex64', 'dtype must be complex64');
  assert(input.rank === 1, 'input rank must be 1');
  const ret = ENV.engine.runKernel(backend => backend.fft(input), {input});
  return ret;
}

export const fft = op({fft_});
