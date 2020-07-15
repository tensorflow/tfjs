/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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

import {complex} from '../ops/complex';
import {imag} from '../ops/imag';
import {op} from '../ops/operation';
import {real} from '../ops/real';
import {Tensor} from '../tensor';
import {assert} from '../util';

import {fft} from './fft';
import {zeros} from './zeros';

/**
 * Real value input fast Fourier transform.
 *
 * Computes the 1-dimensional discrete Fourier transform over the
 * inner-most dimension of the real input.
 *
 * ```js
 * const real = tf.tensor1d([1, 2, 3]);
 *
 * real.rfft().print();
 * ```
 * @param input The real value input to compute an rfft over.
 */
/**
 * @doc {heading: 'Operations', subheading: 'Spectral', namespace: 'spectral'}
 */
function rfft_(input: Tensor, fftLength?: number): Tensor {
  assert(
      input.dtype === 'float32',
      () => `The dtype for rfft() must be real value but got ${input.dtype}`);

  let innerDimensionSize = input.shape[input.shape.length - 1];
  const batch = input.size / innerDimensionSize;

  let adjustedInput: Tensor;
  if (fftLength != null && fftLength < innerDimensionSize) {
    // Need to crop
    const begin = input.shape.map(v => 0);
    const size = input.shape.map(v => v);
    size[input.shape.length - 1] = fftLength;
    adjustedInput = input.slice(begin, size);
    innerDimensionSize = fftLength;
  } else if (fftLength != null && fftLength > innerDimensionSize) {
    // Need to pad with zeros
    const zerosShape = input.shape.map(v => v);
    zerosShape[input.shape.length - 1] = fftLength - innerDimensionSize;
    adjustedInput = input.concat(zeros(zerosShape), input.shape.length - 1);
    innerDimensionSize = fftLength;
  } else {
    adjustedInput = input;
  }

  // Complement the input with zero imaginary numbers.
  const zerosInput = adjustedInput.zerosLike();
  const complexInput =
      complex(adjustedInput, zerosInput).as2D(batch, innerDimensionSize);

  const ret = fft(complexInput);

  // Exclude complex conjugations. These conjugations are put symmetrically.
  const half = Math.floor(innerDimensionSize / 2) + 1;
  const realValues = real(ret);
  const imagValues = imag(ret);
  const realComplexConjugate = realValues.split(
      [half, innerDimensionSize - half], realValues.shape.length - 1);
  const imagComplexConjugate = imagValues.split(
      [half, innerDimensionSize - half], imagValues.shape.length - 1);

  const outputShape = adjustedInput.shape.slice();
  outputShape[adjustedInput.shape.length - 1] = half;

  return complex(realComplexConjugate[0], imagComplexConjugate[0])
      .reshape(outputShape);
}

export const rfft = op({rfft_});
