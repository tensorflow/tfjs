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
import {complex, imag, real} from '../ops/complex_ops';
import {op} from '../ops/operation';
import {Tensor, Tensor2D} from '../tensor';
import {assert} from '../util';
import {scalar} from './tensor_ops';

/**
 * Fast Fourier transform.
 *
 * Computes the 1-dimensional discrete Fourier transform over the inner-most
 * dimension of input.
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
function fft_(input: Tensor): Tensor {
  assert(
      input.dtype === 'complex64',
      () => `The dtype for tf.spectral.fft() must be complex64 ` +
          `but got ${input.dtype}.`);

  // Collapse all outer dimensions to a single batch dimension.
  const innerDimensionSize = input.shape[input.shape.length - 1];
  const batch = input.size / innerDimensionSize;
  const input2D = input.as2D(batch, innerDimensionSize);

  const ret = ENV.engine.runKernel(backend => backend.fft(input2D), {input});

  return ret.reshape(input.shape);
}

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
 */
/**
 * @doc {heading: 'Operations', subheading: 'Spectral', namespace: 'spectral'}
 */
function ifft_(input: Tensor): Tensor {
  assert(
      input.dtype === 'complex64',
      () => `The dtype for tf.spectral.ifft() must be complex64 ` +
          `but got ${input.dtype}.`);

  // Collapse all outer dimensions to a single batch dimension.
  const innerDimensionSize = input.shape[input.shape.length - 1];
  const batch = input.size / innerDimensionSize;
  const input2D = input.as2D(batch, innerDimensionSize);

  const ret = ENV.engine.runKernel(backend => backend.ifft(input2D), {input});

  return ret.reshape(input.shape);
}

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
function rfft_(input: Tensor): Tensor {
  assert(
      input.dtype === 'float32',
      () => `The dtype for rfft() must be real value but got ${input.dtype}`);

  const innerDimensionSize = input.shape[input.shape.length - 1];
  const batch = input.size / innerDimensionSize;

  // Complement the input with zero imaginary numbers.
  const zeros = input.zerosLike();
  const complexInput = complex(input, zeros).as2D(batch, innerDimensionSize);

  const ret = fft(complexInput);

  // Exclude complex conjugations. These conjugations are put symmetrically.
  const half = Math.floor(innerDimensionSize / 2) + 1;
  const realValues = real(ret);
  const imagValues = imag(ret);
  const realComplexConjugate = realValues.split(
      [half, innerDimensionSize - half], realValues.shape.length - 1);
  const imagComplexConjugate = imagValues.split(
      [half, innerDimensionSize - half], imagValues.shape.length - 1);

  const outputShape = input.shape.slice();
  outputShape[input.shape.length - 1] = half;

  return complex(realComplexConjugate[0], imagComplexConjugate[0])
      .reshape(outputShape);
}

/**
 * Inversed real value input fast Fourier transform.
 *
 * Computes the 1-dimensional inversed discrete Fourier transform over the
 * inner-most dimension of the real input.
 *
 * ```js
 * const real = tf.tensor1d([1, 2, 3]);
 *
 * real.irfft().print();
 * ```
 * @param input The real value input to compute an irfft over.
 */
/**
 * @doc {heading: 'Operations', subheading: 'Spectral', namespace: 'spectral'}
 */
function irfft_(input: Tensor): Tensor {
  const innerDimensionSize = input.shape[input.shape.length - 1];
  const batch = input.size / innerDimensionSize;

  if (innerDimensionSize <= 2) {
    const complexInput = input.as2D(batch, innerDimensionSize);
    const ret = ifft(complexInput);
    return real(ret);
  } else {
    // The length of unique components of the DFT of a real-valued signal
    // is 2 * (input_len - 1)
    const outputShape = [batch, 2 * (innerDimensionSize - 1)];
    const realInput = real(input).as2D(batch, innerDimensionSize);
    const imagInput = imag(input).as2D(batch, innerDimensionSize);

    const realConjugate =
        realInput.slice([0, 1], [batch, innerDimensionSize - 2]).reverse(1);
    const imagConjugate =
        imagInput.slice([0, 1], [batch, innerDimensionSize - 2])
            .reverse(1)
            .mul(scalar(-1)) as Tensor2D;

    const r = realInput.concat(realConjugate, 1);
    const i = imagInput.concat(imagConjugate, 1);
    const complexInput = complex(r, i).as2D(outputShape[0], outputShape[1]);
    const ret = ifft(complexInput);
    return real(ret);
  }
}

export const fft = op({fft_});
export const ifft = op({ifft_});
export const rfft = op({rfft_});
export const irfft = op({irfft_});
