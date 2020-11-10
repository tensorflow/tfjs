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

import {Tensor, Tensor2D} from '../../tensor';
import {complex} from '../complex';
import {concat} from '../concat';
import {imag} from '../imag';
import {mul} from '../mul';
import {op} from '../operation';
import {real} from '../real';
import {reshape} from '../reshape';
import {reverse} from '../reverse';
import {scalar} from '../scalar';
import {slice} from '../slice';

import {ifft} from './ifft';

/**
 * Inversed real value input fast Fourier transform.
 *
 * Computes the 1-dimensional inversed discrete Fourier transform over the
 * inner-most dimension of the real input.
 *
 * ```js
 * const real = tf.tensor1d([1, 2, 3]);
 * const imag = tf.tensor1d([0, 0, 0]);
 * const x = tf.complex(real, imag);
 *
 * x.irfft().print();
 * ```
 * @param input The real value input to compute an irfft over.
 *
 * @doc {heading: 'Operations', subheading: 'Spectral', namespace: 'spectral'}
 */
function irfft_(input: Tensor): Tensor {
  const innerDimensionSize = input.shape[input.shape.length - 1];
  const batch = input.size / innerDimensionSize;
  let ret: Tensor;
  if (innerDimensionSize <= 2) {
    const complexInput = reshape(input, [batch, innerDimensionSize]);
    ret = ifft(complexInput);
  } else {
    // The length of unique components of the DFT of a real-valued signal
    // is 2 * (input_len - 1)
    const outputShape = [batch, 2 * (innerDimensionSize - 1)];
    const realInput = reshape(real(input), [batch, innerDimensionSize]);
    const imagInput = reshape(imag(input), [batch, innerDimensionSize]);

    const realConjugate =
        reverse(slice(realInput, [0, 1], [batch, innerDimensionSize - 2]), 1);
    const imagConjugate: Tensor2D = mul(
        reverse(slice(imagInput, [0, 1], [batch, innerDimensionSize - 2]), 1),
        scalar(-1));

    const r = concat([realInput, realConjugate], 1);
    const i = concat([imagInput, imagConjugate], 1);
    const complexInput =
        reshape(complex(r, i), [outputShape[0], outputShape[1]]);
    ret = ifft(complexInput);
  }
  ret = real(ret);
  // reshape the result if the input is 3D tensor.
  if (input.rank === 3 && input.shape[0] !== 0) {
    const temp = ret;
    const batch = input.shape[0];
    ret = reshape(ret, [batch, ret.shape[0] / batch, ret.shape[1]]);
    temp.dispose();
  }
  return ret;
}

export const irfft = op({irfft_});
