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
import {Tensor, Tensor2D} from '../tensor';

import {ifft} from './ifft';
import {scalar} from './scalar';

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
 */
/**
 * @doc {heading: 'Operations', subheading: 'Spectral', namespace: 'spectral'}
 */
function irfft_(input: Tensor): Tensor {
  const innerDimensionSize = input.shape[input.shape.length - 1];
  const batch = input.size / innerDimensionSize;
  let ret: Tensor;
  if (innerDimensionSize <= 2) {
    const complexInput = input.as2D(batch, innerDimensionSize);
    ret = ifft(complexInput);
  } else {
    // The length of unique components of the DFT of a real-valued signal
    // is 2 * (input_len - 1)
    const outputShape = [batch, 2 * (innerDimensionSize - 1)];
    const realInput = real(input).as2D(batch, innerDimensionSize);
    const imagInput = imag(input).as2D(batch, innerDimensionSize);

    const realConjugate =
        realInput.slice([0, 1], [batch, innerDimensionSize - 2]).reverse(1);
    const imagConjugate: Tensor2D =
        imagInput.slice([0, 1], [batch, innerDimensionSize - 2])
            .reverse(1)
            .mul(scalar(-1));

    const r = realInput.concat(realConjugate, 1);
    const i = imagInput.concat(imagConjugate, 1);
    const complexInput = complex(r, i).as2D(outputShape[0], outputShape[1]);
    ret = ifft(complexInput);
  }
  ret = real(ret);
  // reshape the result if the input is 3D tensor.
  if (input.rank === 3 && input.shape[0] !== 0) {
    const temp = ret;
    const batch = input.shape[0];
    ret = ret.reshape([batch, ret.shape[0] / batch, ret.shape[1]]);
    temp.dispose();
  }
  return ret;
}

export const irfft = op({irfft_});
