/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
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

import {op} from '../ops/operation';
import {Tensor, Tensor1D} from '../tensor';

import {concat} from './concat';
import {frame} from './frame';
import {hannWindow} from './hann_window';
import {mul} from './mul';
import {rfft} from './rfft';
import {enclosingPowerOfTwo} from './signal_ops_util';
import {slice} from './slice';

/**
 * Computes the Short-time Fourier Transform of signals
 * See: https://en.wikipedia.org/wiki/Short-time_Fourier_transform
 *
 * ```js
 * const input = tf.tensor1d([1, 1, 1, 1, 1])
 * tf.signal.stft(input, 3, 1).print();
 * ```
 * @param signal 1-dimensional real value tensor.
 * @param frameLength The window length of samples.
 * @param frameStep The number of samples to step.
 * @param fftLength The size of the FFT to apply.
 * @param windowFn A callable that takes a window length and returns 1-d tensor.
 */
/**
 * @doc {heading: 'Operations', subheading: 'Signal', namespace: 'signal'}
 */
function stft_(
    signal: Tensor1D, frameLength: number, frameStep: number,
    fftLength?: number,
    windowFn: (length: number) => Tensor1D = hannWindow): Tensor {
  if (fftLength == null) {
    fftLength = enclosingPowerOfTwo(frameLength);
  }
  const framedSignal = frame(signal, frameLength, frameStep);
  const windowedSignal = mul(framedSignal, windowFn(frameLength));
  const output: Tensor[] = [];
  for (let i = 0; i < framedSignal.shape[0]; i++) {
    output.push(
        rfft(slice(windowedSignal, [i, 0], [1, frameLength]), fftLength));
  }
  return concat(output);
}
export const stft = op({stft_});
