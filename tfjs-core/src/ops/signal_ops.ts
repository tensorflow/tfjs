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
import {fill} from './fill';
import {mul} from './mul';
import {slice} from './slice';
import {rfft} from './spectral_ops';
import {tensor1d, tensor2d} from './tensor_ops';

/**
 * Generate a Hann window.
 *
 * See: https://en.wikipedia.org/wiki/Window_function#Hann_and_Hamming_windows
 *
 * ```js
 * tf.signal.hannWindow(10).print();
 * ```
 * @param The length of window
 */
/**
 * @doc {heading: 'Operations', subheading: 'Signal', namespace: 'signal'}
 */
function hannWindow_(windowLength: number): Tensor1D {
  return cosineWindow(windowLength, 0.5, 0.5);
}

/**
 * Generate a hamming window.
 *
 * See: https://en.wikipedia.org/wiki/Window_function#Hann_and_Hamming_windows
 *
 * ```js
 * tf.signal.hammingWindow(10).print();
 * ```
 * @param The length of window
 */
/**
 * @doc {heading: 'Operations', subheading: 'Signal', namespace: 'signal'}
 */
function hammingWindow_(windowLength: number): Tensor1D {
  return cosineWindow(windowLength, 0.54, 0.46);
}

/**
 * Expands input into frames of frameLength.
 * Slides a window size with frameStep.
 *
 * ```js
 * tf.signal.frame([1, 2, 3], 2, 1).print();
 * ```
 * @param signal The input tensor to be expanded
 * @param frameLength Length of each frame
 * @param frameStep The frame hop size in samples.
 * @param padEnd Whether to pad the end of signal with padValue.
 * @param padValue An number to use where the input signal does
 *     not exist when padEnd is True.
 */
/**
 * @doc {heading: 'Operations', subheading: 'Signal', namespace: 'signal'}
 */
function frame_(
    signal: Tensor1D, frameLength: number, frameStep: number, padEnd = false,
    padValue = 0): Tensor {
  let start = 0;
  const output: Tensor[] = [];
  while (start + frameLength <= signal.size) {
    output.push(slice(signal, start, frameLength));
    start += frameStep;
  }

  if (padEnd) {
    while (start < signal.size) {
      const padLen = (start + frameLength) - signal.size;
      const pad = concat([
        slice(signal, start, frameLength - padLen), fill([padLen], padValue)
      ]);
      output.push(pad);
      start += frameStep;
    }
  }

  if (output.length === 0) {
    return tensor2d([], [0, frameLength]);
  }

  return concat(output).as2D(output.length, frameLength);
}

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
        rfft(windowedSignal.slice([i, 0], [1, frameLength]), fftLength));
  }
  return concat(output);
}

function enclosingPowerOfTwo(value: number) {
  // Return 2**N for integer N such that 2**N >= value.
  return Math.floor(Math.pow(2, Math.ceil(Math.log(value) / Math.log(2.0))));
}

function cosineWindow(windowLength: number, a: number, b: number): Tensor1D {
  const even = 1 - windowLength % 2;
  const newValues = new Float32Array(windowLength);
  for (let i = 0; i < windowLength; ++i) {
    const cosArg = (2.0 * Math.PI * i) / (windowLength + even - 1);
    newValues[i] = a - b * Math.cos(cosArg);
  }
  return tensor1d(newValues, 'float32');
}

export const hannWindow = op({hannWindow_});
export const hammingWindow = op({hammingWindow_});
export const frame = op({frame_});
export const stft = op({stft_});
