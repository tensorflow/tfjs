/**
 * @license
 * Copyright 2026 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

import {Tensor, tensor2d} from '@tensorflow/tfjs-core';

/**
 * Configuration options for `padSequences`.
 */
export interface PadSequencesOptions {
  /**
   * Maximum length of all sequences. If omitted, it defaults to the
   * length of the longest sequence in the input.
   */
  maxlen?: number;
  /**
   * String: 'pre' or 'post'. Pad either before or after each sequence.
   * Defaults to 'pre'.
   */
  padding?: 'pre'|'post';
  /**
   * String: 'pre' or 'post'. Remove values from sequences larger than
   * `maxlen`, either at the beginning or at the end of the sequences.
   * Defaults to 'pre'.
   */
  truncating?: 'pre'|'post';
  /**
   * Float or String: Pad value (must match output dtype).
   * Defaults to 0.
   */
  value?: number;
  /**
   * Data type of the output Tensor.
   * Defaults to 'int32'.
   */
  dtype?: 'int32'|'float32';
}

/**
 * Pads or truncates a list of sequences (nested numeric arrays) to the
 * same length.
 *
 * This function mimics the behavior of Python Keras
 * `tf.keras.utils.pad_sequences`.
 *
 * @param sequences A 2D array of numeric sequences (e.g., tokenized
 *   integer arrays).
 * @param options Configuration options for padding, truncating, max
 *   length, and data type.
 * @returns A 2D Tensor (`Tensor2D`) containing the padded/truncated
 *   sequences with shape `[sequences.length, maxlen]`.
 */
export function padSequences(
    sequences: number[][], options: PadSequencesOptions = {}): Tensor {
  // Extract configuration options with sensible Keras-compatible defaults
  const {padding = 'pre', truncating = 'pre', value = 0, dtype = 'int32'} =
      options;

  // Calculate maximum length: use user-defined maxlen if provided,
  // otherwise find the length of the longest input sequence.
  const lengths = sequences.map(s => s.length);
  const maxlen = options.maxlen ?? Math.max(...lengths, 0);

  // Process each sequence individually
  const padded = sequences.map(seq => {
    let current = [...seq];

    // Truncate sequence if it exceeds maxlen
    if (current.length > maxlen) {
      current = truncating === 'post' ?
          current.slice(0, maxlen) :               // Keep first `maxlen`
          current.slice(current.length - maxlen);  // Keep last `maxlen`
    }

    // Pad sequence if it is shorter than maxlen
    if (current.length < maxlen) {
      const padCount = maxlen - current.length;
      const padArray = new Array(padCount).fill(value);

      current = padding === 'post' ?
          [...current, ...padArray] :  // Append pad values to end
          [...padArray, ...current];   // Prepend pad values to start
    }

    return current;
  });

  // Convert the 2D padded JavaScript array into a 2D TensorFlow Tensor
  return tensor2d(padded, [sequences.length, maxlen], dtype);
}
