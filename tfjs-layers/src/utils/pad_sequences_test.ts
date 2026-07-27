/**
 * @license
 * Copyright 2026 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

import * as tfc from '@tensorflow/tfjs-core';

import {padSequences} from './pad_sequences';
import {expectTensorsClose} from './test_utils';

describe('padSequences', () => {
  it('default pre-padding and automatic maxlen calculation', () => {
    const seqs = [[1, 2], [3, 4, 5]];
    const y = padSequences(seqs);

    expect(y.shape).toEqual([2, 3]);
    expectTensorsClose(
        y, tfc.tensor2d([[0, 1, 2], [3, 4, 5]], [2, 3], 'int32'));
  });

  it('custom post-padding', () => {
    const seqs = [[1, 2], [3, 4, 5]];
    const y = padSequences(seqs, {maxlen: 3, padding: 'post'});

    expectTensorsClose(
        y, tfc.tensor2d([[1, 2, 0], [3, 4, 5]], [2, 3], 'int32'));
  });

  it('pre-truncating when sequence length exceeds maxlen', () => {
    const seqs = [[1, 2, 3, 4, 5]];
    const y = padSequences(seqs, {maxlen: 3, truncating: 'pre'});

    expectTensorsClose(y, tfc.tensor2d([[3, 4, 5]], [1, 3], 'int32'));
  });

  it('post-truncating when sequence length exceeds maxlen', () => {
    const seqs = [[1, 2, 3, 4, 5]];
    const y = padSequences(seqs, {maxlen: 3, truncating: 'post'});

    expectTensorsClose(y, tfc.tensor2d([[1, 2, 3]], [1, 3], 'int32'));
  });

  it('custom pad value and float32 dtype', () => {
    const seqs = [[1.5, 2.5]];
    const y = padSequences(seqs, {maxlen: 4, value: -1, dtype: 'float32'});

    expect(y.dtype).toEqual('float32');
    expectTensorsClose(
        y, tfc.tensor2d([[-1, -1, 1.5, 2.5]], [1, 4], 'float32'));
  });

  it('handles empty input sequences', () => {
    const seqs: number[][] = [];
    const y = padSequences(seqs, {maxlen: 5});

    expect(y.shape).toEqual([0, 5]);
  });
});
