/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
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

import * as tf from '../../index';
import {ALL_ENVS, describeWithFlags} from '../../jasmine_util';
import {expectArraysEqual} from '../../test_util';

async function expectResult(
    result: tf.NamedTensorMap, nGrams: string[], nGramsSplits: number[]) {
  expectArraysEqual(await result.nGrams.data(), nGrams);
  expectArraysEqual(await result.nGramsSplits.data(), nGramsSplits);

  expect(result.nGrams.shape).toEqual([nGrams.length]);
  expect(result.nGramsSplits.shape).toEqual([nGramsSplits.length]);

  expect(result.nGrams.dtype).toEqual('string');
  expect(result.nGramsSplits.dtype).toEqual('int32');
}

describeWithFlags('stringNGrams', ALL_ENVS, () => {
  it('padded trigrams', async () => {
    // Batch items are:
    // 0: "a", "b", "c", "d"
    // 1: "e", "f"
    const result = tf.string.stringNGrams(
        ['a', 'b', 'c', 'd', 'e', 'f'], tf.tensor1d([0, 4, 6], 'int32'), '|',
        [3], 'LP', 'RP', -1, false);
    const nGrams = [
      'LP|LP|a', 'LP|a|b', 'a|b|c', 'b|c|d', 'c|d|RP', 'd|RP|RP',  // 0
      'LP|LP|e', 'LP|e|f', 'e|f|RP', 'f|RP|RP'                     // 1
    ];
    const nGramsSplits = [0, 6, 10];
    await expectResult(result, nGrams, nGramsSplits);
  });

  it('padded bigrams and trigrams', async () => {
    // Batch items are:
    // 0: "a", "b", "c", "d"
    // 1: "e", "f"
    const result = tf.string.stringNGrams(
        ['a', 'b', 'c', 'd', 'e', 'f'], tf.tensor1d([0, 4, 6], 'int32'), '|',
        [2, 3], 'LP', 'RP', -1, false);
    const nGrams = [
      'LP|a', 'a|b', 'b|c', 'c|d', 'd|RP', 'LP|LP|a', 'LP|a|b', 'a|b|c',
      'b|c|d', 'c|d|RP', 'd|RP|RP',                                    // 0
      'LP|e', 'e|f', 'f|RP', 'LP|LP|e', 'LP|e|f', 'e|f|RP', 'f|RP|RP'  // 1
    ];
    const nGramsSplits = [0, 11, 18];
    await expectResult(result, nGrams, nGramsSplits);
  });

  it('padded bigrams', async () => {
    // Batch items are:
    // 0: "a", "b", "c", "d"
    // 1: "e", "f"
    const result = tf.string.stringNGrams(
        ['a', 'b', 'c', 'd', 'e', 'f'], tf.tensor1d([0, 4, 6], 'int32'), '|',
        [2], 'LP', 'RP', -1, false);
    const nGrams = [
      'LP|a', 'a|b', 'b|c', 'c|d', 'd|RP',  // 0
      'LP|e', 'e|f', 'f|RP'                 // 1
    ];
    const nGramsSplits = [0, 5, 8];
    await expectResult(result, nGrams, nGramsSplits);
  });

  it('padding is at most nGramSize - 1', async () => {
    // Batch items are:
    // 0: "a", "b", "c", "d"
    // 1: "e", "f"
    const result = tf.string.stringNGrams(
        ['a', 'b', 'c', 'd', 'e', 'f'], tf.tensor1d([0, 4, 6], 'int32'), '|',
        [2], 'LP', 'RP', 4, false);
    const nGrams = [
      'LP|a', 'a|b', 'b|c', 'c|d', 'd|RP',  // 0
      'LP|e', 'e|f', 'f|RP'                 // 1
    ];
    const nGramsSplits = [0, 5, 8];
    await expectResult(result, nGrams, nGramsSplits);
  });

  it('padded unigram and bigrams', async () => {
    // Batch items are:
    // 0: "a", "b", "c", "d"
    // 1: "e", "f"
    const result = tf.string.stringNGrams(
        ['a', 'b', 'c', 'd', 'e', 'f'], tf.tensor1d([0, 4, 6], 'int32'), '|',
        [1, 2], 'LP', 'RP', -1, false);
    const nGrams = [
      'a', 'b', 'c', 'd', 'LP|a', 'a|b', 'b|c', 'c|d', 'd|RP',  // 0
      'e', 'f', 'LP|e', 'e|f', 'f|RP'                           // 1
    ];
    const nGramsSplits = [0, 9, 14];
    await expectResult(result, nGrams, nGramsSplits);
  });

  it('overlapping padded nGrams', async () => {
    // This test validates that n-grams with both left and right padding in a
    // single ngram token are created correctly.

    // Batch items are:
    // 0: "a"
    // 1: "b", "c", "d"
    // 2: "e", "f"
    const result = tf.string.stringNGrams(
        ['a', 'b', 'c', 'd', 'e', 'f'], tf.tensor1d([0, 1, 4, 6], 'int32'), '|',
        [3], 'LP', 'RP', -1, false);
    const nGrams = [
      'LP|LP|a', 'LP|a|RP', 'a|RP|RP',                    // 0
      'LP|LP|b', 'LP|b|c', 'b|c|d', 'c|d|RP', 'd|RP|RP',  // 1
      'LP|LP|e', 'LP|e|f', 'e|f|RP', 'f|RP|RP'            // 2
    ];
    const nGramsSplits = [0, 3, 8, 12];
    await expectResult(result, nGrams, nGramsSplits);
  });

  it('overlapping padded multi char nGrams', async () => {
    // Batch items are:
    // 0: "a"
    // 1: "b", "c", "d"
    // 2: "e", "f"
    const result = tf.string.stringNGrams(
        ['aa', 'bb', 'cc', 'dd', 'ee', 'ff'],
        tf.tensor1d([0, 1, 4, 6], 'int32'), '|', [3], 'LP', 'RP', -1, false);
    const nGrams = [
      'LP|LP|aa', 'LP|aa|RP', 'aa|RP|RP',                          // 0
      'LP|LP|bb', 'LP|bb|cc', 'bb|cc|dd', 'cc|dd|RP', 'dd|RP|RP',  // 1
      'LP|LP|ee', 'LP|ee|ff', 'ee|ff|RP', 'ff|RP|RP'               // 2
    ];
    const nGramsSplits = [0, 3, 8, 12];
    await expectResult(result, nGrams, nGramsSplits);
  });

  it('multi overlapping padded nGrams', async () => {
    // This test validates that n-grams with more than 1 padding value on each
    // side are created correctly.

    // Batch items are:
    // 0: "a"
    const result = tf.string.stringNGrams(
        ['a'], tf.tensor1d([0, 1], 'int32'), '|', [5], 'LP', 'RP', -1, false);
    const nGrams = [
      'LP|LP|LP|LP|a', 'LP|LP|LP|a|RP', 'LP|LP|a|RP|RP', 'LP|a|RP|RP|RP',
      'a|RP|RP|RP|RP'
    ];
    const nGramsSplits = [0, 5];
    await expectResult(result, nGrams, nGramsSplits);
  });

  it('unpadded trigrams', async () => {
    // Batch items are:
    // 0: "a", "b", "c", "d"
    // 1: "e", "f"
    const result = tf.string.stringNGrams(
        ['a', 'b', 'c', 'd', 'e', 'f'], tf.tensor1d([0, 4, 6], 'int32'), '|',
        [3], '', '', 0, false);
    const nGrams = ['a|b|c', 'b|c|d'];
    const nGramsSplits = [0, 2, 2];
    await expectResult(result, nGrams, nGramsSplits);
  });

  it('unpadded trigrams with empty sequence', async () => {
    // Batch items are:
    // 0: "a", "b", "c", "d"
    // 1: "e", "f"
    const result = tf.string.stringNGrams(
        ['a', 'b', 'c', 'd', 'e', 'f'], tf.tensor1d([0, 4, 4, 6], 'int32'), '|',
        [3], '', '', 0, false);
    const nGrams = ['a|b|c', 'b|c|d'];
    const nGramsSplits = [0, 2, 2, 2];
    await expectResult(result, nGrams, nGramsSplits);
  });

  it('unpadded trigrams with preserve short', async () => {
    // Batch items are:
    // 0: "a", "b", "c", "d"
    // 1: "e", "f"
    const result = tf.string.stringNGrams(
        ['a', 'b', 'c', 'd', 'e', 'f'], tf.tensor1d([0, 4, 6], 'int32'), '|',
        [3], '', '', 0, true);
    const nGrams = ['a|b|c', 'b|c|d', 'e|f'];
    const nGramsSplits = [0, 2, 3];
    await expectResult(result, nGrams, nGramsSplits);
  });

  it('unpadded trigrams with preserve short and empty sequence', async () => {
    // Batch items are:
    // 0: "a", "b", "c", "d"
    // 1: "e", "f"
    const result = tf.string.stringNGrams(
        ['a', 'b', 'c', 'd', 'e', 'f'], tf.tensor1d([0, 4, 4, 6], 'int32'), '|',
        [3], '', '', 0, true);
    const nGrams = ['a|b|c', 'b|c|d', 'e|f'];
    const nGramsSplits = [0, 2, 2, 3];
    await expectResult(result, nGrams, nGramsSplits);
  });

  it('unpadded trigrams and quad grams with preserve short', async () => {
    // Batch items are:
    // 0: "a", "b", "c", "d"
    // 1: "e", "f"
    const result = tf.string.stringNGrams(
        ['a', 'b', 'c', 'd', 'e', 'f'], tf.tensor1d([0, 4, 6], 'int32'), '|',
        [4, 3], '', '', 0, true);
    const nGrams = ['a|b|c|d', 'a|b|c', 'b|c|d', 'e|f'];
    const nGramsSplits = [0, 3, 4];
    await expectResult(result, nGrams, nGramsSplits);
  });

  it('unpadded bigrams and trigrams', async () => {
    // Batch items are:
    // 0: "a", "b", "c", "d"
    // 1: "e", "f"
    const result = tf.string.stringNGrams(
        ['a', 'b', 'c', 'd', 'e', 'f'], tf.tensor1d([0, 4, 6], 'int32'), '|',
        [2, 3], '', '', 0, false);
    const nGrams = ['a|b', 'b|c', 'c|d', 'a|b|c', 'b|c|d', 'e|f'];
    const nGramsSplits = [0, 5, 6];
    await expectResult(result, nGrams, nGramsSplits);
  });

  it('unpadded bigrams and trigrams with preserve short', async () => {
    // Batch items are:
    // 0: "a", "b", "c", "d"
    // 1: "e", "f"
    const result = tf.string.stringNGrams(
        ['a', 'b', 'c', 'd', 'e', 'f'], tf.tensor1d([0, 4, 6], 'int32'), '|',
        [2, 3], '', '', 0, true);
    // Note that in this case, because the bigram 'e|f' was already generated,
    // the op will not generate a special preserveShort bigram.
    const nGrams = ['a|b', 'b|c', 'c|d', 'a|b|c', 'b|c|d', 'e|f'];
    const nGramsSplits = [0, 5, 6];
    await expectResult(result, nGrams, nGramsSplits);
  });

  it('unpadded trigrams and bigrams with preserve short', async () => {
    // Batch items are:
    // 0: "a", "b", "c", "d"
    // 1: "e", "f"
    const result = tf.string.stringNGrams(
        ['a', 'b', 'c', 'd', 'e', 'f'], tf.tensor1d([0, 4, 6], 'int32'), '|',
        [3, 2], '', '', 0, true);
    // Note that in this case, because the bigram 'e|f' was already generated,
    // the op will not generate a special preserveShort bigram.
    const nGrams = ['a|b|c', 'b|c|d', 'a|b', 'b|c', 'c|d', 'e|f'];
    const nGramsSplits = [0, 5, 6];
    await expectResult(result, nGrams, nGramsSplits);
  });

  it('unpadded bigrams', async () => {
    // Batch items are:
    // 0: "a", "b", "c", "d"
    // 1: "e", "f"
    const result = tf.string.stringNGrams(
        ['a', 'b', 'c', 'd', 'e', 'f'], tf.tensor1d([0, 4, 6], 'int32'), '|',
        [2], '', '', 0, false);
    const nGrams = ['a|b', 'b|c', 'c|d', 'e|f'];
    const nGramsSplits = [0, 3, 4];
    await expectResult(result, nGrams, nGramsSplits);
  });

  it('overlapping unpadded nGrams', async () => {
    // Batch items are:
    // 0: "a"
    // 1: "b", "c", "d"
    // 2: "e", "f"
    const result = tf.string.stringNGrams(
        ['a', 'b', 'c', 'd', 'e', 'f'], tf.tensor1d([0, 1, 4, 6], 'int32'), '|',
        [3], '', '', 0, false);
    const nGrams = ['b|c|d'];
    const nGramsSplits = [0, 0, 1, 1];
    await expectResult(result, nGrams, nGramsSplits);
  });

  it('overlapping unpadded nGrams no output', async () => {
    // Batch items are:
    // 0: "a"
    // 1: "b", "c", "d"
    // 2: "e", "f"
    const result = tf.string.stringNGrams(
        ['a', 'b', 'c', 'd', 'e', 'f'], tf.tensor1d([0, 1, 4, 6], 'int32'), '|',
        [5], '', '', 0, false);
    const nGrams: string[] = [];
    const nGramsSplits = [0, 0, 0, 0];
    await expectResult(result, nGrams, nGramsSplits);
  });

  it('singly padded trigrams', async () => {
    // Batch items are:
    // 0: "a", "b", "c", "d"
    // 1: "e", "f"
    const result = tf.string.stringNGrams(
        ['a', 'b', 'c', 'd', 'e', 'f'], tf.tensor1d([0, 4, 6], 'int32'), '|',
        [3], 'LP', 'RP', 1, false);
    const nGrams = [
      'LP|a|b', 'a|b|c', 'b|c|d', 'c|d|RP',  // 0
      'LP|e|f', 'e|f|RP'
    ];
    const nGramsSplits = [0, 4, 6];
    await expectResult(result, nGrams, nGramsSplits);
  });

  it('singly padded bigrams', async () => {
    // Batch items are:
    // 0: "a", "b", "c", "d"
    // 1: "e", "f"
    const result = tf.string.stringNGrams(
        ['a', 'b', 'c', 'd', 'e', 'f'], tf.tensor1d([0, 4, 6], 'int32'), '|',
        [2], 'LP', 'RP', 1, false);
    const nGrams = [
      'LP|a', 'a|b', 'b|c', 'c|d', 'd|RP',  // 0
      'LP|e', 'e|f', 'f|RP'
    ];
    const nGramsSplits = [0, 5, 8];
    await expectResult(result, nGrams, nGramsSplits);
  });

  it('singly padded bigrams and 5grams', async () => {
    // Batch items are:
    // 0: "a", "b", "c", "d"
    // 1: "e", "f"
    const result = tf.string.stringNGrams(
        ['a', 'b', 'c', 'd', 'e', 'f'], tf.tensor1d([0, 4, 6], 'int32'), '|',
        [2, 5], 'LP', 'RP', 1, false);
    const nGrams = [
      'LP|a', 'a|b', 'b|c', 'c|d', 'd|RP', 'LP|a|b|c|d', 'a|b|c|d|RP',  // 0
      'LP|e', 'e|f', 'f|RP'
    ];
    const nGramsSplits = [0, 7, 10];
    await expectResult(result, nGrams, nGramsSplits);
  });

  it('singly padded 5grams with preserve short', async () => {
    // Batch items are:
    // 0: "a", "b", "c", "d"
    // 1: "e", "f"
    const result = tf.string.stringNGrams(
        ['a', 'b', 'c', 'd', 'e', 'f'], tf.tensor1d([0, 4, 6], 'int32'), '|',
        [5], 'LP', 'RP', 1, true);
    const nGrams = [
      'LP|a|b|c|d', 'a|b|c|d|RP',  // 0
      'LP|e|f|RP'
    ];
    const nGramsSplits = [0, 2, 3];
    await expectResult(result, nGrams, nGramsSplits);
  });

  it('overlapping singly padded nGrams', async () => {
    // Batch items are:
    // 0: "a"
    // 1: "b", "c", "d"
    // 2: "e", "f"
    const result = tf.string.stringNGrams(
        ['a', 'b', 'c', 'd', 'e', 'f'], tf.tensor1d([0, 1, 4, 6], 'int32'), '|',
        [3], 'LP', 'RP', 1, false);
    const nGrams = [
      'LP|a|RP',                    // 0
      'LP|b|c', 'b|c|d', 'c|d|RP',  // 1
      'LP|e|f', 'e|f|RP'
    ];
    const nGramsSplits = [0, 1, 4, 6];
    await expectResult(result, nGrams, nGramsSplits);
  });

  it('overlapping singly padded nGrams no output', async () => {
    // Batch items are:
    // 0: "a"
    // 1: "b", "c", "d"
    // 2: "e", "f"
    const result = tf.string.stringNGrams(
        ['a', 'b', 'c', 'd', 'e', 'f'], tf.tensor1d([0, 1, 4, 6], 'int32'), '|',
        [5], 'LP', 'RP', 1, false);
    const nGrams = ['LP|b|c|d|RP'];
    const nGramsSplits = [0, 0, 1, 1];
    await expectResult(result, nGrams, nGramsSplits);
  });

  it('singly padded unigrams', async () => {
    // Batch items are:
    // 0: "a", "b", "c", "d"
    // 1: "e", "f"
    const result = tf.string.stringNGrams(
        ['a', 'b', 'c', 'd', 'e', 'f'], tf.tensor1d([0, 4, 6], 'int32'), '|',
        [1], 'LP', 'RP', 1, false);
    const nGrams = ['a', 'b', 'c', 'd', 'e', 'f'];
    const nGramsSplits = [0, 4, 6];
    await expectResult(result, nGrams, nGramsSplits);
  });

  it('empty input', async () => {
    const result = tf.string.stringNGrams(
        tf.tensor1d([], 'string'), tf.tensor1d([], 'int32'), '|', [1], 'LP',
        'RP', 3, false);
    const nGrams: string[] = [];
    const nGramsSplits: number[] = [];
    await expectResult(result, nGrams, nGramsSplits);
  });

  it('no tokens', async () => {
    // Batch items are:
    // 0:
    // 1: "a"
    const result = tf.string.stringNGrams(
        ['a'], tf.tensor1d([0, 0, 1], 'int32'), '|', [3], 'L', 'R', -1, false);
    const nGrams = [
      'L|L|R', 'L|R|R',          // no input in first split
      'L|L|a', 'L|a|R', 'a|R|R'  // second split
    ];
    const nGramsSplits = [0, 2, 5];
    await expectResult(result, nGrams, nGramsSplits);
  });

  it('no tokens no pad', async () => {
    // Batch items are:
    // 0:
    // 1: "a"
    const result = tf.string.stringNGrams(
        ['a'], tf.tensor1d([0, 0, 1], 'int32'), '|', [3], '', '', 0, false);
    const nGrams: string[] = [];
    const nGramsSplits = [0, 0, 0];
    await expectResult(result, nGrams, nGramsSplits);
  });

  it('throw error if first partition index is not 0', async () => {
    expect(
        () => tf.string.stringNGrams(
            ['a'], tf.tensor1d([1, 1, 1], 'int32'), '|', [3], '', '', 0, false))
        .toThrowError(/First split value must be 0/);
  });

  it('throw error if partition indices are decreasing', async () => {
    expect(
        () => tf.string.stringNGrams(
            ['a'], tf.tensor1d([0, 1, 0], 'int32'), '|', [3], '', '', 0, false))
        .toThrowError(/must be in \[1, 1\]/);
  });

  it('throw error if partition index is >= input size', async () => {
    expect(
        () => tf.string.stringNGrams(
            ['a'], tf.tensor1d([0, 2, 1], 'int32'), '|', [3], '', '', 0, false))
        .toThrowError(/must be in \[0, 1\]/);
  });

  it('throw error if last partition index is !== input size', async () => {
    expect(
        () => tf.string.stringNGrams(
            ['a'], tf.tensor1d([0, 0], 'int32'), '|', [3], '', '', 0, false))
        .toThrowError(/Last split value must be data size/);
  });
});
