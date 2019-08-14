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
 *
 * =============================================================================
 */

import {ChainedIterator, iteratorFromItems} from './lazy_iterator';
import {iteratorFromConcatenatedFunction} from './lazy_iterator';
import {ShuffleIterator} from './lazy_iterator';
import {TestIntegerIterator} from './lazy_iterator_test';

const LONG_STREAM_LENGTH = 100;
const SHORT_STREAM_LENGTH = 15;

describe('ShuffleIterator', () => {
  it('shuffles a stream without replacement', async () => {
    const baseIterator = new TestIntegerIterator(LONG_STREAM_LENGTH);
    const shuffleIterator = new ShuffleIterator(baseIterator, 1000);
    const notExpectedResult: number[] = [];
    for (let i = 0; i < 1; i++) {
      for (let j = 0; j < LONG_STREAM_LENGTH; j++) {
        notExpectedResult[i * LONG_STREAM_LENGTH + j] = j;
      }
    }
    const result = await shuffleIterator.toArrayForTest();
    expect(result).not.toEqual(notExpectedResult);
    expect(result.length).toEqual(LONG_STREAM_LENGTH);
    const counts = new Array<number>(LONG_STREAM_LENGTH);
    result.forEach((x) => {
      counts[x] = (counts[x] || 0) + 1;
    });
    for (let i = 0; i < LONG_STREAM_LENGTH; i++) {
      expect(counts[i]).toEqual(1);
    }
  });

  it('shuffles a single chained stream without replacement', async () => {
    const baseIterator = new ChainedIterator(
        iteratorFromItems([new TestIntegerIterator(SHORT_STREAM_LENGTH)]));
    const shuffleIterator = new ShuffleIterator(baseIterator, 1000);
    const notExpectedResult: number[] = [];
    for (let i = 0; i < 1; i++) {
      for (let j = 0; j < SHORT_STREAM_LENGTH; j++) {
        notExpectedResult[i * SHORT_STREAM_LENGTH + j] = j;
      }
    }
    const result = await shuffleIterator.toArrayForTest();
    expect(result).not.toEqual(notExpectedResult);
    expect(result.length).toEqual(SHORT_STREAM_LENGTH);
    const counts = new Array<number>(SHORT_STREAM_LENGTH);
    result.forEach((x) => {
      counts[x] = (counts[x] || 0) + 1;
    });
    for (let i = 0; i < SHORT_STREAM_LENGTH; i++) {
      expect(counts[i]).toEqual(1);
    }
  });

  it('shuffles multiple chained streams without replacement', async () => {
    const baseIterator = iteratorFromConcatenatedFunction(
        () => (
            {value: new TestIntegerIterator(SHORT_STREAM_LENGTH), done: false}),
        3);
    const shuffleIterator = new ShuffleIterator(baseIterator, 1000);
    const notExpectedResult: number[] = [];
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < SHORT_STREAM_LENGTH; j++) {
        notExpectedResult[i * SHORT_STREAM_LENGTH + j] = j;
      }
    }
    const result = await shuffleIterator.toArrayForTest();
    expect(result).not.toEqual(notExpectedResult);
    expect(result.length).toEqual(3 * SHORT_STREAM_LENGTH);
    const counts = new Array<number>(SHORT_STREAM_LENGTH);
    result.forEach((x) => {
      counts[x] = (counts[x] || 0) + 1;
    });
    for (let i = 0; i < SHORT_STREAM_LENGTH; i++) {
      expect(counts[i]).toEqual(3);
    }
  });
});
