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

import {ChainedStream} from './data_stream';
import {streamFromConcatenatedFunction, streamFromItems} from './data_stream';
import {ShuffleStream} from './data_stream';
import {TestIntegerStream} from './data_stream_test';

describe('ShuffleStream', () => {
  // TODO(davidsoergel): Remove this once we figure out the timeout issue.
  let originalTimeout: number;
  beforeAll(() => {
    originalTimeout = jasmine.DEFAULT_TIMEOUT_INTERVAL;
    jasmine.DEFAULT_TIMEOUT_INTERVAL = 20000;
  });
  afterAll(() => {
    jasmine.DEFAULT_TIMEOUT_INTERVAL = originalTimeout;
  });

  it('shuffles a stream without replacement', done => {
    const baseStream = new TestIntegerStream();
    const shuffleStream = new ShuffleStream(baseStream, 1000);
    const notExpectedResult: number[] = [];
    for (let i = 0; i < 1; i++) {
      for (let j = 0; j < 100; j++) {
        notExpectedResult[i * 100 + j] = j;
      }
    }
    shuffleStream.collectRemaining()
        .then(result => {
          expect(result).not.toEqual(notExpectedResult);
          expect(result.length).toEqual(100);
          const counts = new Array<number>(100);
          result.forEach((x) => {
            counts[x] = (counts[x] || 0) + 1;
          });
          for (let i = 0; i < 100; i++) {
            expect(counts[i]).toEqual(1);
          }
        })
        .then(done)
        .catch(done.fail);
  });

  it('shuffles a single chained stream without replacement', done => {
    const baseStreamPromise =
        ChainedStream.create(streamFromItems([new TestIntegerStream()]));
    const shuffleStreamPromise = baseStreamPromise.then(
        baseStream => new ShuffleStream(baseStream, 1000));
    const notExpectedResult: number[] = [];
    for (let i = 0; i < 1; i++) {
      for (let j = 0; j < 100; j++) {
        notExpectedResult[i * 100 + j] = j;
      }
    }
    shuffleStreamPromise
        .then(shuffleStream => shuffleStream.collectRemaining().then(result => {
          expect(result).not.toEqual(notExpectedResult);
          expect(result.length).toEqual(100);
          const counts = new Array<number>(100);
          result.forEach((x) => {
            counts[x] = (counts[x] || 0) + 1;
          });
          for (let i = 0; i < 100; i++) {
            expect(counts[i]).toEqual(1);
          }
        }))
        .then(done)
        .catch(done.fail);
  });

  it('shuffles multiple chained streams without replacement', done => {
    const baseStreamPromise =
        streamFromConcatenatedFunction(() => new TestIntegerStream(), 6);
    const shuffleStreamPromise = baseStreamPromise.then(
        baseStream => new ShuffleStream(baseStream, 1000));
    const notExpectedResult: number[] = [];
    for (let i = 0; i < 6; i++) {
      for (let j = 0; j < 100; j++) {
        notExpectedResult[i * 100 + j] = j;
      }
    }
    shuffleStreamPromise
        .then(shuffleStream => shuffleStream.collectRemaining().then(result => {
          expect(result).not.toEqual(notExpectedResult);
          expect(result.length).toEqual(600);
          const counts = new Array<number>(100);
          result.forEach((x) => {
            counts[x] = (counts[x] || 0) + 1;
          });
          for (let i = 0; i < 100; i++) {
            expect(counts[i]).toEqual(6);
          }
        }))
        .then(done)
        .catch(done.fail);
  });
});
