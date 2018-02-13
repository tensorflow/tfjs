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

import {streamFromConcatenatedFunction} from './data_stream';
import {PrefetchStream} from './data_stream';
import {TestIntegerStream} from './data_stream_test';

describe('PrefetchStream', () => {
  // TODO(davidsoergel): Remove this once we figure out the timeout issue.
  let originalTimeout: number;
  beforeAll(() => {
    originalTimeout = jasmine.DEFAULT_TIMEOUT_INTERVAL;
    jasmine.DEFAULT_TIMEOUT_INTERVAL = 20000;
  });
  afterAll(() => {
    jasmine.DEFAULT_TIMEOUT_INTERVAL = originalTimeout;
  });

  it('fetches a stream completely (stream size < buffer size)', done => {
    const prefetchStream = new PrefetchStream(new TestIntegerStream(), 500);
    const expectedResult: number[] = [];
    for (let j = 0; j < 100; j++) {
      expectedResult[j] = j;
    }

    prefetchStream.collectRemaining()
        .then(result => {
          expect(result).toEqual(expectedResult);
        })
        .then(done)
        .catch(done.fail);
  });

  it('fetches a chained stream completely (stream size < buffer size)',
     done => {
       const baseStreamPromise = streamFromConcatenatedFunction(() => {
         return new TestIntegerStream();
       }, 7);

       const prefetchStreamPromise = baseStreamPromise.then(
           baseStream => new PrefetchStream(baseStream, 1000));
       const expectedResult: number[] = [];
       for (let i = 0; i < 7; i++) {
         for (let j = 0; j < 100; j++) {
           expectedResult[i * 100 + j] = j;
         }
       }

       prefetchStreamPromise
           .then(
               prefetchStream =>
                   prefetchStream.collectRemaining().then(result => {
                     expect(result).toEqual(expectedResult);
                   }))
           .then(done)
           .catch(done.fail);
     });

  it('fetches a chained stream completely (stream size > buffer size)',
     done => {
       const baseStreamPromise = streamFromConcatenatedFunction(() => {
         return new TestIntegerStream();
       }, 7);

       const prefetchStreamPromise = baseStreamPromise.then(
           baseStream => new PrefetchStream(baseStream, 500));
       const expectedResult: number[] = [];
       for (let i = 0; i < 7; i++) {
         for (let j = 0; j < 100; j++) {
           expectedResult[i * 100 + j] = j;
         }
       }

       prefetchStreamPromise
           .then(
               prefetchStream =>
                   prefetchStream.collectRemaining().then(result => {
                     expect(result).toEqual(expectedResult);
                   }))
           .then(done)
           .catch(done.fail);
     });
});
