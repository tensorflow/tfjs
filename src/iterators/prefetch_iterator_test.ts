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

import {iteratorFromConcatenatedFunction} from './lazy_iterator';
import {PrefetchIterator} from './lazy_iterator';
import {TestIntegerIterator} from './lazy_iterator_test';

describe('PrefetchIterator', () => {
  it('fetches a stream completely (stream size < buffer size)', done => {
    const prefetchIterator =
        new PrefetchIterator(new TestIntegerIterator(), 500);
    const expectedResult: number[] = [];
    for (let j = 0; j < 100; j++) {
      expectedResult[j] = j;
    }

    prefetchIterator.collect()
        .then(result => {
          expect(result).toEqual(expectedResult);
        })
        .then(done)
        .catch(done.fail);
  });

  it('fetches a chained stream completely (stream size < buffer size)',
     async done => {
       const baseIterator = iteratorFromConcatenatedFunction(
           () => ({value: new TestIntegerIterator(), done: false}), 3);

       const prefetchIterator = new PrefetchIterator(baseIterator, 500);

       const expectedResult: number[] = [];
       for (let i = 0; i < 3; i++) {
         for (let j = 0; j < 100; j++) {
           expectedResult[i * 100 + j] = j;
         }
       }

       prefetchIterator.collect()
           .then(result => {
             expect(result).toEqual(expectedResult);
           })
           .then(done)
           .catch(done.fail);
     });

  it('fetches a chained stream completely (stream size > buffer size)',
     done => {
       const baseIterator = iteratorFromConcatenatedFunction(
           () => ({value: new TestIntegerIterator(), done: false}), 3);

       const prefetchIterator = new PrefetchIterator(baseIterator, 122);
       const expectedResult: number[] = [];
       for (let i = 0; i < 3; i++) {
         for (let j = 0; j < 100; j++) {
           expectedResult[i * 100 + j] = j;
         }
       }

       prefetchIterator.collect()
           .then(result => {
             expect(result).toEqual(expectedResult);
           })
           .then(done)
           .catch(done.fail);
     });
});
