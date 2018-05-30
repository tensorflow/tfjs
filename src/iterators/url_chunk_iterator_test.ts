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

import * as fetchMock from 'fetch-mock';

import {URLChunkIterator} from './url_chunk_iterator';

const testString = 'abcdefghijklmnopqrstuvwxyz';

const url = 'mock_url';
fetchMock.get('*', testString);

describe('URLChunkIterator', () => {
  it('Reads the entire file and then closes the stream', done => {
    const readIterator = new URLChunkIterator(url, {chunkSize: 10});
    readIterator.collectRemaining()
        .then(result => {
          expect(result.length).toEqual(3);
          const totalBytes = result.map(x => x.length).reduce((a, b) => a + b);
          expect(totalBytes).toEqual(26);
        })
        .then(done)
        .catch(done.fail);
  });

  it('Reads chunks in order', done => {
    const readIterator = new URLChunkIterator(url, {chunkSize: 10});
    readIterator.collectRemaining()
        .then(result => {
          expect(result[0][0]).toEqual('a'.charCodeAt(0));
          expect(result[1][0]).toEqual('k'.charCodeAt(0));
          expect(result[2][0]).toEqual('u'.charCodeAt(0));
        })
        .then(done)
        .catch(done.fail);
  });

  it('Reads chunks of expected sizes', done => {
    const readIterator = new URLChunkIterator(url, {chunkSize: 10});
    readIterator.collectRemaining()
        .then(result => {
          expect(result[0].length).toEqual(10);
          expect(result[1].length).toEqual(10);
          expect(result[2].length).toEqual(6);
        })
        .then(done)
        .catch(done.fail);
  });
});

fetchMock.reset();
