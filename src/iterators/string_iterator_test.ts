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

import {ENV} from '@tensorflow/tfjs-core';
import {FileChunkIterator} from './file_chunk_iterator';

const lorem = `Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do
eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim
veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo
consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum
dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident,
sunt in culpa qui officia deserunt mollit anim id est laborum.`;

const testData = ENV.get('IS_BROWSER') ? new Blob([lorem]) : Buffer.from(lorem);

describe('StringIterator.split()', () => {
  it('Correctly splits lines', async () => {
    const byteIterator = new FileChunkIterator(testData, {chunkSize: 50});
    const utf8Iterator = byteIterator.decodeUTF8();
    const lineIterator = utf8Iterator.split('\n');
    const expected = lorem.split('\n');

    const result = await lineIterator.collect();
    expect(result.length).toEqual(6);
    const totalCharacters = result.map(x => x.length).reduce((a, b) => a + b);
    expect(totalCharacters).toEqual(440);
    expect(result).toEqual(expected);
    expect(result.join('\n')).toEqual(lorem);
  });

  it('Correctly splits strings even when separators fall on chunk boundaries',
     async () => {
       const byteIterator = new FileChunkIterator(
           ENV.get('IS_BROWSER') ? new Blob(['ab def hi      pq']) :
                                   Buffer.from('ab def hi      pq'),
           {chunkSize: 3});
       // Note the initial chunking will be
       //   ['ab ', 'def', ' hi', '   ', '   ', 'pq],
       // so here we are testing for correct behavior when
       //   * a separator is the last character in a chunk (the first chunk),
       //   * it is the first character (the third chunk), and
       //   * when the entire chunk consists of separators (fourth and fifth).
       const utf8Iterator = byteIterator.decodeUTF8();
       const lineIterator = utf8Iterator.split(' ');
       const expected = ['ab', 'def', 'hi', '', '', '', '', '', 'pq'];

       const result = await lineIterator.collect();
       expect(result.length).toEqual(9);
       const totalCharacters =
           result.map(x => x.length).reduce((a, b) => a + b);
       expect(totalCharacters).toEqual(9);
       expect(result).toEqual(expected);
       expect(result.join(' ')).toEqual('ab def hi      pq');
     });
});
