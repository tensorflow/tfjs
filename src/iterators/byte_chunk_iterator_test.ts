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

import {FileChunkIterator} from './file_chunk_iterator';

const runes = `ᚠᛇᚻ᛫ᛒᛦᚦ᛫ᚠᚱᚩᚠᚢᚱ᛫ᚠᛁᚱᚪ᛫ᚷᛖᚻᚹᛦᛚᚳᚢᛗ
ᛋᚳᛖᚪᛚ᛫ᚦᛖᚪᚻ᛫ᛗᚪᚾᚾᚪ᛫ᚷᛖᚻᚹᛦᛚᚳ᛫ᛗᛁᚳᛚᚢᚾ᛫ᚻᛦᛏ᛫ᛞᚫᛚᚪᚾ
ᚷᛁᚠ᛫ᚻᛖ᛫ᚹᛁᛚᛖ᛫ᚠᚩᚱ᛫ᛞᚱᛁᚻᛏᚾᛖ᛫ᛞᚩᛗᛖᛋ᛫ᚻᛚᛇᛏᚪᚾ᛬`;

const testBlob = new Blob([runes]);

describe('ByteChunkIterator.decodeUTF8()', () => {
  it('Correctly reassembles split characters', done => {
    const byteChunkIterator = new FileChunkIterator(testBlob, {chunkSize: 50});
    const utf8Iterator = byteChunkIterator.decodeUTF8();
    expect(testBlob.size).toEqual(323);

    utf8Iterator.collect()
        .then((result: string[]) => {
          // The test string is 109 characters long; its UTF8 encoding is 323
          // bytes. We read it in chunks of 50 bytes, so there were 7 chunks of
          // bytes. The UTF decoder slightly adjusted the boundaries between the
          // chunks to allow decoding, but did not change the number of chunks,
          // so 7 chunks remain.
          expect(result.length).toEqual(7);
          const totalCharacters =
              result.map(x => x.length).reduce((a, b) => a + b);
          expect(totalCharacters).toEqual(109);
          expect(result.join('')).toEqual(runes);
        })
        .then(done)
        .catch(done.fail);
  });
});
