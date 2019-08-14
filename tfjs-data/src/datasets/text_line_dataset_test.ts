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
import {FileDataSource} from '../sources/file_data_source';
import {TextLineDataset} from './text_line_dataset';

const runes = `ᚠᛇᚻ᛫ᛒᛦᚦ᛫ᚠᚱᚩᚠᚢᚱ᛫ᚠᛁᚱᚪ᛫ᚷᛖᚻᚹᛦᛚᚳᚢᛗ
ᛋᚳᛖᚪᛚ᛫ᚦᛖᚪᚻ᛫ᛗᚪᚾᚾᚪ᛫ᚷᛖᚻᚹᛦᛚᚳ᛫ᛗᛁᚳᛚᚢᚾ᛫ᚻᛦᛏ᛫ᛞᚫᛚᚪᚾ
ᚷᛁᚠ᛫ᚻᛖ᛫ᚹᛁᛚᛖ᛫ᚠᚩᚱ᛫ᛞᚱᛁᚻᛏᚾᛖ᛫ᛞᚩᛗᛖᛋ᛫ᚻᛚᛇᛏᚪᚾ᛬`;

const textWithDOSLineBreaks = 'abc\rdefg\r\nhijklmn\r\nopqrst';

const testBlob = ENV.get('IS_BROWSER') ? new Blob([runes]) : Buffer.from(runes);

const textBlobWithDOSLineBreaks = ENV.get('IS_BROWSER') ?
    new Blob([textWithDOSLineBreaks]) :
    Buffer.from(textWithDOSLineBreaks);

describe('TextLineDataset', () => {
  it('Produces a stream of strings containing UTF8-decoded text lines',
     async () => {
       const source = new FileDataSource(testBlob, {chunkSize: 10});
       const dataset = new TextLineDataset(source);
       const iter = await dataset.iterator();
       const result = await iter.toArrayForTest();

       expect(result).toEqual([
         'ᚠᛇᚻ᛫ᛒᛦᚦ᛫ᚠᚱᚩᚠᚢᚱ᛫ᚠᛁᚱᚪ᛫ᚷᛖᚻᚹᛦᛚᚳᚢᛗ',
         'ᛋᚳᛖᚪᛚ᛫ᚦᛖᚪᚻ᛫ᛗᚪᚾᚾᚪ᛫ᚷᛖᚻᚹᛦᛚᚳ᛫ᛗᛁᚳᛚᚢᚾ᛫ᚻᛦᛏ᛫ᛞᚫᛚᚪᚾ',
         'ᚷᛁᚠ᛫ᚻᛖ᛫ᚹᛁᛚᛖ᛫ᚠᚩᚱ᛫ᛞᚱᛁᚻᛏᚾᛖ᛫ᛞᚩᛗᛖᛋ᛫ᚻᛚᛇᛏᚪᚾ᛬',
       ]);
     });

  it('Parses lines from windows/DOS text correctly', async () => {
    const source =
        new FileDataSource(textBlobWithDOSLineBreaks, {chunkSize: 10});
    const dataset = new TextLineDataset(source);
    const iter = await dataset.iterator();
    const result = await iter.toArrayForTest();

    // \r is retained when not followed by \n
    expect(result[0]).toEqual('abc\rdefg');
    expect(result[1]).toEqual('hijklmn');
    expect(result[2]).toEqual('opqrst');
  });
});
