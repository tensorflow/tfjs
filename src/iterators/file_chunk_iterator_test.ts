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

const range = (start: number, end: number) => {
  return Array.from({length: (end - start)}, (v, k) => k + start);
};

const testData = ENV.get('IS_BROWSER') ?
    new Blob([new Uint8Array(range(0, 55))]) :
    new Uint8Array(range(0, 55));

describe('FileChunkIterator', () => {
  it('Reads the entire file and then closes the stream', async () => {
    const readIterator = new FileChunkIterator(testData, {chunkSize: 10});
    const result = await readIterator.toArrayForTest();
    expect(result.length).toEqual(6);
    const totalBytes = result.map(x => x.length).reduce((a, b) => a + b);
    expect(totalBytes).toEqual(55);
  });

  it('Reads chunks in order', async () => {
    const readIterator = new FileChunkIterator(testData, {chunkSize: 10});
    const result = await readIterator.toArrayForTest();
    expect(result[0][0]).toEqual(0);
    expect(result[1][0]).toEqual(10);
    expect(result[2][0]).toEqual(20);
    expect(result[3][0]).toEqual(30);
    expect(result[4][0]).toEqual(40);
    expect(result[5][0]).toEqual(50);
  });

  it('Reads chunks of expected sizes', async () => {
    const readIterator = new FileChunkIterator(testData, {chunkSize: 10});
    const result = await readIterator.toArrayForTest();
    expect(result[0].length).toEqual(10);
    expect(result[1].length).toEqual(10);
    expect(result[2].length).toEqual(10);
    expect(result[3].length).toEqual(10);
    expect(result[4].length).toEqual(10);
    expect(result[5].length).toEqual(5);
  });

  it('Provides mistyped data throws error', () => {
    expect(() => {
      // tslint:disable-next-line:no-unused-expression
      new FileChunkIterator(null);
    })
        .toThrowError(
            'FileChunkIterator only supports File, Blob and Uint8Array right ' +
            'now.');
  });
});
