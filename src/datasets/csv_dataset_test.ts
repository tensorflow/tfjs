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
import {DType} from '@tensorflow/tfjs-core/dist/types';

import {FileDataSource} from '../sources/file_data_source';

import {CSVDataset} from './csv_dataset';

const csvString = `ab,cd,ef
ghi,,jkl
,mn,op
1.4,7.8,12
qrs,tu,
v,w,x
y,z,`;

const csvStringWithHeaders = `foo,bar,baz
` + csvString;

const csvData =
    ENV.get('IS_BROWSER') ? new Blob([csvString]) : Buffer.from(csvString);

const csvDataWithHeaders = ENV.get('IS_BROWSER') ?
    new Blob([csvStringWithHeaders]) :
    Buffer.from(csvStringWithHeaders);

const csvDataExtra = `A,B,C
1,2,3
2,2,3
3,2,3
4,2,3
5,2,3
6,2,3
7,2,3`;

const csvDataSemicolon = `A;B;C
1;2;3
2;2;3
3;2;3
4;2;3
5;2;3
6;2;3
7;2;3`;

const csvMixedType = `A,B,C,D
1,True,3,1
2,False,2,0
3,True,1,1
1,False,3,0
2,True,2,1
3,False,1,0
1,True,3,1
2,False,2,0`;

const csvDataWithHeadersExtra = ENV.get('IS_BROWSER') ?
    new Blob([csvDataExtra]) :
    Buffer.from(csvDataExtra);
const csvBlobWithSemicolon = ENV.get('IS_BROWSER') ?
    new Blob([csvDataSemicolon]) :
    Buffer.from(csvDataSemicolon);
const csvBlobWithMixedType = ENV.get('IS_BROWSER') ? new Blob([csvMixedType]) :
                                                     Buffer.from(csvMixedType);

describe('CSVDataset', () => {
  it('produces a stream of dicts containing UTF8-decoded csv data',
     async () => {
       const source = new FileDataSource(csvData, {chunkSize: 10});
       const dataset =
           await CSVDataset.create(source, false, ['foo', 'bar', 'baz']);

       expect(dataset.csvColumnNames).toEqual(['foo', 'bar', 'baz']);

       const iter = await dataset.iterator();
       const result = await iter.collect();

       expect(result).toEqual([
         {'foo': 'ab', 'bar': 'cd', 'baz': 'ef'},
         {'foo': 'ghi', 'bar': undefined, 'baz': 'jkl'},
         {'foo': undefined, 'bar': 'mn', 'baz': 'op'},
         {'foo': 1.4, 'bar': 7.8, 'baz': 12},
         {'foo': 'qrs', 'bar': 'tu', 'baz': undefined},
         {'foo': 'v', 'bar': 'w', 'baz': 'x'},
         {'foo': 'y', 'bar': 'z', 'baz': undefined},
       ]);
     });

  it('reads CSV column headers when requested', async () => {
    const source = new FileDataSource(csvDataWithHeaders, {chunkSize: 10});
    const dataset = await CSVDataset.create(source, true);

    expect(dataset.csvColumnNames).toEqual(['foo', 'bar', 'baz']);
    const iter = await dataset.iterator();
    const result = await iter.collect();

    expect(result).toEqual([
      {'foo': 'ab', 'bar': 'cd', 'baz': 'ef'},
      {'foo': 'ghi', 'bar': undefined, 'baz': 'jkl'},
      {'foo': undefined, 'bar': 'mn', 'baz': 'op'},
      {'foo': 1.4, 'bar': 7.8, 'baz': 12},
      {'foo': 'qrs', 'bar': 'tu', 'baz': undefined},
      {'foo': 'v', 'bar': 'w', 'baz': 'x'},
      {'foo': 'y', 'bar': 'z', 'baz': undefined},
    ]);
  });

  it('throw error when column configs mismatch column names', async done => {
    try {
      const source = new FileDataSource(csvData, {chunkSize: 10});
      await CSVDataset.create(
          source, false, ['foo', 'bar', 'baz'], {'A': {required: true}});
      done.fail();
    } catch (error) {
      expect(error.message).toBe('Column config does not match column names.');
      done();
    }
  });

  it('throw error when no header line and no column names provided',
     async done => {
       try {
         const source = new FileDataSource(csvData, {chunkSize: 10});
         await CSVDataset.create(source);
         done.fail();
       } catch (error) {
         expect(error.message)
             .toBe('Column names must be provided if there is no header line.');
         done();
       }
     });

  it('emits rows in order despite async requests', async () => {
    const source = new FileDataSource(csvDataWithHeadersExtra, {chunkSize: 10});
    const ds = await CSVDataset.create(source, true);
    expect(ds.csvColumnNames).toEqual(['A', 'B', 'C']);
    const csvIterator = await ds.iterator();
    const promises = [
      csvIterator.next(), csvIterator.next(), csvIterator.next(),
      csvIterator.next(), csvIterator.next()
    ];
    const elements = await Promise.all(promises);
    expect(elements[0].value).toEqual({A: 1, B: 2, C: 3});
    expect(elements[1].value).toEqual({A: 2, B: 2, C: 3});
    expect(elements[2].value).toEqual({A: 3, B: 2, C: 3});
    expect(elements[3].value).toEqual({A: 4, B: 2, C: 3});
    expect(elements[4].value).toEqual({A: 5, B: 2, C: 3});
  });

  it('throw error when required column is empty', async done => {
    try {
      const source = new FileDataSource(csvData, {chunkSize: 10});
      const dataset = await CSVDataset.create(
          source, false, ['foo', 'bar', 'baz'], {'foo': {required: true}});
      expect(dataset.csvColumnNames).toEqual(['foo', 'bar', 'baz']);
      const iter = await dataset.iterator();
      await iter.collect();
      done.fail();
    } catch (error) {
      expect(error.message)
          .toBe('Required column foo is empty in this line: ,mn,op');
      done();
    }
  });

  it('fill default value when provided', async () => {
    const source = new FileDataSource(csvData, {chunkSize: 10});
    const dataset = await CSVDataset.create(
        source, false, ['foo', 'bar', 'baz'],
        {'foo': {default: 'abc'}, 'baz': {default: 123}});

    expect(dataset.csvColumnNames).toEqual(['foo', 'bar', 'baz']);
    const iter = await dataset.iterator();
    const result = await iter.collect();

    expect(result).toEqual([
      {'foo': 'ab', 'bar': 'cd', 'baz': 'ef'},
      {'foo': 'ghi', 'bar': undefined, 'baz': 'jkl'},
      {'foo': 'abc', 'bar': 'mn', 'baz': 'op'},
      {'foo': 1.4, 'bar': 7.8, 'baz': 12},
      {'foo': 'qrs', 'bar': 'tu', 'baz': 123},
      {'foo': 'v', 'bar': 'w', 'baz': 'x'},
      {'foo': 'y', 'bar': 'z', 'baz': 123},
    ]);
  });

  it('provide delimiter through parameter', async () => {
    const source = new FileDataSource(csvBlobWithSemicolon, {chunkSize: 10});
    const dataset =
        await CSVDataset.create(source, true, null, null, false, ';');
    expect(dataset.csvColumnNames).toEqual(['A', 'B', 'C']);
    const iter = await dataset.iterator();
    const result = await iter.collect();

    expect(result[0]).toEqual({A: 1, B: 2, C: 3});
    expect(result[1]).toEqual({A: 2, B: 2, C: 3});
    expect(result[2]).toEqual({A: 3, B: 2, C: 3});
    expect(result[3]).toEqual({A: 4, B: 2, C: 3});
    expect(result[4]).toEqual({A: 5, B: 2, C: 3});
  });

  it('provide datatype through parameter to parse different types',
     async () => {
       const source = new FileDataSource(csvBlobWithMixedType, {chunkSize: 10});
       const dataset = await CSVDataset.create(source, true, null, {
         'A': {dtype: DType.int32},
         'B': {dtype: DType.bool},
         'C': {dtype: DType.int32},
         'D': {dtype: DType.bool}
       });
       expect(dataset.csvColumnNames).toEqual(['A', 'B', 'C', 'D']);
       const iter = await dataset.iterator();
       const result = await iter.collect();

       expect(result).toEqual([
         {'A': 1, 'B': 1, 'C': 3, 'D': 1}, {'A': 2, 'B': 0, 'C': 2, 'D': 0},
         {'A': 3, 'B': 1, 'C': 1, 'D': 1}, {'A': 1, 'B': 0, 'C': 3, 'D': 0},
         {'A': 2, 'B': 1, 'C': 2, 'D': 1}, {'A': 3, 'B': 0, 'C': 1, 'D': 0},
         {'A': 1, 'B': 1, 'C': 3, 'D': 1}, {'A': 2, 'B': 0, 'C': 2, 'D': 0}
       ]);
     });

  it('reads CSV with selected column in order', async () => {
    const source = new FileDataSource(csvDataWithHeaders, {chunkSize: 10});
    const dataset = await CSVDataset.create(
        source, true, null, {'bar': {}, 'foo': {}}, true);

    expect(dataset.csvColumnNames).toEqual(['bar', 'foo']);
    const iter = await dataset.iterator();
    const result = await iter.collect();

    expect(result).toEqual([
      {'bar': 'cd', 'foo': 'ab'},
      {'bar': undefined, 'foo': 'ghi'},
      {'bar': 'mn', 'foo': undefined},
      {'bar': 7.8, 'foo': 1.4},
      {'bar': 'tu', 'foo': 'qrs'},
      {'bar': 'w', 'foo': 'v'},
      {'bar': 'z', 'foo': 'y'},
    ]);
  });

  it('reads CSV with wrong column', async done => {
    try {
      const source = new FileDataSource(csvDataWithHeaders, {chunkSize: 10});
      await CSVDataset.create(source, true, ['bar', 'foooooooo']);
      done.fail();
    } catch (e) {
      expect(e.message).toEqual(
          'Provided column names does not match header line.');
      done();
    }
  });

  it('reads CSV with missing label value', async done => {
    try {
      const source = new FileDataSource(csvDataWithHeaders, {chunkSize: 10});
      const dataset =
          await CSVDataset.create(source, true, null, {'baz': {isLabel: true}});
      expect(dataset.csvColumnNames).toEqual(['foo', 'bar', 'baz']);
      const iter = await dataset.iterator();
      await iter.collect(1000, 0);
      done.fail();
    } catch (e) {
      expect(e.message).toEqual(
          'Required column baz is empty in this line: qrs,tu,');
      done();
    }
  });

  it('reads CSV with label column', async () => {
    const source = new FileDataSource(csvDataWithHeadersExtra, {chunkSize: 10});
    const dataset =
        await CSVDataset.create(source, true, null, {'C': {isLabel: true}});
    expect(dataset.csvColumnNames).toEqual(['A', 'B', 'C']);
    const iter = await dataset.iterator();
    const result = await iter.collect();

    expect(result).toEqual([
      [{'A': 1, 'B': 2}, {'C': 3}], [{'A': 2, 'B': 2}, {'C': 3}],
      [{'A': 3, 'B': 2}, {'C': 3}], [{'A': 4, 'B': 2}, {'C': 3}],
      [{'A': 5, 'B': 2}, {'C': 3}], [{'A': 6, 'B': 2}, {'C': 3}],
      [{'A': 7, 'B': 2}, {'C': 3}]
    ]);
  });
});
