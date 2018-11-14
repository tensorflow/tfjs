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

const csvWithQuote = `A,B,C
1,"2",3
2,2,3
3,"""2",3
4,"2,",
"5"",2,3
6,2,"345"123,456""
7,"2",3`;

const csvDataWithHeadersExtra = ENV.get('IS_BROWSER') ?
    new Blob([csvDataExtra]) :
    Buffer.from(csvDataExtra);
const csvDataWithSemicolon = ENV.get('IS_BROWSER') ?
    new Blob([csvDataSemicolon]) :
    Buffer.from(csvDataSemicolon);
const csvDataWithMixedType = ENV.get('IS_BROWSER') ? new Blob([csvMixedType]) :
                                                     Buffer.from(csvMixedType);
const csvDataWithQuote = ENV.get('IS_BROWSER') ? new Blob([csvWithQuote]) :
                                                 Buffer.from(csvWithQuote);

describe('CSVDataset', () => {
  it('produces a stream of dicts containing UTF8-decoded csv data',
     async () => {
       const source = new FileDataSource(csvData, {chunkSize: 10});
       const dataset = new CSVDataset(
           source, {hasHeader: false, columnNames: ['foo', 'bar', 'baz']});

       expect(await dataset.columnNames()).toEqual(['foo', 'bar', 'baz']);

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
    const dataset = new CSVDataset(source);

    expect(await dataset.columnNames()).toEqual(['foo', 'bar', 'baz']);
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
      const dataset = new CSVDataset(source, {
        hasHeader: false,
        columnNames: ['foo', 'bar', 'baz'],
        columnConfigs: {'A': {required: true}}
      });
      await dataset.columnNames();
      done.fail();
    } catch (error) {
      expect(error.message)
          .toBe(
              'The key "A" provided in columnConfigs does not ' +
              'match any of the column names (foo,bar,baz).');
      done();
    }
  });

  it('throw error when no header line and no column names provided',
     async done => {
       try {
         const source = new FileDataSource(csvData, {chunkSize: 10});
         const dataset = new CSVDataset(source, {hasHeader: false});
         await dataset.columnNames();
         done.fail();
       } catch (error) {
         expect(error.message)
             .toBe('Column names must be provided if there is no header line.');
         done();
       }
     });

  it('take first line as columnNames by default', async () => {
    const source = new FileDataSource(csvDataWithHeaders, {chunkSize: 10});
    const dataset = new CSVDataset(source);
    expect(await dataset.columnNames()).toEqual(['foo', 'bar', 'baz']);
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

  it('emits rows in order despite async requests', async () => {
    const source = new FileDataSource(csvDataWithHeadersExtra, {chunkSize: 10});
    const ds = new CSVDataset(source);
    expect(await ds.columnNames()).toEqual(['A', 'B', 'C']);
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
      const dataset = new CSVDataset(source, {
        hasHeader: false,
        columnNames: ['foo', 'bar', 'baz'],
        columnConfigs: {'foo': {required: true}}
      });
      expect(await dataset.columnNames()).toEqual(['foo', 'bar', 'baz']);
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
    const dataset = new CSVDataset(source, {
      hasHeader: false,
      columnNames: ['foo', 'bar', 'baz'],
      columnConfigs: {'foo': {default: 'abc'}, 'baz': {default: 123}}
    });

    expect(await dataset.columnNames()).toEqual(['foo', 'bar', 'baz']);
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
    const source = new FileDataSource(csvDataWithSemicolon, {chunkSize: 10});
    const dataset = new CSVDataset(source, {delimiter: ';'});
    expect(await dataset.columnNames()).toEqual(['A', 'B', 'C']);
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
       const source = new FileDataSource(csvDataWithMixedType, {chunkSize: 10});
       const dataset = new CSVDataset(source, {
         columnConfigs: {
           'A': {dtype: DType.int32},
           'B': {dtype: DType.bool},
           'C': {dtype: DType.int32},
           'D': {dtype: DType.bool}
         }
       });
       expect(await dataset.columnNames()).toEqual(['A', 'B', 'C', 'D']);
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
    const dataset = new CSVDataset(
        source,
        {columnConfigs: {'bar': {}, 'foo': {}}, configuredColumnsOnly: true});

    expect(await dataset.columnNames()).toEqual(['bar', 'foo']);
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
      const dataset =
          new CSVDataset(source, {columnNames: ['bar', 'foooooooo']});
      await dataset.columnNames();
      done.fail();
    } catch (e) {
      expect(e.message).toEqual(
          'The length of provided columnNames (2) does not match the length ' +
          'of the header line read from file (3).');
      done();
    }
  });

  it('reads CSV with column names override header', async () => {
    const source = new FileDataSource(csvDataWithHeaders, {chunkSize: 10});
    const dataset = new CSVDataset(source, {columnNames: ['a', 'b', 'c']});
    expect(await dataset.columnNames()).toEqual(['a', 'b', 'c']);
    const iter = await dataset.iterator();
    const result = await iter.collect();

    expect(result).toEqual([
      {'a': 'ab', 'b': 'cd', 'c': 'ef'},
      {'a': 'ghi', 'b': undefined, 'c': 'jkl'},
      {'a': undefined, 'b': 'mn', 'c': 'op'},
      {'a': 1.4, 'b': 7.8, 'c': 12},
      {'a': 'qrs', 'b': 'tu', 'c': undefined},
      {'a': 'v', 'b': 'w', 'c': 'x'},
      {'a': 'y', 'b': 'z', 'c': undefined},
    ]);
  });

  it('reads CSV with missing label value', async done => {
    try {
      const source = new FileDataSource(csvDataWithHeaders, {chunkSize: 10});
      const dataset =
          new CSVDataset(source, {columnConfigs: {'baz': {isLabel: true}}});
      expect(await dataset.columnNames()).toEqual(['foo', 'bar', 'baz']);
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
        new CSVDataset(source, {columnConfigs: {'C': {isLabel: true}}});
    expect(await dataset.columnNames()).toEqual(['A', 'B', 'C']);
    const iter = await dataset.iterator();
    const result = await iter.collect();

    expect(result).toEqual([
      [{'A': 1, 'B': 2}, {'C': 3}], [{'A': 2, 'B': 2}, {'C': 3}],
      [{'A': 3, 'B': 2}, {'C': 3}], [{'A': 4, 'B': 2}, {'C': 3}],
      [{'A': 5, 'B': 2}, {'C': 3}], [{'A': 6, 'B': 2}, {'C': 3}],
      [{'A': 7, 'B': 2}, {'C': 3}]
    ]);
  });

  it('reads CSV with quote', async () => {
    const source = new FileDataSource(csvDataWithQuote, {chunkSize: 10});
    const dataset = new CSVDataset(source);
    expect(await dataset.columnNames()).toEqual(['A', 'B', 'C']);
    const iter = await dataset.iterator();
    const result = await iter.collect();

    expect(result).toEqual([
      {'A': 1, 'B': 2, 'C': 3}, {'A': 2, 'B': 2, 'C': 3},
      {'A': 3, 'B': '""2', 'C': 3}, {'A': 4, 'B': '2,', 'C': undefined},
      {'A': '"5""', 'B': 2, 'C': 3}, {'A': 6, 'B': 2, 'C': '345"123,456"'},
      {'A': 7, 'B': 2, 'C': 3}
    ]);
  });

  it('check duplicate column names', async done => {
    try {
      const csvStringWithDuplicateColumnNames = `foo,bar,foo
    ` + csvString;
      const csvDataWithDuplicateColumnNames = ENV.get('IS_BROWSER') ?
          new Blob([csvStringWithDuplicateColumnNames]) :
          Buffer.from(csvStringWithDuplicateColumnNames);
      const source =
          new FileDataSource(csvDataWithDuplicateColumnNames, {chunkSize: 10});
      const dataset = new CSVDataset(source);
      await dataset.columnNames();
    } catch (e) {
      expect(e.message).toEqual('Duplicate column names found: foo');
      done();
    }
  });
});
