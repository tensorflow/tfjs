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

import {FileDataSource} from '../sources/file_data_source';

import {CSVDataset, CsvHeaderConfig} from './csv_dataset';

const csvData = `ab,cd,ef
ghi,,jkl
,mn,op
1.4,7.8,12
qrs,tu,
v,w,x
y,z`;

const csvDataWithHeaders = `foo,bar,baz
` + csvData;

const csvBlob = new Blob([csvData]);

const csvBlobWithHeaders = new Blob([csvDataWithHeaders]);

const csvDataExtra = `A,B,C
1,2,3
2,2,3
3,2,3
4,2,3
5,2,3
6,2,3
7,2,3`;

const csvBlobWithHeadersExtra = new Blob([csvDataExtra]);

describe('CSVDataset', () => {
  it('produces a stream of dicts containing UTF8-decoded csv data',
     async () => {
       const source = new FileDataSource(csvBlob, {chunkSize: 10});
       const dataset = await CSVDataset.create(source, ['foo', 'bar', 'baz']);

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
    const source = new FileDataSource(csvBlobWithHeaders, {chunkSize: 10});
    const dataset =
        await CSVDataset.create(source, CsvHeaderConfig.READ_FIRST_LINE);

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

  it('numbers CSV columns by default', async () => {
    const source = new FileDataSource(csvBlob, {chunkSize: 10});
    const dataset = await CSVDataset.create(source);
    expect(dataset.csvColumnNames).toEqual(['0', '1', '2']);
    const iter = await dataset.iterator();
    const result = await iter.collect();

    expect(result).toEqual([
      {'0': 'ab', '1': 'cd', '2': 'ef'},
      {'0': 'ghi', '1': undefined, '2': 'jkl'},
      {'0': undefined, '1': 'mn', '2': 'op'},
      {'0': 1.4, '1': 7.8, '2': 12},
      {'0': 'qrs', '1': 'tu', '2': undefined},
      {'0': 'v', '1': 'w', '2': 'x'},
      {'0': 'y', '1': 'z', '2': undefined},
    ]);
  });

  it('emits rows in order despite async requests', async () => {
    const source = new FileDataSource(csvBlobWithHeadersExtra, {chunkSize: 10});
    const ds = await CSVDataset.create(source, CsvHeaderConfig.READ_FIRST_LINE);
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
});
