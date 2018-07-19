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

import {Dataset} from '../dataset';
import {DataSource} from '../datasource';
import {LazyIterator} from '../iterators/lazy_iterator';
import {DataElement, ElementArray} from '../types';

import {TextLineDataset} from './text_line_dataset';

export enum CsvHeaderConfig {
  READ_FIRST_LINE,
  NUMBERED
  // PROVIDED // This is just represented as string[]
}

/**
 * Represents a potentially large collection of delimited text records.
 *
 * The produced `DataElement`s each contain one key-value pair for
 * every column of the table.  When a field is empty in the incoming data, the
 * resulting value is `undefined`.  Values that can be parsed as numbers are
 * emitted as type `number`; otherwise they are left as `string`.
 *
 * The results are not batched.
 */
export class CSVDataset extends Dataset<DataElement> {
  base: TextLineDataset;
  private hasHeaderLine = false;
  private _csvColumnNames: string[];

  /**
   * Create a `CSVDataset`.  Note this CSVDataset cannot be used until
   * setCsvColumnNames() is called; that is an async method and so cannot be
   * incorporated into the constructor.  The static async create() method
   * solves this issue.
   *
   * @param input A `DataSource` providing a chunked, UTF8-encoded byte stream.
   */
  private constructor(protected readonly input: DataSource) {
    super();
    this.base = new TextLineDataset(input);
  }

  get csvColumnNames(): string[] {
    return this._csvColumnNames;
  }

  private async setCsvColumnNames(csvColumnNames: CsvHeaderConfig|string[]) {
    if (csvColumnNames == null || csvColumnNames === CsvHeaderConfig.NUMBERED) {
      const iter = await this.base.iterator();
      const firstElement = await iter.next();
      if (firstElement.done) {
        throw new Error('No data was found for CSV parsing.');
      }
      const firstLine: string = firstElement.value;
      this._csvColumnNames =
          Array.from(firstLine.split(',').keys()).map(x => x.toString());
    } else if (csvColumnNames === CsvHeaderConfig.READ_FIRST_LINE) {
      const iter = await this.base.iterator();
      const firstElement = await iter.next();
      if (firstElement.done) {
        throw new Error('No data was found for CSV parsing.');
      }
      const firstLine: string = firstElement.value;
      this._csvColumnNames = firstLine.split(',');
      this.hasHeaderLine = true;
    } else {
      this._csvColumnNames = csvColumnNames;
    }
  }

  /**
   * Create a `CSVDataset`.
   *
   * @param input A `DataSource` providing a chunked, UTF8-encoded byte stream.
   * @param csvColumnNames The keys to use for the columns, in order.  If this
   *   argument is provided, it is assumed that the input file does not have a
   *   header line providing the column names.  If this argument is not provided
   *   (or is null or undefined), then the column names are read from the first
   *   line of the input.
   */
  static async create(
      input: DataSource,
      csvColumnNames: CsvHeaderConfig|string[] = CsvHeaderConfig.NUMBERED) {
    const result = new CSVDataset(input);
    await result.setCsvColumnNames(csvColumnNames);
    return result;
  }

  async iterator(): Promise<LazyIterator<DataElement>> {
    let lines = await this.base.iterator();
    if (this.hasHeaderLine) {
      // We previously read the first line to get the headers.
      // Now that we're providing data, skip it.
      lines = lines.skip(1);
    }
    return lines.map(x => this.makeDataElement(x));
  }

  makeDataElement(line: string): DataElement {
    // TODO(soergel): proper CSV parsing with escaping, quotes, etc.
    // TODO(soergel): alternate separators, e.g. for TSV
    const values = line.split(',');
    const result: {[key: string]: ElementArray} = {};
    for (let i = 0; i < this._csvColumnNames.length; i++) {
      const value = values[i];
      // TODO(soergel): specify data type using a schema
      if (value === '') {
        result[this._csvColumnNames[i]] = undefined;
      } else {
        const valueAsNum = Number(value);
        if (isNaN(valueAsNum)) {
          result[this._csvColumnNames[i]] = value;
        } else {
          result[this._csvColumnNames[i]] = valueAsNum;
        }
      }
    }
    return result;
  }
}

// TODO(soergel): add more basic datasets for parity with tf.data
// tf.data.FixedLengthRecordDataset()
// tf.data.TFRecordDataset()
