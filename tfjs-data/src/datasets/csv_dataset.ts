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

import {TensorContainer, util} from '@tensorflow/tfjs-core';
import {Dataset} from '../dataset';
import {DataSource} from '../datasource';
import {LazyIterator} from '../iterators/lazy_iterator';
import {ColumnConfig, CSVConfig} from '../types';
import {TextLineDataset} from './text_line_dataset';

const CODE_QUOTE = '"';
const STATE_OUT = Symbol('out');
const STATE_FIELD = Symbol('field');
const STATE_QUOTE = Symbol('quote');
const STATE_QUOTE_AFTER_QUOTE = Symbol('quoteafterquote');
const STATE_WITHIN_QUOTE_IN_QUOTE = Symbol('quoteinquote');

/**
 * Represents a potentially large collection of delimited text records.
 *
 * The produced `TensorContainer`s each contain one key-value pair for
 * every column of the table.  When a field is empty in the incoming data, the
 * resulting value is `undefined`, or throw error if it is required.  Values
 * that can be parsed as numbers are emitted as type `number`, other values
 * are parsed as `string`.
 *
 * The results are not batched.
 *
 * @doc {heading: 'Data', subheading: 'Classes', namespace: 'data'}
 */
export class CSVDataset extends Dataset<TensorContainer> {
  base: TextLineDataset;
  private hasHeader = true;
  private fullColumnNames: string[] = null;
  private columnNamesValidated = false;
  private columnConfigs: {[key: string]: ColumnConfig} = null;
  private configuredColumnsOnly = false;
  private delimiter = ',';
  private delimWhitespace = false;

  /**
   * Returns column names of the csv dataset. If `configuredColumnsOnly` is
   * true, return column names in `columnConfigs`. If `configuredColumnsOnly` is
   * false and `columnNames` is provided, `columnNames`. If
   * `configuredColumnsOnly` is false and `columnNames` is not provided, return
   * all column names parsed from the csv file. For example usage please go to
   * `tf.data.csv`.
   *
   * @doc {heading: 'Data', subheading: 'Classes'}
   */
  async columnNames() {
    if (!this.columnNamesValidated) {
      await this.setColumnNames();
    }
    return this.configuredColumnsOnly ? Object.keys(this.columnConfigs) :
                                        this.fullColumnNames;
  }

  /* 1) If `columnNames` is provided as string[], use this string[] as output
   * keys in corresponding order. The length must match the number of inferred
   * columns if `hasHeader` is true .
   * 2) If `columnNames` is not provided, parse header line as `columnNames` if
   * hasHeader is true. If `hasHeader` is false, throw an error.
   * 3) If `columnConfigs` is provided, all the keys in `columnConfigs` must
   * exist in parsed `columnNames`.
   */
  private async setColumnNames() {
    const columnNamesFromFile = await this.maybeReadHeaderLine();
    if (!this.fullColumnNames && !columnNamesFromFile) {
      // Throw an error if columnNames is not provided and no header line.
      throw new Error(
          'Column names must be provided if there is no header line.');
    } else if (this.fullColumnNames && columnNamesFromFile) {
      // Check provided columnNames match header line.
      util.assert(
          columnNamesFromFile.length === this.fullColumnNames.length,
          () => 'The length of provided columnNames (' +
              this.fullColumnNames.length.toString() +
              ') does not match the length of the header line read from ' +
              'file (' + columnNamesFromFile.length.toString() + ').');
    }
    if (!this.fullColumnNames) {
      this.fullColumnNames = columnNamesFromFile;
    }
    // Check if there are duplicate column names.
    const counts: {[key: string]: number} = this.fullColumnNames.reduce(
        (countAcc: {[key: string]: number}, name) => {
          countAcc[name] = (countAcc[name] + 1) || 1;
          return countAcc;
        },
        {});
    const duplicateNames =
        Object.keys(counts).filter((name) => (counts[name] > 1));
    util.assert(
        duplicateNames.length === 0,
        () => 'Duplicate column names found: ' + duplicateNames.toString());
    // Check if keys in columnConfigs match columnNames.
    if (this.columnConfigs) {
      for (const key of Object.keys(this.columnConfigs)) {
        const index = this.fullColumnNames.indexOf(key);
        if (index === -1) {
          throw new Error(
              'The key "' + key +
              '" provided in columnConfigs does not match any of the column ' +
              'names (' + this.fullColumnNames.toString() + ').');
        }
      }
    }
    this.columnNamesValidated = true;
  }

  private async maybeReadHeaderLine() {
    if (this.hasHeader) {
      const iter = await this.base.iterator();
      const firstElement = await iter.next();
      if (firstElement.done) {
        throw new Error('No data was found for CSV parsing.');
      }
      const firstLine: string = firstElement.value;
      const headers = this.parseRow(firstLine, false);
      return headers;
    } else {
      return null;
    }
  }

  /**
   * Create a `CSVDataset`.
   *
   * @param input A `DataSource` providing a chunked, UTF8-encoded byte stream.
   * @param csvConfig (Optional) A CSVConfig object that contains configurations
   *     of reading and decoding from CSV file(s).
   *
   *     hasHeader: (Optional) A boolean value that indicates whether the first
   *     row of provided CSV file is a header line with column names, and should
   *     not be included in the data. Defaults to `true`.
   *
   *     columnNames: (Optional) A list of strings that corresponds to
   *     the CSV column names, in order. If provided, it ignores the column
   *     names inferred from the header row. If not provided, infers the column
   *     names from the first row of the records. If hasHeader is false and
   *     columnNames is not provided, this method throws an error.
   *
   *     columnConfigs: (Optional) A dictionary whose key is column names, value
   *     is an object stating if this column is required, column's data type,
   *     default value, and if this column is label. If provided, keys must
   *     correspond to names provided in columnNames or inferred from the file
   *     header lines. If isLabel is true any column, returns an array of two
   *     items: the first item is a dict of features key/value pairs, the second
   *     item is a dict of labels key/value pairs. If no feature is marked as
   *     label, returns a dict of features only.
   *
   *     configuredColumnsOnly (Optional) If true, only columns provided in
   *     columnConfigs will be parsed and provided during iteration.
   *
   *     delimiter (Optional) The string used to parse each line of the input
   *     file. Defaults to `,`.
   */
  constructor(protected readonly input: DataSource, csvConfig?: CSVConfig) {
    super();
    this.base = new TextLineDataset(input);
    if (!csvConfig) {
      csvConfig = {};
    }
    this.hasHeader = csvConfig.hasHeader === false ? false : true;
    this.fullColumnNames = csvConfig.columnNames;
    this.columnConfigs = csvConfig.columnConfigs;
    this.configuredColumnsOnly = csvConfig.configuredColumnsOnly;
    if (csvConfig.delimWhitespace) {
      util.assert(
          csvConfig.delimiter == null,
          () =>
              'Delimiter should not be provided when delimWhitespace is true.');
      this.delimWhitespace = true;
      this.delimiter = ' ';
    } else {
      this.delimiter = csvConfig.delimiter ? csvConfig.delimiter : ',';
    }
  }

  async iterator(): Promise<LazyIterator<TensorContainer>> {
    if (!this.columnNamesValidated) {
      await this.setColumnNames();
    }
    let lines = await this.base.iterator();
    if (this.hasHeader) {
      // We previously read the first line to get the columnNames.
      // Now that we're providing data, skip it.
      lines = lines.skip(1);
    }
    return lines.map(x => this.makeDataElement(x));
  }

  makeDataElement(line: string): TensorContainer {
    const values = this.parseRow(line);
    const features: {[key: string]: TensorContainer} = {};
    const labels: {[key: string]: TensorContainer} = {};

    for (let i = 0; i < this.fullColumnNames.length; i++) {
      const key = this.fullColumnNames[i];
      const config = this.columnConfigs ? this.columnConfigs[key] : null;
      if (this.configuredColumnsOnly && !config) {
        // This column is not selected.
        continue;
      } else {
        const value = values[i];
        let parsedValue = null;
        if (value === '') {
          // If default value is provided, use it. If default value is not
          // provided, set as undefined.
          if (config && config.default !== undefined) {
            parsedValue = config.default;
          } else if (config && (config.required || config.isLabel)) {
            throw new Error(
                `Required column ${key} is empty in this line: ${line}`);
          } else {
            parsedValue = undefined;
          }
        } else {
          // A value is present, so parse it based on type
          const valueAsNum = Number(value);
          if (isNaN(valueAsNum)) {
            // The value is a string and this column is declared as boolean
            // in config, parse it as boolean.
            if (config && config.dtype === 'bool') {
              parsedValue = this.getBoolean(value);
            } else {
              // Set value as string
              parsedValue = value;
            }
          } else if (!config || !config.dtype) {
            // If this value is a number and no type config is provided, return
            // it as number.
            parsedValue = valueAsNum;
          } else {
            // If this value is a number and data type is provided, parse it
            // according to provided data type.
            switch (config.dtype) {
              case 'float32':
                parsedValue = valueAsNum;
                break;
              case 'int32':
                parsedValue = Math.floor(valueAsNum);
                break;
              case 'bool':
                parsedValue = this.getBoolean(value);
                break;
              default:
                parsedValue = valueAsNum;
            }
          }
        }
        // Check if this column is label.
        (config && config.isLabel) ? labels[key] = parsedValue :
                                     features[key] = parsedValue;
      }
    }
    // If label exists, return an object of features and labels as {xs:features,
    // ys:labels}, otherwise return features only.
    if (Object.keys(labels).length === 0) {
      return features;

    } else {
      return {xs: features, ys: labels};
    }
  }

  private getBoolean(value: string): number {
    if (value === '1' || value.toLowerCase() === 'true') {
      return 1;
    } else {
      return 0;
    }
  }

  // adapted from https://beta.observablehq.com/@mbostock/streaming-csv
  private parseRow(line: string, validateElementCount = true): string[] {
    const result: string[] = [];
    let readOffset = 0;
    const readLength = line.length;
    let currentState = STATE_OUT;
    // Goes through the line to parse quote.
    for (let i = 0; i < readLength; i++) {
      switch (currentState) {
        // Before enter a new field
        case STATE_OUT:
          switch (line.charAt(i)) {
            // Enter a quoted field
            case CODE_QUOTE:
              readOffset = i + 1;
              currentState = STATE_QUOTE;
              break;
            // Read an empty field
            case this.delimiter:
              readOffset = i + 1;
              // If delimiter is white space and configured to collapse
              // multiple white spaces, ignore this white space.
              if (this.delimiter === ' ' && this.delimWhitespace) {
                break;
              }
              result.push('');
              currentState = STATE_OUT;
              break;
            // Enter an unquoted field
            default:
              currentState = STATE_FIELD;
              readOffset = i;
              break;
          }
          break;
        // In an unquoted field
        case STATE_FIELD:
          switch (line.charAt(i)) {
            // Exit an unquoted field, add it to result
            case this.delimiter:
              result.push(line.substring(readOffset, i));
              currentState = STATE_OUT;
              readOffset = i + 1;
              break;
            default:
          }
          break;
        // In a quoted field
        case STATE_QUOTE:
          switch (line.charAt(i)) {
            // Read a quote after a quote
            case CODE_QUOTE:
              currentState = STATE_QUOTE_AFTER_QUOTE;
              break;
            default:
          }
          break;
        // This state means it's right after a second quote in a field
        case STATE_QUOTE_AFTER_QUOTE:
          switch (line.charAt(i)) {
            // Finished a quoted field
            case this.delimiter:
              result.push(line.substring(readOffset, i - 1));
              currentState = STATE_OUT;
              readOffset = i + 1;
              break;
            // Finished a quoted part in a quoted field
            case CODE_QUOTE:
              currentState = STATE_QUOTE;
              break;
            // In a quoted part in a quoted field
            default:
              currentState = STATE_WITHIN_QUOTE_IN_QUOTE;
              break;
          }
          break;
        case STATE_WITHIN_QUOTE_IN_QUOTE:
          switch (line.charAt(i)) {
            // Exit a quoted part in a quoted field
            case CODE_QUOTE:
              currentState = STATE_QUOTE;
              break;
            default:
          }
          break;
        default:
      }
    }
    // Adds last item based on if it is quoted.
    if (currentState === STATE_QUOTE_AFTER_QUOTE) {
      result.push(line.substring(readOffset, readLength - 1));
    } else {
      result.push(line.substring(readOffset));
    }
    // Check if each row has the same number of elements as column names.
    if (validateElementCount && result.length !== this.fullColumnNames.length) {
      throw new Error(`Invalid row in csv file. Should have ${
          this.fullColumnNames.length} elements in a row, but got ${result}`);
    }
    return result;
  }
}

// TODO(soergel): add more basic datasets for parity with tf.data
// tf.data.FixedLengthRecordDataset()
// tf.data.TFRecordDataset()
