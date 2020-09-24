/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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

import {TensorContainerObject, tensor, scalar} from '@tensorflow/tfjs-core';
import {maskedCrc32c} from '../proto/crc32c';
import {LazyIterator} from './lazy_iterator';
import {FeatureProto, Feature} from '../types';

// tslint:disable-next-line:no-require-imports
const messages = require('../proto/api_pb.js');

/**
 * Provide tenserflow.Example from a File path.
 * @param file Local file path.
 * @returns a lazy Iterator of tenserflow.Example.
 */
export class TFRecordIterator extends LazyIterator<TensorContainerObject> {
  // The file descriptor used to read records.
  private fd: number;
  // True when the reader is closed.
  private closed: boolean;

  // Buffer used to read the 8-byte length and 4-byte CRC32C header.
  private lengthAndCrcBuffer: Uint8Array;
  // DataView used to extract the 8-byte length and 4-byte CRC32C header.
  private lengthAndCrc: DataView;
  // node.js Buffer pointing at length for CRC32C computations.
  private lengthBuffer: Buffer;
  // Bytes length of buffer: 8-byte length.
  private lengthBufferLength = 8;
  // Bytes length of buffer: 4-byte CRC32C header.
  private crcBufferLength = 4;
  // Bytes length of read buffer: 8-byte length and 4-byte CRC32C header.
  private readByteLength = 12;

  // Buffer used to read records.
  private dataBuffer: Uint8Array;
  // DataVieww used to extract the 4-byte CRC32C at the end of the record.
  private dataBufferView: DataView;

  constructor(protected file: string) {
    super();
    this.closed = false;
    // tslint:disable-next-line:no-require-imports
    const fs = require('fs');
    this.fd = fs.openSync(file.substr(7), 'r');

    this.dataBuffer = new Uint8Array(1);
    this.dataBufferView = new DataView(this.dataBuffer.buffer, 0, 1);

    const metadataBuffer = new ArrayBuffer(this.readByteLength);
    this.lengthAndCrcBuffer = new Uint8Array(
      metadataBuffer, 0, this.readByteLength);
    this.lengthAndCrc = new DataView(metadataBuffer, 0, this.readByteLength);
    this.lengthBuffer = Buffer.from(metadataBuffer, 0, this.lengthBufferLength);
  }

  summary() {
    return `DataBuffer ${this.dataBuffer}`;
  }

  async next(): Promise<IteratorResult<TensorContainerObject>> {
    if (this.closed) {
      return {value: null, done: true};
    }
    // tslint:disable-next-line:no-require-imports
    const fs = require('fs');
    let bytesRead = fs.readSync(
      this.fd, this.lengthAndCrcBuffer, 0, this.readByteLength, null);
    if (bytesRead === 0) {
      fs.closeSync(this.fd);
      this.closed = true;
      return {value: null, done: true};
    }
    if (bytesRead !== this.readByteLength) {
      fs.closeSync(this.fd);
      this.closed = true;
      throw new Error(
        `Incomplete read; expected ${this.readByteLength} bytes,` +
        `got ${bytesRead}`);
    }

    const length = this.lengthAndCrc.getUint32(0, true);
    const length64 = this.lengthAndCrc.getUint32(4, true);
    const lengthCrc = this.lengthAndCrc.getUint32(8, true);

    if (length64 !== 0) {
      fs.closeSync(this.fd);
      this.closed = true;
      throw new Error(`4GB+ records not supported`);
    }

    if (lengthCrc !== maskedCrc32c(this.lengthBuffer)) {
      fs.closeSync(this.fd);
      this.closed = true;
      throw new Error(`Incorrect record length CRC32C`);
    }
    // Need to read the CRC32C as well.
    const readLength = length + this.crcBufferLength;
    if (readLength > this.dataBuffer.length) {
      // Grow the buffer.
      let newLength = this.dataBuffer.length;
      while (newLength < readLength) {
        newLength *= 2;
      }
      this.dataBuffer = new Uint8Array(newLength);
      this.dataBufferView = new DataView(this.dataBuffer.buffer, 0, newLength);
    }

    bytesRead = fs.readSync(this.fd, this.dataBuffer, 0, readLength, null);
    if (bytesRead !== readLength) {
      fs.closeSync(this.fd);
      this.closed = true;
      throw new Error(
        `Incomplete read; expected ${readLength} bytes,got ${bytesRead}`);
    }

    const recordData = new Uint8Array(this.dataBuffer.buffer, 0, length);
    const recordCrc = this.dataBufferView.getUint32(length, true);

    const recordBuffer = Buffer.from(
      this.dataBuffer.buffer as ArrayBuffer,
      0,
      length
    );
    if (recordCrc !== maskedCrc32c(recordBuffer)) {
      fs.closeSync(this.fd);
      this.closed = true;
      throw new Error(`Incorrect record CRC32C`);
    }

    const result: TensorContainerObject = {};
    messages.Example.deserializeBinary(recordData)
      .getFeatures()
      .getFeatureMap()
      .forEach((item: Feature, index: string) => {
        const itemObj: FeatureProto = item.toObject();
        if (itemObj.bytesList) {
          const bytesBuff = Buffer.from(
            itemObj.bytesList.valueList[0],
            'base64'
          );
          result[index] = tensor(new Uint8Array(bytesBuff));
        } else if (itemObj.floatList) {
          result[index] = scalar(itemObj.floatList.valueList[0], 'float32');
        } else if (itemObj.int64List) {
          result[index] = scalar(itemObj.int64List.valueList[0], 'int32');
        }
      });
    return {value: result, done: false};
  }
}
