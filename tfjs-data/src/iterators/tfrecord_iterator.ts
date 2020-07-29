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

import {TensorContainerObject, env, tensor, scalar} from '@tensorflow/tfjs-core';
import {isLocalPath} from '../util/source_util';
import {maskedCrc32c} from '../proto/crc32c';
import {LazyIterator} from './lazy_iterator';
import {IFeatureProto, IFeature} from '../types';

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

  // Buffer used to read records.
  private dataBuffer: Uint8Array;
  // DataVieww used to extract the 4-byte CRC32C at the end of the record.
  private dataBufferView: DataView;

  constructor(protected file: string) {
    super();
    this.closed = false;
    if (isLocalPath(file) && env().get('IS_NODE')) {
      // tslint:disable-next-line:no-require-imports
      const fs = require('fs');
      this.fd = fs.openSync(file.substr(7), 'r');
    }

    this.dataBuffer = new Uint8Array(1);
    this.dataBufferView = new DataView(this.dataBuffer.buffer, 0, 1);

    const metadataBuffer = new ArrayBuffer(12);
    this.lengthAndCrcBuffer = new Uint8Array(metadataBuffer, 0, 12);
    this.lengthAndCrc = new DataView(metadataBuffer, 0, 12);
    this.lengthBuffer = Buffer.from(metadataBuffer, 0, 8);
  }

  summary() {
    return `DataBuffer ${this.dataBuffer}`;
  }

  async next(): Promise<IteratorResult<TensorContainerObject>> {
    if (!isLocalPath(this.file) || !env().get('IS_NODE')) {
      return {value: null, done: true};
    }
    if (this.closed) return {value: null, done: true};
    // tslint:disable-next-line:no-require-imports
    const fs = require('fs');
    let bytesRead = fs.readSync(this.fd, this.lengthAndCrcBuffer, 0, 12, null);
    if (bytesRead === 0) {
      fs.closeSync(this.fd);
      this.closed = true;
      return {value: null, done: true};
    }
    if (bytesRead !== 12) {
      fs.closeSync(this.fd);
      this.closed = true;
      // error msg: `Incomplete read; expected 12 bytes, got ${bytesRead}`
      return {value: null, done: true};
    }

    const length = this.lengthAndCrc.getUint32(0, true);
    const length64 = this.lengthAndCrc.getUint32(4, true);
    const lengthCrc = this.lengthAndCrc.getUint32(8, true);

    if (length64 !== 0) {
      fs.closeSync(this.fd);
      this.closed = true;
      // error msg: `4GB+ records not supported`
      return {value: null, done: true};
    }

    if (lengthCrc !== maskedCrc32c(this.lengthBuffer)) {
      fs.closeSync(this.fd);
      this.closed = true;
      // error msg: `Incorrect record length CRC32C`
      return {value: null, done: true};
    }

    const readLength = length + 4; // Need to read the CRC32C as well.
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
      // error msg: `Incomplete read; expected ${readLength} bytes, got ${bytesRead}`
      return {value: null, done: true};
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
      // error msg: `Incorrect record CRC32C`
      return {value: null, done: true};
    }

    let result: TensorContainerObject = {};
    messages.Example.deserializeBinary(recordData)
      .getFeatures()
      .getFeatureMap()
      .forEach((item: IFeature, index: string) => {
        const itemObj: IFeatureProto = item.toObject();
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
