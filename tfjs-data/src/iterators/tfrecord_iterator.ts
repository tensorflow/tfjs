import * as fs from 'fs';
import { promisify } from 'util';

import { maskedCrc32c } from '../util/protobuf/crc32c';
import { tensorflow } from '../util/protobuf/protos';
import { LazyIterator } from './lazy_iterator';

const fsRead = promisify(fs.read);
const fsClose = promisify(fs.close);

export class TFRecordIterator extends LazyIterator<any> {
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

  constructor(protected file: any, protected options: any) {
    super();
    this.closed = false;
    if (options.fd !== undefined) {
      this.fd = options.fd;
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

  async next(): Promise<IteratorResult<tensorflow.Example>> {
    // TODO: Only local is considered here
    if (this.closed) return { value: null, done: true };
    let { bytesRead } = await fsRead(this.fd, this.lengthAndCrcBuffer, 0, 12, null);
    if (bytesRead === 0) {
      fsClose(this.fd);
      this.closed = true;
      return { value: null, done: true };
    }
    if (bytesRead !== 12) {
      fsClose(this.fd);
      this.closed = true;
      return {
        value: `Incomplete read; expected 12 bytes, got ${bytesRead}`,
        done: true,
      };
    }

    const length = this.lengthAndCrc.getUint32(0, true);
    const length64 = this.lengthAndCrc.getUint32(4, true);
    const lengthCrc = this.lengthAndCrc.getUint32(8, true);

    if (length64 !== 0) {
      fsClose(this.fd);
      this.closed = true;
      return { value: `4GB+ records not supported`, done: true };
    }

    if (lengthCrc !== maskedCrc32c(this.lengthBuffer)) {
      fsClose(this.fd);
      this.closed = true;
      return { value: `Incorrect record length CRC32C`, done: true };
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

    ({ bytesRead } = await fsRead(
      this.fd,
      this.dataBuffer,
      0,
      readLength,
      null
    ));
    if (bytesRead !== readLength) {
      fsClose(this.fd);
      this.closed = true;
      return {
        value: `Incomplete read; expected ${readLength} bytes, got ${bytesRead}`,
        done: true,
      };
    }

    const recordData = new Uint8Array(this.dataBuffer.buffer, 0, length);
    const recordCrc = this.dataBufferView.getUint32(length, true);

    // TODO: Check CRC.
    const recordBuffer = Buffer.from(this.dataBuffer.buffer as ArrayBuffer, 0, length);
    if (recordCrc !== maskedCrc32c(recordBuffer)) {
      fsClose(this.fd);
      this.closed = true;
      return { value: `Incorrect record CRC32C`, done: true };
    }
    return { value: tensorflow.Example.decode(recordData), done: false };
  }
}
