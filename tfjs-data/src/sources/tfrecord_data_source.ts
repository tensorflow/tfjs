import * as fs from 'fs';

import { TFRecordIterator } from '../iterators/tfrecord_iterator';

export class TFRecordDataSource {
  /**
   * Create a `TFRecordDataSource`.
   *
   * @param input Local file path, eg: `./path to file`.
   *     Only works in node environment.
   */
  constructor(protected input: any) {}

  async iterator(): Promise<TFRecordIterator> {
    // TODO: Only local file is considered here
    const fd = fs.openSync(this.input, 'r');
    return new TFRecordIterator(this.input, { fd: fd });
  }
}
