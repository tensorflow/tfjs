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

// inspired by https://github.com/maxogden/filereader-stream
import {ENV, util} from '@tensorflow/tfjs-core';
import {FileElement} from '../types';
import {ByteChunkIterator} from './byte_chunk_iterator';

export interface FileChunkIteratorOptions {
  /** The byte offset at which to begin reading the File or Blob. Default 0. */
  offset?: number;
  /** The number of bytes to read at a time. Default 1MB. */
  chunkSize?: number;
}

/**
 * Provide a stream of chunks from a File, Blob, or Uint8Array.
 * @param file The source File, Blob or Uint8Array.
 * @param options Optional settings controlling file reading.
 * @returns a lazy Iterator of Uint8Arrays containing sequential chunks of the
 *   input File, Blob or Uint8Array.
 */
export class FileChunkIterator extends ByteChunkIterator {
  offset: number;
  chunkSize: number;

  constructor(
      protected file: FileElement,
      protected options: FileChunkIteratorOptions = {}) {
    super();
    util.assert(
        (file instanceof Uint8Array) ||
            (ENV.get('IS_BROWSER') ?
                 (file instanceof File || file instanceof Blob) :
                 false),
        () => 'FileChunkIterator only supports File, Blob and Uint8Array ' +
            'right now.');
    this.offset = options.offset || 0;
    // default 1MB chunk has tolerable perf on large files
    this.chunkSize = options.chunkSize || 1024 * 1024;
  }

  summary() {
    return `FileChunks ${this.file}`;
  }

  async next(): Promise<IteratorResult<Uint8Array>> {
    if (this.offset >= ((this.file instanceof Uint8Array) ?
                            this.file.byteLength :
                            this.file.size)) {
      return {value: null, done: true};
    }
    const chunk = new Promise<Uint8Array>((resolve, reject) => {
      const end = this.offset + this.chunkSize;
      if (this.file instanceof Uint8Array) {
        // Note if end > this.uint8Array.byteLength, we just get a small last
        // chunk.
        resolve(new Uint8Array(this.file.slice(this.offset, end)));
      } else {
        // This branch assumes that this.file type is File or Blob, which
        // means it is in the browser environment.

        // TODO(soergel): is this a performance issue?
        const fileReader = new FileReader();
        fileReader.onload = (event) => {
          let data: string|ArrayBuffer|Uint8Array = fileReader.result;
          // Not sure we can trust the return type of
          // FileReader.readAsArrayBuffer See e.g.
          // https://github.com/node-file-api/FileReader/issues/2
          if (data instanceof ArrayBuffer) {
            data = new Uint8Array(data);
          }
          if (!(data instanceof Uint8Array)) {
            return reject(new TypeError('FileReader returned unknown type.'));
          }
          resolve(data);
        };
        fileReader.onabort = (event) => {
          return reject(new Error('Aborted'));
        };
        fileReader.onerror = (event) => {
          return reject(new Error(event.type));
        };
        // TODO(soergel): better handle onabort, onerror
        // Note if end > this.file.size, we just get a small last chunk.
        const slice = this.file.slice(this.offset, end);
        // We can't use readAsText here (even if we know the file is text)
        // because the slice boundary may fall within a multi-byte character.
        fileReader.readAsArrayBuffer(slice);
      }
      this.offset = end;
    });
    return {value: (await chunk), done: false};
  }
}
