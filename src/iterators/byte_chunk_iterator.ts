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

import * as utf8 from 'utf8';

import {LazyIterator, OneToManyIterator} from './lazy_iterator';
import {StringIterator} from './string_iterator';

export abstract class ByteChunkIterator extends LazyIterator<Uint8Array> {
  /**
   * Decode a stream of UTF8-encoded byte arrays to a stream of strings.
   *
   * The byte arrays producetd from the ByteChunkIterator on which this is
   * called will be interpreted as concatenated.  No assumptions are made about
   * the boundaries of the incoming chunks, so a multi-byte UTF8 encoding of a
   * character may span the boundary between chunks.  This naturally happens,
   * for instance, when reading fixed-size byte arrays from a file.
   */
  decodeUTF8(): StringIterator {
    return new Utf8Iterator(this);
  }
}

// ============================================================================
// The following private classes serve to implement the chainable methods
// on ByteChunkIterator.  Unfortunately they can't be placed in separate files,
// due to resulting trouble with circular imports.
// ============================================================================

// We wanted multiple inheritance, e.g.
//   class Utf8Iterator extends QueueIterator<string>, StringIterator
// but the TypeScript mixin approach is a bit hacky, so we take this adapter
// approach instead.

class Utf8Iterator extends StringIterator {
  private impl: Utf8IteratorImpl;

  constructor(upstream: LazyIterator<Uint8Array>) {
    super();
    this.impl = new Utf8IteratorImpl(upstream);
  }

  async next() {
    return this.impl.next();
  }
}

/**
 * Decode a stream of UTF8-encoded byte arrays to a stream of strings.
 *
 * This is tricky because the incoming byte array boundaries may disrupt a
 * multi-byte UTF8 character. Thus any incomplete character data at the end of
 * a chunk must be carried over and prepended to the next chunk before
 * decoding.
 *
 * In the context of an input pipeline for machine learning, UTF8 decoding is
 * needed to parse text files containing training examples or prediction
 * requests (e.g., formatted as CSV or JSON).  We cannot use the built-in
 * decoding provided by FileReader.readAsText() because here we are in a
 * streaming context, which FileReader does not support.
 *
 * @param upstream A `LazyIterator` of `Uint8Arrays` containing UTF8-encoded
 *   text, which should be interpreted as concatenated.  No assumptions are
 *   made about the boundaries of the incoming chunks, so a multi-byte UTF8
 *   encoding of a character may span the boundary between chunks.  This
 *   naturally happens, for instance, when reading fixed-size byte arrays from a
 *   file.
 */
class Utf8IteratorImpl extends OneToManyIterator<string> {
  // An array of the full required width of the split character, if any.
  partial: Uint8Array = new Uint8Array([]);
  // The number of bytes of that array that are populated so far.
  partialBytesValid = 0;

  constructor(protected readonly upstream: LazyIterator<Uint8Array>) {
    super();
  }

  async pump(): Promise<boolean> {
    const chunkResult = await this.upstream.next();
    let chunk;
    if (chunkResult.done) {
      if (this.partial.length === 0) {
        return false;
      }
      // Pretend that the pump succeeded in order to emit the small last batch.
      // The next pump() call will actually fail.
      chunk = new Uint8Array([]);
    } else {
      chunk = chunkResult.value;
    }
    const partialBytesRemaining = this.partial.length - this.partialBytesValid;
    let nextIndex = partialBytesRemaining;
    let okUpToIndex = nextIndex;
    let splitUtfWidth = 0;

    while (nextIndex < chunk.length) {
      okUpToIndex = nextIndex;
      splitUtfWidth = utfWidth(chunk[nextIndex]);
      nextIndex = okUpToIndex + splitUtfWidth;
    }
    if (nextIndex === chunk.length) {
      okUpToIndex = nextIndex;
    }

    // decode most of the chunk without copying it first
    const bulk: string = utf8.decode(String.fromCharCode.apply(
        null, chunk.slice(partialBytesRemaining, okUpToIndex)));

    if (partialBytesRemaining > 0) {
      // Reassemble the split character
      this.partial.set(
          chunk.slice(0, partialBytesRemaining), this.partialBytesValid);
      // Too bad about the string concat.
      const reassembled: string =
          utf8.decode(String.fromCharCode.apply(null, this.partial));
      this.outputQueue.push(reassembled + bulk);
    } else {
      this.outputQueue.push(bulk);
    }

    if (okUpToIndex === chunk.length) {
      this.partial = new Uint8Array([]);
      this.partialBytesValid = 0;
    } else {
      // prepare the next split character
      this.partial = new Uint8Array(new ArrayBuffer(splitUtfWidth));
      this.partial.set(chunk.slice(okUpToIndex), 0);
      this.partialBytesValid = chunk.length - okUpToIndex;
    }

    return true;
  }
}

function utfWidth(firstByte: number): number {
  if (firstByte >= 252) {
    return 6;
  } else if (firstByte >= 248) {
    return 5;
  } else if (firstByte >= 240) {
    return 4;
  } else if (firstByte >= 224) {
    return 3;
  } else if (firstByte >= 192) {
    return 2;
  } else {
    return 1;
  }
}
