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
 * =============================================================================
 */

/**
 * Convert an ArrayBuffer to a Buffer.
 */
export function toBuffer(ab: ArrayBuffer): Buffer {
  const view = new Uint8Array(ab);
  return Buffer.from(view);  // copies data
}

/**
 * Convert a Buffer or an Array of Buffers to an ArrayBuffer.
 *
 * If the input is an Array of Buffers, they will be concatenated in the
 * specified order to form the output ArrayBuffer.
 */
export function toArrayBuffer(buf: Buffer|Buffer[]): ArrayBuffer {
  if (Array.isArray(buf)) {
    // An Array of Buffers.
    let totalLength = 0;
    for (const buffer of buf) {
      totalLength += buffer.length;
    }

    const ab = new ArrayBuffer(totalLength);
    const view = new Uint8Array(ab);
    let pos = 0;
    for (const buffer of buf) {
      pos += buffer.copy(view, pos);
    }
    return ab;
  } else {
    // A single Buffer. Return a copy of the underlying ArrayBuffer slice.
    return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
  }
}
