/**
 * @license
 * Copyright 2018 Google Inc. All Rights Reserved.
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

import {toArrayBuffer, toBuffer} from './io_utils';

describe('toBuffer', () => {
  it('Simple case', () => {
    const ab = new Uint8Array([3, 2, 1]).buffer;
    const buffer = toBuffer(ab);
    expect(new Uint8Array(buffer)).toEqual(new Uint8Array([3, 2, 1]));
  });
});

describe('toArrayBuffer', () => {
  it('Single Buffer', () => {
    const buf = Buffer.from([10, 20, 30]);
    const ab = toArrayBuffer(buf);
    expect(new Uint8Array(ab)).toEqual(new Uint8Array([10, 20, 30]));
  });

  it('Two Buffers', () => {
    const buf1 = Buffer.from([10, 20, 30]);
    const buf2 = Buffer.from([40, 50, 60]);
    const ab = toArrayBuffer([buf1, buf2]);
    expect(new Uint8Array(ab)).toEqual(new Uint8Array([
      10, 20, 30, 40, 50, 60
    ]));
  });

  it('Three Buffers', () => {
    const buf1 = Buffer.from([10, 20, 30]);
    const buf2 = Buffer.from([40, 50, 60]);
    const buf3 = Buffer.from([3, 2, 1]);
    const ab = toArrayBuffer([buf1, buf2, buf3]);
    expect(new Uint8Array(ab)).toEqual(new Uint8Array([
      10, 20, 30, 40, 50, 60, 3, 2, 1
    ]));
  });

  it('Zero buffers', () => {
    const ab = toArrayBuffer([]);
    expect(new Uint8Array(ab)).toEqual(new Uint8Array([]));
  });
});
