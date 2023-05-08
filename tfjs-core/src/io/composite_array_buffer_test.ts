/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
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
import {expectArraysEqual} from '../test_util';
import {CompositeArrayBuffer} from './composite_array_buffer';

describe('CompositeArrayBuffer', () => {
  const uniformBuffers = [
    new Uint8Array([0, 1, 2, 3]).buffer,
    new Uint8Array([4, 5, 6, 7]).buffer,
    new Uint8Array([8, 9, 10, 11]).buffer,
    new Uint8Array([12, 13, 14, 15]).buffer,
    new Uint8Array([16]).buffer,
  ];

  const nonUniformBuffers = [
    new Uint8Array([0, 1, 2]).buffer,
    new Uint8Array([3, 4, 5, 6, 7]).buffer,
    new Uint8Array([8, 9, 10, 11]).buffer,
    new Uint8Array([12, 13, 14, 15, 16]).buffer,
  ];

  const bufferTestCases = [
    ['uniform', uniformBuffers],
    ['non-uniform', nonUniformBuffers]
  ] as const;

  for (const [buffersType, buffers] of bufferTestCases) {
    let composite: CompositeArrayBuffer;
    beforeEach(() => {
      composite = new CompositeArrayBuffer(buffers);
    });

    it(`${buffersType}: slices across multiple buffers`, () => {
      expectArraysEqual(new Uint8Array(composite.slice(1, 13)),
                        [1,2,3,4,5,6,7,8,9,10,11,12]);
    });

    it(`${buffersType}: slices to the end of the array when \'end\' is not ` +
      'specified', () => {
        expectArraysEqual(new Uint8Array(composite.slice(5)),
                          [5,6,7,8,9,10,11,12,13,14,15,16]);
      });

    it(`${buffersType}: makes a copy when slice() is called with no arguments`,
       () => {
         expectArraysEqual(new Uint8Array(composite.slice()),
                           [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]);
       });

    it(`${buffersType}: slices from zero when start is negative`, () => {
      expectArraysEqual(new Uint8Array(composite.slice(-4, 5)),
                        [0,1,2,3,4]);
    });

    it(`${buffersType}: slices to the end when end is greater than length`,
       () => {
         expectArraysEqual(new Uint8Array(composite.slice(7, 1000)),
                           [7,8,9,10,11,12,13,14,15,16]);
       });

    it(`${buffersType}: slices multiple ranges out of order`, () => {
      expectArraysEqual(new Uint8Array(composite.slice(13, 15)), [13, 14]);
      expectArraysEqual(new Uint8Array(composite.slice(0, 2)), [0, 1]);
      expectArraysEqual(new Uint8Array(composite.slice(9, 13)),
                        [9, 10, 11, 12]);
    });
  }

  it('can be created from an empty arraybuffer', () => {
    const array = new Uint8Array([]);
    const singleComposite = new CompositeArrayBuffer(array.buffer);
    expectArraysEqual(new Uint8Array(singleComposite.slice()), []);
  });

  it('can be created from a single array', () => {
    const array = new Uint8Array([1,2,3]);
    const singleComposite = new CompositeArrayBuffer(array.buffer);
    expectArraysEqual(new Uint8Array(singleComposite.slice()), array);
  });

  it('can be created from zero arrays', () => {
    const singleComposite = new CompositeArrayBuffer([]);
    expectArraysEqual(new Uint8Array(singleComposite.slice()),
                      new Uint8Array());
  });

  it('can be created from undefined input', () => {
    const singleComposite = new CompositeArrayBuffer();
    expectArraysEqual(new Uint8Array(singleComposite.slice()),
                      new Uint8Array());
  });

  it('treats NaN as zero when passed as the start of slice', () => {
    const array = new Uint8Array([1,2,3]);
    const composite = new CompositeArrayBuffer(array.buffer);
    expectArraysEqual(new Uint8Array(composite.slice(NaN, 2)), [1,2]);
  });

  it('treats NaN as zero when passed as the end of slice', () => {
    const array = new Uint8Array([1,2,3]);
    const composite = new CompositeArrayBuffer(array.buffer);
    expectArraysEqual(new Uint8Array(composite.slice(0, NaN)), []);
  });

  it('supports TypedArray input', () => {
    // This support is necessary for some tests in tfjs-converter. Maybe those
    // tests are misconfigured?
    const array = new Uint8Array([1,2,3]);
    const composite = new CompositeArrayBuffer(array);
    expectArraysEqual(new Uint8Array(composite.slice(0, 2)), [1,2]);
  });
});
