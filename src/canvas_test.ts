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

import * as tf from '@tensorflow/tfjs-core';

class MockContext {
  getImageData(x: number, y: number, width: number, height: number) {
    const data = new Uint8ClampedArray(width * height * 4);
    for (let i = 0; i < data.length; ++i) {
      data[i] = i + 1;
    }
    return {data};
  }
}

class MockCanvas {
  constructor(public width: number, public height: number) {}
  getContext(type: '2d'): MockContext {
    return new MockContext();
  }
}

describe('tf.browser.fromPixels with polyfills', () => {
  it('accepts a canvas-like element', () => {
    const c = new MockCanvas(2, 2);
    // tslint:disable-next-line:no-any
    const t = tf.browser.fromPixels(c as any);
    expect(t.dtype).toBe('int32');
    expect(t.shape).toEqual([2, 2, 3]);
    tf.test_util.expectArraysEqual(
        t, [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15]);
  });

  it('accepts a canvas-like element, numChannels=4', () => {
    const c = new MockCanvas(2, 2);
    // tslint:disable-next-line:no-any
    const t = tf.browser.fromPixels(c as any, 4);
    expect(t.dtype).toBe('int32');
    expect(t.shape).toEqual([2, 2, 4]);
    tf.test_util.expectArraysEqual(
        t, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
  });

  it('errors when passed a non-canvas object', () => {
    const c = 5;
    // tslint:disable-next-line:no-any
    expect(() => tf.browser.fromPixels(c as any))
        .toThrowError(
            /When running in node, pixels must be an HTMLCanvasElement/);
  });
});
