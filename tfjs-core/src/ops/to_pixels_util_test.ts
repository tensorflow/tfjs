/**
 * @license
 * Copyright 2023 Google Inc.
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

import * as tf from '../index';
import {expectPromiseToFail} from '../test_util';

function getDataFromCanvas(canvas: HTMLCanvasElement): Uint8ClampedArray {
  const testcanvas = document.createElement('canvas');
  const testContext = testcanvas.getContext('2d', {alpha: false});
  const width = canvas.width;
  const height = canvas.height;
  testContext.canvas.width = width;
  testContext.canvas.height = height;
  testContext.drawImage(canvas, 0, 0, width, height);
  return new Uint8ClampedArray(
      testContext.getImageData(0, 0, width, height).data);
}

export async function toPixelsWithCanvas(
    x: tf.Tensor2D|tf.Tensor3D|tf.TensorLike, genData: boolean) {
  let canvas = null;
  if (typeof document !== 'undefined') {
    canvas = document.createElement('canvas');
    if (genData) {
      await tf.browser.toPixels(x, canvas, true);
    } else {
      await tf.browser.toPixels(x, canvas, false);
    }
    return getDataFromCanvas(canvas);
  }
  return tf.browser.toPixels(x);
}

export async function toPixelsNoCanvas(x: tf.Tensor2D|tf.Tensor3D|
                                       tf.TensorLike) {
  return tf.browser.toPixels(x);
}

export function toPixelsTestCase(
    toPixels: (x: tf.Tensor2D|tf.Tensor3D|tf.TensorLike, genData: boolean) =>
        Promise<{}>,
    genData = false) {
  it('draws a rank-2 float32 tensor', async () => {
    const x = tf.tensor2d([.15, .2], [2, 1], 'float32');
    const data = await toPixels(x, genData);
    const expected = new Uint8ClampedArray([
      Math.round(.15 * 255), Math.round(.15 * 255), Math.round(.15 * 255), 255,
      Math.round(.2 * 255), Math.round(.2 * 255), Math.round(.2 * 255), 255
    ]);
    expect(data).toEqual(expected);
  });

  it('draws a rank-2 int32 tensor', async () => {
    const x = tf.tensor2d([10, 20], [2, 1], 'int32');
    const data = await toPixels(x, genData);

    const expected = new Uint8ClampedArray([10, 10, 10, 255, 20, 20, 20, 255]);
    expect(data).toEqual(expected);
  });

  it('draws a rank-3 float32 tensor, 1 channel', async () => {
    const x = tf.tensor3d([.15, .2], [2, 1, 1], 'float32');
    const data = await toPixels(x, genData);

    const expected = new Uint8ClampedArray([
      Math.round(.15 * 255), Math.round(.15 * 255), Math.round(.15 * 255), 255,
      Math.round(.2 * 255), Math.round(.2 * 255), Math.round(.2 * 255), 255
    ]);
    expect(data).toEqual(expected);
  });

  it('draws a rank-3 int32 tensor, 1 channel', async () => {
    const x = tf.tensor3d([10, 20], [2, 1, 1], 'int32');
    const data = await toPixels(x, genData);

    const expected = new Uint8ClampedArray([10, 10, 10, 255, 20, 20, 20, 255]);
    expect(data).toEqual(expected);
  });

  it('draws a rank-3 float32 tensor, 3 channel', async () => {
    // 0.1 and 0.3 are changed to 0.1001 and 0.3001 to avoid boundary conditions
    // such as Math.round(~25.5) which on Mobile Safari gives 25 and Desktop
    // gives 26.
    const x =
        tf.tensor3d([.05, .1001, .15, .2, .25, .3001], [2, 1, 3], 'float32');
    const data = await toPixels(x, genData);
    const expected = new Uint8ClampedArray([
      Math.round(.05 * 255), Math.round(.1001 * 255), Math.round(.15 * 255),
      255, Math.round(.2 * 255), Math.round(.25 * 255), Math.round(.3001 * 255),
      255
    ]);
    expect(data).toEqual(expected);
  });

  it('draws a rank-3 int32 tensor, 3 channel', async () => {
    const x = tf.tensor3d([10, 20, 30, 40, 50, 60], [2, 1, 3], 'int32');
    const data = await toPixels(x, genData);
    const expected = new Uint8ClampedArray([10, 20, 30, 255, 40, 50, 60, 255]);
    expect(data).toEqual(expected);
  });

  it('throws for scalars', done => {
    // tslint:disable-next-line:no-any
    const x = tf.scalar(1) as any;
    expectPromiseToFail(() => toPixels(x, genData), done);
  });

  it('throws for rank-1 tensors', done => {
    // tslint:disable-next-line:no-any
    const x = tf.tensor1d([1]) as any;
    expectPromiseToFail(() => toPixels(x, genData), done);
  });
  it('throws for rank-4 tensors', done => {
    // tslint:disable-next-line:no-any
    const x = tf.tensor4d([1], [1, 1, 1, 1]) as any;
    expectPromiseToFail(() => toPixels(x, genData), done);
  });
  it('throws for bool dtype', done => {
    const x = tf.tensor2d([1], [1, 1], 'bool');
    expectPromiseToFail(() => toPixels(x, genData), done);
  });
  it('throws for rank-3 depth = 2', done => {
    const x = tf.tensor3d([1, 2], [1, 1, 2]);
    expectPromiseToFail(() => toPixels(x, genData), done);
  });
  it('throws for rank-3 depth = 5', done => {
    const x = tf.tensor3d([1, 2, 3, 4, 5], [1, 1, 5]);
    expectPromiseToFail(() => toPixels(x, genData), done);
  });
  it('throws for float32 tensor with values not in [0 - 1]', done => {
    const x = tf.tensor2d([-1, .5], [1, 2]);
    expectPromiseToFail(() => toPixels(x, genData), done);
  });
  it('throws for int32 tensor with values not in [0 - 255]', done => {
    const x = tf.tensor2d([-1, 100], [1, 2], 'int32');
    expectPromiseToFail(() => toPixels(x, genData), done);
  });
  it('throws when passed a non-tensor', done => {
    // tslint:disable-next-line:no-any
    const x = {} as any;
    expectPromiseToFail(() => toPixels(x, genData), done);
  });

  it('accepts a tensor-like object', async () => {
    const x = [[10], [20]];  // 2x1;
    const data = await tf.browser.toPixels(x);

    const expected = new Uint8ClampedArray([10, 10, 10, 255, 20, 20, 20, 255]);
    expect(data).toEqual(expected);
  });

  it('does not leak memory', async () => {
    const x = tf.tensor2d([[.1], [.2]], [2, 1]);
    const startNumTensors = tf.memory().numTensors;
    await toPixels(x, genData);
    expect(tf.memory().numTensors).toEqual(startNumTensors);
  });

  it('does not leak memory given a tensor-like object', async () => {
    const x = [[10], [20]];  // 2x1;
    const startNumTensors = tf.memory().numTensors;
    await toPixels(x, genData);
    expect(tf.memory().numTensors).toEqual(startNumTensors);
  });
}
