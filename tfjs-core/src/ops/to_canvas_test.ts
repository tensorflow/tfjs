/**
 * @license
 * Copyright 2021 Google Inc. All Rights Reserved.
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
import {ALL_ENVS, describeWithFlags} from '../jasmine_util';
import {expectPromiseToFail} from '../test_util';

function getDataFromCanvas(canvas: HTMLCanvasElement): Uint8ClampedArray {
  const testcanvas = document.createElement('canvas');
  const testContext = testcanvas.getContext('2d', {alpha: false});
  const width = canvas.width;
  const height = canvas.height;
  testContext.canvas.width = width;
  testContext.canvas.height = height;
  testContext.drawImage(canvas as HTMLCanvasElement, 0, 0, width, height);
  return new Uint8ClampedArray(
      testContext.getImageData(0, 0, width, height).data);
}

describeWithFlags('toCanvas', ALL_ENVS, () => {
  it('draws a rank-2 float32 tensor', async () => {
    const x = tf.tensor2d([.15, .2], [2, 1], 'float32');

    const canvas = await tf.browser.toCanvas(x);
    const data = getDataFromCanvas(canvas);
    const expected = new Uint8ClampedArray([
      Math.round(.15 * 255), Math.round(.15 * 255), Math.round(.15 * 255), 255,
      Math.round(.2 * 255), Math.round(.2 * 255), Math.round(.2 * 255), 255
    ]);
    expect(data).toEqual(expected);
  });

  it('draws a rank-2 int32 tensor', async () => {
    const x = tf.tensor2d([10, 20], [2, 1], 'int32');
    const canvas = await tf.browser.toCanvas(x);
    const data = getDataFromCanvas(canvas);

    const expected = new Uint8ClampedArray([10, 10, 10, 255, 20, 20, 20, 255]);
    expect(data).toEqual(expected);
  });

  it('draws a rank-3 float32 tensor, 1 channel', async () => {
    const x = tf.tensor3d([.15, .2], [2, 1, 1], 'float32');

    const canvas = await tf.browser.toCanvas(x);
    const data = getDataFromCanvas(canvas);

    const expected = new Uint8ClampedArray([
      Math.round(.15 * 255), Math.round(.15 * 255), Math.round(.15 * 255), 255,
      Math.round(.2 * 255), Math.round(.2 * 255), Math.round(.2 * 255), 255
    ]);
    expect(data).toEqual(expected);
  });

  it('draws a rank-3 int32 tensor, 1 channel', async () => {
    const x = tf.tensor3d([10, 20], [2, 1, 1], 'int32');

    const canvas = await tf.browser.toCanvas(x);
    const data = getDataFromCanvas(canvas);

    const expected = new Uint8ClampedArray([10, 10, 10, 255, 20, 20, 20, 255]);
    expect(data).toEqual(expected);
  });

  it('draws a rank-3 float32 tensor, 3 channel', async () => {
    // 0.1 and 0.3 are changed to 0.1001 and 0.3001 to avoid boundary conditions
    // such as Math.round(~25.5) which on Mobile Safari gives 25 and Desktop
    // gives 26.
    const x =
        tf.tensor3d([.05, .1001, .15, .2, .25, .3001], [2, 1, 3], 'float32');

    const canvas = await tf.browser.toCanvas(x);
    const data = getDataFromCanvas(canvas);
    const expected = new Uint8ClampedArray([
      Math.round(.05 * 255), Math.round(.1001 * 255), Math.round(.15 * 255),
      255, Math.round(.2 * 255), Math.round(.25 * 255), Math.round(.3001 * 255),
      255
    ]);
    expect(data).toEqual(expected);
  });

  it('draws a rank-3 int32 tensor, 3 channel', async () => {
    const x = tf.tensor3d([10, 20, 30, 40, 50, 60], [2, 1, 3], 'int32');

    const canvas = await tf.browser.toCanvas(x);
    const data = getDataFromCanvas(canvas);
    const expected = new Uint8ClampedArray([10, 20, 30, 255, 40, 50, 60, 255]);
    expect(data).toEqual(expected);
  });

  /*
  * For 2d convas, if we set premultiplyalpha as false,
  * it will always set alpha to 255. So we can't get correct value for alpha.
  * So temporary suppress theses two cases. Next I will use webgl context cnavas
  * as the test canvas so that we can get correct result.
  it('draws a rank-3 float32 tensor, 4 channel', async () => {
    const x = tf.tensor3d(
        [.05, .1001, .15, .2, .25, .3001, .35, .4], [2, 1, 4], 'float32');

    const canvas = await tf.browser.toCanvas(x);
    const data = getDataFromCanvas(canvas);
    const expected = new Uint8ClampedArray([
      Math.round(.05 * 255), Math.round(.1001 * 255), Math.round(.15 * 255),
      Math.round(.20 * 255), Math.round(.25 * 255), Math.round(.3001 * 255),
      Math.round(.35 * 255), Math.round(.4 * 255)
    ]);
    expect(data).toEqual(expected);
  });

  it('draws a rank-3 int32 tensor, 4 channel', async () => {
    const x = tf.tensor3d([10, 20, 30, 40, 50, 60, 70, 80], [2, 1, 4], 'int32');

    const canvas = await tf.browser.toCanvas(x);
    const data = getDataFromCanvas(canvas);
    const expected = new Uint8ClampedArray([10, 20, 30, 40, 50, 60, 70, 80]);
    expect(data).toEqual(expected);
  });
*/
  it('throws for scalars', done => {
    // tslint:disable-next-line:no-any
    expectPromiseToFail(() => tf.browser.toCanvas(tf.scalar(1) as any), done);
  });

  it('throws for rank-1 tensors', done => {
    expectPromiseToFail(
        // tslint:disable-next-line:no-any
        () => tf.browser.toCanvas(tf.tensor1d([1]) as any), done);
  });
  it('throws for rank-4 tensors', done => {
    expectPromiseToFail(
        // tslint:disable-next-line:no-any
        () => tf.browser.toCanvas(tf.tensor4d([1], [1, 1, 1, 1]) as any), done);
  });
  it('throws for bool dtype', done => {
    expectPromiseToFail(
        () => tf.browser.toCanvas(tf.tensor2d([1], [1, 1], 'bool')), done);
  });
  it('throws for rank-3 depth = 2', done => {
    expectPromiseToFail(
        () => tf.browser.toCanvas(tf.tensor3d([1, 2], [1, 1, 2])), done);
  });
  it('throws for rank-3 depth = 5', done => {
    expectPromiseToFail(
        () => tf.browser.toCanvas(tf.tensor3d([1, 2, 3, 4, 5], [1, 1, 5])),
        done);
  });
  it('throws when passed a non-tensor', done => {
    // tslint:disable-next-line:no-any
    expectPromiseToFail(() => tf.browser.toCanvas({} as any), done);
  });

  it('accepts a tensor-like object', async () => {
    const x = [[10], [20]];  // 2x1;
    const canvas = await tf.browser.toCanvas(x);
    const data = getDataFromCanvas(canvas);

    const expected = new Uint8ClampedArray([10, 10, 10, 255, 20, 20, 20, 255]);
    expect(data).toEqual(expected);
  });

  it('does not leak memory', async () => {
    const x = tf.tensor2d([[.1], [.2]], [2, 1]);
    const startNumTensors = tf.memory().numTensors;
    await tf.browser.toCanvas(x);
    expect(tf.memory().numTensors).toEqual(startNumTensors);
  });

  it('does not leak memory given a tensor-like object', async () => {
    const x = [[10], [20]];  // 2x1;
    const startNumTensors = tf.memory().numTensors;
    await tf.browser.toCanvas(x);
    expect(tf.memory().numTensors).toEqual(startNumTensors);
  });
});
