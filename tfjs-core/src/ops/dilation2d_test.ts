/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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
import {expectArraysClose} from '../test_util';

describeWithFlags('dilation2d', ALL_ENVS, () => {
  it('valid padding.', async () => {
    const inputShape: [number, number, number, number] = [1, 2, 2, 1];
    const filterShape: [number, number, number] = [2, 2, 1];
    const x = tf.tensor4d([.1, .2, .3, .4], inputShape);
    const filter = tf.tensor3d([.4, .3, .1, .0], filterShape);

    const result = tf.dilation2d(x, filter, 1 /* strides */, 'valid');

    expect(result.shape).toEqual([1, 1, 1, 1]);
    expectArraysClose(await result.data(), [.5]);
  });

  it('same padding.', async () => {
    const inputShape: [number, number, number, number] = [1, 2, 2, 1];
    const filterShape: [number, number, number] = [2, 2, 1];
    const x = tf.tensor4d([.1, .2, .3, .4], inputShape);
    const filter = tf.tensor3d([.4, .3, .1, .0], filterShape);

    const result = tf.dilation2d(x, filter, 1 /* strides */, 'same');

    expect(result.shape).toEqual([1, 2, 2, 1]);
    expectArraysClose(await result.data(), [.5, .6, .7, .8]);
  });

  it('same padding depth 3.', async () => {
    const inputShape: [number, number, number, number] = [1, 2, 2, 3];
    const filterShape: [number, number, number] = [2, 2, 3];
    const x = tf.tensor4d(
        [.1, .2, .0, .2, .3, .1, .3, .4, .2, .4, .5, .3], inputShape);
    const filter = tf.tensor3d(
        [.4, .5, .3, .3, .4, .2, .1, .2, .0, .0, .1, -.1], filterShape);

    const result = tf.dilation2d(x, filter, 1 /* strides */, 'same');

    expect(result.shape).toEqual([1, 2, 2, 3]);
    expectArraysClose(
        await result.data(), [.5, .7, .3, .6, .8, .4, .7, .9, .5, .8, 1., .6]);
  });

  it('same padding batch 2.', async () => {
    const inputShape: [number, number, number, number] = [2, 2, 2, 1];
    const filterShape: [number, number, number] = [2, 2, 1];
    const x = tf.tensor4d([.1, .2, .3, .4, .2, .3, .4, .5], inputShape);
    const filter = tf.tensor3d([.4, .3, .1, .0], filterShape);

    const result = tf.dilation2d(x, filter, 1 /* strides */, 'same');

    expect(result.shape).toEqual([2, 2, 2, 1]);
    expectArraysClose(await result.data(), [.5, .6, .7, .8, .6, .7, .8, .9]);
  });

  it('same padding filter 2.', async () => {
    const inputShape: [number, number, number, number] = [1, 3, 3, 1];
    const filterShape: [number, number, number] = [2, 2, 1];
    const x = tf.tensor4d([.1, .2, .3, .4, .5, .6, .7, .8, .9], inputShape);
    const filter = tf.tensor3d([.4, .3, .1, .2], filterShape);

    const result = tf.dilation2d(x, filter, 1 /* strides */, 'same');

    expect(result.shape).toEqual([1, 3, 3, 1]);
    expectArraysClose(
        await result.data(), [.7, .8, .7, 1, 1.1, 1, 1.1, 1.2, 1.3]);
  });

  it('valid padding non-square-window.', async () => {
    const inputShape: [number, number, number, number] = [1, 2, 2, 1];
    const filterShape: [number, number, number] = [1, 2, 1];
    const x = tf.tensor4d([.1, .2, .3, .4], inputShape);
    const filter = tf.tensor3d([.4, .3], filterShape);

    const result = tf.dilation2d(x, filter, 1 /* strides */, 'valid');

    expect(result.shape).toEqual([1, 2, 1, 1]);
    expectArraysClose(await result.data(), [.5, .7]);
  });

  it('same padding dilations 2.', async () => {
    const inputShape: [number, number, number, number] = [1, 3, 3, 1];
    const filterShape: [number, number, number] = [2, 2, 1];
    const x = tf.tensor4d([.1, .2, .3, .4, .5, .6, .7, .8, .9], inputShape);
    const filter = tf.tensor3d([.4, .3, .1, .2], filterShape);

    const result = tf.dilation2d(x, filter, 1 /* strides */, 'same', 2);

    // Because dilations = 2, the effective filter is [3, 3, 1]:
    // filter_eff = [[[.4], [.0], [.3]],
    //               [[.0], [.0], [.0]],
    //               [[.1], [.0], [.2]]]
    expect(result.shape).toEqual([1, 3, 3, 1]);
    expectArraysClose(
        await result.data(), [.7, .8, .6, 1., 1.1, .9, .8, .9, .9]);
  });

  it('valid padding uneven stride.', async () => {
    const inputShape: [number, number, number, number] = [1, 3, 4, 1];
    const filterShape: [number, number, number] = [2, 2, 1];
    const x = tf.tensor4d(
        [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1., 1.1, 1.2], inputShape);
    const filter = tf.tensor3d([.4, .3, .1, .2], filterShape);

    const result = tf.dilation2d(x, filter, [1, 2] /* strides */, 'valid');

    expect(result.shape).toEqual([1, 2, 2, 1]);
    expectArraysClose(await result.data(), [.8, 1., 1.2, 1.4]);
  });

  it('throws when input rank is not 3 or 4.', async () => {
    const filterShape: [number, number, number] = [1, 1, 1];
    // tslint:disable-next-line:no-any
    const x: any = tf.tensor1d([.5]);
    const filter = tf.tensor3d([.4], filterShape);

    expect(() => tf.dilation2d(x, filter, 1, 'valid')).toThrowError();
  });

  it('thorws when filter is not rank 3.', async () => {
    const inputShape: [number, number, number, number] = [1, 2, 2, 1];
    const filterShape: [number, number] = [2, 2];
    const x = tf.tensor4d([.1, .2, .3, .4], inputShape);
    // tslint:disable-next-line:no-any
    const filter: any = tf.tensor2d([.4, .3, .1, .0], filterShape);

    expect(() => tf.dilation2d(x, filter, 1, 'valid')).toThrowError();
  });

  it('throws when data format is not NHWC.', async () => {
    const inputShape: [number, number, number, number] = [1, 2, 2, 1];
    const filterShape: [number, number, number] = [2, 2, 1];
    const x = tf.tensor4d([.1, .2, .3, .4], inputShape);
    const filter = tf.tensor3d([.4, .3, .1, .0], filterShape);
    // tslint:disable-next-line:no-any
    const dataFormat: any = 'NCHW';

    expect(
        () => tf.dilation2d(x, filter, 1 /* strides */, 'valid', 1, dataFormat))
        .toThrowError();
  });

  it('dilation gradient valid padding.', async () => {
    const inputShape: [number, number, number, number] = [1, 3, 3, 1];
    const filterShape: [number, number, number] = [1, 1, 1];
    const x = tf.tensor4d([.1, .2, .3, .4, .5, .6, .7, .8, .9], inputShape);
    const filter = tf.tensor3d([.5], filterShape);
    const dy = tf.tensor4d([.2, .3, .4, .2, .1, 1., .2, .3, .4], inputShape);

    const grads = tf.grads(
        (x: tf.Tensor4D, filter: tf.Tensor3D) =>
            x.dilation2d(filter, 1, 'valid'));

    const [dx, dfilter] = grads([x, filter], dy);

    expect(dx.shape).toEqual(x.shape);
    expectArraysClose(await dx.data(), [.2, .3, .4, .2, .1, 1., .2, .3, .4]);

    expect(dfilter.shape).toEqual(filterShape);
    expectArraysClose(await dfilter.data(), [3.1]);
  });

  it('dilation gradient same padding.', async () => {
    const inputShape: [number, number, number, number] = [1, 3, 3, 1];
    const filterShape: [number, number, number] = [1, 1, 1];
    const x = tf.tensor4d([.1, .2, .3, .4, .5, .6, .7, .8, .9], inputShape);
    const filter = tf.tensor3d([.5], filterShape);
    const dy = tf.tensor4d([.2, .3, .4, .2, .1, 1., .2, .3, .4], inputShape);

    const grads = tf.grads(
        (x: tf.Tensor4D, filter: tf.Tensor3D) =>
            x.dilation2d(filter, 1, 'same'));

    const [dx, dfilter] = grads([x, filter], dy);

    expect(dx.shape).toEqual(x.shape);
    expectArraysClose(await dx.data(), [.2, .3, .4, .2, .1, 1., .2, .3, .4]);

    expect(dfilter.shape).toEqual(filterShape);
    expectArraysClose(await dfilter.data(), [3.1]);
  });

  it('dilation gradient same padding depth 2.', async () => {
    const inputShape: [number, number, number, number] = [1, 2, 2, 3];
    const filterShape: [number, number, number] = [1, 1, 3];
    const x = tf.tensor4d(
        [.1, .2, .0, .2, .3, .1, .3, .4, .2, .4, .5, .3], inputShape);
    const filter = tf.tensor3d([.4, .5, .6], filterShape);
    const dy = tf.tensor4d(
        [.2, .3, .4, .2, .1, 1., .2, .3, .4, .8, -.1, .1], inputShape);

    const grads = tf.grads(
        (x: tf.Tensor4D, filter: tf.Tensor3D) =>
            x.dilation2d(filter, 1, 'same'));

    const [dx, dfilter] = grads([x, filter], dy);

    expect(dx.shape).toEqual(x.shape);
    expectArraysClose(
        await dx.data(), [.2, .3, .4, .2, .1, 1., .2, .3, .4, .8, -.1, .1]);

    expect(dfilter.shape).toEqual(filterShape);
    expectArraysClose(await dfilter.data(), [1.4, .6, 1.9]);
  });

  it('dilation gradient valid padding filter 2.', async () => {
    const inputShape: [number, number, number, number] = [1, 3, 3, 1];
    const filterShape: [number, number, number] = [2, 2, 1];
    const dyShape: [number, number, number, number] = [1, 2, 2, 1];
    const x = tf.tensor4d([.1, .2, .3, .4, .5, .6, .7, .8, .9], inputShape);
    const filter = tf.tensor3d([.4, .3, .1, .2], filterShape);
    const dy = tf.tensor4d([.2, .3, .4, .2], dyShape);

    const grads = tf.grads(
        (x: tf.Tensor4D, filter: tf.Tensor3D) =>
            x.dilation2d(filter, 1, 'valid'));

    const [dx, dfilter] = grads([x, filter], dy);

    expect(dx.shape).toEqual(x.shape);
    expectArraysClose(await dx.data(), [0, 0, 0, 0, .2, .3, 0, .4, .2]);

    expect(dfilter.shape).toEqual(filterShape);
    expectArraysClose(await dfilter.data(), [0, 0, 0, 1.1]);
  });

  it('dilation gradient same padding filter 2 depth 3.', async () => {
    const inputShape: [number, number, number, number] = [1, 3, 3, 3];
    const filterShape: [number, number, number] = [2, 2, 3];
    const x = tf.tensor4d(
        [
          .1, .2, .3, .4, .5, .6, .7, .8, .9, .3, .2, .3, .4, .5,
          .1, .9, .6, .3, .4, .5, .6, .2, .3, .5, .1, .2, .3
        ],
        inputShape);
    const filter = tf.tensor3d(
        [.4, .3, .1, .2, .2, .1, .7, .3, .8, .4, .9, .1], filterShape);
    const dy = tf.tensor4d(
        [
          .2, .3, .4, .2, .1, .5, 0,  .8, .7, .1, .2, .1, .2, .3,
          .4, .5, .6, .6, .6, .7, .8, .3, .2, .1, .2, .4, .2
        ],
        inputShape);

    const grads = tf.grads(
        (x: tf.Tensor4D, filter: tf.Tensor3D) =>
            x.dilation2d(filter, 1, 'same'));

    const [dx, dfilter] = grads([x, filter], dy);

    expect(dx.shape).toEqual(x.shape);
    expectArraysClose(await dx.data(), [
      0, 0,  0,  0,  0,  0,  0,  .8, .5, .2, 0,  .4, 0, .3,
      0, .9, .7, .7, .7, .7, .9, .3, .4, .5, .2, .7, .8
    ]);

    expect(dfilter.shape).toEqual(filterShape);
    expectArraysClose(
        await dfilter.data(),
        [1.6, 2.7, 1.1, .2, 0, .5, .3, 0, 2.2, .2, .9, 0]);
  });
});
