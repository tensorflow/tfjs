/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
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
import {Rank} from '../types';

describeWithFlags('conv1d', ALL_ENVS, () => {
  it('conv1d input=2x2x1,d2=1,f=1,s=1,d=1,p=explicit', async () => {
    const inputDepth = 1;
    const inputShape: [number, number, number] = [2, 2, inputDepth];
    const outputDepth = 1;
    const fSize = 1;
    const pad =
        [[0, 0], [0, 0], [0, 0], [0, 0]] as tf.backend_util.ExplicitPadding;
    const stride = 1;
    const dataFormat = 'NWC';
    const dilation = 1;

    const x = tf.tensor3d([1, 2, 3, 4], inputShape);
    const w = tf.tensor3d([3], [fSize, inputDepth, outputDepth]);

    const result = tf.conv1d(x, w, stride, pad, dataFormat, dilation);

    expect(result.shape).toEqual([2, 2, 1]);
    expectArraysClose(await result.data(), [3, 6, 9, 12]);
  });

  it('conv1d input=2x2x1,d2=1,f=1,s=1,d=1,p=same', async () => {
    const inputDepth = 1;
    const inputShape: [number, number, number] = [2, 2, inputDepth];
    const outputDepth = 1;
    const fSize = 1;
    const pad = 'same';
    const stride = 1;
    const dataFormat = 'NWC';
    const dilation = 1;

    const x = tf.tensor3d([1, 2, 3, 4], inputShape);
    const w = tf.tensor3d([3], [fSize, inputDepth, outputDepth]);

    const result = tf.conv1d(x, w, stride, pad, dataFormat, dilation);

    expect(result.shape).toEqual([2, 2, 1]);
    expectArraysClose(await result.data(), [3, 6, 9, 12]);
  });

  it('conv1d input=4x1,d2=1,f=2x1x1,s=1,d=1,p=valid', async () => {
    const inputDepth = 1;
    const inputShape: [number, number] = [4, inputDepth];
    const outputDepth = 1;
    const fSize = 2;
    const pad = 'valid';
    const stride = 1;
    const dataFormat = 'NWC';
    const dilation = 1;

    const x = tf.tensor2d([1, 2, 3, 4], inputShape);
    const w = tf.tensor3d([2, 1], [fSize, inputDepth, outputDepth]);

    const result = tf.conv1d(x, w, stride, pad, dataFormat, dilation);

    expect(result.shape).toEqual([3, 1]);
    expectArraysClose(await result.data(), [4, 7, 10]);
  });

  it('conv1d input=4x1,d2=1,f=2x1x1,s=1,d=2,p=valid', async () => {
    const inputDepth = 1;
    const inputShape: [number, number] = [4, inputDepth];
    const outputDepth = 1;
    const fSize = 2;
    const fSizeDilated = 3;
    const pad = 'valid';
    const stride = 1;
    const dataFormat = 'NWC';
    const dilation = 2;
    const dilationWEffective = 1;

    const x = tf.tensor2d([1, 2, 3, 4], inputShape);
    const w = tf.tensor3d([2, 1], [fSize, inputDepth, outputDepth]);
    // adding a dilation rate is equivalent to using a filter
    // with 0s for the dilation rate
    const wDilated =
        tf.tensor3d([2, 0, 1], [fSizeDilated, inputDepth, outputDepth]);

    const result = tf.conv1d(x, w, stride, pad, dataFormat, dilation);
    const expectedResult =
        tf.conv1d(x, wDilated, stride, pad, dataFormat, dilationWEffective);

    expect(result.shape).toEqual(expectedResult.shape);
    expectArraysClose(await result.data(), await expectedResult.data());
  });

  it('conv1d input=14x1,d2=1,f=3x1x1,s=1,d=3,p=valid', async () => {
    const inputDepth = 1;
    const inputShape: [number, number] = [14, inputDepth];
    const outputDepth = 1;
    const fSize = 3;
    const fSizeDilated = 7;
    const pad = 'valid';
    const stride = 1;
    const dataFormat = 'NWC';
    const dilation = 3;
    const dilationWEffective = 1;

    const x = tf.tensor2d(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], inputShape);
    const w = tf.tensor3d([3, 2, 1], [fSize, inputDepth, outputDepth]);
    // adding a dilation rate is equivalent to using a filter
    // with 0s for the dilation rate
    const wDilated = tf.tensor3d(
        [3, 0, 0, 2, 0, 0, 1], [fSizeDilated, inputDepth, outputDepth]);

    const result = tf.conv1d(x, w, stride, pad, dataFormat, dilation);
    const expectedResult =
        tf.conv1d(x, wDilated, stride, pad, dataFormat, dilationWEffective);

    expect(result.shape).toEqual(expectedResult.shape);
    expectArraysClose(await result.data(), await expectedResult.data());
  });

  it('TensorLike', async () => {
    const pad = 'same';
    const stride = 1;
    const dataFormat = 'NWC';
    const dilation = 1;

    const x = [[[1], [2]], [[3], [4]]];
    const w = [[[3]]];

    const result = tf.conv1d(x, w, stride, pad, dataFormat, dilation);

    expect(result.shape).toEqual([2, 2, 1]);
    expectArraysClose(await result.data(), [3, 6, 9, 12]);
  });
  it('TensorLike Chained', async () => {
    const inputDepth = 1;
    const inputShape: [number, number, number] = [2, 2, inputDepth];
    const pad = 'same';
    const stride = 1;
    const dataFormat = 'NWC';
    const dilation = 1;

    const x = tf.tensor3d([1, 2, 3, 4], inputShape);
    const w = [[[3]]];

    const result = x.conv1d(w, stride, pad, dataFormat, dilation);

    expect(result.shape).toEqual([2, 2, 1]);
    expectArraysClose(await result.data(), [3, 6, 9, 12]);
  });

  it('throws when x is not rank 3', () => {
    const inputDepth = 1;
    const outputDepth = 1;
    const fSize = 2;
    const pad = 0;
    const stride = 1;
    const dataFormat = 'NWC';
    const dilation = 1;

    // tslint:disable-next-line:no-any
    const x: any = tf.tensor2d([1, 2, 3, 4], [2, 2]);
    const w = tf.tensor3d([3, 1], [fSize, inputDepth, outputDepth]);

    expect(() => tf.conv1d(x, w, stride, pad, dataFormat, dilation))
        .toThrowError();
  });

  it('throws when weights is not rank 3', () => {
    const inputDepth = 1;
    const inputShape: [number, number, number] = [2, 2, inputDepth];
    const pad = 0;
    const stride = 1;
    const dataFormat = 'NWC';
    const dilation = 1;

    const x = tf.tensor3d([1, 2, 3, 4], inputShape);
    // tslint:disable-next-line:no-any
    const w: any = tf.tensor4d([3, 1, 5, 0], [2, 2, 1, 1]);

    expect(() => tf.conv1d(x, w, stride, pad, dataFormat, dilation))
        .toThrowError();
  });

  it('throws when x depth does not match weight depth', () => {
    const inputDepth = 1;
    const wrongInputDepth = 5;
    const inputShape: [number, number, number] = [2, 2, inputDepth];
    const outputDepth = 1;
    const fSize = 2;
    const pad = 0;
    const stride = 1;
    const dataFormat = 'NWC';
    const dilation = 1;

    const x = tf.tensor3d([1, 2, 3, 4], inputShape);
    const w = tf.randomNormal<Rank.R3>([fSize, wrongInputDepth, outputDepth]);

    expect(() => tf.conv1d(x, w, stride, pad, dataFormat, dilation))
        .toThrowError();
  });

  it('throws when both stride and dilation are greater than 1', () => {
    const inputDepth = 1;
    const inputShape: [number, number, number] = [2, 2, inputDepth];
    const outputDepth = 1;
    const fSize = 1;
    const pad = 'same';
    const stride = 2;
    const dataFormat = 'NWC';
    const dilation = 2;

    const x = tf.tensor3d([1, 2, 3, 4], inputShape);
    const w = tf.tensor3d([3], [fSize, inputDepth, outputDepth]);

    expect(() => tf.conv1d(x, w, stride, pad, dataFormat, dilation))
        .toThrowError();
  });

  it('throws when passed x as a non-tensor', () => {
    const inputDepth = 1;
    const outputDepth = 1;
    const fSize = 1;
    const pad = 'same';
    const stride = 2;
    const dataFormat = 'NWC';
    const dilation = 2;

    const w = tf.tensor3d([3], [fSize, inputDepth, outputDepth]);

    expect(
        () =>
            tf.conv1d({} as tf.Tensor3D, w, stride, pad, dataFormat, dilation))
        .toThrowError(/Argument 'x' passed to 'conv1d' must be a Tensor/);
  });

  it('throws when passed filter as a non-tensor', () => {
    const inputDepth = 1;
    const inputShape: [number, number, number] = [2, 2, inputDepth];
    const pad = 'same';
    const stride = 2;
    const dataFormat = 'NWC';
    const dilation = 2;

    const x = tf.tensor3d([1, 2, 3, 4], inputShape);

    expect(
        () =>
            tf.conv1d(x, {} as tf.Tensor3D, stride, pad, dataFormat, dilation))
        .toThrowError(/Argument 'filter' passed to 'conv1d' must be a Tensor/);
  });

  it('accepts a tensor-like object', async () => {
    const pad = 'same';
    const stride = 1;
    const dataFormat = 'NWC';
    const dilation = 1;
    const x = [[[1], [2]], [[3], [4]]];  // 2x2x1
    const w = [[[3]]];                   // 1x1x1

    const result = tf.conv1d(x, w, stride, pad, dataFormat, dilation);

    expect(result.shape).toEqual([2, 2, 1]);
    expectArraysClose(await result.data(), [3, 6, 9, 12]);
  });

  it('gradient with clones, input=2x2x1,d2=1,f=1,s=1,d=1,p=same', async () => {
    const inputDepth = 1;
    const inputShape: [number, number, number] = [2, 2, inputDepth];
    const outputDepth = 1;
    const fSize = 1;
    const filterShape: [number, number, number] =
        [fSize, inputDepth, outputDepth];
    const pad = 'same';
    const stride = 1;
    const dataFormat = 'NWC';
    const dilation = 1;

    const x = tf.tensor3d([1, 2, 3, 4], inputShape);
    const w = tf.tensor3d([3], filterShape);

    const dy = tf.tensor3d([3, 2, 1, 0], inputShape);

    const grads = tf.grads(
        (x: tf.Tensor3D, w: tf.Tensor3D) =>
            tf.conv1d(x.clone(), w.clone(), stride, pad, dataFormat, dilation)
                .clone());
    const [dx, dw] = grads([x, w], dy);

    expect(dx.shape).toEqual(x.shape);
    expectArraysClose(await dx.data(), [9, 6, 3, 0]);

    expect(dw.shape).toEqual(w.shape);
    expectArraysClose(await dw.data(), [10]);
  });

  it('conv1d gradients input=14x1,d2=1,f=3x1x1,s=1,p=valid', async () => {
    const inputDepth = 1;
    const inputShape: [number, number] = [14, inputDepth];

    const outputDepth = 1;
    const fSize = 3;
    const pad = 'valid';
    const stride = 1;
    const dataFormat = 'NWC';

    const x = tf.tensor2d(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], inputShape);
    const w = tf.tensor3d([3, 2, 1], [fSize, inputDepth, outputDepth]);

    const dy =
        tf.tensor2d([3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0], [12, inputDepth]);

    const grads = tf.grads(
        (x: tf.Tensor2D, w: tf.Tensor3D) =>
            tf.conv1d(x, w, stride, pad, dataFormat));
    const [dx, dw] = grads([x, w], dy);

    expect(dx.shape).toEqual(x.shape);
    expectArraysClose(
        await dx.data(), [9, 12, 10, 4, 10, 12, 10, 4, 10, 12, 10, 4, 1, 0]);

    expect(dw.shape).toEqual(w.shape);
    expectArraysClose(await dw.data(), [102, 120, 138]);
  });
});
