/**
 * @license
 * Copyright 2017 Google Inc. All Rights Reserved.
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
import {describeWithFlags} from '../jasmine_util';
import {ALL_ENVS, expectArraysClose} from '../test_util';
import {Rank} from '../types';

describeWithFlags('conv2dTranspose', ALL_ENVS, () => {
  it('input=2x2x1,d2=1,f=2,s=1,p=0', () => {
    const origInputDepth = 1;
    const origOutputDepth = 1;
    const inputShape: [number, number, number] = [1, 1, origOutputDepth];
    const fSize = 2;
    const origPad = 0;
    const origStride = 1;

    const x = tf.tensor3d([2], inputShape);
    const w = tf.tensor4d(
        [3, 1, 5, 0], [fSize, fSize, origInputDepth, origOutputDepth]);

    const result = tf.conv2dTranspose(x, w, [2, 2, 1], origStride, origPad);
    const expected = [6, 2, 10, 0];

    expect(result.shape).toEqual([2, 2, 1]);
    expectArraysClose(result, expected);
  });

  it('input=2x2x1,d2=1,f=2,s=1,p=0, batch=2', () => {
    const origInputDepth = 1;
    const origOutputDepth = 1;
    const inputShape: [number, number, number, number] =
        [2, 1, 1, origOutputDepth];
    const fSize = 2;
    const origPad = 0;
    const origStride = 1;

    const x = tf.tensor4d([2, 3], inputShape);
    const w = tf.tensor4d(
        [3, 1, 5, 0], [fSize, fSize, origInputDepth, origOutputDepth]);

    const result = tf.conv2dTranspose(x, w, [2, 2, 2, 1], origStride, origPad);
    const expected = [6, 2, 10, 0, 9, 3, 15, 0];

    expect(result.shape).toEqual([2, 2, 2, 1]);
    expectArraysClose(result, expected);
  });

  it('throws when x is not rank 3', () => {
    const origInputDepth = 1;
    const origOutputDepth = 1;
    const fSize = 2;
    const origPad = 0;
    const origStride = 1;

    // tslint:disable-next-line:no-any
    const x: any = tf.tensor2d([2, 2], [2, 1]);
    const w = tf.tensor4d(
        [3, 1, 5, 0], [fSize, fSize, origInputDepth, origOutputDepth]);

    expect(() => tf.conv2dTranspose(x, w, [2, 2, 1], origStride, origPad))
        .toThrowError();
  });

  it('throws when weights is not rank 4', () => {
    const origInputDepth = 1;
    const origOutputDepth = 1;
    const inputShape: [number, number, number] = [1, 1, origOutputDepth];
    const fSize = 2;
    const origPad = 0;
    const origStride = 1;

    const x = tf.tensor3d([2], inputShape);
    // tslint:disable-next-line:no-any
    const w: any = tf.tensor3d([3, 1, 5, 0], [fSize, fSize, origInputDepth]);

    expect(() => tf.conv2dTranspose(x, w, [2, 2, 1], origStride, origPad))
        .toThrowError();
  });

  it('throws when x depth does not match weights original output depth', () => {
    const origInputDepth = 1;
    const origOutputDepth = 2;
    const wrongOrigOutputDepth = 3;
    const inputShape: [number, number, number] = [1, 1, origOutputDepth];
    const fSize = 2;
    const origPad = 0;
    const origStride = 1;

    const x = tf.tensor3d([2, 2], inputShape);
    const w = tf.randomNormal<Rank.R4>(
        [fSize, fSize, origInputDepth, wrongOrigOutputDepth]);

    expect(() => tf.conv2dTranspose(x, w, [2, 2, 2], origStride, origPad))
        .toThrowError();
  });

  it('throws when passed x as a non-tensor', () => {
    const origInputDepth = 1;
    const origOutputDepth = 1;
    const fSize = 2;
    const origPad = 0;
    const origStride = 1;

    const w = tf.tensor4d(
        [3, 1, 5, 0], [fSize, fSize, origInputDepth, origOutputDepth]);

    expect(
        () => tf.conv2dTranspose(
            {} as tf.Tensor3D, w, [2, 2, 1], origStride, origPad))
        .toThrowError(
            /Argument 'x' passed to 'conv2dTranspose' must be a Tensor/);
  });

  it('throws when passed filter as a non-tensor', () => {
    const origOutputDepth = 1;
    const inputShape: [number, number, number] = [1, 1, origOutputDepth];
    const origPad = 0;
    const origStride = 1;

    const x = tf.tensor3d([2], inputShape);

    expect(
        () => tf.conv2dTranspose(
            x, {} as tf.Tensor4D, [2, 2, 1], origStride, origPad))
        .toThrowError(
            /Argument 'filter' passed to 'conv2dTranspose' must be a Tensor/);
  });

  it('accepts a tensor-like object', () => {
    const origPad = 0;
    const origStride = 1;

    const x = [[[2]]];                           // 1x1x1
    const w = [[[[3]], [[1]]], [[[5]], [[0]]]];  // 2x2x1x1

    const result = tf.conv2dTranspose(x, w, [2, 2, 1], origStride, origPad);
    const expected = [6, 2, 10, 0];

    expect(result.shape).toEqual([2, 2, 1]);
    expectArraysClose(result, expected);
  });
});
