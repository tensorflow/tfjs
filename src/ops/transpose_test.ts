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
// tslint:disable-next-line:max-line-length
import {ALL_ENVS, expectArraysClose} from '../test_util';

describeWithFlags('transpose', ALL_ENVS, () => {
  it('of scalar is no-op', () => {
    const a = tf.scalar(3);
    expectArraysClose(tf.transpose(a), [3]);
  });

  it('of 1D is no-op', () => {
    const a = tf.tensor1d([1, 2, 3]);
    expectArraysClose(tf.transpose(a), [1, 2, 3]);
  });

  it('of scalar with perm of incorrect rank throws error', () => {
    const a = tf.scalar(3);
    const perm = [0];  // Should be empty array.
    expect(() => tf.transpose(a, perm)).toThrowError();
  });

  it('of 1d with perm out of bounds throws error', () => {
    const a = tf.tensor1d([1, 2, 3]);
    const perm = [1];
    expect(() => tf.transpose(a, perm)).toThrowError();
  });

  it('of 1d with perm incorrect rank throws error', () => {
    const a = tf.tensor1d([1, 2, 3]);
    const perm = [0, 0];  // Should be of length 1.
    expect(() => tf.transpose(a, perm)).toThrowError();
  });

  it('2D (no change)', () => {
    const t = tf.tensor2d([1, 11, 2, 22, 3, 33, 4, 44], [2, 4]);
    const t2 = tf.transpose(t, [0, 1]);

    expect(t2.shape).toEqual(t.shape);
    expectArraysClose(t2, t);
  });

  it('2D (transpose)', () => {
    const t = tf.tensor2d([1, 11, 2, 22, 3, 33, 4, 44], [2, 4]);
    const t2 = tf.transpose(t, [1, 0]);

    expect(t2.shape).toEqual([4, 2]);
    expectArraysClose(t2, [1, 3, 11, 33, 2, 4, 22, 44]);
  });

  it('3D [r, c, d] => [d, r, c]', () => {
    const t = tf.tensor3d([1, 11, 2, 22, 3, 33, 4, 44], [2, 2, 2]);
    const t2 = tf.transpose(t, [2, 0, 1]);

    expect(t2.shape).toEqual([2, 2, 2]);
    expectArraysClose(t2, [1, 2, 3, 4, 11, 22, 33, 44]);
  });

  it('3D [r, c, d] => [d, c, r]', () => {
    const t = tf.tensor3d([1, 11, 2, 22, 3, 33, 4, 44], [2, 2, 2]);
    const t2 = tf.transpose(t, [2, 1, 0]);

    expect(t2.shape).toEqual([2, 2, 2]);
    expectArraysClose(t2, [1, 3, 2, 4, 11, 33, 22, 44]);
  });

  it('5D [r, c, d, e, f] => [r, c, f, e, d]', () => {
    const t = tf.tensor5d([1, 11, 2, 22, 3, 33, 4, 44], [1, 1, 2, 2, 2]);
    const t2 = tf.transpose(t, [0, 1, 4, 3, 2]);

    expect(t2.shape).toEqual([1, 1, 2, 2, 2]);
    expectArraysClose(t2, [1, 3, 2, 4, 11, 33, 22, 44]);
  });

  it('5D [r, c, d, e, f] => [r, c, d, f, e]', () => {
    const t = tf.tensor5d([1, 11, 2, 22, 3, 33, 4, 44], [1, 1, 2, 2, 2]);
    const t2 = tf.transpose(t, [0, 1, 4, 2, 3]);

    expect(t2.shape).toEqual([1, 1, 2, 2, 2]);
    expectArraysClose(t2, [1, 2, 3, 4, 11, 22, 33, 44]);
  });

  it('5D [r, c, d, e, f] => [c, r, d, e, f]', () => {
    const t = tf.tensor5d([1, 11, 2, 22, 3, 33, 4, 44], [1, 1, 2, 2, 2]);
    const t2 = tf.transpose(t, [1, 0, 2, 3, 4]);

    expect(t2.shape).toEqual([1, 1, 2, 2, 2]);
    expectArraysClose(t2, [1, 11, 2, 22, 3, 33, 4, 44]);
  });

  it('6D [r, c, d, e, f, g] => [g, c, f, e, d, r]', () => {
    const t = tf.tensor6d(
        [1, 11, 2, 22, 3, 33, 4, 44, 1, 12, 3, 23, 4, 34, 5, 45],
        [1, 1, 2, 2, 2, 2]);
    const t2 = tf.transpose(t, [5, 1, 4, 3, 2, 0]);

    expect(t2.shape).toEqual([2, 1, 2, 2, 2, 1]);
    expectArraysClose(
        t2, [1, 1, 3, 4, 2, 3, 4, 5, 11, 12, 33, 34, 22, 23, 44, 45]);
  });

  it('6D [r, c, d, e, f, g] => [r, c, d, f, g, e]', () => {
    const t = tf.tensor6d(
        [1, 11, 2, 22, 3, 33, 4, 44, 1, 12, 3, 23, 4, 34, 5, 45],
        [1, 1, 2, 2, 2, 2]);
    const t2 = tf.transpose(t, [0, 1, 5, 2, 3, 4]);

    expect(t2.shape).toEqual([1, 1, 2, 2, 2, 2]);
    expectArraysClose(
        t2, [1, 2, 3, 4, 1, 3, 4, 5, 11, 22, 33, 44, 12, 23, 34, 45]);
  });

  it('6D [r, c, d, e, f, g] => [c, r, g, e, f, d]', () => {
    const t = tf.tensor6d(
        [1, 11, 2, 22, 3, 33, 4, 44, 1, 12, 3, 23, 4, 34, 5, 45],
        [1, 1, 2, 2, 2, 2]);
    const t2 = tf.transpose(t, [1, 0, 5, 3, 4, 2]);

    expect(t2.shape).toEqual([1, 1, 2, 2, 2, 2]);
    expectArraysClose(
        t2, [1, 1, 2, 3, 3, 4, 4, 5, 11, 12, 22, 23, 33, 34, 44, 45]);
  });

  it('gradient 3D [r, c, d] => [d, c, r]', () => {
    const t = tf.tensor3d([1, 11, 2, 22, 3, 33, 4, 44], [2, 2, 2]);
    const perm = [2, 1, 0];
    const dy = tf.tensor3d([111, 211, 121, 221, 112, 212, 122, 222], [2, 2, 2]);
    const dt = tf.grad(t => t.transpose(perm))(t, dy);
    expect(dt.shape).toEqual(t.shape);
    expect(dt.dtype).toEqual('float32');
    expectArraysClose(dt, [111, 112, 121, 122, 211, 212, 221, 222]);
  });

  it('throws when passed a non-tensor', () => {
    expect(() => tf.transpose({} as tf.Tensor))
        .toThrowError(/Argument 'x' passed to 'transpose' must be a Tensor/);
  });
});
