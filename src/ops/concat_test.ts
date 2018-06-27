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

describeWithFlags('concat1d', ALL_ENVS, () => {
  it('3 + 5', () => {
    const a = tf.tensor1d([3]);
    const b = tf.tensor1d([5]);

    const result = tf.concat1d([a, b]);
    const expected = [3, 5];
    expectArraysClose(result, expected);
  });

  it('3 + [5,7]', () => {
    const a = tf.tensor1d([3]);
    const b = tf.tensor1d([5, 7]);

    const result = tf.concat1d([a, b]);
    const expected = [3, 5, 7];
    expectArraysClose(result, expected);
  });

  it('[3,5] + 7', () => {
    const a = tf.tensor1d([3, 5]);
    const b = tf.tensor1d([7]);

    const result = tf.concat1d([a, b]);
    const expected = [3, 5, 7];
    expectArraysClose(result, expected);
  });

  it('3 + 5 + 7 + 9', () => {
    const a = tf.tensor1d([3]);
    const b = tf.tensor1d([5]);
    const c = tf.tensor1d([7]);
    const d = tf.tensor1d([9]);

    const result = tf.concat1d([a, b, c, d]);
    expectArraysClose(result, [3, 5, 7, 9]);
  });

  it('single tensor', () => {
    const a = tf.tensor1d([3]);

    const result = tf.concat1d([a]);
    expectArraysClose(result, [3]);
  });

  it('accepts a tensor-like object', () => {
    const a = [3];
    const b = [5];

    const result = tf.concat1d([a, b]);
    const expected = [3, 5];
    expectArraysClose(result, expected);
  });
});

describeWithFlags('concat2d', ALL_ENVS, () => {
  it('[[3]] + [[5]], axis=0', () => {
    const axis = 0;
    const a = tf.tensor2d([3], [1, 1]);
    const b = tf.tensor2d([5], [1, 1]);

    const result = tf.concat2d([a, b], axis);
    const expected = [3, 5];

    expect(result.shape).toEqual([2, 1]);
    expectArraysClose(result, expected);
  });

  it('[[3]] + [[5]], axis=1', () => {
    const axis = 1;
    const a = tf.tensor2d([3], [1, 1]);
    const b = tf.tensor2d([5], [1, 1]);

    const result = tf.concat2d([a, b], axis);
    const expected = [3, 5];

    expect(result.shape).toEqual([1, 2]);
    expectArraysClose(result, expected);
  });

  it('[[1, 2], [3, 4]] + [[5, 6]], axis=0', () => {
    const axis = 0;
    const a = tf.tensor2d([[1, 2], [3, 4]], [2, 2]);
    const b = tf.tensor2d([[5, 6]], [1, 2]);

    const result = tf.concat2d([a, b], axis);
    const expected = [1, 2, 3, 4, 5, 6];

    expect(result.shape).toEqual([3, 2]);
    expectArraysClose(result, expected);
  });

  it('[[1, 2],[3, 4]] + [[5, 6]] + [[7, 8]], axis=0', () => {
    const axis = 0;
    const a = tf.tensor2d([[1, 2], [3, 4]]);
    const b = tf.tensor2d([[5, 6]]);
    const c = tf.tensor2d([[7, 8]]);

    const result = tf.concat2d([a, b, c], axis);
    const expected = [1, 2, 3, 4, 5, 6, 7, 8];

    expect(result.shape).toEqual([4, 2]);
    expectArraysClose(result, expected);
  });

  it('[[1, 2], [3, 4]] + [[5, 6]], axis=1 throws error', () => {
    const axis = 1;
    const a = tf.tensor2d([[1, 2], [3, 4]], [2, 2]);
    const b = tf.tensor2d([[5, 6]], [1, 2]);

    expect(() => tf.concat2d([a, b], axis)).toThrowError();
  });

  it('[[1, 2], [3, 4]] + [[5, 6], [7, 8]], axis=1', () => {
    const axis = 1;
    const a = tf.tensor2d([[1, 2], [3, 4]], [2, 2]);
    const b = tf.tensor2d([[5, 6], [7, 8]], [2, 2]);

    const result = tf.concat2d([a, b], axis);
    const expected = [1, 2, 5, 6, 3, 4, 7, 8];

    expect(result.shape).toEqual([2, 4]);
    expectArraysClose(result, expected);
  });

  it('[[1, 2],[3, 4]] + [[5, 6],[7, 8]] + [[9, 10],[11, 12]], axis=1', () => {
    const axis = 1;
    const a = tf.tensor2d([[1, 2], [3, 4]]);
    const b = tf.tensor2d([[5, 6], [7, 8]]);
    const c = tf.tensor2d([[9, 10], [11, 12]]);

    const result = tf.concat2d([a, b, c], axis);
    const expected = [1, 2, 5, 6, 9, 10, 3, 4, 7, 8, 11, 12];

    expect(result.shape).toEqual([2, 6]);
    expectArraysClose(result, expected);
  });

  it('accepts a tensor-like object', () => {
    const axis = 0;
    const a = [[3]];
    const b = [[5]];

    const result = tf.concat2d([a, b], axis);
    const expected = [3, 5];

    expect(result.shape).toEqual([2, 1]);
    expectArraysClose(result, expected);
  });
});

describeWithFlags('concat3d', ALL_ENVS, () => {
  it('shapes correct concat axis=0', () => {
    const tensor1 = tf.tensor3d([1, 2, 3], [1, 1, 3]);
    const tensor2 = tf.tensor3d([4, 5, 6], [1, 1, 3]);
    const values = tf.concat3d([tensor1, tensor2], 0);
    expect(values.shape).toEqual([2, 1, 3]);
    expectArraysClose(values, [1, 2, 3, 4, 5, 6]);
  });

  it('concat axis=0', () => {
    const tensor1 = tf.tensor3d([1, 11, 111, 2, 22, 222], [1, 2, 3]);
    const tensor2 = tf.tensor3d(
        [5, 55, 555, 6, 66, 666, 7, 77, 777, 8, 88, 888], [2, 2, 3]);
    const values = tf.concat3d([tensor1, tensor2], 0);
    expect(values.shape).toEqual([3, 2, 3]);
    expectArraysClose(values, [
      1, 11, 111, 2, 22, 222, 5, 55, 555, 6, 66, 666, 7, 77, 777, 8, 88, 888
    ]);
  });

  it('shapes correct concat axis=1', () => {
    const tensor1 = tf.tensor3d([1, 2, 3], [1, 1, 3]);
    const tensor2 = tf.tensor3d([4, 5, 6], [1, 1, 3]);
    const values = tf.concat3d([tensor1, tensor2], 1);
    expect(values.shape).toEqual([1, 2, 3]);
    expectArraysClose(values, [1, 2, 3, 4, 5, 6]);
  });

  it('concat axis=1', () => {
    const tensor1 = tf.tensor3d([1, 11, 111, 3, 33, 333], [2, 1, 3]);
    const tensor2 = tf.tensor3d(
        [5, 55, 555, 6, 66, 666, 7, 77, 777, 8, 88, 888], [2, 2, 3]);
    const values = tf.concat3d([tensor1, tensor2], 1);
    expect(values.shape).toEqual([2, 3, 3]);
    expectArraysClose(values, [
      1, 11, 111, 5, 55, 555, 6, 66, 666, 3, 33, 333, 7, 77, 777, 8, 88, 888
    ]);
  });

  it('shapes correct concat axis=2', () => {
    const tensor1 = tf.tensor3d([1, 2, 3], [1, 1, 3]);
    const tensor2 = tf.tensor3d([4, 5, 6], [1, 1, 3]);
    const values = tf.concat3d([tensor1, tensor2], 2);
    expect(values.shape).toEqual([1, 1, 6]);
    expectArraysClose(values, [1, 2, 3, 4, 5, 6]);
  });

  it('concat axis=2', () => {
    const tensor1 = tf.tensor3d([1, 11, 2, 22, 3, 33, 4, 44], [2, 2, 2]);
    const tensor2 = tf.tensor3d(
        [5, 55, 555, 6, 66, 666, 7, 77, 777, 8, 88, 888], [2, 2, 3]);
    const values = tf.concat3d([tensor1, tensor2], 2);
    expect(values.shape).toEqual([2, 2, 5]);
    expectArraysClose(values, [
      1, 11, 5, 55, 555, 2, 22, 6, 66, 666,
      3, 33, 7, 77, 777, 4, 44, 8, 88, 888
    ]);
  });

  it('concat throws when invalid non-axis shapes, axis=0', () => {
    const axis = 0;
    const x1 = tf.tensor3d([1, 11, 111], [1, 1, 3]);
    const x2 = tf.tensor3d(
        [5, 55, 555, 6, 66, 666, 7, 77, 777, 8, 88, 888], [2, 2, 3]);
    expect(() => tf.concat3d([x1, x2], axis)).toThrowError();
  });

  it('concat throws when invalid non-axis shapes, axis=1', () => {
    const axis = 1;
    const x1 = tf.tensor3d([1, 11, 111], [1, 1, 3]);
    const x2 = tf.tensor3d(
        [5, 55, 555, 6, 66, 666, 7, 77, 777, 8, 88, 888], [2, 2, 3]);
    expect(() => tf.concat3d([x1, x2], axis)).toThrowError();
  });

  it('concat throws when invalid non-axis shapes, axis=2', () => {
    const axis = 2;
    const x1 = tf.tensor3d([1, 11, 2, 22], [1, 2, 2]);
    const x2 = tf.tensor3d(
        [5, 55, 555, 6, 66, 666, 7, 77, 777, 8, 88, 888], [2, 2, 3]);
    expect(() => tf.concat3d([x1, x2], axis)).toThrowError();
  });

  it('gradient concat axis=0', () => {
    const x1 = tf.tensor3d([1, 11, 2, 22], [1, 2, 2]);
    const x2 = tf.tensor3d([5, 55, 6, 66, 7, 77, 8, 88], [2, 2, 2]);
    const dy =
        tf.tensor3d([66, 6, 55, 5, 44, 4, 33, 3, 22, 2, 11, 1], [3, 2, 2]);
    const axis = 0;

    const grads = tf.grads(
        (x1: tf.Tensor3D, x2: tf.Tensor3D) => tf.concat3d([x1, x2], axis));
    const [dx1, dx2] = grads([x1, x2], dy);

    expect(dx1.shape).toEqual(x1.shape);
    expectArraysClose(dx1, [66, 6, 55, 5]);

    expect(dx2.shape).toEqual(x2.shape);
    expectArraysClose(dx2, [44, 4, 33, 3, 22, 2, 11, 1]);
  });

  it('gradient concat axis=1', () => {
    const x1 = tf.tensor3d([1, 11, 2, 22], [2, 1, 2]);
    const x2 = tf.tensor3d([3, 33, 4, 44, 5, 55, 6, 66], [2, 2, 2]);
    const dy =
        tf.tensor3d([66, 6, 55, 5, 44, 4, 33, 3, 22, 2, 11, 1], [2, 3, 2]);
    const axis = 1;

    const grads = tf.grads(
        (x1: tf.Tensor3D, x2: tf.Tensor3D) => tf.concat3d([x1, x2], axis));
    const [dx1, dx2] = grads([x1, x2], dy);

    expect(dx1.shape).toEqual(x1.shape);
    expectArraysClose(dx1, [66, 6, 33, 3]);

    expect(dx2.shape).toEqual(x2.shape);
    expectArraysClose(dx2, [55, 5, 44, 4, 22, 2, 11, 1]);
  });

  it('gradient concat axis=2', () => {
    const x1 = tf.tensor3d([1, 2, 3, 4], [2, 2, 1]);
    const x2 = tf.tensor3d([5, 55, 6, 66, 7, 77, 8, 88], [2, 2, 2]);
    const dy = tf.tensor3d(
        [4, 40, 400, 3, 30, 300, 2, 20, 200, 1, 10, 100], [2, 2, 3]);
    const axis = 2;

    const grads = tf.grads(
        (x1: tf.Tensor3D, x2: tf.Tensor3D) => tf.concat3d([x1, x2], axis));
    const [dx1, dx2] = grads([x1, x2], dy);

    expect(dx1.shape).toEqual(x1.shape);
    expectArraysClose(dx1, [4, 3, 2, 1]);

    expect(dx2.shape).toEqual(x2.shape);
    expectArraysClose(dx2, [40, 400, 30, 300, 20, 200, 10, 100]);
  });

  it('accepts a tensor-like object', () => {
    const tensor1 = [[[1, 2, 3]]];  // 1x1x3
    const tensor2 = [[[4, 5, 6]]];  // 1x1x3
    const values = tf.concat3d([tensor1, tensor2], 0);
    expect(values.shape).toEqual([2, 1, 3]);
    expectArraysClose(values, [1, 2, 3, 4, 5, 6]);
  });
});

describeWithFlags('concat throws for non-tensors', ALL_ENVS, () => {
  it('throws when passed a non-tensor', () => {
    expect(() => tf.concat([{} as tf.Tensor1D]))
        .toThrowError(
            /Argument 'tensors\[0\]' passed to 'concat' must be a Tensor/);
  });

  it('accepts a tensor-like object', () => {
    const tensor1 = [[[1, 2, 3, 4]]];  // 1x1x4
    const tensor2 = [[[4, 5, 6, 7]]];  // 1x1x4
    const values = tf.concat([tensor1, tensor2], 0);
    expect(values.shape).toEqual([2, 1, 4]);
    expectArraysClose(values, [1, 2, 3, 4, 4, 5, 6, 7]);
  });
});
