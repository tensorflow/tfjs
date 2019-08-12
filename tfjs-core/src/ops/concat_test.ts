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
import {ALL_ENVS, describeWithFlags} from '../jasmine_util';
import {expectArraysClose, expectArraysEqual} from '../test_util';

describeWithFlags('concat1d', ALL_ENVS, () => {
  it('3 + 5', async () => {
    const a = tf.tensor1d([3]);
    const b = tf.tensor1d([5]);

    const result = tf.concat1d([a, b]);
    const expected = [3, 5];
    expectArraysClose(await result.data(), expected);
  });
  it('TensorLike 3 + 5', async () => {
    const a = [3];
    const b = [5];

    const result = tf.concat1d([a, b]);
    const expected = [3, 5];
    expectArraysClose(await result.data(), expected);
  });
  it('TensorLike Chained 3 + 5', async () => {
    const a = tf.tensor1d([3]);
    const b = [5];

    const result = a.concat([b]);
    const expected = [3, 5];
    expectArraysClose(await result.data(), expected);
  });
  it('3 + [5,7]', async () => {
    const a = tf.tensor1d([3]);
    const b = tf.tensor1d([5, 7]);

    const result = tf.concat1d([a, b]);
    const expected = [3, 5, 7];
    expectArraysClose(await result.data(), expected);
  });

  it('[3,5] + 7', async () => {
    const a = tf.tensor1d([3, 5]);
    const b = tf.tensor1d([7]);

    const result = tf.concat1d([a, b]);
    const expected = [3, 5, 7];
    expectArraysClose(await result.data(), expected);
  });

  it('3 + 5 + 7 + 9', async () => {
    const a = tf.tensor1d([3]);
    const b = tf.tensor1d([5]);
    const c = tf.tensor1d([7]);
    const d = tf.tensor1d([9]);

    const result = tf.concat1d([a, b, c, d]);
    expectArraysClose(await result.data(), [3, 5, 7, 9]);
  });

  it('single tensor', async () => {
    const a = tf.tensor1d([3]);

    const result = tf.concat1d([a]);
    expectArraysClose(await result.data(), [3]);
  });

  it('accepts a tensor-like object', async () => {
    const a = [3];
    const b = [5];

    const result = tf.concat1d([a, b]);
    const expected = [3, 5];
    expectArraysClose(await result.data(), expected);
  });

  it('concat complex input', async () => {
    // [1+1j, 2+2j]
    const c1 = tf.complex([1, 2], [1, 2]);
    // [3+3j, 4+4j]
    const c2 = tf.complex([3, 4], [3, 4]);

    const axis = 0;
    const result = tf.concat([c1, c2], axis);
    const expected = [1, 1, 2, 2, 3, 3, 4, 4];
    expect(result.dtype).toEqual('complex64');
    expectArraysClose(await result.data(), expected);
  });
});

describeWithFlags('concat2d', ALL_ENVS, () => {
  it('[[3]] + [[5]], axis=0', async () => {
    const axis = 0;
    const a = tf.tensor2d([3], [1, 1]);
    const b = tf.tensor2d([5], [1, 1]);

    const result = tf.concat2d([a, b], axis);
    const expected = [3, 5];

    expect(result.shape).toEqual([2, 1]);
    expectArraysClose(await result.data(), expected);
  });
  it('TensorLike [[3]] + [[5]], axis=0', async () => {
    const axis = 0;
    const a = [[3]];
    const b = [[5]];

    const result = tf.concat2d([a, b], axis);
    const expected = [3, 5];

    expect(result.shape).toEqual([2, 1]);
    expectArraysClose(await result.data(), expected);
  });
  it('TensorLike Chained [[3]] + [[5]], axis=0', async () => {
    const axis = 0;
    const a = tf.tensor2d([3], [1, 1]);
    const b = [[5]];

    const result = a.concat([b], axis);
    const expected = [3, 5];

    expect(result.shape).toEqual([2, 1]);
    expectArraysClose(await result.data(), expected);
  });

  it('[[3]] + [[5]], axis=1', async () => {
    const axis = 1;
    const a = tf.tensor2d([3], [1, 1]);
    const b = tf.tensor2d([5], [1, 1]);

    const result = tf.concat2d([a, b], axis);
    const expected = [3, 5];

    expect(result.shape).toEqual([1, 2]);
    expectArraysClose(await result.data(), expected);
  });

  it('[[1, 2], [3, 4]] + [[5, 6]], axis=0', async () => {
    const axis = 0;
    const a = tf.tensor2d([[1, 2], [3, 4]], [2, 2]);
    const b = tf.tensor2d([[5, 6]], [1, 2]);

    const result = tf.concat2d([a, b], axis);
    const expected = [1, 2, 3, 4, 5, 6];

    expect(result.shape).toEqual([3, 2]);
    expectArraysClose(await result.data(), expected);
  });

  it('[[1, 2],[3, 4]] + [[5, 6]] + [[7, 8]], axis=0', async () => {
    const axis = 0;
    const a = tf.tensor2d([[1, 2], [3, 4]]);
    const b = tf.tensor2d([[5, 6]]);
    const c = tf.tensor2d([[7, 8]]);

    const result = tf.concat2d([a, b, c], axis);
    const expected = [1, 2, 3, 4, 5, 6, 7, 8];

    expect(result.shape).toEqual([4, 2]);
    expectArraysClose(await result.data(), expected);
  });

  it('[[1, 2], [3, 4]] + [[5, 6]], axis=1 throws error', () => {
    const axis = 1;
    const a = tf.tensor2d([[1, 2], [3, 4]], [2, 2]);
    const b = tf.tensor2d([[5, 6]], [1, 2]);

    expect(() => tf.concat2d([a, b], axis)).toThrowError();
  });

  it('[[1, 2], [3, 4]] + [[5, 6], [7, 8]], axis=1', async () => {
    const axis = 1;
    const a = tf.tensor2d([[1, 2], [3, 4]], [2, 2]);
    const b = tf.tensor2d([[5, 6], [7, 8]], [2, 2]);

    const result = tf.concat2d([a, b], axis);
    const expected = [1, 2, 5, 6, 3, 4, 7, 8];

    expect(result.shape).toEqual([2, 4]);
    expectArraysClose(await result.data(), expected);
  });

  it('[[1, 2],[3, 4]] + [[5, 6],[7, 8]] + [[9, 10],[11, 12]], axis=1',
     async () => {
       const axis = 1;
       const a = tf.tensor2d([[1, 2], [3, 4]]);
       const b = tf.tensor2d([[5, 6], [7, 8]]);
       const c = tf.tensor2d([[9, 10], [11, 12]]);

       const result = tf.concat2d([a, b, c], axis);
       const expected = [1, 2, 5, 6, 9, 10, 3, 4, 7, 8, 11, 12];

       expect(result.shape).toEqual([2, 6]);
       expectArraysClose(await result.data(), expected);
     });

  it('accepts a tensor-like object', async () => {
    const axis = 0;
    const a = [[3]];
    const b = [[5]];

    const result = tf.concat2d([a, b], axis);
    const expected = [3, 5];

    expect(result.shape).toEqual([2, 1]);
    expectArraysClose(await result.data(), expected);
  });

  it('concat zero-sized tensors', async () => {
    const a = tf.tensor2d([], [0, 5]);
    const b = tf.tensor2d([], [0, 5]);
    const c = tf.tensor2d([], [0, 5]);

    const res = tf.concat([a, b, c], /* axis */ 0);
    expect(res.shape).toEqual([0, 5]);
    expectArraysEqual(await res.data(), []);

    const res2 = tf.concat([a, b, c], /* axis */ 1);
    expect(res2.shape).toEqual([0, 15]);
    expectArraysEqual(await res2.data(), []);
  });

  it('concat complex input axis=0', async () => {
    // [[1+1j, 2+2j], [3+3j, 4+4j]]
    const c1 = tf.complex([[1, 2], [3, 4]], [[1, 2], [3, 4]]);
    // [[5+5j, 6+6j], [7+7j, 8+8j]]
    const c2 = tf.complex([[5, 6], [7, 8]], [[5, 6], [7, 8]]);

    const axis = 0;
    const result = tf.concat([c1, c2], axis);
    const expected = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8];
    expect(result.dtype).toEqual('complex64');
    expectArraysClose(await result.data(), expected);
  });

  it('concat complex input axis=1', async () => {
    // [[1+1j, 2+2j], [3+3j, 4+4j]]
    const c1 = tf.complex([[1, 2], [3, 4]], [[1, 2], [3, 4]]);
    // [[5+5j, 6+6j], [7+7j, 8+8j]]
    const c2 = tf.complex([[5, 6], [7, 8]], [[5, 6], [7, 8]]);

    const axis = 1;
    const result = tf.concat([c1, c2], axis);
    const expected = [1, 1, 2, 2, 5, 5, 6, 6, 3, 3, 4, 4, 7, 7, 8, 8];
    expect(result.dtype).toEqual('complex64');
    expectArraysClose(await result.data(), expected);
  });
});

describeWithFlags('concat3d', ALL_ENVS, () => {
  it('shapes correct concat axis=0', async () => {
    const tensor1 = tf.tensor3d([1, 2, 3], [1, 1, 3]);
    const tensor2 = tf.tensor3d([4, 5, 6], [1, 1, 3]);
    const values = tf.concat3d([tensor1, tensor2], 0);
    expect(values.shape).toEqual([2, 1, 3]);
    expectArraysClose(await values.data(), [1, 2, 3, 4, 5, 6]);
  });

  it('concat axis=0', async () => {
    const tensor1 = tf.tensor3d([1, 11, 111, 2, 22, 222], [1, 2, 3]);
    const tensor2 = tf.tensor3d(
        [5, 55, 555, 6, 66, 666, 7, 77, 777, 8, 88, 888], [2, 2, 3]);
    const values = tf.concat3d([tensor1, tensor2], 0);
    expect(values.shape).toEqual([3, 2, 3]);
    expectArraysClose(await values.data(), [
      1, 11, 111, 2, 22, 222, 5, 55, 555, 6, 66, 666, 7, 77, 777, 8, 88, 888
    ]);
  });

  it('TensorLike concat axis=0', async () => {
    const tensor1 = [[[1, 11, 111], [2, 22, 222]]];
    const tensor2 =
        [[[5, 55, 555], [6, 66, 666]], [[7, 77, 777], [8, 88, 888]]];
    const values = tf.concat3d([tensor1, tensor2], 0);
    expect(values.shape).toEqual([3, 2, 3]);
    expectArraysClose(await values.data(), [
      1, 11, 111, 2, 22, 222, 5, 55, 555, 6, 66, 666, 7, 77, 777, 8, 88, 888
    ]);
  });
  it('TensorLike Chained concat axis=0', async () => {
    const tensor1 = tf.tensor3d([1, 11, 111, 2, 22, 222], [1, 2, 3]);
    const tensor2 =
        [[[5, 55, 555], [6, 66, 666]], [[7, 77, 777], [8, 88, 888]]];
    const values = tensor1.concat([tensor2], 0);
    expect(values.shape).toEqual([3, 2, 3]);
    expectArraysClose(await values.data(), [
      1, 11, 111, 2, 22, 222, 5, 55, 555, 6, 66, 666, 7, 77, 777, 8, 88, 888
    ]);
  });

  it('shapes correct concat axis=1', async () => {
    const tensor1 = tf.tensor3d([1, 2, 3], [1, 1, 3]);
    const tensor2 = tf.tensor3d([4, 5, 6], [1, 1, 3]);
    const values = tf.concat3d([tensor1, tensor2], 1);
    expect(values.shape).toEqual([1, 2, 3]);
    expectArraysClose(await values.data(), [1, 2, 3, 4, 5, 6]);
  });

  it('concat axis=1', async () => {
    const tensor1 = tf.tensor3d([1, 11, 111, 3, 33, 333], [2, 1, 3]);
    const tensor2 = tf.tensor3d(
        [5, 55, 555, 6, 66, 666, 7, 77, 777, 8, 88, 888], [2, 2, 3]);
    const values = tf.concat3d([tensor1, tensor2], 1);
    expect(values.shape).toEqual([2, 3, 3]);
    expectArraysClose(await values.data(), [
      1, 11, 111, 5, 55, 555, 6, 66, 666, 3, 33, 333, 7, 77, 777, 8, 88, 888
    ]);
  });

  it('shapes correct concat axis=2', async () => {
    const tensor1 = tf.tensor3d([1, 2, 3], [1, 1, 3]);
    const tensor2 = tf.tensor3d([4, 5, 6], [1, 1, 3]);
    const values = tf.concat3d([tensor1, tensor2], 2);
    expect(values.shape).toEqual([1, 1, 6]);
    expectArraysClose(await values.data(), [1, 2, 3, 4, 5, 6]);
  });

  it('concat a large number of tensors, axis=0', async () => {
    const tensors = [];
    const expected = [];
    for (let i = 0; i < 100; i++) {
      tensors.push(tf.tensor([i], [1]));
      expected.push(i);
    }
    const axis = 0;
    const res = tf.concat(tensors, axis);
    expect(res.shape).toEqual([100]);
    expect(res.dtype).toBe('float32');
    expectArraysClose(await res.data(), expected);
  });

  it('concat a large number of tensors, axis=1', async () => {
    const tensors = [];
    const expected = [];
    for (let i = 0; i < 100; i++) {
      tensors.push(tf.tensor([i], [1, 1]));
      expected.push(i);
    }
    const axis = 1;
    const res = tf.concat(tensors, axis);
    expect(res.shape).toEqual([1, 100]);
    expect(res.dtype).toBe('float32');
    expectArraysClose(await res.data(), expected);
  });

  it('concat axis=2', async () => {
    const tensor1 = tf.tensor3d([1, 11, 2, 22, 3, 33, 4, 44], [2, 2, 2]);
    const tensor2 = tf.tensor3d(
        [5, 55, 555, 6, 66, 666, 7, 77, 777, 8, 88, 888], [2, 2, 3]);
    const values = tf.concat3d([tensor1, tensor2], 2);
    expect(values.shape).toEqual([2, 2, 5]);
    expectArraysClose(await values.data(), [
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

  it('gradient concat axis=0', async () => {
    const x1 = tf.tensor3d([1, 11, 2, 22], [1, 2, 2]);
    const x2 = tf.tensor3d([5, 55, 6, 66, 7, 77, 8, 88], [2, 2, 2]);
    const dy =
        tf.tensor3d([66, 6, 55, 5, 44, 4, 33, 3, 22, 2, 11, 1], [3, 2, 2]);
    const axis = 0;

    const grads = tf.grads(
        (x1: tf.Tensor3D, x2: tf.Tensor3D) => tf.concat3d([x1, x2], axis));
    const [dx1, dx2] = grads([x1, x2], dy);

    expect(dx1.shape).toEqual(x1.shape);
    expectArraysClose(await dx1.data(), [66, 6, 55, 5]);

    expect(dx2.shape).toEqual(x2.shape);
    expectArraysClose(await dx2.data(), [44, 4, 33, 3, 22, 2, 11, 1]);
  });

  it('gradient with clones', async () => {
    const x1 = tf.tensor3d([1, 11, 2, 22], [1, 2, 2]);
    const x2 = tf.tensor3d([5, 55, 6, 66, 7, 77, 8, 88], [2, 2, 2]);
    const dy =
        tf.tensor3d([66, 6, 55, 5, 44, 4, 33, 3, 22, 2, 11, 1], [3, 2, 2]);
    const axis = 0;

    const grads = tf.grads(
        (x1: tf.Tensor3D, x2: tf.Tensor3D) =>
            tf.concat3d([x1.clone(), x2.clone()], axis).clone());
    const [dx1, dx2] = grads([x1, x2], dy);

    expect(dx1.shape).toEqual(x1.shape);
    expectArraysClose(await dx1.data(), [66, 6, 55, 5]);

    expect(dx2.shape).toEqual(x2.shape);
    expectArraysClose(await dx2.data(), [44, 4, 33, 3, 22, 2, 11, 1]);
  });

  it('gradient concat axis=1', async () => {
    const x1 = tf.tensor3d([1, 11, 2, 22], [2, 1, 2]);
    const x2 = tf.tensor3d([3, 33, 4, 44, 5, 55, 6, 66], [2, 2, 2]);
    const dy =
        tf.tensor3d([66, 6, 55, 5, 44, 4, 33, 3, 22, 2, 11, 1], [2, 3, 2]);
    const axis = 1;

    const grads = tf.grads(
        (x1: tf.Tensor3D, x2: tf.Tensor3D) => tf.concat3d([x1, x2], axis));
    const [dx1, dx2] = grads([x1, x2], dy);

    expect(dx1.shape).toEqual(x1.shape);
    expectArraysClose(await dx1.data(), [66, 6, 33, 3]);

    expect(dx2.shape).toEqual(x2.shape);
    expectArraysClose(await dx2.data(), [55, 5, 44, 4, 22, 2, 11, 1]);
  });

  it('gradient concat axis=2', async () => {
    const x1 = tf.tensor3d([1, 2, 3, 4], [2, 2, 1]);
    const x2 = tf.tensor3d([5, 55, 6, 66, 7, 77, 8, 88], [2, 2, 2]);
    const dy = tf.tensor3d(
        [4, 40, 400, 3, 30, 300, 2, 20, 200, 1, 10, 100], [2, 2, 3]);
    const axis = 2;

    const grads = tf.grads(
        (x1: tf.Tensor3D, x2: tf.Tensor3D) => tf.concat3d([x1, x2], axis));
    const [dx1, dx2] = grads([x1, x2], dy);

    expect(dx1.shape).toEqual(x1.shape);
    expectArraysClose(await dx1.data(), [4, 3, 2, 1]);

    expect(dx2.shape).toEqual(x2.shape);
    expectArraysClose(await dx2.data(), [40, 400, 30, 300, 20, 200, 10, 100]);
  });

  it('gradient concat axis=-1', async () => {
    const x1 = tf.tensor3d([1, 2, 3, 4], [2, 2, 1]);
    const x2 = tf.tensor3d([5, 55, 6, 66, 7, 77, 8, 88], [2, 2, 2]);
    const dy = tf.tensor3d(
        [4, 40, 400, 3, 30, 300, 2, 20, 200, 1, 10, 100], [2, 2, 3]);
    const axis = -1;

    const grads = tf.grads(
        (x1: tf.Tensor3D, x2: tf.Tensor3D) => tf.concat3d([x1, x2], axis));
    const [dx1, dx2] = grads([x1, x2], dy);

    expect(dx1.shape).toEqual(x1.shape);
    expectArraysClose(await dx1.data(), [4, 3, 2, 1]);

    expect(dx2.shape).toEqual(x2.shape);
    expectArraysClose(await dx2.data(), [40, 400, 30, 300, 20, 200, 10, 100]);
  });

  it('accepts a tensor-like object', async () => {
    const tensor1 = [[[1, 2, 3]]];  // 1x1x3
    const tensor2 = [[[4, 5, 6]]];  // 1x1x3
    const values = tf.concat3d([tensor1, tensor2], 0);
    expect(values.shape).toEqual([2, 1, 3]);
    expectArraysClose(await values.data(), [1, 2, 3, 4, 5, 6]);
  });

  it('concat tensors with 0 in their shape', async () => {
    const tensor1 = tf.tensor3d([1, 2, 3, 4, 5, 6], [2, 3, 1]);
    const tensor2 = tf.tensor3d([], [0, 3, 1]);
    const values = tf.concat3d([tensor1, tensor2], 0);
    expect(values.shape).toEqual([2, 3, 1]);
    expectArraysClose(await values.data(), [1, 2, 3, 4, 5, 6]);
  });

  it('concat complex input axis=0', async () => {
    // [[[1+1j, 2+2j], [3+3j, 4+4j], [5+5j, 6+6j]]]
    const c1 =
        tf.complex([[[1, 2], [3, 4], [5, 6]]], [[[1, 2], [3, 4], [5, 6]]]);
    // [[[7+7j, 8+8j], [9+9j, 10+10j], [11+11j, 12+12j]]]
    const c2 = tf.complex(
        [[[7, 8], [9, 10], [11, 12]]], [[[7, 8], [9, 10], [11, 12]]]);

    const axis = 0;
    const result = tf.concat([c1, c2], axis);
    const expected = [
      1, 1, 2, 2, 3, 3, 4,  4,  5,  5,  6,  6,
      7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12
    ];
    expect(result.dtype).toEqual('complex64');
    expectArraysClose(await result.data(), expected);
  });

  it('concat complex input axis=1', async () => {
    // [[[1+1j, 2+2j], [3+3j, 4+4j], [5+5j, 6+6j]]]
    const c1 =
        tf.complex([[[1, 2], [3, 4], [5, 6]]], [[[1, 2], [3, 4], [5, 6]]]);
    // [[[7+7j, 8+8j], [9+9j, 10+10j], [11+11j, 12+12j]]]
    const c2 = tf.complex(
        [[[7, 8], [9, 10], [11, 12]]], [[[7, 8], [9, 10], [11, 12]]]);

    const axis = 1;
    const result = tf.concat([c1, c2], axis);
    const expected = [
      1, 1, 2, 2, 3, 3, 4,  4,  5,  5,  6,  6,
      7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12
    ];
    expect(result.dtype).toEqual('complex64');
    expectArraysClose(await result.data(), expected);
  });

  it('concat complex input axis=1', async () => {
    // [[[1+1j, 2+2j], [3+3j, 4+4j], [5+5j, 6+6j]]]
    const c1 =
        tf.complex([[[1, 2], [3, 4], [5, 6]]], [[[1, 2], [3, 4], [5, 6]]]);
    // [[[7+7j, 8+8j], [9+9j, 10+10j], [11+11j, 12+12j]]]
    const c2 = tf.complex(
        [[[7, 8], [9, 10], [11, 12]]], [[[7, 8], [9, 10], [11, 12]]]);

    const axis = 2;
    const result = tf.concat([c1, c2], axis);
    const expected = [
      1, 1, 2,  2,  7, 7, 8, 8, 3,  3,  4,  4,
      9, 9, 10, 10, 5, 5, 6, 6, 11, 11, 12, 12
    ];
    expect(result.dtype).toEqual('complex64');
    expectArraysClose(await result.data(), expected);
  });
});

describeWithFlags('concat throws for non-tensors', ALL_ENVS, () => {
  it('throws when passed a non-tensor', () => {
    expect(() => tf.concat([{} as tf.Tensor1D]))
        .toThrowError(
            /Argument 'tensors\[0\]' passed to 'concat' must be a Tensor/);
  });

  it('accepts a tensor-like object', async () => {
    const tensor1 = [[[1, 2, 3, 4]]];  // 1x1x4
    const tensor2 = [[[4, 5, 6, 7]]];  // 1x1x4
    const values = tf.concat([tensor1, tensor2], 0);
    expect(values.shape).toEqual([2, 1, 4]);
    expectArraysClose(await values.data(), [1, 2, 3, 4, 4, 5, 6, 7]);
  });
});
