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

import * as tf from '../index';
import {ALL_ENVS, describeWithFlags} from '../jasmine_util';
import {expectArraysClose} from '../test_util';

describeWithFlags('logicalNot', ALL_ENVS, () => {
  it('Tensor1D.', async () => {
    let a = tf.tensor1d([1, 0, 0], 'bool');
    expectArraysClose(await tf.logicalNot(a).data(), [0, 1, 1]);

    a = tf.tensor1d([0, 0, 0], 'bool');
    expectArraysClose(await tf.logicalNot(a).data(), [1, 1, 1]);

    a = tf.tensor1d([1, 1], 'bool');
    expectArraysClose(await tf.logicalNot(a).data(), [0, 0]);
  });
  it('Tests chaining in Tensor1D', async () => {
    let a = tf.tensor1d([1, 0, 0], 'bool');
    expectArraysClose(await a.logicalNot().data(), [0, 1, 1]);

    a = tf.tensor1d([0, 0, 0], 'bool');
    expectArraysClose(await a.logicalNot().data(), [1, 1, 1]);

    a = tf.tensor1d([1, 1], 'bool');
    expectArraysClose(await a.logicalNot().data(), [0, 0]);
  });

  it('Tensor2D', async () => {
    let a = tf.tensor2d([[1, 0, 1], [0, 0, 0]], [2, 3], 'bool');
    expectArraysClose(await tf.logicalNot(a).data(), [0, 1, 0, 1, 1, 1]);

    a = tf.tensor2d([[0, 0, 0], [1, 1, 1]], [2, 3], 'bool');
    expectArraysClose(await tf.logicalNot(a).data(), [1, 1, 1, 0, 0, 0]);
  });

  it('Tensor3D', async () => {
    let a = tf.tensor3d([[[1], [0], [1]], [[0], [0], [0]]], [2, 3, 1], 'bool');
    expectArraysClose(await tf.logicalNot(a).data(), [0, 1, 0, 1, 1, 1]);

    a = tf.tensor3d([[[0], [0], [0]], [[1], [1], [1]]], [2, 3, 1], 'bool');
    expectArraysClose(await tf.logicalNot(a).data(), [1, 1, 1, 0, 0, 0]);
  });

  it('Tensor4D', async () => {
    let a = tf.tensor4d([1, 0, 1, 0], [2, 2, 1, 1], 'bool');
    expectArraysClose(await tf.logicalNot(a).data(), [0, 1, 0, 1]);

    a = tf.tensor4d([0, 0, 0, 0], [2, 2, 1, 1], 'bool');
    expectArraysClose(await tf.logicalNot(a).data(), [1, 1, 1, 1]);

    a = tf.tensor4d([1, 1, 1, 1], [2, 2, 1, 1], 'bool');
    expectArraysClose(await tf.logicalNot(a).data(), [0, 0, 0, 0]);
  });

  it('Tensor6D', async () => {
    let a = tf.tensor6d([1, 0, 1, 0], [2, 2, 1, 1, 1, 1], 'bool');
    expectArraysClose(await tf.logicalNot(a).data(), [0, 1, 0, 1]);

    a = tf.zeros([2, 2, 2, 2, 2, 2]).cast('bool');
    let expectedResult = new Uint8Array(64).fill(1);
    expectedResult = expectedResult.fill(1);
    expectArraysClose(await tf.logicalNot(a).data(), expectedResult);

    a = tf.ones([2, 2, 2, 2, 2, 2]).cast('bool');
    expectedResult = expectedResult.fill(0);
    expectArraysClose(await tf.logicalNot(a).data(), expectedResult);
  });

  it('throws when passed a non-tensor', () => {
    expect(() => tf.logicalNot({} as tf.Tensor))
        .toThrowError(/Argument 'x' passed to 'logicalNot' must be a Tensor/);
  });

  it('accepts a tensor-like object', async () => {
    const a = [1, 0, 0];
    expectArraysClose(await tf.logicalNot(a).data(), [0, 1, 1]);
  });
});

describeWithFlags('logicalAnd', ALL_ENVS, () => {
  it('Tensor1D.', async () => {
    let a = tf.tensor1d([1, 0, 0], 'bool');
    let b = tf.tensor1d([0, 1, 0], 'bool');
    expectArraysClose(await tf.logicalAnd(a, b).data(), [0, 0, 0]);

    a = tf.tensor1d([0, 0, 0], 'bool');
    b = tf.tensor1d([0, 0, 0], 'bool');
    expectArraysClose(await tf.logicalAnd(a, b).data(), [0, 0, 0]);

    a = tf.tensor1d([1, 1], 'bool');
    b = tf.tensor1d([1, 1], 'bool');
    expectArraysClose(await tf.logicalAnd(a, b).data(), [1, 1]);
  });
  it('mismatched Tensor1D shapes', () => {
    const a = tf.tensor1d([1, 0], 'bool');
    const b = tf.tensor1d([0, 1, 0], 'bool');
    const f = () => {
      tf.logicalAnd(a, b);
    };
    expect(f).toThrowError();
  });

  it('Tensor2D', async () => {
    let a = tf.tensor2d([[1, 0, 1], [0, 0, 0]], [2, 3], 'bool');
    let b = tf.tensor2d([[0, 0, 0], [0, 1, 0]], [2, 3], 'bool');
    expectArraysClose(await tf.logicalAnd(a, b).data(), [0, 0, 0, 0, 0, 0]);

    a = tf.tensor2d([[0, 0, 0], [1, 1, 1]], [2, 3], 'bool');
    b = tf.tensor2d([[0, 0, 0], [1, 1, 1]], [2, 3], 'bool');
    expectArraysClose(await tf.logicalAnd(a, b).data(), [0, 0, 0, 1, 1, 1]);
  });
  it('broadcasting Tensor2D shapes', async () => {
    const a = tf.tensor2d([[1], [0]], [2, 1], 'bool');
    const b = tf.tensor2d([[0, 1, 0], [0, 1, 0]], [2, 3], 'bool');
    expectArraysClose(await tf.logicalAnd(a, b).data(), [0, 1, 0, 0, 0, 0]);
  });

  it('Tensor3D', async () => {
    let a = tf.tensor3d([[[1], [0], [1]], [[0], [0], [1]]], [2, 3, 1], 'bool');
    let b = tf.tensor3d([[[0], [0], [1]], [[1], [0], [0]]], [2, 3, 1], 'bool');
    expectArraysClose(await tf.logicalAnd(a, b).data(), [0, 0, 1, 0, 0, 0]);

    a = tf.tensor3d([[[0], [0], [0]], [[1], [1], [1]]], [2, 3, 1], 'bool');
    b = tf.tensor3d([[[0], [0], [0]], [[1], [1], [1]]], [2, 3, 1], 'bool');
    expectArraysClose(await tf.logicalAnd(a, b).data(), [0, 0, 0, 1, 1, 1]);
  });
  it('broadcasting Tensor3D shapes', async () => {
    const a = tf.tensor3d(
        [[[1, 0], [0, 0], [1, 1]], [[0, 0], [0, 1], [0, 0]]], [2, 3, 2],
        'bool');
    const b =
        tf.tensor3d([[[0], [0], [1]], [[1], [0], [0]]], [2, 3, 1], 'bool');
    expectArraysClose(
        await tf.logicalAnd(a, b).data(), [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]);
  });

  it('Tensor4D', async () => {
    let a = tf.tensor4d([1, 0, 1, 0], [2, 2, 1, 1], 'bool');
    let b = tf.tensor4d([0, 1, 1, 0], [2, 2, 1, 1], 'bool');
    expectArraysClose(await tf.logicalAnd(a, b).data(), [0, 0, 1, 0]);

    a = tf.tensor4d([0, 0, 0, 0], [2, 2, 1, 1], 'bool');
    b = tf.tensor4d([0, 0, 0, 0], [2, 2, 1, 1], 'bool');
    expectArraysClose(await tf.logicalAnd(a, b).data(), [0, 0, 0, 0]);

    a = tf.tensor4d([1, 1, 1, 1], [2, 2, 1, 1], 'bool');
    b = tf.tensor4d([1, 1, 1, 1], [2, 2, 1, 1], 'bool');
    expectArraysClose(await tf.logicalAnd(a, b).data(), [1, 1, 1, 1]);
  });
  it('broadcasting Tensor4D shapes', async () => {
    const a = tf.tensor4d([1, 0, 1, 0], [2, 2, 1, 1], 'bool');
    const b = tf.tensor4d(
        [[[[1, 0]], [[0, 0]]], [[[0, 0]], [[1, 1]]]], [2, 2, 1, 2], 'bool');
    expectArraysClose(
        await tf.logicalAnd(a, b).data(), [1, 0, 0, 0, 0, 0, 0, 0]);
  });

  it('TensorLike', async () => {
    const a = [true, false, false];
    const b = [false, true, false];
    expectArraysClose(await tf.logicalAnd(a, b).data(), [0, 0, 0]);
  });

  it('TensorLike Chained', async () => {
    const a = tf.tensor1d([1, 0, 0], 'bool');
    const b = [false, true, false];
    expectArraysClose(await a.logicalAnd(b).data(), [0, 0, 0]);
  });

  it('throws when passed a as a non-tensor', () => {
    expect(() => tf.logicalAnd({} as tf.Tensor, tf.scalar(1, 'bool')))
        .toThrowError(/Argument 'a' passed to 'logicalAnd' must be a Tensor/);
  });
  it('throws when passed b as a non-tensor', () => {
    expect(() => tf.logicalAnd(tf.scalar(1, 'bool'), {} as tf.Tensor))
        .toThrowError(/Argument 'b' passed to 'logicalAnd' must be a Tensor/);
  });

  it('accepts a tensor-like object', async () => {
    const a = [1, 0, 0, 1];
    const b = [0, 1, 0, 1];
    expectArraysClose(await tf.logicalAnd(a, b).data(), [0, 0, 0, 1]);
  });
});

describeWithFlags('logicalOr', ALL_ENVS, () => {
  it('Tensor1D.', async () => {
    let a = tf.tensor1d([1, 0, 0], 'bool');
    let b = tf.tensor1d([0, 1, 0], 'bool');
    expectArraysClose(await tf.logicalOr(a, b).data(), [1, 1, 0]);

    a = tf.tensor1d([0, 0, 0], 'bool');
    b = tf.tensor1d([0, 0, 0], 'bool');
    expectArraysClose(await tf.logicalOr(a, b).data(), [0, 0, 0]);

    a = tf.tensor1d([1, 1], 'bool');
    b = tf.tensor1d([1, 1], 'bool');
    expectArraysClose(await tf.logicalOr(a, b).data(), [1, 1]);
  });
  it('mismatched Tensor1D shapes', () => {
    const a = tf.tensor1d([1, 0], 'bool');
    const b = tf.tensor1d([0, 1, 0], 'bool');
    const f = () => {
      tf.logicalOr(a, b);
    };
    expect(f).toThrowError();
  });

  it('Tensor2D', async () => {
    let a = tf.tensor2d([[1, 0, 1], [0, 0, 0]], [2, 3], 'bool');
    let b = tf.tensor2d([[0, 0, 0], [0, 1, 0]], [2, 3], 'bool');
    expectArraysClose(await tf.logicalOr(a, b).data(), [1, 0, 1, 0, 1, 0]);

    a = tf.tensor2d([[0, 0, 0], [1, 1, 1]], [2, 3], 'bool');
    b = tf.tensor2d([[0, 0, 0], [1, 1, 1]], [2, 3], 'bool');
    expectArraysClose(await tf.logicalOr(a, b).data(), [0, 0, 0, 1, 1, 1]);
  });
  it('broadcasting Tensor2D shapes', async () => {
    const a = tf.tensor2d([[1], [0]], [2, 1], 'bool');
    const b = tf.tensor2d([[0, 0, 0], [0, 1, 0]], [2, 3], 'bool');
    expectArraysClose(await tf.logicalOr(a, b).data(), [1, 1, 1, 0, 1, 0]);
  });

  it('Tensor3D', async () => {
    let a = tf.tensor3d([[[1], [0], [1]], [[0], [0], [0]]], [2, 3, 1], 'bool');
    let b = tf.tensor3d([[[0], [0], [1]], [[1], [0], [0]]], [2, 3, 1], 'bool');
    expectArraysClose(await tf.logicalOr(a, b).data(), [1, 0, 1, 1, 0, 0]);

    a = tf.tensor3d([[[0], [0], [0]], [[1], [1], [1]]], [2, 3, 1], 'bool');
    b = tf.tensor3d([[[0], [0], [0]], [[1], [1], [1]]], [2, 3, 1], 'bool');
    expectArraysClose(await tf.logicalOr(a, b).data(), [0, 0, 0, 1, 1, 1]);
  });
  it('broadcasting Tensor3D shapes', async () => {
    const a = tf.tensor3d(
        [[[1, 0], [0, 0], [1, 1]], [[0, 0], [0, 1], [0, 0]]], [2, 3, 2],
        'bool');
    const b =
        tf.tensor3d([[[0], [0], [1]], [[1], [0], [0]]], [2, 3, 1], 'bool');
    expectArraysClose(
        await tf.logicalOr(a, b).data(), [1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0]);
  });

  it('Tensor4D', async () => {
    let a = tf.tensor4d([1, 0, 1, 0], [2, 2, 1, 1], 'bool');
    let b = tf.tensor4d([0, 1, 0, 0], [2, 2, 1, 1], 'bool');
    expectArraysClose(await tf.logicalOr(a, b).data(), [1, 1, 1, 0]);

    a = tf.tensor4d([0, 0, 0, 0], [2, 2, 1, 1], 'bool');
    b = tf.tensor4d([0, 0, 0, 0], [2, 2, 1, 1], 'bool');
    expectArraysClose(await tf.logicalOr(a, b).data(), [0, 0, 0, 0]);

    a = tf.tensor4d([1, 1, 1, 1], [2, 2, 1, 1], 'bool');
    b = tf.tensor4d([1, 1, 1, 1], [2, 2, 1, 1], 'bool');
    expectArraysClose(await tf.logicalOr(a, b).data(), [1, 1, 1, 1]);
  });
  it('broadcasting Tensor4D shapes', async () => {
    const a = tf.tensor4d([1, 0, 1, 0], [2, 2, 1, 1], 'bool');
    const b = tf.tensor4d(
        [[[[1, 0]], [[0, 0]]], [[[0, 0]], [[1, 1]]]], [2, 2, 1, 2], 'bool');
    expectArraysClose(
        await tf.logicalOr(a, b).data(), [1, 1, 0, 0, 1, 1, 1, 1]);
  });

  it('TensorLike', async () => {
    const a = [true, false, false];
    const b = [false, true, false];
    expectArraysClose(await tf.logicalOr(a, b).data(), [1, 1, 0]);
  });

  it('TensorLike Chained', async () => {
    const a = tf.tensor1d([1, 0, 0], 'bool');
    const b = [false, true, false];
    expectArraysClose(await a.logicalOr(b).data(), [1, 1, 0]);
  });

  it('throws when passed a as a non-tensor', () => {
    expect(() => tf.logicalOr({} as tf.Tensor, tf.scalar(1, 'bool')))
        .toThrowError(/Argument 'a' passed to 'logicalOr' must be a Tensor/);
  });
  it('throws when passed b as a non-tensor', () => {
    expect(() => tf.logicalOr(tf.scalar(1, 'bool'), {} as tf.Tensor))
        .toThrowError(/Argument 'b' passed to 'logicalOr' must be a Tensor/);
  });

  it('accepts a tensor-like object', async () => {
    const a = [1, 0, 0, 1];
    const b = [0, 1, 0, 1];
    expectArraysClose(await tf.logicalOr(a, b).data(), [1, 1, 0, 1]);
  });
});

describeWithFlags('logicalXor', ALL_ENVS, () => {
  it('Tensor1D.', async () => {
    let a = tf.tensor1d([1, 0, 0], 'bool');
    let b = tf.tensor1d([0, 1, 0], 'bool');
    expectArraysClose(await tf.logicalXor(a, b).data(), [1, 1, 0]);

    a = tf.tensor1d([0, 0, 0], 'bool');
    b = tf.tensor1d([0, 0, 0], 'bool');
    expectArraysClose(await tf.logicalXor(a, b).data(), [0, 0, 0]);

    a = tf.tensor1d([1, 1], 'bool');
    b = tf.tensor1d([1, 1], 'bool');
    expectArraysClose(await tf.logicalXor(a, b).data(), [0, 0]);
  });
  it('mismatched Tensor1D shapes', () => {
    const a = tf.tensor1d([1, 0], 'bool');
    const b = tf.tensor1d([0, 1, 0], 'bool');
    const f = () => {
      tf.logicalXor(a, b);
    };
    expect(f).toThrowError();
  });

  // Tensor2D:
  it('Tensor2D', async () => {
    let a = tf.tensor2d([[1, 0, 1], [0, 0, 0]], [2, 3], 'bool');
    let b = tf.tensor2d([[0, 0, 0], [0, 1, 0]], [2, 3], 'bool');
    expectArraysClose(await tf.logicalXor(a, b).data(), [1, 0, 1, 0, 1, 0]);

    a = tf.tensor2d([[0, 0, 0], [1, 1, 1]], [2, 3], 'bool');
    b = tf.tensor2d([[0, 0, 0], [1, 1, 1]], [2, 3], 'bool');
    expectArraysClose(await tf.logicalXor(a, b).data(), [0, 0, 0, 0, 0, 0]);
  });
  it('broadcasting Tensor2D shapes', async () => {
    const a = tf.tensor2d([[1], [0]], [2, 1], 'bool');
    const b = tf.tensor2d([[0, 0, 0], [0, 1, 0]], [2, 3], 'bool');
    expectArraysClose(await tf.logicalXor(a, b).data(), [1, 1, 1, 0, 1, 0]);
  });

  // Tensor3D:
  it('Tensor3D', async () => {
    let a = tf.tensor3d([[[1], [0], [1]], [[0], [0], [0]]], [2, 3, 1], 'bool');
    let b = tf.tensor3d([[[0], [0], [1]], [[1], [0], [0]]], [2, 3, 1], 'bool');
    expectArraysClose(await tf.logicalXor(a, b).data(), [1, 0, 0, 1, 0, 0]);

    a = tf.tensor3d([[[0], [0], [0]], [[1], [1], [1]]], [2, 3, 1], 'bool');
    b = tf.tensor3d([[[0], [0], [0]], [[1], [1], [1]]], [2, 3, 1], 'bool');
    expectArraysClose(await tf.logicalXor(a, b).data(), [0, 0, 0, 0, 0, 0]);
  });
  it('broadcasting Tensor3D shapes', async () => {
    const a = tf.tensor3d(
        [[[1, 0], [0, 0], [1, 1]], [[0, 0], [0, 1], [0, 0]]], [2, 3, 2],
        'bool');
    const b =
        tf.tensor3d([[[0], [0], [1]], [[1], [0], [0]]], [2, 3, 1], 'bool');
    expectArraysClose(
        await tf.logicalXor(a, b).data(), [1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0]);
  });

  // Tensor4D:
  it('Tensor4D', async () => {
    let a = tf.tensor4d([1, 0, 1, 0], [2, 2, 1, 1], 'bool');
    let b = tf.tensor4d([0, 1, 1, 0], [2, 2, 1, 1], 'bool');
    expectArraysClose(await tf.logicalXor(a, b).data(), [1, 1, 0, 0]);

    a = tf.tensor4d([0, 0, 0, 0], [2, 2, 1, 1], 'bool');
    b = tf.tensor4d([0, 0, 0, 0], [2, 2, 1, 1], 'bool');
    expectArraysClose(await tf.logicalXor(a, b).data(), [0, 0, 0, 0]);

    a = tf.tensor4d([1, 1, 1, 1], [2, 2, 1, 1], 'bool');
    b = tf.tensor4d([1, 1, 1, 1], [2, 2, 1, 1], 'bool');
    expectArraysClose(await tf.logicalXor(a, b).data(), [0, 0, 0, 0]);
  });
  it('broadcasting Tensor4D shapes', async () => {
    const a = tf.tensor4d([1, 0, 1, 0], [2, 2, 1, 1], 'bool');
    const b = tf.tensor4d(
        [[[[1, 0]], [[0, 0]]], [[[0, 0]], [[1, 1]]]], [2, 2, 1, 2], 'bool');
    expectArraysClose(
        await tf.logicalXor(a, b).data(), [0, 1, 0, 0, 1, 1, 1, 1]);
  });

  it('TensorLike', async () => {
    const a = [true, false, false];
    const b = [false, true, false];
    expectArraysClose(await tf.logicalXor(a, b).data(), [1, 1, 0]);
  });

  it('TensorLike Chained', async () => {
    const a = tf.tensor1d([1, 0, 0], 'bool');
    const b = [false, true, false];
    expectArraysClose(await a.logicalXor(b).data(), [1, 1, 0]);
  });

  it('throws when passed a as a non-tensor', () => {
    expect(() => tf.logicalXor({} as tf.Tensor, tf.scalar(1, 'bool')))
        .toThrowError(/Argument 'a' passed to 'logicalXor' must be a Tensor/);
  });
  it('throws when passed b as a non-tensor', () => {
    expect(() => tf.logicalXor(tf.scalar(1, 'bool'), {} as tf.Tensor))
        .toThrowError(/Argument 'b' passed to 'logicalXor' must be a Tensor/);
  });
  it('accepts a tensor-like object', async () => {
    const a = [1, 0, 0, 1];
    const b = [0, 1, 0, 1];
    expectArraysClose(await tf.logicalXor(a, b).data(), [1, 1, 0, 0]);
  });
});

describeWithFlags('where', ALL_ENVS, () => {
  it('Scalars.', async () => {
    const a = tf.scalar(10);
    const b = tf.scalar(20);
    const c = tf.scalar(1, 'bool');

    expectArraysClose(await tf.where(c, a, b).data(), [10]);
  });

  it('Invalid condition type', () => {
    const c = tf.tensor1d([1, 0, 1, 0], 'int32');
    const a = tf.tensor1d([10, 10, 10, 10], 'bool');
    const b = tf.tensor1d([20, 20, 20, 20], 'bool');
    const f = () => {
      tf.where(c, a, b);
    };
    expect(f).toThrowError();
  });

  it('Tensor1D', async () => {
    const c = tf.tensor1d([1, 0, 1, 0], 'bool');
    const a = tf.tensor1d([10, 10, 10, 10]);
    const b = tf.tensor1d([20, 20, 20, 20]);
    expectArraysClose(await tf.where(c, a, b).data(), [10, 20, 10, 20]);
  });

  it('Tensor1D different a/b shapes', () => {
    let c = tf.tensor1d([1, 0, 1, 0], 'bool');
    let a = tf.tensor1d([10, 10, 10]);
    let b = tf.tensor1d([20, 20, 20, 20]);
    let f = () => {
      tf.where(c, a, b);
    };
    expect(f).toThrowError();

    c = tf.tensor1d([1, 0, 1, 0], 'bool');
    a = tf.tensor1d([10, 10, 10, 10]);
    b = tf.tensor1d([20, 20, 20]);
    f = () => {
      tf.where(c, a, b);
    };
  });

  it('Tensor1D different condition/a shapes', () => {
    const c = tf.tensor1d([1, 0, 1, 0], 'bool');
    const a = tf.tensor1d([10, 10, 10]);
    const b = tf.tensor1d([20, 20, 20]);
    const f = () => {
      tf.where(c, a, b);
    };
    expect(f).toThrowError();
  });

  it('Tensor2D', async () => {
    const c = tf.tensor2d([[1, 0], [0, 1]], [2, 2], 'bool');
    const a = tf.tensor2d([[10, 10], [10, 10]], [2, 2]);
    const b = tf.tensor2d([[5, 5], [5, 5]], [2, 2]);
    expectArraysClose(await tf.where(c, a, b).data(), [10, 5, 5, 10]);
  });

  it('Tensor2D different a/b shapes', () => {
    let c = tf.tensor2d([[1, 1], [0, 0]], [2, 2], 'bool');
    let a = tf.tensor2d([[5, 5, 5], [5, 5, 5]], [2, 3]);
    let b = tf.tensor2d([[4, 4], [4, 4]], [2, 2]);
    let f = () => {
      tf.where(c, a, b);
    };
    expect(f).toThrowError();

    c = tf.tensor2d([[1, 1], [0, 0]], [2, 2], 'bool');
    a = tf.tensor2d([[5, 5], [5, 5]], [2, 2]);
    b = tf.tensor2d([[4, 4, 4], [4, 4, 4]], [2, 3]);
    f = () => {
      tf.where(c, a, b);
    };
    expect(f).toThrowError();
  });

  it('Tensor2D different condition/a shapes', () => {
    const c = tf.tensor2d([[1, 0], [0, 1]], [2, 2], 'bool');
    const a = tf.tensor2d([[10, 10, 10], [10, 10, 10]], [2, 3]);
    const b = tf.tensor2d([[5, 5, 5], [5, 5, 5]], [2, 3]);
    const f = () => {
      tf.where(c, a, b);
    };
    expect(f).toThrowError();
  });

  it('Tensor2D different `a` dimension w/ condition rank=1', async () => {
    const c = tf.tensor1d([1, 0, 1, 0], 'bool');
    let a = tf.tensor2d([[10, 10], [10, 10]], [2, 2]);
    let b = tf.tensor2d([[5, 5], [5, 5]], [2, 2]);
    const f = () => {
      tf.where(c, a, b);
    };
    expect(f).toThrowError();

    a = tf.tensor2d([[10], [10], [10], [10]], [4, 1]);
    b = tf.tensor2d([[5], [5], [5], [5]], [4, 1]);
    expectArraysClose(await tf.where(c, a, b).data(), [10, 5, 10, 5]);

    a = tf.tensor2d([[10, 10], [10, 10], [10, 10], [10, 10]], [4, 2]);
    b = tf.tensor2d([[5, 5], [5, 5], [5, 5], [5, 5]], [4, 2]);
    expectArraysClose(
        await tf.where(c, a, b).data(), [10, 10, 5, 5, 10, 10, 5, 5]);
  });

  it('Tensor3D', async () => {
    const c =
        tf.tensor3d([[[1], [0], [1]], [[0], [0], [0]]], [2, 3, 1], 'bool');
    const a = tf.tensor3d([[[5], [5], [5]], [[5], [5], [5]]], [2, 3, 1]);
    const b = tf.tensor3d([[[3], [3], [3]], [[3], [3], [3]]], [2, 3, 1]);
    expectArraysClose(await tf.where(c, a, b).data(), [5, 3, 5, 3, 3, 3]);
  });

  it('Tensor3D different a/b shapes', () => {
    const c =
        tf.tensor3d([[[1], [0], [1]], [[0], [0], [0]]], [2, 3, 1], 'bool');
    let a = tf.tensor3d([[[5], [5]], [[5], [5]]], [2, 2, 1]);
    let b = tf.tensor3d([[[3], [3], [3]], [[3], [3], [3]]], [2, 3, 1]);
    let f = () => {
      tf.where(c, a, b);
    };
    expect(f).toThrowError();

    a = tf.tensor3d([[[5], [5], [5]], [[5], [5], [5]]], [2, 3, 1]);
    b = tf.tensor3d([[[3], [3]], [[3], [3]]], [2, 2, 1]);
    f = () => {
      tf.where(c, a, b);
    };
    expect(f).toThrowError();
  });

  it('Tensor3D different condition/a shapes', () => {
    const c = tf.tensor3d([[[1], [0]], [[0], [0]]], [2, 2, 1], 'bool');
    const a = tf.tensor3d([[[5], [5], [5]], [[5], [5], [5]]], [2, 3, 1]);
    const b = tf.tensor3d([[[3], [3], [3]], [[3], [3], [3]]], [2, 3, 1]);
    const f = () => {
      tf.where(c, a, b);
    };
    expect(f).toThrowError();
  });

  it('Tensor3D different `a` dimension w/ condition rank=1', async () => {
    const c = tf.tensor1d([1, 0, 1, 0], 'bool');
    let a = tf.tensor3d([[[9, 9], [9, 9]], [[9, 9], [9, 9]]], [2, 2, 2]);
    let b = tf.tensor3d([[[8, 8], [8, 8]], [[8, 8], [8, 8]]], [2, 2, 2]);
    const f = () => {
      tf.where(c, a, b);
    };
    expect(f).toThrowError();

    a = tf.tensor3d([[[9]], [[9]], [[9]], [[9]]], [4, 1, 1]);
    b = tf.tensor3d([[[8]], [[8]], [[8]], [[8]]], [4, 1, 1]);
    expectArraysClose(await tf.where(c, a, b).data(), [9, 8, 9, 8]);

    a = tf.tensor3d(
        [[[9], [9]], [[9], [9]], [[9], [9]], [[9], [9]]], [4, 2, 1]);
    b = tf.tensor3d(
        [[[8], [8]], [[8], [8]], [[8], [8]], [[8], [8]]], [4, 2, 1]);
    expectArraysClose(await tf.where(c, a, b).data(), [9, 9, 8, 8, 9, 9, 8, 8]);
  });

  it('Tensor4D', async () => {
    const c = tf.tensor4d([1, 0, 1, 1], [2, 2, 1, 1], 'bool');
    const a = tf.tensor4d([7, 7, 7, 7], [2, 2, 1, 1]);
    const b = tf.tensor4d([3, 3, 3, 3], [2, 2, 1, 1]);
    expectArraysClose(await tf.where(c, a, b).data(), [7, 3, 7, 7]);
  });

  it('Tensor4D different a/b shapes', () => {
    const c = tf.tensor4d([1, 0, 1, 1], [2, 2, 1, 1], 'bool');
    let a = tf.tensor4d([7, 7, 7, 7, 7, 7, 7, 7], [2, 2, 2, 1]);
    let b = tf.tensor4d([3, 3, 3, 3], [2, 2, 1, 1]);
    let f = () => {
      tf.where(c, a, b);
    };
    expect(f).toThrowError();

    a = tf.tensor4d([7, 7, 7, 7], [2, 2, 1, 1]);
    b = tf.tensor4d([3, 3, 3, 3, 3, 3, 3, 3], [2, 2, 2, 1]);
    f = () => {
      tf.where(c, a, b);
    };
    expect(f).toThrowError();
  });

  it('Tensor4D different condition/a shapes', () => {
    const c = tf.tensor4d([1, 0, 1, 1, 1, 0, 1, 1], [2, 2, 2, 1], 'bool');
    const a = tf.tensor4d([7, 7, 7, 7], [2, 2, 1, 1]);
    const b = tf.tensor4d([3, 3, 3, 3], [2, 2, 1, 1]);
    const f = () => {
      tf.where(c, a, b);
    };
    expect(f).toThrowError();
  });

  it('Tensor4D different `a` dimension w/ condition rank=1', async () => {
    const c = tf.tensor1d([1, 0, 1, 0], 'bool');
    let a = tf.tensor4d([7, 7, 7, 7, 7, 7, 7, 7], [2, 2, 2, 1]);
    let b = tf.tensor4d([3, 3, 3, 3, 3, 3, 3, 3], [2, 2, 2, 1]);
    const f = () => {
      tf.where(c, a, b);
    };
    expect(f).toThrowError();

    a = tf.tensor4d([7, 7, 7, 7], [4, 1, 1, 1]);
    b = tf.tensor4d([3, 3, 3, 3], [4, 1, 1, 1]);
    expectArraysClose(await tf.where(c, a, b).data(), [7, 3, 7, 3]);

    a = tf.tensor4d([7, 7, 7, 7, 7, 7, 7, 7], [4, 2, 1, 1]);
    b = tf.tensor4d([3, 3, 3, 3, 3, 3, 3, 3], [4, 2, 1, 1]);
    expectArraysClose(await tf.where(c, a, b).data(), [7, 7, 3, 3, 7, 7, 3, 3]);
  });

  it('TensorLike', async () => {
    expectArraysClose(await tf.where(true, 10, 20).data(), [10]);
  });

  it('TensorLike Chained', async () => {
    const a = tf.scalar(10);
    expectArraysClose(await a.where(true, 20).data(), [10]);
  });

  it('throws when passed condition as a non-tensor', () => {
    expect(
        () => tf.where(
            {} as tf.Tensor, tf.scalar(1, 'bool'), tf.scalar(1, 'bool')))
        .toThrowError(
            /Argument 'condition' passed to 'where' must be a Tensor/);
  });
  it('throws when passed a as a non-tensor', () => {
    expect(
        () => tf.where(
            tf.scalar(1, 'bool'), {} as tf.Tensor, tf.scalar(1, 'bool')))
        .toThrowError(/Argument 'a' passed to 'where' must be a Tensor/);
  });
  it('throws when passed b as a non-tensor', () => {
    expect(
        () => tf.where(
            tf.scalar(1, 'bool'), tf.scalar(1, 'bool'), {} as tf.Tensor))
        .toThrowError(/Argument 'b' passed to 'where' must be a Tensor/);
  });

  it('accepts a tensor-like object', async () => {
    const a = 10;
    const b = 20;
    const c = 1;
    expectArraysClose(await tf.where(c, a, b).data(), [10]);
  });

  it('1D gradient', async () => {
    const c = tf.tensor1d([1, 0, 1], 'bool');
    const a = tf.tensor1d([1, 2, 3]);
    const b = tf.tensor1d([4, 5, 6]);
    const dy = tf.tensor1d([1, 2, 3]);
    const grads = tf.grads((c, a, b) => tf.where(c, a, b));
    const [dc, da, db] = grads([c, a, b], dy);
    expectArraysClose(await dc.data(), [0, 0, 0]);
    expectArraysClose(await da.data(), [1, 0, 3]);
    expectArraysClose(await db.data(), [0, 2, 0]);
    expect(dc.shape).toEqual(c.shape);
    expect(da.shape).toEqual(a.shape);
    expect(db.shape).toEqual(b.shape);
  });

  it('gradient with clones', async () => {
    const c = tf.tensor1d([1, 0, 1], 'bool');
    const a = tf.tensor1d([1, 2, 3]);
    const b = tf.tensor1d([4, 5, 6]);
    const dy = tf.tensor1d([1, 2, 3]);
    const grads = tf.grads(
        (c, a, b) => tf.where(c.clone(), a.clone(), b.clone()).clone());
    const [dc, da, db] = grads([c, a, b], dy);
    expectArraysClose(await dc.data(), [0, 0, 0]);
    expectArraysClose(await da.data(), [1, 0, 3]);
    expectArraysClose(await db.data(), [0, 2, 0]);
    expect(dc.shape).toEqual(c.shape);
    expect(da.shape).toEqual(a.shape);
    expect(db.shape).toEqual(b.shape);
  });

  it('2D gradient', async () => {
    const c = tf.tensor2d([1, 0, 1, 1, 1, 0], [2, 3], 'bool');
    const a = tf.tensor2d([1, 2, 3, 4, 5, 6], [2, 3]);
    const b = tf.tensor2d([7, 8, 9, 10, 11, 12], [2, 3]);
    const dy = tf.tensor2d([1, 2, 3, 4, 5, 6], [2, 3]);
    const grads = tf.grads((c, a, b) => tf.where(c, a, b));
    const [dc, da, db] = grads([c, a, b], dy);
    expectArraysClose(await dc.data(), [0, 0, 0, 0, 0, 0]);
    expectArraysClose(await da.data(), [1, 0, 3, 4, 5, 0]);
    expectArraysClose(await db.data(), [0, 2, 0, 0, 0, 6]);
    expect(dc.shape).toEqual(c.shape);
    expect(da.shape).toEqual(a.shape);
    expect(db.shape).toEqual(b.shape);
  });
  it('3D gradient', async () => {
    const c = tf.tensor3d([1, 1, 0, 1, 1, 0], [2, 3, 1], 'bool');
    const a = tf.tensor3d([1, 2, 3, 4, 5, 6], [2, 3, 1]);
    const b = tf.tensor3d([7, 8, 9, 10, 11, 12], [2, 3, 1]);
    const dy = tf.tensor3d([1, 2, 3, 4, 5, 6], [2, 3, 1]);
    const grads = tf.grads((c, a, b) => tf.where(c, a, b));
    const [dc, da, db] = grads([c, a, b], dy);
    expectArraysClose(await dc.data(), [0, 0, 0, 0, 0, 0]);
    expectArraysClose(await da.data(), [1, 2, 0, 4, 5, 0]);
    expectArraysClose(await db.data(), [0, 0, 3, 0, 0, 6]);
    expect(dc.shape).toEqual(c.shape);
    expect(da.shape).toEqual(a.shape);
    expect(db.shape).toEqual(b.shape);
  });
  it('4D gradient', async () => {
    const c = tf.tensor4d([1, 1, 0, 1], [2, 2, 1, 1], 'bool');
    const a = tf.tensor4d([1, 2, 3, 4], [2, 2, 1, 1]);
    const b = tf.tensor4d([5, 6, 7, 8], [2, 2, 1, 1]);
    const dy = tf.tensor4d([1, 2, 3, 4], [2, 2, 1, 1]);
    const grads = tf.grads((c, a, b) => tf.where(c, a, b));
    const [dc, da, db] = grads([c, a, b], dy);
    expectArraysClose(await dc.data(), [0, 0, 0, 0]);
    expectArraysClose(await da.data(), [1, 2, 0, 4]);
    expectArraysClose(await db.data(), [0, 0, 3, 0]);
    expect(dc.shape).toEqual(c.shape);
    expect(da.shape).toEqual(a.shape);
    expect(db.shape).toEqual(b.shape);
  });
});

describeWithFlags('whereAsync', ALL_ENVS, () => {
  it('1d tensor', async () => {
    const condition = tf.tensor1d([true, false, true, true], 'bool');
    const res = await tf.whereAsync(condition);
    expect(res.dtype).toBe('int32');
    expect(res.shape).toEqual([3, 1]);
    expectArraysClose(await res.data(), [0, 2, 3]);
  });

  it('2d tensor', async () => {
    const condition = tf.tensor2d(
        [[true, false, false], [false, true, true]], [2, 3], 'bool');
    const res = await tf.whereAsync(condition);
    expect(res.dtype).toBe('int32');
    expect(res.shape).toEqual([3, 2]);
    expectArraysClose(await res.data(), [0, 0, 1, 1, 1, 2]);
  });

  it('3d tensor', async () => {
    const condition = tf.tensor3d(
        [
          [[true, false, false], [false, true, true]],
          [[false, false, false], [true, true, false]]
        ],
        [2, 2, 3], 'bool');
    const res = await tf.whereAsync(condition);
    expect(res.dtype).toBe('int32');
    expect(res.shape).toEqual([5, 3]);
    expectArraysClose(
        await res.data(), [0, 0, 0, 0, 1, 1, 0, 1, 2, 1, 1, 0, 1, 1, 1]);
  });

  it('accepts a tensor-like object', async () => {
    const condition = [true, false, true];
    const res = await tf.whereAsync(condition);
    expect(res.dtype).toBe('int32');
    expect(res.shape).toEqual([2, 1]);
    expectArraysClose(await res.data(), [0, 2]);
  });

  it('throws error if condition is not of type bool', async () => {
    const condition = tf.tensor1d([1, 0, 1]);
    // expect(...).toThrowError() does not support async functions.
    // See https://github.com/jasmine/jasmine/issues/1410
    try {
      await tf.whereAsync(condition);
      throw new Error('The line above should have thrown an error');
    } catch (ex) {
      expect(ex.message)
          .toMatch(/Argument 'condition' passed to 'whereAsync' must be bool/);
    }
  });

  it('returns tensor with 0 in shape when no values are true', async () => {
    const condition = [[[false]], [[false]], [[false]]];
    const res = await tf.whereAsync(condition);
    expect(res.dtype).toBe('int32');
    expect(res.shape).toEqual([0, 3]);
    expectArraysClose(await res.data(), []);
  });
});
