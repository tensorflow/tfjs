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

describeWithFlags('equalStrict', ALL_ENVS, () => {
  it('Tensor1D - int32', async () => {
    let a = tf.tensor1d([1, 4, 5], 'int32');
    let b = tf.tensor1d([2, 3, 5], 'int32');
    expectArraysClose(await tf.equalStrict(a, b).data(), [0, 0, 1]);

    a = tf.tensor1d([2, 2, 2], 'int32');
    b = tf.tensor1d([2, 2, 2], 'int32');
    expectArraysClose(await tf.equalStrict(a, b).data(), [1, 1, 1]);

    a = tf.tensor1d([0, 0], 'int32');
    b = tf.tensor1d([3, 3], 'int32');
    expectArraysClose(await tf.equalStrict(a, b).data(), [0, 0]);
  });
  it('Tensor1D - float32', async () => {
    let a = tf.tensor1d([1.1, 4.1, 5.1], 'float32');
    let b = tf.tensor1d([2.2, 3.2, 5.1], 'float32');
    expectArraysClose(await tf.equalStrict(a, b).data(), [0, 0, 1]);

    a = tf.tensor1d([2.31, 2.31, 2.31], 'float32');
    b = tf.tensor1d([2.31, 2.31, 2.31], 'float32');
    expectArraysClose(await tf.equalStrict(a, b).data(), [1, 1, 1]);

    a = tf.tensor1d([0.45, 0.123], 'float32');
    b = tf.tensor1d([3.123, 3.321], 'float32');
    expectArraysClose(await tf.equalStrict(a, b).data(), [0, 0]);
  });
  it('mismatched Tensor1D shapes - int32', () => {
    const a = tf.tensor1d([1, 2], 'int32');
    const b = tf.tensor1d([1, 2, 3], 'int32');
    const f = () => {
      tf.equalStrict(a, b);
    };
    expect(f).toThrowError();
  });
  it('mismatched Tensor1D shapes - float32', () => {
    const a = tf.tensor1d([1.1, 2.1], 'float32');
    const b = tf.tensor1d([1.1, 2.1, 3.1], 'float32');
    const f = () => {
      tf.equalStrict(a, b);
    };
    expect(f).toThrowError();
  });
  it('NaNs in Tensor1D - float32', async () => {
    const a = tf.tensor1d([1.1, NaN, 2.1], 'float32');
    const b = tf.tensor1d([2.1, 3.1, NaN], 'float32');
    expectArraysClose(await tf.equalStrict(a, b).data(), [0, 0, 0]);
  });

  // Tensor2D:
  it('Tensor2D - int32', async () => {
    let a = tf.tensor2d([[1, 4, 5], [8, 9, 12]], [2, 3], 'int32');
    let b = tf.tensor2d([[2, 3, 6], [7, 10, 11]], [2, 3], 'int32');
    expectArraysClose(await tf.equalStrict(a, b).data(), [0, 0, 0, 0, 0, 0]);

    a = tf.tensor2d([[0, 0], [1, 1]], [2, 2], 'int32');
    b = tf.tensor2d([[0, 0], [1, 1]], [2, 2], 'int32');
    expectArraysClose(await tf.equalStrict(a, b).data(), [1, 1, 1, 1]);
  });
  it('Tensor2D - float32', async () => {
    let a = tf.tensor2d([[1.1, 4.1, 5.1], [8.1, 9.1, 12.1]], [2, 3], 'float32');
    let b =
        tf.tensor2d([[2.1, 4.1, 5.1], [7.1, 10.1, 11.1]], [2, 3], 'float32');
    expectArraysClose(await tf.equalStrict(a, b).data(), [0, 1, 1, 0, 0, 0]);

    a = tf.tensor2d([[0.2, 0.2], [1.2, 1.2]], [2, 2], 'float32');
    b = tf.tensor2d([[0.2, 0.2], [1.2, 1.2]], [2, 2], 'float32');
    expectArraysClose(await tf.equalStrict(a, b).data(), [1, 1, 1, 1]);
  });
  it('mismatch Tensor2D shapes - int32', () => {
    const a = tf.tensor2d([[3], [7]], [2, 1], 'int32');
    const b = tf.tensor2d([[2, 3, 4], [7, 8, 9]], [2, 3], 'int32');

    const f = () => {
      tf.equalStrict(a, b);
    };
    expect(f).toThrowError();
  });
  it('mismatch Tensor2D shapes - float32', () => {
    const a = tf.tensor2d([[1.1], [7.1]], [2, 1], 'float32');
    const b =
        tf.tensor2d([[0.1, 1.1, 2.1], [7.1, 8.1, 9.1]], [2, 3], 'float32');
    const f = () => {
      tf.equalStrict(a, b);
    };
    expect(f).toThrowError();
  });
  it('NaNs in Tensor2D - float32', async () => {
    const a = tf.tensor2d([[1.1, NaN], [1.1, NaN]], [2, 2], 'float32');
    const b = tf.tensor2d([[0.1, NaN], [1.1, NaN]], [2, 2], 'float32');
    expectArraysClose(await tf.equalStrict(a, b).data(), [0, 0, 1, 0]);
  });

  // Tensor3D:
  it('Tensor3D - int32', async () => {
    let a =
        tf.tensor3d([[[1], [4], [5]], [[8], [9], [12]]], [2, 3, 1], 'int32');
    let b =
        tf.tensor3d([[[2], [3], [6]], [[7], [10], [12]]], [2, 3, 1], 'int32');
    expectArraysClose(await tf.equalStrict(a, b).data(), [0, 0, 0, 0, 0, 1]);

    a = tf.tensor3d([[[0], [0], [0]], [[1], [1], [1]]], [2, 3, 1], 'int32');
    b = tf.tensor3d([[[0], [0], [0]], [[1], [1], [1]]], [2, 3, 1], 'int32');
    expectArraysClose(await tf.equalStrict(a, b).data(), [1, 1, 1, 1, 1, 1]);
  });
  it('Tensor3D - float32', async () => {
    let a = tf.tensor3d(
        [[[1.1], [4.1], [5.1]], [[8.1], [9.1], [12.1]]], [2, 3, 1], 'float32');
    let b = tf.tensor3d(
        [[[2.1], [3.1], [6.1]], [[7.1], [10.1], [12.1]]], [2, 3, 1], 'float32');
    expectArraysClose(await tf.equalStrict(a, b).data(), [0, 0, 0, 0, 0, 1]);

    a = tf.tensor3d(
        [[[0.1], [0.1], [0.1]], [[1.1], [1.1], [1.1]]], [2, 3, 1], 'float32');
    b = tf.tensor3d(
        [[[0.1], [0.1], [0.1]], [[1.1], [1.1], [1.1]]], [2, 3, 1], 'float32');
    expectArraysClose(await tf.equalStrict(a, b).data(), [1, 1, 1, 1, 1, 1]);
  });
  it('mismatch Tensor3D shapes - int32', () => {
    const a = tf.tensor3d(
        [[[1, 0], [2, 3], [4, 5]], [[6, 7], [9, 8], [10, 11]]], [2, 3, 2],
        'int32');
    const b =
        tf.tensor3d([[[1], [2], [3]], [[7], [10], [9]]], [2, 3, 1], 'int32');

    const f = () => {
      tf.equalStrict(a, b);
    };
    expect(f).toThrowError();
  });
  it('mismatch Tensor3D shapes - float32', () => {
    const a = tf.tensor3d(
        [
          [[1.1, 0.1], [2.1, 3.1], [4.1, 5.1]],
          [[6.1, 7.1], [9.1, 8.1], [10.1, 11.1]]
        ],
        [2, 3, 2], 'float32');
    const b = tf.tensor3d(
        [[[1.1], [2.1], [3.1]], [[7.1], [10.1], [9.1]]], [2, 3, 1], 'float32');

    const f = () => {
      tf.equalStrict(a, b);
    };
    expect(f).toThrowError();
  });
  it('NaNs in Tensor3D - float32', async () => {
    const a = tf.tensor3d(
        [[[1.1], [NaN], [1.1]], [[0.1], [0.1], [0.1]]], [2, 3, 1], 'float32');
    const b = tf.tensor3d(
        [[[0.1], [0.1], [1.1]], [[1.1], [0.1], [NaN]]], [2, 3, 1], 'float32');
    expectArraysClose(await tf.equalStrict(a, b).data(), [0, 0, 1, 0, 1, 0]);
  });

  // Tensor4D:
  it('Tensor4D - int32', async () => {
    let a = tf.tensor4d([1, 4, 5, 8], [2, 2, 1, 1], 'int32');
    let b = tf.tensor4d([2, 3, 6, 8], [2, 2, 1, 1], 'int32');
    expectArraysClose(await tf.equalStrict(a, b).data(), [0, 0, 0, 1]);

    a = tf.tensor4d([0, 1, 2, 3], [2, 2, 1, 1], 'int32');
    b = tf.tensor4d([0, 1, 2, 3], [2, 2, 1, 1], 'int32');
    expectArraysClose(await tf.equalStrict(a, b).data(), [1, 1, 1, 1]);

    a = tf.tensor4d([1, 1, 1, 1], [2, 2, 1, 1], 'int32');
    b = tf.tensor4d([2, 2, 2, 2], [2, 2, 1, 1], 'int32');
    expectArraysClose(await tf.equalStrict(a, b).data(), [0, 0, 0, 0]);
  });
  it('Tensor4D - float32', async () => {
    let a = tf.tensor4d([1.1, 4.1, 5.1, 8.1], [2, 2, 1, 1], 'float32');
    let b = tf.tensor4d([2.1, 3.1, 6.1, 8.1], [2, 2, 1, 1], 'float32');
    expectArraysClose(await tf.equalStrict(a, b).data(), [0, 0, 0, 1]);

    a = tf.tensor4d([0.1, 1.1, 2.2, 3.3], [2, 2, 1, 1], 'float32');
    b = tf.tensor4d([0.1, 1.1, 2.2, 3.3], [2, 2, 1, 1], 'float32');
    expectArraysClose(await tf.equalStrict(a, b).data(), [1, 1, 1, 1]);

    a = tf.tensor4d([0.1, 0.1, 0.1, 0.1], [2, 2, 1, 1], 'float32');
    b = tf.tensor4d([1.1, 1.1, 1.1, 1.1], [2, 2, 1, 1], 'float32');
    expectArraysClose(await tf.equalStrict(a, b).data(), [0, 0, 0, 0]);
  });
  it('mismatch Tensor4D shapes - int32', () => {
    const a = tf.tensor4d([1, 2, 5, 9], [2, 2, 1, 1], 'int32');
    const b = tf.tensor4d(
        [[[[1, 2]], [[3, 4]]], [[[5, 6]], [[7, 8]]]], [2, 2, 1, 2], 'int32');

    const f = () => {
      tf.equalStrict(a, b);
    };
    expect(f).toThrowError();
  });
  it('mismatch Tensor4D shapes - float32', () => {
    const a = tf.tensor4d([1.1, 2.1, 5.1, 9.1], [2, 2, 1, 1], 'float32');
    const b = tf.tensor4d(
        [[[[1.1, 2.1]], [[3.1, 4.1]]], [[[5.1, 6.1]], [[7.1, 8.1]]]],
        [2, 2, 1, 2], 'float32');

    const f = () => {
      tf.equalStrict(a, b);
    };
    expect(f).toThrowError();
  });
  it('NaNs in Tensor4D - float32', async () => {
    const a = tf.tensor4d([1.1, NaN, 1.1, 0.1], [2, 2, 1, 1], 'float32');
    const b = tf.tensor4d([0.1, 1.1, 1.1, NaN], [2, 2, 1, 1], 'float32');
    expectArraysClose(await tf.equalStrict(a, b).data(), [0, 0, 1, 0]);
  });

  it('accepts a tensor-like object', async () => {
    const a = [1, 4, 5];
    const b = [2, 3, 5];
    expectArraysClose(await tf.equalStrict(a, b).data(), [0, 0, 1]);
  });
});

describeWithFlags('notEqualStrict', ALL_ENVS, () => {
  it('Tensor1D - int32', async () => {
    let a = tf.tensor1d([1, 4, 5], 'int32');
    let b = tf.tensor1d([2, 3, 5], 'int32');
    expectArraysClose(await tf.notEqualStrict(a, b).data(), [1, 1, 0]);

    a = tf.tensor1d([2, 2, 2], 'int32');
    b = tf.tensor1d([2, 2, 2], 'int32');
    expectArraysClose(await tf.notEqualStrict(a, b).data(), [0, 0, 0]);

    a = tf.tensor1d([0, 0], 'int32');
    b = tf.tensor1d([3, 3], 'int32');
    expectArraysClose(await tf.notEqualStrict(a, b).data(), [1, 1]);
  });
  it('Tensor1D - float32', async () => {
    let a = tf.tensor1d([1.1, 4.1, 5.1], 'float32');
    let b = tf.tensor1d([2.2, 3.2, 5.1], 'float32');
    expectArraysClose(await tf.notEqualStrict(a, b).data(), [1, 1, 0]);

    a = tf.tensor1d([2.31, 2.31, 2.31], 'float32');
    b = tf.tensor1d([2.31, 2.31, 2.31], 'float32');
    expectArraysClose(await tf.notEqualStrict(a, b).data(), [0, 0, 0]);

    a = tf.tensor1d([0.45, 0.123], 'float32');
    b = tf.tensor1d([3.123, 3.321], 'float32');
    expectArraysClose(await tf.notEqualStrict(a, b).data(), [1, 1]);
  });
  it('mismatched Tensor1D shapes - int32', () => {
    const a = tf.tensor1d([1, 2], 'int32');
    const b = tf.tensor1d([1, 2, 3], 'int32');
    const f = () => {
      tf.notEqualStrict(a, b);
    };
    expect(f).toThrowError();
  });
  it('mismatched Tensor1D shapes - float32', () => {
    const a = tf.tensor1d([1.1, 2.1], 'float32');
    const b = tf.tensor1d([1.1, 2.1, 3.1], 'float32');
    const f = () => {
      tf.notEqualStrict(a, b);
    };
    expect(f).toThrowError();
  });
  it('NaNs in Tensor1D - float32', async () => {
    const a = tf.tensor1d([1.1, NaN, 2.1], 'float32');
    const b = tf.tensor1d([2.1, 3.1, NaN], 'float32');
    expectArraysClose(await tf.notEqualStrict(a, b).data(), [1, 1, 1]);
  });
  it('strict version throws when x and y are different shape', () => {
    const a = tf.tensor1d([2]);
    const b = tf.tensor1d([4, 2, -1]);

    expect(() => tf.notEqualStrict(a, b)).toThrowError();
    expect(() => tf.notEqualStrict(b, a)).toThrowError();
  });

  // Tensor2D:
  it('Tensor2D - int32', async () => {
    let a = tf.tensor2d([[1, 4, 5], [8, 9, 12]], [2, 3], 'int32');
    let b = tf.tensor2d([[2, 3, 6], [7, 10, 11]], [2, 3], 'int32');
    expectArraysClose(await tf.notEqualStrict(a, b).data(), [1, 1, 1, 1, 1, 1]);

    a = tf.tensor2d([[0, 0], [1, 1]], [2, 2], 'int32');
    b = tf.tensor2d([[0, 0], [1, 1]], [2, 2], 'int32');
    expectArraysClose(await tf.notEqualStrict(a, b).data(), [0, 0, 0, 0]);
  });
  it('Tensor2D - float32', async () => {
    let a = tf.tensor2d([[1.1, 4.1, 5.1], [8.1, 9.1, 12.1]], [2, 3], 'float32');
    let b =
        tf.tensor2d([[2.1, 4.1, 5.1], [7.1, 10.1, 11.1]], [2, 3], 'float32');
    expectArraysClose(await tf.notEqualStrict(a, b).data(), [1, 0, 0, 1, 1, 1]);

    a = tf.tensor2d([[0.2, 0.2], [1.2, 1.2]], [2, 2], 'float32');
    b = tf.tensor2d([[0.2, 0.2], [1.2, 1.2]], [2, 2], 'float32');
    expectArraysClose(await tf.notEqualStrict(a, b).data(), [0, 0, 0, 0]);
  });
  it('mismatch Tensor2D shapes - int32', () => {
    const a = tf.tensor2d([[3], [7]], [2, 1], 'int32');
    const b = tf.tensor2d([[2, 3, 4], [7, 8, 9]], [2, 3], 'int32');

    const f = () => {
      tf.notEqualStrict(a, b);
    };
    expect(f).toThrowError();
  });
  it('mismatch Tensor2D shapes - float32', () => {
    const a = tf.tensor2d([[1.1], [7.1]], [2, 1], 'float32');
    const b =
        tf.tensor2d([[0.1, 1.1, 2.1], [7.1, 8.1, 9.1]], [2, 3], 'float32');
    const f = () => {
      tf.notEqualStrict(a, b);
    };
    expect(f).toThrowError();
  });
  it('NaNs in Tensor2D - float32', async () => {
    const a = tf.tensor2d([[1.1, NaN], [1.1, NaN]], [2, 2], 'float32');
    const b = tf.tensor2d([[0.1, NaN], [1.1, NaN]], [2, 2], 'float32');
    expectArraysClose(await tf.notEqualStrict(a, b).data(), [1, 1, 0, 1]);
  });

  // Tensor3D:
  it('Tensor3D - int32', async () => {
    let a =
        tf.tensor3d([[[1], [4], [5]], [[8], [9], [12]]], [2, 3, 1], 'int32');
    let b =
        tf.tensor3d([[[2], [3], [6]], [[7], [10], [12]]], [2, 3, 1], 'int32');
    expectArraysClose(await tf.notEqualStrict(a, b).data(), [1, 1, 1, 1, 1, 0]);

    a = tf.tensor3d([[[0], [0], [0]], [[1], [1], [1]]], [2, 3, 1], 'int32');
    b = tf.tensor3d([[[0], [0], [0]], [[1], [1], [1]]], [2, 3, 1], 'int32');
    expectArraysClose(await tf.notEqualStrict(a, b).data(), [0, 0, 0, 0, 0, 0]);
  });
  it('Tensor3D - float32', async () => {
    let a = tf.tensor3d(
        [[[1.1], [4.1], [5.1]], [[8.1], [9.1], [12.1]]], [2, 3, 1], 'float32');
    let b = tf.tensor3d(
        [[[2.1], [3.1], [6.1]], [[7.1], [10.1], [12.1]]], [2, 3, 1], 'float32');
    expectArraysClose(await tf.notEqualStrict(a, b).data(), [1, 1, 1, 1, 1, 0]);

    a = tf.tensor3d(
        [[[0.1], [0.1], [0.1]], [[1.1], [1.1], [1.1]]], [2, 3, 1], 'float32');
    b = tf.tensor3d(
        [[[0.1], [0.1], [0.1]], [[1.1], [1.1], [1.1]]], [2, 3, 1], 'float32');
    expectArraysClose(await tf.notEqualStrict(a, b).data(), [0, 0, 0, 0, 0, 0]);
  });
  it('mismatch Tensor3D shapes - int32', () => {
    const a = tf.tensor3d(
        [[[1, 0], [2, 3], [4, 5]], [[6, 7], [9, 8], [10, 11]]], [2, 3, 2],
        'int32');
    const b =
        tf.tensor3d([[[1], [2], [3]], [[7], [10], [9]]], [2, 3, 1], 'int32');

    const f = () => {
      tf.notEqualStrict(a, b);
    };
    expect(f).toThrowError();
  });
  it('mismatch Tensor3D shapes - float32', () => {
    const a = tf.tensor3d(
        [
          [[1.1, 0.1], [2.1, 3.1], [4.1, 5.1]],
          [[6.1, 7.1], [9.1, 8.1], [10.1, 11.1]]
        ],
        [2, 3, 2], 'float32');
    const b = tf.tensor3d(
        [[[1.1], [2.1], [3.1]], [[7.1], [10.1], [9.1]]], [2, 3, 1], 'float32');

    const f = () => {
      tf.notEqualStrict(a, b);
    };
    expect(f).toThrowError();
  });
  it('NaNs in Tensor3D - float32', async () => {
    const a = tf.tensor3d(
        [[[1.1], [NaN], [1.1]], [[0.1], [0.1], [0.1]]], [2, 3, 1], 'float32');
    const b = tf.tensor3d(
        [[[0.1], [0.1], [1.1]], [[1.1], [0.1], [NaN]]], [2, 3, 1], 'float32');
    expectArraysClose(await tf.notEqualStrict(a, b).data(), [1, 1, 0, 1, 0, 1]);
  });

  // Tensor4D:
  it('Tensor4D - int32', async () => {
    let a = tf.tensor4d([1, 4, 5, 8], [2, 2, 1, 1], 'int32');
    let b = tf.tensor4d([2, 3, 6, 8], [2, 2, 1, 1], 'int32');
    expectArraysClose(await tf.notEqualStrict(a, b).data(), [1, 1, 1, 0]);

    a = tf.tensor4d([0, 1, 2, 3], [2, 2, 1, 1], 'int32');
    b = tf.tensor4d([0, 1, 2, 3], [2, 2, 1, 1], 'int32');
    expectArraysClose(await tf.notEqualStrict(a, b).data(), [0, 0, 0, 0]);

    a = tf.tensor4d([1, 1, 1, 1], [2, 2, 1, 1], 'int32');
    b = tf.tensor4d([2, 2, 2, 2], [2, 2, 1, 1], 'int32');
    expectArraysClose(await tf.notEqualStrict(a, b).data(), [1, 1, 1, 1]);
  });
  it('Tensor4D - float32', async () => {
    let a = tf.tensor4d([1.1, 4.1, 5.1, 8.1], [2, 2, 1, 1], 'float32');
    let b = tf.tensor4d([2.1, 3.1, 6.1, 8.1], [2, 2, 1, 1], 'float32');
    expectArraysClose(await tf.notEqualStrict(a, b).data(), [1, 1, 1, 0]);

    a = tf.tensor4d([0.1, 1.1, 2.2, 3.3], [2, 2, 1, 1], 'float32');
    b = tf.tensor4d([0.1, 1.1, 2.2, 3.3], [2, 2, 1, 1], 'float32');
    expectArraysClose(await tf.notEqualStrict(a, b).data(), [0, 0, 0, 0]);

    a = tf.tensor4d([0.1, 0.1, 0.1, 0.1], [2, 2, 1, 1], 'float32');
    b = tf.tensor4d([1.1, 1.1, 1.1, 1.1], [2, 2, 1, 1], 'float32');
    expectArraysClose(await tf.notEqualStrict(a, b).data(), [1, 1, 1, 1]);
  });
  it('mismatch Tensor4D shapes - int32', () => {
    const a = tf.tensor4d([1, 2, 5, 9], [2, 2, 1, 1], 'int32');
    const b = tf.tensor4d(
        [[[[1, 2]], [[3, 4]]], [[[5, 6]], [[7, 8]]]], [2, 2, 1, 2], 'int32');

    const f = () => {
      tf.notEqualStrict(a, b);
    };
    expect(f).toThrowError();
  });
  it('mismatch Tensor4D shapes - float32', () => {
    const a = tf.tensor4d([1.1, 2.1, 5.1, 9.1], [2, 2, 1, 1], 'float32');
    const b = tf.tensor4d(
        [[[[1.1, 2.1]], [[3.1, 4.1]]], [[[5.1, 6.1]], [[7.1, 8.1]]]],
        [2, 2, 1, 2], 'float32');

    const f = () => {
      tf.notEqualStrict(a, b);
    };
    expect(f).toThrowError();
  });
  it('NaNs in Tensor4D - float32', async () => {
    const a = tf.tensor4d([1.1, NaN, 1.1, 0.1], [2, 2, 1, 1], 'float32');
    const b = tf.tensor4d([0.1, 1.1, 1.1, NaN], [2, 2, 1, 1], 'float32');
    expectArraysClose(await tf.notEqualStrict(a, b).data(), [1, 1, 0, 1]);
  });

  it('accepts a tensor-like object', async () => {
    const a = [1, 4, 5];
    const b = [2, 3, 5];
    expectArraysClose(await tf.notEqualStrict(a, b).data(), [1, 1, 0]);
  });
});

describeWithFlags('lessStrict', ALL_ENVS, () => {
  it('Tensor1D - strict version throws when a and b are different shape',
     () => {
       const a = tf.tensor1d([2]);
       const b = tf.tensor1d([4, 2, -1]);

       expect(() => tf.lessStrict(a, b)).toThrowError();
       expect(() => tf.lessStrict(b, a)).toThrowError();
     });

  // Tensor2D:
  it('Tensor2D - strict version throws when a and b are different shape',
     () => {
       const a = tf.tensor2d([[1.1], [7.1]], [2, 1], 'float32');
       const b =
           tf.tensor2d([[0.1, 1.1, 2.1], [7.1, 8.1, 9.1]], [2, 3], 'float32');

       expect(() => tf.lessStrict(a, b)).toThrowError();
       expect(() => tf.lessStrict(b, a)).toThrowError();
     });

  // Tensor3D:
  it('Tensor3D - strict version throws when a and b are different shape',
     () => {
       const a = tf.tensor3d(
           [
             [[1.1, 0.1], [2.1, 3.1], [4.1, 5.1]],
             [[6.1, 7.1], [9.1, 8.1], [10.1, 11.1]]
           ],
           [2, 3, 2], 'float32');
       const b = tf.tensor3d(
           [[[1.1], [2.1], [3.1]], [[7.1], [10.1], [9.1]]], [2, 3, 1],
           'float32');

       expect(() => tf.lessStrict(a, b)).toThrowError();
       expect(() => tf.lessStrict(b, a)).toThrowError();
     });

  // Tensor4D:
  it('Tensor4D - strict version throws when a and b are different shape',
     () => {
       const a = tf.tensor4d([1.1, 2.1, 5.1, 9.1], [2, 2, 1, 1], 'float32');
       const b = tf.tensor4d(
           [[[[1.1, 2.1]], [[3.1, 4.1]]], [[[5.1, 6.1]], [[7.1, 8.1]]]],
           [2, 2, 1, 2], 'float32');

       expect(() => tf.lessStrict(a, b)).toThrowError();
       expect(() => tf.lessStrict(b, a)).toThrowError();
     });

  it('accepts a tensor-like object', async () => {
    const a = [1, 4, 5];
    const b = [2, 3, 5];

    const res = tf.lessStrict(a, b);
    expect(res.dtype).toBe('bool');
    expectArraysClose(await res.data(), [1, 0, 0]);
  });
});

describeWithFlags('lessEqualStrict', ALL_ENVS, () => {
  it('Tensor1D - strict version throws when a and b are different shape',
     () => {
       const a = tf.tensor1d([2]);
       const b = tf.tensor1d([4, 2, -1]);

       expect(() => tf.lessEqualStrict(a, b)).toThrowError();
       expect(() => tf.lessEqualStrict(b, a)).toThrowError();
     });

  // Tensor2D:
  it('Tensor2D - strict version throws when a and b are different shape',
     () => {
       const a = tf.tensor2d([[1.1], [7.1]], [2, 1], 'float32');
       const b =
           tf.tensor2d([[0.1, 1.1, 2.1], [7.1, 8.1, 9.1]], [2, 3], 'float32');

       expect(() => tf.lessEqualStrict(a, b)).toThrowError();
       expect(() => tf.lessEqualStrict(b, a)).toThrowError();
     });

  // Tensor3D:
  it('Tensor3D - strict version throws when a and b are different shape',
     () => {
       const a = tf.tensor3d(
           [
             [[1.1, 0.1], [2.1, 3.1], [4.1, 5.1]],
             [[6.1, 7.1], [9.1, 8.1], [10.1, 11.1]]
           ],
           [2, 3, 2], 'float32');
       const b = tf.tensor3d(
           [[[1.1], [2.1], [3.1]], [[7.1], [10.1], [9.1]]], [2, 3, 1],
           'float32');

       expect(() => tf.lessEqualStrict(a, b)).toThrowError();
       expect(() => tf.lessEqualStrict(b, a)).toThrowError();
     });

  // Tensor4D:
  it('Tensor4D - strict version throws when a and b are different shape',
     () => {
       const a = tf.tensor4d([1.1, 2.1, 5.1, 9.1], [2, 2, 1, 1], 'float32');
       const b = tf.tensor4d(
           [[[[1.1, 2.1]], [[3.1, 4.1]]], [[[5.1, 6.1]], [[7.1, 8.1]]]],
           [2, 2, 1, 2], 'float32');

       expect(() => tf.lessEqualStrict(a, b)).toThrowError();
       expect(() => tf.lessEqualStrict(b, a)).toThrowError();
     });

  it('accepts a tensor-like object', async () => {
    const a = [1, 4, 5];
    const b = [2, 3, 5];

    const res = tf.lessStrict(a, b);
    expect(res.dtype).toBe('bool');
    expectArraysClose(await res.data(), [1, 0, 0]);
  });
});

describeWithFlags('lessEqualStrict', ALL_ENVS, () => {
  it('Tensor1D - strict version throws when a and b are different shape',
     () => {
       const a = tf.tensor1d([2]);
       const b = tf.tensor1d([4, 2, -1]);

       expect(() => tf.lessEqualStrict(a, b)).toThrowError();
       expect(() => tf.lessEqualStrict(b, a)).toThrowError();
     });

  // Tensor2D:
  it('Tensor2D - strict version throws when a and b are different shape',
     () => {
       const a = tf.tensor2d([[1.1], [7.1]], [2, 1], 'float32');
       const b =
           tf.tensor2d([[0.1, 1.1, 2.1], [7.1, 8.1, 9.1]], [2, 3], 'float32');

       expect(() => tf.lessEqualStrict(a, b)).toThrowError();
       expect(() => tf.lessEqualStrict(b, a)).toThrowError();
     });

  // Tensor3D:
  it('Tensor3D - strict version throws when a and b are different shape',
     () => {
       const a = tf.tensor3d(
           [
             [[1.1, 0.1], [2.1, 3.1], [4.1, 5.1]],
             [[6.1, 7.1], [9.1, 8.1], [10.1, 11.1]]
           ],
           [2, 3, 2], 'float32');
       const b = tf.tensor3d(
           [[[1.1], [2.1], [3.1]], [[7.1], [10.1], [9.1]]], [2, 3, 1],
           'float32');

       expect(() => tf.lessEqualStrict(a, b)).toThrowError();
       expect(() => tf.lessEqualStrict(b, a)).toThrowError();
     });

  // Tensor4D:
  it('Tensor4D - strict version throws when a and b are different shape',
     () => {
       const a = tf.tensor4d([1.1, 2.1, 5.1, 9.1], [2, 2, 1, 1], 'float32');
       const b = tf.tensor4d(
           [[[[1.1, 2.1]], [[3.1, 4.1]]], [[[5.1, 6.1]], [[7.1, 8.1]]]],
           [2, 2, 1, 2], 'float32');

       expect(() => tf.lessEqualStrict(a, b)).toThrowError();
       expect(() => tf.lessEqualStrict(b, a)).toThrowError();
     });

  it('accepts a tensor-like object', async () => {
    const a = [1, 4, 5];
    const b = [2, 3, 5];
    const res = tf.lessEqualStrict(a, b);

    expect(res.dtype).toBe('bool');
    expectArraysClose(await res.data(), [1, 0, 1]);
  });
});

describeWithFlags('greaterStrict', ALL_ENVS, () => {
  it('Tensor1D - strict version throws when a and b are different shape',
     () => {
       const a = tf.tensor1d([2]);
       const b = tf.tensor1d([4, 2, -1]);

       expect(() => tf.greaterStrict(a, b)).toThrowError();
       expect(() => tf.greaterStrict(b, a)).toThrowError();
     });

  // Tensor2D:
  it('Tensor2D - strict version throws when a and b are different shape',
     () => {
       const a = tf.tensor2d([[1.1], [7.1]], [2, 1], 'float32');
       const b =
           tf.tensor2d([[0.1, 1.1, 2.1], [7.1, 8.1, 9.1]], [2, 3], 'float32');

       expect(() => tf.greaterStrict(a, b)).toThrowError();
       expect(() => tf.greaterStrict(b, a)).toThrowError();
     });

  // Tensor3D:
  it('Tensor3D - strict version throws when a and b are different shape',
     () => {
       const a = tf.tensor3d(
           [
             [[1.1, 0.1], [2.1, 3.1], [4.1, 5.1]],
             [[6.1, 7.1], [9.1, 8.1], [10.1, 11.1]]
           ],
           [2, 3, 2], 'float32');
       const b = tf.tensor3d(
           [[[1.1], [2.1], [3.1]], [[7.1], [10.1], [9.1]]], [2, 3, 1],
           'float32');

       expect(() => tf.greaterStrict(a, b)).toThrowError();
       expect(() => tf.greaterStrict(b, a)).toThrowError();
     });

  // Tensor4D:
  it('Tensor4D - strict version throws when a and b are different shape',
     () => {
       const a = tf.tensor4d([1.1, 2.1, 5.1, 9.1], [2, 2, 1, 1], 'float32');
       const b = tf.tensor4d(
           [[[[1.1, 2.1]], [[3.1, 4.1]]], [[[5.1, 6.1]], [[7.1, 8.1]]]],
           [2, 2, 1, 2], 'float32');

       expect(() => tf.greaterStrict(a, b)).toThrowError();
       expect(() => tf.greaterStrict(b, a)).toThrowError();
     });

  it('accepts a tensor-like object', async () => {
    const a = [1, 4, 5];
    const b = [2, 3, 5];

    const res = tf.greaterStrict(a, b);
    expect(res.dtype).toBe('bool');
    expectArraysClose(await res.data(), [0, 1, 0]);
  });
});

describeWithFlags('greaterEqualStrict', ALL_ENVS, () => {
  it('Tensor1D - strict version throws when a and b are different shape',
     () => {
       const a = tf.tensor1d([2]);
       const b = tf.tensor1d([4, 2, -1]);

       expect(() => tf.greaterEqualStrict(a, b)).toThrowError();
       expect(() => tf.greaterEqualStrict(b, a)).toThrowError();
     });

  // Tensor2D:
  it('Tensor2D - strict version throws when a and b are different shape',
     () => {
       const a = tf.tensor2d([[1.1], [7.1]], [2, 1], 'float32');
       const b =
           tf.tensor2d([[0.1, 1.1, 2.1], [7.1, 8.1, 9.1]], [2, 3], 'float32');

       expect(() => tf.greaterEqualStrict(a, b)).toThrowError();
       expect(() => tf.greaterEqualStrict(b, a)).toThrowError();
     });

  // Tensor3D:
  it('Tensor3D - strict version throws when a and b are different shape',
     () => {
       const a = tf.tensor3d(
           [
             [[1.1, 0.1], [2.1, 3.1], [4.1, 5.1]],
             [[6.1, 7.1], [9.1, 8.1], [10.1, 11.1]]
           ],
           [2, 3, 2], 'float32');
       const b = tf.tensor3d(
           [[[1.1], [2.1], [3.1]], [[7.1], [10.1], [9.1]]], [2, 3, 1],
           'float32');

       expect(() => tf.greaterEqualStrict(a, b)).toThrowError();
       expect(() => tf.greaterEqualStrict(b, a)).toThrowError();
     });

  // Tensor4D:
  it('Tensor4D - strict version throws when a and b are different shape',
     () => {
       const a = tf.tensor4d([1.1, 2.1, 5.1, 9.1], [2, 2, 1, 1], 'float32');
       const b = tf.tensor4d(
           [[[[1.1, 2.1]], [[3.1, 4.1]]], [[[5.1, 6.1]], [[7.1, 8.1]]]],
           [2, 2, 1, 2], 'float32');

       expect(() => tf.greaterEqualStrict(a, b)).toThrowError();
       expect(() => tf.greaterEqualStrict(b, a)).toThrowError();
     });

  it('accepts a tensor-like object', async () => {
    const a = [1, 4, 5];
    const b = [2, 3, 5];
    const res = tf.greaterEqualStrict(a, b);

    expect(res.dtype).toBe('bool');
    expectArraysClose(await res.data(), [0, 1, 1]);
  });
});
