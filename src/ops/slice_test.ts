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
import {ALL_ENVS, describeWithFlags, SYNC_BACKEND_ENVS} from '../jasmine_util';
import {expectArraysClose} from '../test_util';
import {Rank} from '../types';

describeWithFlags('slice1d', ALL_ENVS, () => {
  it('slices 1x1 into 1x1 (effectively a copy)', async () => {
    const a = tf.tensor1d([5]);
    const result = tf.slice1d(a, 0, 1);

    expect(result.shape).toEqual([1]);
    expectArraysClose(await result.data(), 5);
  });

  it('slices 5x1 into shape 2x1 starting at 3', async () => {
    const a = tf.tensor1d([1, 2, 3, 4, 5]);
    const result = tf.slice1d(a, 3, 2);

    expect(result.shape).toEqual([2]);
    expectArraysClose(await result.data(), [4, 5]);
  });

  it('slices 5x1 into shape 3x1 starting at 1', async () => {
    const a = tf.tensor1d([1, 2, 3, 4, 5]);
    const result = tf.slice1d(a, 1, 3);

    expect(result.shape).toEqual([3]);
    expectArraysClose(await result.data(), [2, 3, 4]);
  });

  it('grad', async () => {
    const a = tf.tensor1d([1, 2, 3, 4, 5]);
    const dy = tf.tensor1d([10, 100]);
    const da = tf.grad((a: tf.Tensor1D) => tf.slice1d(a, 1, 2))(a, dy);
    expect(da.shape).toEqual([5]);
    expectArraysClose(await da.data(), [0, 10, 100, 0, 0]);
  });

  it('gradient with clones', async () => {
    const a = tf.tensor1d([1, 2, 3, 4, 5]);
    const dy = tf.tensor1d([10, 100]);
    const da =
        tf.grad((a: tf.Tensor1D) => tf.slice1d(a.clone(), 1, 2).clone())(a, dy);
    expect(da.shape).toEqual([5]);
    expectArraysClose(await da.data(), [0, 10, 100, 0, 0]);
  });

  it('accepts a tensor-like object', async () => {
    const a = [5];
    const result = tf.slice1d(a, 0, 1);
    expect(result.shape).toEqual([1]);
    expectArraysClose(await result.data(), 5);
  });
});

describeWithFlags('slice2d', ALL_ENVS, () => {
  it('slicing a 1x1 from a 1x1 returns a 1x1', () => {
    const a = tf.tensor2d([0], [1, 1]);
    const b = tf.slice2d(a, [0, 0], [1, 1]);
    expect(b.shape).toEqual([1, 1]);
  });

  it('returns a tensor of slice size', () => {
    const a = tf.zeros<Rank.R2>([100, 100]);
    const b = tf.slice2d(a, [0, 0], [12, 34]);
    expect(b.shape).toEqual([12, 34]);
  });

  it('returns the upper-left submatrix when begin is [0, 0]', async () => {
    const a = tf.randomUniform<Rank.R2>([10, 10], -1, 1);
    const b = tf.slice2d(a, [0, 0], [2, 2]);
    const aValues = await a.data();

    expectArraysClose(
        await b.data(), [aValues[0], aValues[1], aValues[10], aValues[11]]);
  });

  it('returns the rectangle specified', async () => {
    const a = tf.tensor2d([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [4, 3]);
    const b = tf.slice2d(a, [1, 1], [3, 2]);

    expectArraysClose(await b.data(), [5, 6, 8, 9, 11, 12]);
  });

  it('throws when requesting out of bounds slice', () => {
    const a = tf.tensor2d([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [4, 3]);
    expect(() => tf.slice2d(a, [1, 1], [10, 10])).toThrowError();
  });

  it('grad', async () => {
    const a = tf.tensor2d([[1, 2, 3], [4, 5, 6]]);
    const dy = tf.tensor2d([[20], [50]]);
    const da =
        tf.grad((x: tf.Tensor2D) => tf.slice2d(a, [0, 1], [2, 1]))(a, dy);
    expect(da.shape).toEqual([2, 3]);
    expectArraysClose(await da.data(), [0, 20, 0, 0, 50, 0]);
  });

  it('accepts a tensor-like object', () => {
    const a = [[0]];  // 1x1
    const b = tf.slice2d(a, [0, 0], [1, 1]);
    expect(b.shape).toEqual([1, 1]);
  });

  it('slice an already sliced tensor, first was not continous', async () => {
    const a = [
      [1, 2, 3, 4],
      [5, 6, 7, 8],
      [9, 10, 11, 12],
    ];  // 3x4.
    const b = tf.slice(a, [0, 1]);
    const c = tf.slice(b, [1, 1], [1, 1]);
    expect(c.shape).toEqual([1, 1]);
    expectArraysClose(await c.data(), [7]);
  });

  it('slice an already sliced tensor, first was continous', async () => {
    const a = [
      [1, 2, 3, 4],
      [5, 6, 7, 8],
      [9, 10, 11, 12],
    ];  // 3x4.
    const b = tf.slice(a, [1, 0]);
    const c = tf.slice(b, [1, 0]);
    expect(c.shape).toEqual([1, 4]);
    expectArraysClose(await c.data(), [9, 10, 11, 12]);
  });

  it('slice an already sliced tensor and do async read', async () => {
    const a = [
      [1, 2, 3, 4],
      [5, 6, 7, 8],
      [9, 10, 11, 12],
    ];  // 3x4.
    const b = tf.slice(a, [0, 1]);
    const c = tf.slice(b, [1, 1], [1, 1]);
    expect(c.shape).toEqual([1, 1]);
    expectArraysClose(await c.data(), new Float32Array([7]));
  });

  it('square a sliced texture, followed by non-sliced texture of same shape',
     async () => {  // Tests collisions in the shader cache.
       // Make a 2x3 tensor, upload to gpu and reshape to 3x2.
       const input = tf.tensor([[1, 2, 3], [4, 5, 6]]).abs().as2D(3, 2);
       const slicedInput = tf.slice(input, [0, 0], [3, 2]);
       // First square program takes the sliced input.
       const a = slicedInput.square();
       expectArraysClose(await a.data(), [1, 4, 9, 16, 25, 36]);
       // Second square program takes the non-sliced input.
       const b = tf.square(input);
       expectArraysClose(await b.data(), [1, 4, 9, 16, 25, 36]);
     });

  it('square a non-sliced texture, followed by a sliced texture of same shape',
     async () => {  // Tests collisions in the shader cache.
       // Make a 2x3 tensor, upload to gpu and reshape to 3x2.
       const input = tf.tensor([[1, 2, 3], [4, 5, 6]]).abs().as2D(3, 2);
       // Make a sliced version of the same tensor with the same shape.
       const slicedInput = tf.slice(input, [0, 0], [3, 2]);
       // First square program takes the non-sliced input.
       const a = input.square();
       expectArraysClose(await a.data(), [1, 4, 9, 16, 25, 36]);
       // Second square program takes the sliced input.
       const b = tf.square(slicedInput);
       expectArraysClose(await b.data(), [1, 4, 9, 16, 25, 36]);
     });

  it('slice a tensor and do async read', async () => {
    const a = [
      [1, 2, 3, 4],
      [5, 6, 7, 8],
      [9, 10, 11, 12],
    ];  // 3x4.
    const b = tf.slice(a, [0, 1], [3, 2]);
    expect(b.shape).toEqual([3, 2]);
    const vals = await b.data();
    expectArraysClose(vals, new Float32Array([2, 3, 6, 7, 10, 11]));
  });

  it('flatten a sliced tensor that was continous in memory', async () => {
    const a = [
      [1, 2, 3, 4],
      [5, 6, 7, 8],
      [9, 10, 11, 12],
    ];  // 3x4.
    const b = tf.slice(a, [1, 0]).flatten();
    expect(b.shape).toEqual([8]);
    expectArraysClose(await b.data(), [5, 6, 7, 8, 9, 10, 11, 12]);
  });

  it('slice a tensor that was not continous in memory', async () => {
    const a = [
      [1, 2, 3, 4],
      [5, 6, 7, 8],
      [9, 10, 11, 12],
    ];  // 3x4.
    const b = tf.slice(a, [0, 1]);
    expect(b.shape).toEqual([3, 3]);
    expectArraysClose(await b.data(), [2, 3, 4, 6, 7, 8, 10, 11, 12]);
  });

  it('flatten a sliced tensor that was not continous in memory', async () => {
    const a = [
      [1, 2, 3, 4],
      [5, 6, 7, 8],
      [9, 10, 11, 12],
    ];  // 3x4.
    const b = tf.slice(a, [0, 1]).flatten();
    expect(b.shape).toEqual([9]);
    expectArraysClose(await b.data(), [2, 3, 4, 6, 7, 8, 10, 11, 12]);
  });

  it('flatten a sliced tensor not continous in memory and run program',
     async () => {
       const a = [
         [1, 2, 3, 4],
         [5, 6, 7, 8],
         [9, 10, 11, 12],
       ];  // 3x4.
       const b = tf.slice(a, [0, 1]).flatten();
       const c = tf.square(b);
       expectArraysClose(await c.data(), [4, 9, 16, 36, 49, 64, 100, 121, 144]);
     });

  it('reshape a sliced 1d into a 2d tensor', async () => {
    const a = [1, 2, 3, 4, 5];
    const b = tf.slice(a, 1).as2D(2, 2);
    expect(b.shape).toEqual([2, 2]);
    expectArraysClose(await b.data(), [2, 3, 4, 5]);
  });

  it('reshape a sliced 1d into a 2d tensor and run program', async () => {
    const a = [1, 2, 3, 4, 5];
    const b = tf.slice(a, 1).as2D(2, 2).square();
    expect(b.shape).toEqual([2, 2]);
    expectArraysClose(await b.data(), [4, 9, 16, 25]);
  });

  it('broadcast the original with the sliced tensor', async () => {
    const a = [[1, 2], [3, 4]];
    const b = tf.slice(a, [0, 1]);
    const c = tf.add(a, b);
    expect(c.shape).toEqual([2, 2]);
    expectArraysClose(await c.data(), [3, 4, 7, 8]);
  });

  it('zero-sized slice out of a non-zero sized tensor', async () => {
    const a = tf.zeros([4, 2]);
    const res = tf.slice(a, [0, 0], [0, 2]);
    expect(res.shape).toEqual([0, 2]);
    expectArraysClose(await res.data(), []);
  });

  it('zero-sized slice out of a zero-sized tensor', async () => {
    const a = tf.zeros([0, 4]);
    const res = tf.slice(a, [0, 1], [0, 3]);
    expect(res.shape).toEqual([0, 3]);
    expectArraysClose(await res.data(), []);
  });
});

describeWithFlags('slice3d', ALL_ENVS, () => {
  it('slices 1x1x1 into shape 1x1x1 (effectively a copy)', async () => {
    const a = tf.tensor3d([[[5]]], [1, 1, 1]);
    const result = tf.slice3d(a, [0, 0, 0], [1, 1, 1]);

    expect(result.shape).toEqual([1, 1, 1]);
    expectArraysClose(await result.data(), [5]);
  });

  it('slices 2x2x2 array into 1x2x2 starting at [1, 0, 0]', async () => {
    const a = tf.tensor3d([1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2]);
    const result = tf.slice3d(a, [1, 0, 0], [1, 2, 2]);

    expect(result.shape).toEqual([1, 2, 2]);
    expectArraysClose(await result.data(), [5, 6, 7, 8]);
  });

  it('slices 2x2x2 array into 2x1x1 starting at [0, 1, 1]', async () => {
    const a = tf.tensor3d([1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2]);
    const result = tf.slice3d(a, [0, 1, 1], [2, 1, 1]);

    expect(result.shape).toEqual([2, 1, 1]);
    expectArraysClose(await result.data(), [4, 8]);
  });

  it('accepts a tensor-like object', async () => {
    const a = [[[5]]];  // 1x1x1
    const result = tf.slice3d(a, [0, 0, 0], [1, 1, 1]);

    expect(result.shape).toEqual([1, 1, 1]);
    expectArraysClose(await result.data(), [5]);
  });
});

describeWithFlags('slice4d', ALL_ENVS, () => {
  it('slices 1x1x1x1 into shape 1x1x1x1 (effectively a copy)', async () => {
    const a = tf.tensor4d([[[[5]]]], [1, 1, 1, 1]);
    const result = tf.slice4d(a, [0, 0, 0, 0], [1, 1, 1, 1]);

    expect(result.shape).toEqual([1, 1, 1, 1]);
    expectArraysClose(await result.data(), [5]);
  });

  it('slices 2x2x2x2 array into 1x2x2x2 starting at [1, 0, 0, 0]', async () => {
    const a = tf.tensor4d(
        [1, 2, 3, 4, 5, 6, 7, 8, 11, 22, 33, 44, 55, 66, 77, 88],
        [2, 2, 2, 2],
    );
    const result = tf.slice4d(a, [1, 0, 0, 0], [1, 2, 2, 2]);

    expect(result.shape).toEqual([1, 2, 2, 2]);
    expectArraysClose(await result.data(), [11, 22, 33, 44, 55, 66, 77, 88]);
  });

  it('slices 2x2x2x2 array into 2x1x1x1 starting at [0, 1, 1, 1]', async () => {
    const a = tf.tensor4d(
        [1, 2, 3, 4, 5, 6, 7, 8, 11, 22, 33, 44, 55, 66, 77, 88], [2, 2, 2, 2]);
    const result = tf.slice4d(a, [0, 1, 1, 1], [2, 1, 1, 1]);

    expect(result.shape).toEqual([2, 1, 1, 1]);
    expectArraysClose(await result.data(), [8, 88]);
  });

  it('accepts a tensor-like object', async () => {
    const a = [[[[5]]]];  // 1x1x1x1
    const result = tf.slice4d(a, [0, 0, 0, 0], [1, 1, 1, 1]);

    expect(result.shape).toEqual([1, 1, 1, 1]);
    expectArraysClose(await result.data(), [5]);
  });
});

describeWithFlags('slice5d', ALL_ENVS, () => {
  it('slices 1x1x1x1x1 into shape 1x1x1x1x1 (effectively a copy)', async () => {
    const a = tf.tensor5d([[[[[5]]]]], [1, 1, 1, 1, 1]);
    const result = tf.slice(a, [0, 0, 0, 0, 0], [1, 1, 1, 1, 1]);

    expect(result.shape).toEqual([1, 1, 1, 1, 1]);
    expectArraysClose(await result.data(), [5]);
  });

  it('slices 2x2x2x2x2 array into 1x2x2x2x2 starting at [1,0,0,0,0]',
     async () => {
       const a = tf.tensor5d(
           [
             1,  2,  3,   4,   5,   6,   7,   8,   9,   10, 11,
             12, 13, 14,  15,  16,  11,  22,  33,  44,  55, 66,
             77, 88, 111, 222, 333, 444, 555, 666, 777, 888
           ],
           [2, 2, 2, 2, 2]);
       const result = tf.slice(a, [1, 0, 0, 0, 0], [1, 2, 2, 2, 2]);

       expect(result.shape).toEqual([1, 2, 2, 2, 2]);
       expectArraysClose(await result.data(), [
         11, 22, 33, 44, 55, 66, 77, 88, 111, 222, 333, 444, 555, 666, 777, 888
       ]);
     });

  it('slices 2x2x2x2x2 array into 2x1x1x1x1 starting at [0,1,1,1,1]',
     async () => {
       const a = tf.tensor5d(
           [
             1,  2,  3,   4,   5,   6,   7,   8,   9,   10, 11,
             12, 13, 14,  15,  16,  11,  22,  33,  44,  55, 66,
             77, 88, 111, 222, 333, 444, 555, 666, 777, 888
           ],
           [2, 2, 2, 2, 2]);
       const result = tf.slice(a, [0, 1, 1, 1, 1], [2, 1, 1, 1, 1]);

       expect(result.shape).toEqual([2, 1, 1, 1, 1]);
       expectArraysClose(await result.data(), [16, 888]);
     });

  it('accepts a tensor-like object', async () => {
    const a = [[[[[5]]]]];  // 1x1x1x1x1
    const result = tf.slice(a, [0, 0, 0, 0, 0], [1, 1, 1, 1, 1]);

    expect(result.shape).toEqual([1, 1, 1, 1, 1]);
    expectArraysClose(await result.data(), [5]);
  });
});

describeWithFlags('slice6d', ALL_ENVS, () => {
  it('slices 1x1x1x1x1x1 into shape 1x1x1x1x1x1 (effectively a copy)',
     async () => {
       const a = tf.tensor6d([[[[[[5]]]]]], [1, 1, 1, 1, 1, 1]);
       const result = tf.slice(a, [0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1]);

       expect(result.shape).toEqual([1, 1, 1, 1, 1, 1]);
       expectArraysClose(await result.data(), [5]);
     });

  it('slices 2x2x2x2x2x2 array into 1x2x2x2x2x2 starting at [1,0,0,0,0,0]',
     async () => {
       const a = tf.tensor6d(
           [
             31,  32,  33,   34,   35,   36,   37,   38,   39,   310,  311,
             312, 313, 314,  315,  316,  311,  322,  333,  344,  355,  366,
             377, 388, 3111, 3222, 3333, 3444, 3555, 3666, 3777, 3888,

             1,   2,   3,    4,    5,    6,    7,    8,    9,    10,   11,
             12,  13,  14,   15,   16,   11,   22,   33,   44,   55,   66,
             77,  88,  111,  222,  333,  444,  555,  666,  777,  888
           ],
           [2, 2, 2, 2, 2, 2]);
       const result = tf.slice(a, [1, 0, 0, 0, 0, 0], [1, 2, 2, 2, 2, 2]);

       expect(result.shape).toEqual([1, 2, 2, 2, 2, 2]);
       expectArraysClose(await result.data(), [
         1,  2,  3,   4,   5,   6,   7,   8,   9,   10, 11,
         12, 13, 14,  15,  16,  11,  22,  33,  44,  55, 66,
         77, 88, 111, 222, 333, 444, 555, 666, 777, 888
       ]);
     });

  it('slices 2x2x2x2x2x2 array into 2x1x1x1x1x1 starting at [0,1,1,1,1,1]',
     async () => {
       const a = tf.tensor6d(
           [
             31,  32,  33,   34,   35,   36,   37,   38,   39,   310,  311,
             312, 313, 314,  315,  316,  311,  322,  333,  344,  355,  366,
             377, 388, 3111, 3222, 3333, 3444, 3555, 3666, 3777, 3888,

             1,   2,   3,    4,    5,    6,    7,    8,    9,    10,   11,
             12,  13,  14,   15,   16,   11,   22,   33,   44,   55,   66,
             77,  88,  111,  222,  333,  444,  555,  666,  777,  888
           ],
           [2, 2, 2, 2, 2, 2]);
       const result = tf.slice(a, [0, 1, 1, 1, 1, 1], [2, 1, 1, 1, 1, 1]);

       expect(result.shape).toEqual([2, 1, 1, 1, 1, 1]);
       expectArraysClose(await result.data(), [3888, 888]);
     });

  it('accepts a tensor-like object', async () => {
    const a = [[[[[[5]]]]]];  // 1x1x1x1x1x1
    const result = tf.slice(a, [0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1]);

    expect(result.shape).toEqual([1, 1, 1, 1, 1, 1]);
    expectArraysClose(await result.data(), [5]);
  });
});

describeWithFlags('slice ergonomics', ALL_ENVS, () => {
  it('slices 2x2x2 array into 2x1x1 no size', async () => {
    const a = tf.tensor3d([1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2]);
    const result = a.slice([0, 1, 1]);
    expect(result.shape).toEqual([2, 1, 1]);
    expectArraysClose(await result.data(), [4, 8]);
  });

  it('slices 2x2x2 array into 1x2x2 with scalar begin no size', async () => {
    const a = tf.tensor3d([1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2]);
    const result = a.slice(1);
    expect(result.shape).toEqual([1, 2, 2]);
    expectArraysClose(await result.data(), [5, 6, 7, 8]);
  });

  it('slices 2x2x2 array using 2d size and 2d size', async () => {
    const a = tf.tensor3d([1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2]);
    const result = a.slice([0, 1]);
    expect(result.shape).toEqual([2, 1, 2]);
    expectArraysClose(await result.data(), [3, 4, 7, 8]);
  });

  it('slices 2x2x2 array using negative size', async () => {
    const a = tf.tensor3d([1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2]);
    const result = a.slice([0, 1], [-1, 1]);
    expect(result.shape).toEqual([2, 1, 2]);
    expectArraysClose(await result.data(), [3, 4, 7, 8]);
  });

  it('slices 2x2x2 array using 1d size', async () => {
    const a = tf.tensor3d([1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2]);
    const result = a.slice(0, 1);
    expect(result.shape).toEqual([1, 2, 2]);
    expectArraysClose(await result.data(), [1, 2, 3, 4]);
  });

  it('throws when passed a non-tensor', () => {
    expect(() => tf.slice({} as tf.Tensor, 0, 0))
        .toThrowError(/Argument 'x' passed to 'slice' must be a Tensor/);
  });

  it('accepts a tensor-like object', async () => {
    const a = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]];  // 2x2x2
    const result = tf.slice(a, [0, 1, 1]);
    expect(result.shape).toEqual([2, 1, 1]);
    expectArraysClose(await result.data(), [4, 8]);
  });

  it('should match source tensor dtype', () => {
    const a = tf.tensor1d([1, 2, 3, 4, 5], 'int32');
    const b = a.asType('float32');

    expect(tf.slice(b, 0).dtype).toEqual('float32');
  });
});

describeWithFlags('shallow slicing', ALL_ENVS, () => {
  beforeAll(() => {
    tf.ENV.set('WEBGL_CPU_FORWARD', false);
  });

  it('shallow slice an input that was cast', async () => {
    const a = tf.tensor([[1, 2], [3, 4]], [2, 2], 'int32');
    const b = a.toFloat();
    const c = b.slice(1, 1);
    expect(c.dtype).toBe('float32');
    expect(c.shape).toEqual([1, 2]);
    expectArraysClose(await c.data(), [3, 4]);
  });

  it('delayed async read of sliced tensor has no mem leak', async () => {
    const a = tf.zeros([10]);
    const b = tf.slice(a, 0, 1);
    const nBefore = tf.memory().numTensors;
    expect(nBefore).toBe(2);
    await b.data();
    const nAfter = tf.memory().numTensors;
    expect(nAfter).toBe(2);
    tf.dispose([a, b]);
    expect(tf.memory().numTensors).toBe(0);
  });
});

describeWithFlags('shallow slicing', SYNC_BACKEND_ENVS, () => {
  it('delayed sync read of sliced tensor has no mem leak', () => {
    const a = tf.zeros([10]);
    const b = tf.slice(a, 0, 1);
    const nBefore = tf.memory().numTensors;
    expect(nBefore).toBe(2);
    b.dataSync();
    const nAfter = tf.memory().numTensors;
    expect(nAfter).toBe(2);
    tf.dispose([a, b]);
    expect(tf.memory().numTensors).toBe(0);
  });
});
