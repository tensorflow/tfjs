/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
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

import {tensor1d, tensor2d, tensor3d} from './ops';

describeWithFlags('einsum', ALL_ENVS, () => {
  it('two scalars', async () => {
    const x = tf.scalar(2);
    const y = tf.scalar(3);
    const out = tf.einsum(',->', x, y);
    expectArraysClose(await out.data(), 6);
  });

  it('1D tensor and scalars: reduce', async () => {
    const x = tensor1d([2, 3]);
    const y = tf.scalar(4);
    const out = tf.einsum('i,->', x, y);
    expectArraysClose(await out.data(), 20);
  });

  it('1D tensor and scalars: multiply', async () => {
    const x = tensor1d([2, 3]);
    const y = tf.scalar(4);
    const out = tf.einsum('i,->i', x, y);
    expectArraysClose(await out.data(), [8, 12]);
  });

  it('1d reduce sum', async () => {
    const x = tensor1d([2, 4, 6]);
    const out = tf.einsum('i->', x);
    expectArraysClose(await out.data(), 12);
  });

  it('2d matrix reduce sum', async () => {
    const x = tensor2d([[1, 2], [3, 4]]);
    const out = tf.einsum('ij->', x);
    expectArraysClose(await out.data(), 10);
  });

  it('2d matrices multiply and reduce summing', async () => {
    const x = tensor2d([[1, 2], [3, 4]]);
    const y = tensor2d([[4, 3], [2, 1]]);
    const out = tf.einsum('ij,ji->', x, y);
    expectArraysClose(await out.data(), 21);
  });

  it('2d matrix times scalar and reduce summing', async () => {
    const x = tensor2d([[1, 2], [3, 4]]);
    const y = tf.scalar(5);
    const out = tf.einsum('ij,->', x, y);
    expectArraysClose(await out.data(), 50);
  });

  it('two 1d tensors dot', async () => {
    const x = tensor1d([1, 3, 5]);
    const y = tensor1d([2, 4, 6]);
    const out = tf.einsum('i,i->', x, y);
    expectArraysClose(await out.data(), 44);
  });

  it('two 1d tensors outer', async () => {
    const x = tensor1d([1, 3, 5]);
    const y = tensor1d([2, 4, 6]);
    const out = tf.einsum('i,j->ij', x, y);
    expectArraysClose(await out.data(), [[2, 4, 6], [6, 12, 18], [10, 20, 30]]);
  });

  it('2d matrix calculate trace: duplicate axes not implemented yet', () => {
    const x = tensor2d([[1, 2], [3, 4]]);
    expect(() => tf.einsum('ii->', x)).toThrowError(/not implemented yet/);
  });

  it('2d and 1d matrix & vector multiply', async () => {
    const x = tensor2d([[1, 2, 3], [4, 5, 6]]);
    const y = tensor1d([2, 4, 6]);
    const out = tf.einsum('ij,j->i', x, y);
    expectArraysClose(await out.data(), [28, 64]);
  });

  it('2d matrix sum along columns', async () => {
    const x = tensor2d([[1, 2, 3], [4, 5, 6]]);
    const out = tf.einsum('ij->j', x);
    expectArraysClose(await out.data(), [5, 7, 9]);
  });

  it('2d matrix sum along rows', async () => {
    const x = tensor2d([[1, 2, 3], [4, 5, 6]]);
    const out = tf.einsum('ij->i', x);
    expectArraysClose(await out.data(), [6, 15]);
  });

  it('2d matrix transposing', async () => {
    const x = tensor2d([[1, 2, 3], [4, 5, 6]]);
    const out = tf.einsum('ij->ji', x);
    expectArraysClose(await out.data(), [[1, 4], [2, 5], [3, 6]]);
  });

  it('2d matrix multiply', async () => {
    const x = tensor2d([[1, 2, 3], [4, 5, 6]]);
    const y = tensor2d([[0, 1], [2, 3], [4, 5]]);
    const out = tf.einsum('ij,jk->ik', x, y);
    expectArraysClose(await out.data(), [[16, 22], [34, 49]]);
  });

  it('2d matrix multiply and transposing', async () => {
    const x = tensor2d([[1, 2, 3], [4, 5, 6]]);
    const y = tensor2d([[0, 1], [2, 3], [4, 5]]);
    const out = tf.einsum('ij,jk->ki', x, y);
    expectArraysClose(await out.data(), [[16, 34], [22, 49]]);
  });

  it('two 2d matrices batch dot', async () => {
    const x = tensor2d([[1, 2, 3], [4, 5, 6]]);
    const y = tensor2d([[0, 1, 2], [3, 4, 5]]);
    const out = tf.einsum('bi,bi->b', x, y);
    expectArraysClose(await out.data(), [8, 62]);
  });

  it('two 2d matrices batch outer', async () => {
    const x = tensor2d([[1, 2, 3], [4, 5, 6]]);
    const y = tensor2d([[0, 1, 2], [3, 4, 5]]);
    const out = tf.einsum('bi,bj->bij', x, y);
    expectArraysClose(await out.data(), [
      [[0, 1, 2], [0, 2, 4], [0, 3, 6]],
      [[12, 16, 20], [15, 20, 25], [18, 24, 30]]
    ]);
  });

  it('two 3d tensors batch matmul', async () => {
    const x = tf.reshape(tf.range(1, 13), [2, 2, 3]);
    const y = tf.reshape(tf.range(1, 19), [2, 3, 3]);
    const out = tf.einsum('bij,bjk->bik', x, y);
    expectArraysClose(
        await out.data(),
        [[[30, 36, 42], [66, 81, 96]], [[318, 342, 366], [435, 468, 501]]]);
  });

  it('two 3d tensors A', async () => {
    const x = tf.reshape(tf.range(1, 9), [2, 2, 2]);
    const y = tf.reshape(tf.range(1, 13), [2, 3, 2]);
    const out = tf.einsum('adc,abc->abd', x, y);
    expectArraysClose(
        await out.data(),
        [[[5, 11], [11, 25], [17, 39]], [[83, 113], [105, 143], [127, 173]]]);
  });

  it('two 3d tensors B', async () => {
    const x = tf.reshape(tf.range(1, 9), [2, 2, 2]);
    const y = tf.reshape(tf.range(1, 13), [2, 3, 2]);
    const out = tf.einsum('adc,abc->adb', x, y);
    expectArraysClose(
        await out.data(),
        [[[5, 11, 17], [11, 25, 39]], [[83, 105, 127], [113, 143, 173]]]);
  });

  it('one 3d tensor: batch matrix transposing', async () => {
    const x = tensor3d([[[1, 2], [3, 4]], [[-1, -2], [-3, -4]]]);
    const out = tf.einsum('bij->bji', x);
    expectArraysClose(
        await out.data(), [[[1, 3], [2, 4]], [[-1, -3], [-2, -4]]]);
  });

  it('4d tensor and 3d tensor, contracting two dimensions', async () => {
    const x = tf.reshape(tf.range(1, 33), [2, 4, 2, 2]);
    const y = tf.reshape(tf.range(1, 9), [2, 2, 2]);
    const out = tf.einsum('abcd,cde->abe', x, y);
    expectArraysClose(await out.data(), [
      [[50, 60], [114, 140], [178, 220], [242, 300]],
      [[306, 380], [370, 460], [434, 540], [498, 620]]
    ]);
  });

  it('two 4d tensors, contracting one dimension', async () => {
    const x = tf.reshape(tf.range(1, 33), [2, 4, 2, 2]);
    const y = tf.reshape(tf.range(1, 25), [2, 3, 2, 2]);
    const out = tf.einsum('aecd,abcd->acbe', x, y);
    expectArraysClose(await out.data(), [
      [
        [[5, 17, 29, 41], [17, 61, 105, 149], [29, 105, 181, 257]],
        [[25, 53, 81, 109], [53, 113, 173, 233], [81, 173, 265, 357]]
      ],
      [
        [[473, 581, 689, 797], [613, 753, 893, 1033], [753, 925, 1097, 1269]],
        [[605, 729, 853, 977], [761, 917, 1073, 1229], [917, 1105, 1293, 1481]]
      ]
    ]);
  });

  it('two 4d tensors, contracting two dimensions', async () => {
    const x = tf.reshape(tf.range(1, 33), [2, 4, 2, 2]);
    const y = tf.reshape(tf.range(1, 25), [2, 3, 2, 2]);
    const out = tf.einsum('aecd,abcd->abe', x, y);
    expectArraysClose(await out.data(), [
      [[30, 70, 110, 150], [70, 174, 278, 382], [110, 278, 446, 614]],
      [
        [1078, 1310, 1542, 1774], [1374, 1670, 1966, 2262],
        [1670, 2030, 2390, 2750]
      ]
    ]);
  });

  it('mismatched dimensions throws error', () => {
    const x = tensor2d([[1, 2, 3], [4, 5, 6]]);
    const y = tensor2d([[0, 1], [2, 3]]);
    expect(() => tf.einsum('ij,jk->ik', x, y))
        .toThrowError(
            'Expected dimension 3 at axis 0 of input shaped [2,2], ' +
            'but got dimension 2');
  });

  it('incorrect equation throws error', () => {
    const x = tensor2d([[1, 2], [3, 4]]);
    const y = tensor2d([[0, 1], [2, 3]]);
    expect(() => tf.einsum('', x, y))
        .toThrowError('Equations without an arrow are not supported.');
    expect(() => tf.einsum('ij,jk>ik', x, y))
        .toThrowError('Equations without an arrow are not supported.');
  });

  it('incorrect number of tensors throws error', () => {
    const x = tensor2d([[1, 2], [3, 4]]);
    const y = tensor2d([[0, 1], [2, 3]]);
    expect(() => tf.einsum('ij->ji', x, y))
        .toThrowError('Expected 1 input tensors, received 2');
  });

  it('more than two input tensors throws error', async () => {
    const x = tensor2d([[1, 2], [3, 4]]);
    const y = tensor2d([[0, 1], [2, 3]]);
    const z = tensor2d([[-1, 0], [1, 2]]);
    expect(() => tf.einsum('ij,jk,kl->il', x, y, z))
        .toThrowError(/more than 2 input tensors/);
  });

  it('nonexistent dimension throws error', async () => {
    const x = tensor2d([[1, 2, 3], [4, 5, 6]]);
    const y = tensor2d([[0, 1], [2, 3], [4, 5]]);
    expect(() => tf.einsum('ij,jk->in', x, y))
        .toThrowError(
            'Output subscripts contain the label n not present in ' +
            'the input subscripts.');
  });

  it('two arrows in equation throws error', async () => {
    const x = tensor2d([[1, 2, 3], [4, 5, 6]]);
    const y = tensor2d([[0, 1], [2, 3], [4, 5]]);
    expect(() => tf.einsum('ij,jk->ik->i', x, y))
        .toThrowError(/exactly one arrow/);
  });
});
