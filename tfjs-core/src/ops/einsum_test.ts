/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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

import {tensor1d, tensor2d} from './ops';

describeWithFlags('einsum', ALL_ENVS, () => {
  fit('1d reduce sum', async () => {
    const x = tensor1d([2, 4, 6]);
    const out = tf.einsum('i->', x);
    expectArraysClose(await out.data(), 12);
  });

  fit('2d matrix reduce sum', async () => {
    const x = tensor2d([[1, 2], [3, 4]]);
    const out = tf.einsum('ij->', x);
    expectArraysClose(await out.data(), 10);
  });

  fit('two 1d tensors dot product', async () => {
    const x = tensor1d([1, 3, 5]);
    const y = tensor1d([2, 4, 6]);
    const out = tf.einsum('i,i->', x, y);
    expectArraysClose(await out.data(), 44);
  });

  fit('two 1d tensors outer product', async () => {
    const x = tensor1d([1, 3, 5]);
    const y = tensor1d([2, 4, 6]);
    const out = tf.einsum('i,j->ij', x, y);
    expectArraysClose(await out.data(), [[2, 4, 6], [6, 12, 18], [10, 20, 30]]);
  });

  fit('2d matrix calculate trace: not implemented yet', () => {
    const x = tensor2d([[1, 2], [3, 4]]);
    expect(() => tf.einsum('ii->', x)).toThrowError(/not implemented yet/);
  });

  fit('2d and 1d matrix & vector multiply', async () => {
    const x = tensor2d([[1, 2, 3], [4, 5, 6]], [2, 3]);
    const y = tensor1d([2, 4, 6]);
    const out = tf.einsum('ij,j->i', x, y);
    expectArraysClose(await out.data(), [28, 64]);
  });

  fit('2d matrix sum over rows', async () => {
    const x = tensor2d([[1, 2, 3], [4, 5, 6]], [2, 3]);
    const out = tf.einsum('ij->j', x);
    expectArraysClose(await out.data(), [5, 7, 9]);
  });

  fit('2d matrix sum over rows', async () => {
    const x = tensor2d([[1, 2, 3], [4, 5, 6]], [2, 3]);
    const out = tf.einsum('ij->i', x);
    expectArraysClose(await out.data(), [6, 15]);
  });

  fit('2d matrix transpose', async () => {
    const x = tensor2d([[1, 2, 3], [4, 5, 6]], [2, 3]);
    const out = tf.einsum('ij->ji', x);
    expectArraysClose(await out.data(), [[1, 4], [2, 5], [3, 6]]);
  });

  fit('2d matrix multiply', async () => {
    const x = tensor2d([[1, 2, 3], [4, 5, 6]], [2, 3]);
    const y = tensor2d([[0, 1], [2, 3], [4, 5]], [3, 2]);
    const out = tf.einsum('ij,jk->ik', x, y);
    expectArraysClose(await out.data(), [[16, 22], [34, 49]]);
  });

  fit('2d matrix multiply and transpose', async () => {
    const x = tensor2d([[1, 2, 3], [4, 5, 6]], [2, 3]);
    const y = tensor2d([[0, 1], [2, 3], [4, 5]], [3, 2]);
    const out = tf.einsum('ij,jk->ki', x, y);
    expectArraysClose(await out.data(), [[16, 34], [22, 49]]);
  });

  fit('two 2d matrices batch dot product', async () => {
    const x = tensor2d([[1, 2, 3], [4, 5, 6]], [2, 3]);
    const y = tensor2d([[0, 1, 2], [3, 4, 5]], [2, 3]);
    const out = tf.einsum('bi,bi->b', x, y);
    expectArraysClose(await out.data(), [8, 62]);
  });

  fit('two 2d matrices batch output product product', async () => {
    const x = tensor2d([[1, 2, 3], [4, 5, 6]], [2, 3]);
    const y = tensor2d([[0, 1, 2], [3, 4, 5]], [2, 3]);
    const out = tf.einsum('bi,bj->bij', x, y);
    expectArraysClose(await out.data(), [
      [[0, 1, 2], [0, 2, 4], [0, 3, 6]],
      [[12, 16, 20], [15, 20, 25], [18, 24, 30]]
    ]);
  });

  fit('two 3d tensors', async () => {
    const x = tf.reshape(tf.range(1, 9), [2, 2, 2]);
    const y = tf.reshape(tf.range(1, 13), [2, 3, 2]);
    expect(() => tf.einsum('adc,abc->ac', x, y))
        .toThrowError(/not implemented for >1 input tensors/);
  });

  fit('two 3d tensors batch matmul', async () => {
    const x = tf.reshape(tf.range(1, 13), [2, 2, 3]);
    const y = tf.reshape(tf.range(1, 19), [2, 3, 3]);
    const out = tf.einsum('bij,bjk->bik', x, y);
    expectArraysClose(
        await out.data(),
        [[[30, 36, 42], [66, 81, 96]], [[318, 342, 366], [435, 468, 501]]]);
  });

  fit('two 3d tensors A', async () => {
    const x = tf.reshape(tf.range(1, 9), [2, 2, 2]);
    const y = tf.reshape(tf.range(1, 13), [2, 3, 2]);
    const out = tf.einsum('adc,abc->abd', x, y);
    expectArraysClose(
        await out.data(),
        [[[5, 11], [11, 25], [17, 39]], [[83, 113], [105, 143], [127, 173]]]);
  });

  fit('two 3d tensors B', async () => {
    const x = tf.reshape(tf.range(1, 9), [2, 2, 2]);
    const y = tf.reshape(tf.range(1, 13), [2, 3, 2]);
    const out = tf.einsum('adc,abc->adb', x, y);
    expectArraysClose(
        await out.data(),
        [[[5, 11, 17], [11, 25, 39]], [[83, 105, 127], [113, 143, 173]]]);
  });

  fit('two 4d tensors', async () => {
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

  fit('mismatched dimensions throws error', () => {
    const x = tensor2d([[1, 2, 3], [4, 5, 6]], [2, 3]);
    const y = tensor2d([[0, 1], [2, 3]], [2, 2]);
    expect(() => tf.einsum('ij,jk->ik', x, y))
        .toThrowError(
            'Expected dimension 3 at axis 0 of input shaped [2,2], ' +
            'but got dimension 2');
  });

  fit('incorrect equation throws error', () => {
    const x = tensor2d([[1, 2], [3, 4]], [2, 2]);
    const y = tensor2d([[0, 1], [2, 3]], [2, 2]);
    expect(() => tf.einsum('', x, y))
        .toThrowError('Equations without an arrow is not supported');
    expect(() => tf.einsum('ij,jk>ik', x, y))
        .toThrowError('Equations without an arrow is not supported');
  });

  fit('incorrect number of tensors throws error', () => {
    const x = tensor2d([[1, 2], [3, 4]], [2, 2]);
    const y = tensor2d([[0, 1], [2, 3]], [2, 2]);
    expect(() => tf.einsum('ij->ji', x, y))
        .toThrowError('Expected 1 input tensors, received 2');
  });

  fit('more than two input tensors throws error', async () => {
    const x = tensor2d([[1, 2], [3, 4]], [2, 2]);
    const y = tensor2d([[0, 1], [2, 3]], [2, 2]);
    const z = tensor2d([[1, 2], [3, 4]], [2, 2]);
    expect(() => tf.einsum('ij,jk,kl->il', x, y, z))
        .toThrowError(/more than 2 input tensors/);
  });
});
