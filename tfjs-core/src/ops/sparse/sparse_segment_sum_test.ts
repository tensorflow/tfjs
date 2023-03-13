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

import * as tf from '../../index';
import {ALL_ENVS, describeWithFlags} from '../../jasmine_util';
import {expectArraysClose} from '../../test_util';

function TensorValue3x4() {
  return tf.tensor2d([[1, 2, 3, 4], [-1, -2, -3, -4], [5, 6, 7, 8]]);
}

function TensorValue10() {
  return tf.tensor1d(Array.from(Array(10), (_, i) => i + 1));
}

function TensorValue10x4() {
  return tf.tensor2d(Array.from(Array(40), (_, i) => i + 1), [10, 4]);
}

function TensorValue10x2x4() {
  return tf.tensor3d(Array.from(Array(80), (_, i) => i + 1), [10, 2, 4]);
}

describeWithFlags('sparseSegmentSum', ALL_ENVS, () => {
  it('two rows one segment', async () => {
    const result = tf.sparse.sparseSegmentSum(TensorValue3x4(), [0, 1], [0, 0]);
    expectArraysClose(await result.data(), [[0, 0, 0, 0]]);
  });

  it('two rows two segments', async () => {
    const result = tf.sparse.sparseSegmentSum(TensorValue3x4(), [0, 1], [0, 1]);
    expectArraysClose(await result.data(), [[1, 2, 3, 4], [-1, -2, -3, -4]]);
  });

  it('all rows one segment', async () => {
    const result =
        tf.sparse.sparseSegmentSum(TensorValue3x4(), [0, 1, 2], [0, 0, 1]);
    expectArraysClose(await result.data(), [[0, 0, 0, 0], [5, 6, 7, 8]]);
  });

  it('0 dimensional input invalid', async () => {
    expect(() => tf.sparse.sparseSegmentSum(tf.scalar(1), [], []))
        .toThrowError(/should be at least 1 dimensional/);
  });

  it('1 dimensional input', async () => {
    const result =
        tf.sparse.sparseSegmentSum(TensorValue10(), [8, 3, 0, 9], [0, 1, 2, 2]);
    expectArraysClose(await result.data(), [9, 4, 11]);
  });

  it('3 dimensional input', async () => {
    const result = tf.sparse.sparseSegmentSum(
        TensorValue10x2x4(), [8, 3, 0, 9], [0, 1, 2, 2]);
    expectArraysClose(await result.data(), [
      [[65, 66, 67, 68], [69, 70, 71, 72]],
      [[25, 26, 27, 28], [29, 30, 31, 32]], [[74, 76, 78, 80], [82, 84, 86, 88]]
    ]);
  });

  it('segment ids hole', async () => {
    const result = tf.sparse.sparseSegmentSum(
        TensorValue10x4(), [8, 3, 0, 9], [0, 3, 3, 3]);
    expectArraysClose(
        await result.data(),
        [[33, 34, 35, 36], [0, 0, 0, 0], [0, 0, 0, 0], [51, 54, 57, 60]]);
  });

  it('segment ids > zero', async () => {
    const result = tf.sparse.sparseSegmentSum(
        TensorValue10x4(), [8, 3, 0, 9], [2, 3, 3, 3]);
    expectArraysClose(
        await result.data(),
        [[0, 0, 0, 0], [0, 0, 0, 0], [33, 34, 35, 36], [51, 54, 57, 60]]);
  });

  it('baseline valid', async () => {
    // Baseline for the *invalid* tests below.
    const result = tf.sparse.sparseSegmentSum(
        TensorValue10x4(), [8, 3, 0, 9], [0, 1, 2, 2]);
    expectArraysClose(
        await result.data(),
        [[33, 34, 35, 36], [13, 14, 15, 16], [38, 40, 42, 44]]);
  });

  it('does not have memory leak.', async () => {
    const beforeDataIds = tf.engine().backend.numDataIds();

    const data = TensorValue3x4();
    const indices = tf.tensor1d([0, 1], 'int32');
    const segmentIds = tf.tensor1d([0, 0], 'int32');
    const result = tf.sparse.sparseSegmentSum(data, indices, segmentIds);

    await result.data();

    const afterResDataIds = tf.engine().backend.numDataIds();
    expect(afterResDataIds).toEqual(beforeDataIds + 4);

    data.dispose();
    indices.dispose();
    segmentIds.dispose();
    result.dispose();

    const afterDisposeDataIds = tf.engine().backend.numDataIds();
    expect(afterDisposeDataIds).toEqual(beforeDataIds);
  });

  it('indices invalid 1', async () => {
    expect(() => tf.sparse.sparseSegmentSum(TensorValue10x4(), [8, -1, 0, 9], [
      0, 1, 2, 2
    ])).toThrowError(/indices\[1\] == -1 out of range \[0, 10\)/);
  });

  it('indices invalid 2', async () => {
    expect(() => tf.sparse.sparseSegmentSum(TensorValue10x4(), [8, 3, 0, 10], [
      0, 1, 2, 2
    ])).toThrowError(/indices\[3\] == 10 out of range \[0, 10\)/);
  });

  it('segments invalid 2', async () => {
    expect(() => tf.sparse.sparseSegmentSum(TensorValue10x4(), [8, 3, 0, 9], [
      0, 1, 0, 1
    ])).toThrowError('segment ids are not increasing');
  });

  it('segments invalid 3', async () => {
    expect(
        () => tf.sparse.sparseSegmentSum(
            TensorValue10x4(), [8, 3, 0, 9], [0, 1, 2, 0]))
        .toThrowError(
            'Segment id 1 out of range [0, 1), possibly because segmentIds ' +
            'input is not sorted.');
  });

  it('segments invalid 4', async () => {
    expect(
        () => tf.sparse.sparseSegmentSum(
            TensorValue10x4(), [8, 3, 0, 9], [-1, 0, 1, 1]))
        .toThrowError(
            'Segment id -1 out of range [0, 2), possibly because segmentIds ' +
            'input is not sorted.');
  });

  it('segments invalid 6', async () => {
    expect(() => tf.sparse.sparseSegmentSum(TensorValue10x4(), [8, 3, 0, 9], [
      0, 0, 0, -1
    ])).toThrowError('segment ids must be >= 0');
  });

  it('segments invalid 7', async () => {
    expect(() => tf.sparse.sparseSegmentSum(TensorValue10x4(), [8, 3, 0, 9], [
      0, 0, 0, -2
    ])).toThrowError('segment ids must be >= 0');
  });

  it('indices invalid rank', async () => {
    expect(() => tf.sparse.sparseSegmentSum(TensorValue10x4(), [[8, 3, 0, 9]], [
      0, 0, 0, -2
    ])).toThrowError(/should be Tensor1D/);
  });

  it('segments invalid rank', async () => {
    expect(() => tf.sparse.sparseSegmentSum(TensorValue10x4(), [8, 3, 0, 9], [
      [0, 0, 0, -2]
    ])).toThrowError(/should be Tensor1D/);
  });
});
