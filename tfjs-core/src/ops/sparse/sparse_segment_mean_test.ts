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

function TensorValue3x4Float() {
  return tf.tensor2d([[1, 2, 3, 4], [-1, -2, -3, -4], [6, 7, 8, 9]]);
}

function TensorValue3x4Integer() {
  return tf.tensor2d(
      [[1, 2, 3, 4], [-1, -2, -3, -4], [6, 7, 8, 9]], [3, 4], 'int32');
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

describeWithFlags('sparseSegmentMean', ALL_ENVS, () => {
  it('two rows one segment', async () => {
    const result =
        tf.sparse.sparseSegmentMean(TensorValue3x4Float(), [0, 1], [0, 0]);
    expectArraysClose(await result.data(), [[0, 0, 0, 0]]);
  });

  it('two rows two segments', async () => {
    const result =
        tf.sparse.sparseSegmentMean(TensorValue3x4Float(), [0, 1], [0, 1]);
    expectArraysClose(await result.data(), [[1, 2, 3, 4], [-1, -2, -3, -4]]);
  });

  it('all rows two segments', async () => {
    const result = tf.sparse.sparseSegmentMean(
        TensorValue3x4Float(), [0, 1, 2], [0, 1, 1]);
    expectArraysClose(
        await result.data(), [[1.0, 2.0, 3.0, 4.0], [2.5, 2.5, 2.5, 2.5]]);
  });

  it('integer data', async () => {
    const result = tf.sparse.sparseSegmentMean(
        TensorValue3x4Integer(), [0, 1, 2], [0, 1, 1]);
    expectArraysClose(await result.data(), [[1, 2, 3, 4], [2, 2, 2, 2]]);
  });

  it('0 dimensional input invalid', async () => {
    expect(() => tf.sparse.sparseSegmentMean(tf.scalar(1), [], []))
        .toThrowError(/should be at least 1 dimensional/);
  });

  it('1 dimensional input', async () => {
    const result = tf.sparse.sparseSegmentMean(
        TensorValue10(), [8, 3, 0, 9], [0, 1, 2, 2]);
    expectArraysClose(await result.data(), [9, 4, 5.5]);
  });

  it('3 dimensional input', async () => {
    const result = tf.sparse.sparseSegmentMean(
        TensorValue10x2x4(), [8, 3, 0, 9], [0, 1, 2, 2]);
    expectArraysClose(await result.data(), [
      [[65, 66, 67, 68], [69, 70, 71, 72]],
      [[25, 26, 27, 28], [29, 30, 31, 32]], [[37, 38, 39, 40], [41, 42, 43, 44]]
    ]);
  });

  it('segment ids hole', async () => {
    const result = tf.sparse.sparseSegmentMean(
        TensorValue10x4(), [8, 3, 0, 9], [0, 3, 3, 3]);
    expectArraysClose(
        await result.data(),
        [[33, 34, 35, 36], [0, 0, 0, 0], [0, 0, 0, 0], [17, 18, 19, 20]]);
  });

  it('segment ids > zero', async () => {
    const result = tf.sparse.sparseSegmentMean(
        TensorValue10x4(), [8, 3, 0, 9], [2, 3, 3, 3]);
    expectArraysClose(
        await result.data(),
        [[0, 0, 0, 0], [0, 0, 0, 0], [33, 34, 35, 36], [17, 18, 19, 20]]);
  });

  it('baseline valid', async () => {
    // Baseline for the *invalid* tests below.
    const result = tf.sparse.sparseSegmentMean(
        TensorValue10x4(), [8, 3, 0, 9], [0, 1, 2, 2]);
    expectArraysClose(
        await result.data(),
        [[33, 34, 35, 36], [13, 14, 15, 16], [19, 20, 21, 22]]);
  });

  it('does not have memory leak.', async () => {
    const beforeDataIds = tf.engine().backend.numDataIds();

    const data = TensorValue3x4Float();
    const indices = tf.tensor1d([0, 1], 'int32');
    const segmentIds = tf.tensor1d([0, 0], 'int32');
    const result = tf.sparse.sparseSegmentMean(data, indices, segmentIds);

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

  it('throw error if indices < 0', async () => {
    expect(() => tf.sparse.sparseSegmentMean(TensorValue10x4(), [8, -1, 0, 9], [
      0, 1, 2, 2
    ])).toThrowError(/indices\[1\] == -1 out of range \[0, 10\)/);
  });

  it('throw error if indices >= max rows', async () => {
    expect(() => tf.sparse.sparseSegmentMean(TensorValue10x4(), [8, 3, 0, 10], [
      0, 1, 2, 2
    ])).toThrowError(/indices\[3\] == 10 out of range \[0, 10\)/);
  });

  it('throw error if segment ids are not increasing', async () => {
    expect(() => tf.sparse.sparseSegmentMean(TensorValue10x4(), [8, 3, 0, 9], [
      0, 1, 0, 1
    ])).toThrowError('segment ids are not increasing');
  });

  it('throw error if segment id is out of range', async () => {
    expect(
        () => tf.sparse.sparseSegmentMean(
            TensorValue10x4(), [8, 3, 0, 9], [0, 1, 2, 0]))
        .toThrowError(
            'Segment id 1 out of range [0, 1), possibly because segmentIds ' +
            'input is not sorted.');
  });

  it('throw error if segment id is out of range and negative', async () => {
    expect(
        () => tf.sparse.sparseSegmentMean(
            TensorValue10x4(), [8, 3, 0, 9], [-1, 0, 1, 1]))
        .toThrowError(
            'Segment id -1 out of range [0, 2), possibly because segmentIds ' +
            'input is not sorted.');
  });

  it('throw error if segment id is negative', async () => {
    expect(() => tf.sparse.sparseSegmentMean(TensorValue10x4(), [8, 3, 0, 9], [
      0, 0, 0, -1
    ])).toThrowError('segment ids must be >= 0');
  });

  it('throw error if segment id is negative and there is a hole', async () => {
    expect(() => tf.sparse.sparseSegmentMean(TensorValue10x4(), [8, 3, 0, 9], [
      0, 0, 0, -2
    ])).toThrowError('segment ids must be >= 0');
  });

  it('throw error if indices has invalid rank', async () => {
    expect(
        () => tf.sparse.sparseSegmentMean(
            TensorValue10x4(), [[8, 3, 0, 9]], [0, 0, 0, -2]))
        .toThrowError(/should be Tensor1D/);
  });

  it('throw error if segments has invalid rank', async () => {
    expect(() => tf.sparse.sparseSegmentMean(TensorValue10x4(), [8, 3, 0, 9], [
      [0, 0, 0, -2]
    ])).toThrowError(/should be Tensor1D/);
  });
});
