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
import {PARALLELIZE_THRESHOLD} from './reduce_util';

describeWithFlags('unsortedSegmentSum', ALL_ENVS, () => {
  it('tensor1D', async () => {
    const t = tf.tensor1d([1, 2, 3, 4]);
    const segmentIds = tf.tensor1d([0, 2, 0, 1], 'int32');
    const numSegments = 3;
    const res = tf.unsortedSegmentSum(t, segmentIds, numSegments);

    expect(res.shape).toEqual([numSegments]);
    expectArraysClose(await res.data(), [4, 4, 2]);
  });

  it('tensor2D', async () => {
    const t = tf.tensor2d([1, 2, 3, 4], [2, 2]);
    const segmentIds = tf.tensor1d([0, 0], 'int32');
    const numSegments = 2;
    const res = tf.unsortedSegmentSum(t, segmentIds, numSegments);

    expect(res.shape).toEqual([numSegments, 2]);
    expectArraysClose(await res.data(), [4, 6, 0, 0]);
  });

  it('tensor3D', async () => {
    const t = tf.tensor3d([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [3, 2, 2]);
    const segmentIds = tf.tensor1d([2, 1, 2], 'int32');
    const numSegments = 3;
    const res = tf.unsortedSegmentSum(t, segmentIds, numSegments);

    expect(res.shape).toEqual([numSegments, 2, 2]);
    expectArraysClose(
        await res.data(), [0, 0, 0, 0, 5, 6, 7, 8, 10, 12, 14, 16]);
  });

  it('N > than parallelization threshold, tensor1D', async () => {
    const n = PARALLELIZE_THRESHOLD * 2;
    const values = new Float32Array(n);
    const numSegments = 5;
    const segmentIdValues = new Float32Array(n);
    const vals = new Float32Array(numSegments);
    for (let i = 0; i < n; i++) {
      values[i] = i;
      segmentIdValues[i] = i % numSegments;
      vals[i % numSegments] += i;
    }
    const t = tf.tensor1d(values);
    const segmentIds = tf.tensor1d(segmentIdValues, 'int32');
    const res = tf.unsortedSegmentSum(t, segmentIds, numSegments);

    expect(res.shape).toEqual([numSegments]);
    expectArraysClose(await res.data(), vals);
  });

  it('ignores negative segmentIds', async () => {
    const t = tf.tensor1d([1, 2, 3, 4]);
    const segmentIds = tf.tensor1d([0, 2, -1, 1], 'int32');
    const numSegments = 3;

    const res = tf.unsortedSegmentSum(t, segmentIds, numSegments);

    expect(res.shape).toEqual([numSegments]);
    expectArraysClose(await res.data(), [1, 4, 2]);
  });

  it('gradient ignores negative segmentIds', async () => {
    const t = tf.tensor1d([1, 2, 3, 4]);
    const segmentIds = tf.tensor1d([0, 2, -1, 1], 'int32');
    const numSegments = 3;

    const dy = tf.tensor1d([11, 2, 7]);
    const gradient =
        tf.grad(a => tf.unsortedSegmentSum(a, segmentIds, numSegments))(t, dy);

    expect(gradient.shape).toEqual(t.shape);
    expectArraysClose(await gradient.data(), [11, 7, 0, 2]);
  });

  it('tensor1D gradient', async () => {
    const t = tf.tensor1d([1, 2, 3, 4]);
    const segmentIds = tf.tensor1d([0, 2, 0, 1], 'int32');
    const numSegments = 3;

    const dy = tf.tensor1d([11, 2, 7]);
    const gradient =
        tf.grad(a => tf.unsortedSegmentSum(a, segmentIds, numSegments))(t, dy);

    expect(gradient.shape).toEqual(t.shape);
    expectArraysClose(await gradient.data(), [11, 7, 11, 2]);
  });

  it('gradient with clones', async () => {
    const t = tf.tensor1d([1, 2, 3, 4]);
    const segmentIds = tf.tensor1d([0, 2, 0, 1], 'int32');
    const numSegments = 3;

    const dy = tf.tensor1d([11, 2, 7]);
    const gradient = tf.grad(
        a => tf.unsortedSegmentSum(a.clone(), segmentIds.clone(), numSegments)
                 .clone())(t, dy);

    expect(gradient.shape).toEqual(t.shape);
    expectArraysClose(await gradient.data(), [11, 7, 11, 2]);
  });

  it('tensor2D gradient', async () => {
    const t = tf.tensor2d([1, 2, 3, 4], [2, 2]);
    const segmentIds = tf.tensor1d([0, 0], 'int32');
    const numSegments = 2;

    const dy = tf.tensor2d([11, 2, 4, 5], [2, 2]);
    const gradient =
        tf.grad(a => tf.unsortedSegmentSum(a, segmentIds, numSegments))(t, dy);

    expect(gradient.shape).toEqual(t.shape);
    expectArraysClose(await gradient.data(), [11, 2, 11, 2]);
  });

  it('tensor3D gradient', async () => {
    const t = tf.tensor3d([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [3, 2, 2]);
    const segmentIds = tf.tensor1d([2, 1, 2], 'int32');
    const numSegments = 3;

    const dy =
        tf.tensor3d([11, 2, 4, 5, 17, 31, 1, 0, -1, 14, 3, 28], [3, 2, 2]);
    const gradient =
        tf.grad(a => tf.unsortedSegmentSum(a, segmentIds, numSegments))(t, dy);

    expect(gradient.shape).toEqual(t.shape);
    expectArraysClose(
        await gradient.data(), [-1, 14, 3, 28, 17, 31, 1, 0, -1, 14, 3, 28]);
  });

  it('accepts a tensor-like object', async () => {
    const x = [1, 2, 3, 4];
    const segmentIds = [0, 2, 0, 1];
    const numSegments = 3;
    const res = tf.unsortedSegmentSum(x, segmentIds, numSegments);
    expect(res.shape).toEqual([3]);
    expectArraysClose(await res.data(), [4, 4, 2]);
  });

  it('accepts a tensor-like object chained', async () => {
    const x = tf.tensor1d([1, 2, 3, 4]);
    const segmentIds = [0, 2, 0, 1];
    const numSegments = 3;
    const res = x.unsortedSegmentSum(segmentIds, numSegments);

    expect(res.shape).toEqual([3]);
    expectArraysClose(await res.data(), [4, 4, 2]);
  });
});
