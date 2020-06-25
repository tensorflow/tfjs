/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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
import {Rank} from '../types';

describeWithFlags('basicLSTMCell', ALL_ENVS, () => {
  it('basicLSTMCell with batch=2', async () => {
    const lstmKernel = tf.randomNormal<Rank.R2>([3, 4]);
    const lstmBias = tf.randomNormal<Rank.R1>([4]);
    const forgetBias = tf.scalar(1.0);

    const data = tf.randomNormal<Rank.R2>([1, 2]);
    const batchedData = tf.concat2d([data, data], 0);  // 2x2
    const c = tf.randomNormal<Rank.R2>([1, 1]);
    const batchedC = tf.concat2d([c, c], 0);  // 2x1
    const h = tf.randomNormal<Rank.R2>([1, 1]);
    const batchedH = tf.concat2d([h, h], 0);  // 2x1
    const [newC, newH] = tf.basicLSTMCell(
        forgetBias, lstmKernel, lstmBias, batchedData, batchedC, batchedH);
    const newCVals = await newC.array();
    const newHVals = await newH.array();
    expect(newCVals[0][0]).toEqual(newCVals[1][0]);
    expect(newHVals[0][0]).toEqual(newHVals[1][0]);
  });

  it('basicLSTMCell accepts a tensor-like object', async () => {
    const lstmKernel = tf.randomNormal<Rank.R2>([3, 4]);
    const lstmBias = [0, 0, 0, 0];
    const forgetBias = 1;

    const data = [[0, 0]];                             // 1x2
    const batchedData = tf.concat2d([data, data], 0);  // 2x2
    const c = [[0]];                                   // 1x1
    const batchedC = tf.concat2d([c, c], 0);           // 2x1
    const h = [[0]];                                   // 1x1
    const batchedH = tf.concat2d([h, h], 0);           // 2x1
    const [newC, newH] = tf.basicLSTMCell(
        forgetBias, lstmKernel, lstmBias, batchedData, batchedC, batchedH);
    const newCVals = await newC.array();
    const newHVals = await newH.array();
    expect(newCVals[0][0]).toEqual(newCVals[1][0]);
    expect(newHVals[0][0]).toEqual(newHVals[1][0]);
  });
});
describeWithFlags('basicLSTMCell throws with non-tensor', ALL_ENVS, () => {
  it('input: forgetBias', () => {
    const lstmKernel = tf.randomNormal<Rank.R2>([3, 4]);
    const lstmBias = tf.randomNormal<Rank.R1>([4]);

    const data = tf.randomNormal<Rank.R2>([1, 2]);
    const batchedData = tf.concat2d([data, data], 0);  // 2x2
    const c = tf.randomNormal<Rank.R2>([1, 1]);
    const batchedC = tf.concat2d([c, c], 0);  // 2x1
    const h = tf.randomNormal<Rank.R2>([1, 1]);
    const batchedH = tf.concat2d([h, h], 0);  // 2x1
    expect(
        () => tf.basicLSTMCell(
            {} as tf.Scalar, lstmKernel, lstmBias, batchedData, batchedC,
            batchedH))
        .toThrowError(
            /Argument 'forgetBias' passed to 'basicLSTMCell' must be a Tensor/);
  });
  it('input: lstmKernel', () => {
    const lstmBias = tf.randomNormal<Rank.R1>([4]);
    const forgetBias = tf.scalar(1.0);

    const data = tf.randomNormal<Rank.R2>([1, 2]);
    const batchedData = tf.concat2d([data, data], 0);  // 2x2
    const c = tf.randomNormal<Rank.R2>([1, 1]);
    const batchedC = tf.concat2d([c, c], 0);  // 2x1
    const h = tf.randomNormal<Rank.R2>([1, 1]);
    const batchedH = tf.concat2d([h, h], 0);  // 2x1
    expect(
        () => tf.basicLSTMCell(
            forgetBias, {} as tf.Tensor2D, lstmBias, batchedData, batchedC,
            batchedH))
        .toThrowError(
            /Argument 'lstmKernel' passed to 'basicLSTMCell' must be a Tensor/);
  });
  it('input: lstmBias', () => {
    const lstmKernel = tf.randomNormal<Rank.R2>([3, 4]);
    const forgetBias = tf.scalar(1.0);

    const data = tf.randomNormal<Rank.R2>([1, 2]);
    const batchedData = tf.concat2d([data, data], 0);  // 2x2
    const c = tf.randomNormal<Rank.R2>([1, 1]);
    const batchedC = tf.concat2d([c, c], 0);  // 2x1
    const h = tf.randomNormal<Rank.R2>([1, 1]);
    const batchedH = tf.concat2d([h, h], 0);  // 2x1
    expect(
        () => tf.basicLSTMCell(
            forgetBias, lstmKernel, {} as tf.Tensor1D, batchedData, batchedC,
            batchedH))
        .toThrowError(
            /Argument 'lstmBias' passed to 'basicLSTMCell' must be a Tensor/);
  });
  it('input: data', () => {
    const lstmKernel = tf.randomNormal<Rank.R2>([3, 4]);
    const lstmBias = tf.randomNormal<Rank.R1>([4]);
    const forgetBias = tf.scalar(1.0);

    const c = tf.randomNormal<Rank.R2>([1, 1]);
    const batchedC = tf.concat2d([c, c], 0);  // 2x1
    const h = tf.randomNormal<Rank.R2>([1, 1]);
    const batchedH = tf.concat2d([h, h], 0);  // 2x1
    expect(
        () => tf.basicLSTMCell(
            forgetBias, lstmKernel, lstmBias, {} as tf.Tensor2D, batchedC,
            batchedH))
        .toThrowError(
            /Argument 'data' passed to 'basicLSTMCell' must be a Tensor/);
  });
  it('input: c', () => {
    const lstmKernel = tf.randomNormal<Rank.R2>([3, 4]);
    const lstmBias = tf.randomNormal<Rank.R1>([4]);
    const forgetBias = tf.scalar(1.0);

    const data = tf.randomNormal<Rank.R2>([1, 2]);
    const batchedData = tf.concat2d([data, data], 0);  // 2x2
    const h = tf.randomNormal<Rank.R2>([1, 1]);
    const batchedH = tf.concat2d([h, h], 0);  // 2x1
    expect(
        () => tf.basicLSTMCell(
            forgetBias, lstmKernel, lstmBias, batchedData, {} as tf.Tensor2D,
            batchedH))
        .toThrowError(
            /Argument 'c' passed to 'basicLSTMCell' must be a Tensor/);
  });
  it('input: h', () => {
    const lstmKernel = tf.randomNormal<Rank.R2>([3, 4]);
    const lstmBias = tf.randomNormal<Rank.R1>([4]);
    const forgetBias = tf.scalar(1.0);

    const data = tf.randomNormal<Rank.R2>([1, 2]);
    const batchedData = tf.concat2d([data, data], 0);  // 2x2
    const c = tf.randomNormal<Rank.R2>([1, 1]);
    const batchedC = tf.concat2d([c, c], 0);  // 2x1
    expect(
        () => tf.basicLSTMCell(
            forgetBias, lstmKernel, lstmBias, batchedData, batchedC,
            {} as tf.Tensor2D))
        .toThrowError(
            /Argument 'h' passed to 'basicLSTMCell' must be a Tensor/);
  });
});
