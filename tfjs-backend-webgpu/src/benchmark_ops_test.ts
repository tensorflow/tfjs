/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
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

import * as tf from '@tensorflow/tfjs-core';

import {benchmarkAndLog, describeWebGPU} from './test_util';

const getInputInfo = (inputs: tf.Tensor[]) => {
  const shapes = inputs.map(input => input.shape);
  let info = '';
  for (let i = 0; i < shapes.length; i++) {
    info += ` [${shapes[i].join(',')}]`;
  }
  return info;
};

describeWebGPU('Ops benchmarks', () => {
  fit('argMax', async () => {
    const doTest = async (axis: number) => {
      const a = tf.randomNormal([100, 100, 100]);

      await benchmarkAndLog(`argMax axis=${axis}${getInputInfo([a])}`, () => {
        return tf.argMax(a, axis);
      });
    };

    await doTest(0);
    await doTest(1);
    await doTest(2);
  });

  fit('matMul', async () => {
    const a = tf.randomNormal([500, 500]);
    const b = tf.randomNormal([500, 500]);

    await benchmarkAndLog(
        `matMul${getInputInfo([a, b])}`, () => tf.matMul(a, b));
  });

  fit('add', async () => {
    const a = tf.randomNormal([1, 65, 65, 256]);
    const b = tf.randomNormal([1, 65, 65, 256]);

    await benchmarkAndLog(`add${getInputInfo([a, b])}`, () => tf.add(a, b));
  });

  fit('add', async () => {
    const a = tf.randomNormal([1, 129, 129, 64]);
    const b = tf.randomNormal([64]);
    await benchmarkAndLog(`add${getInputInfo([a, b])}`, () => tf.add(a, b));
  });

  fit('conv2d', async () => {
    const a = tf.randomNormal<tf.Rank.R4>([1, 128, 128, 4]);
    const b = tf.randomNormal<tf.Rank.R4>([25, 25, 4, 4]);

    await benchmarkAndLog(
        `conv2d${getInputInfo([a, b])}`, () => tf.conv2d(a, b, 1, 'same'));
  });

  fit('conv2d', async () => {
    const a = tf.randomNormal<tf.Rank.R4>([1, 263, 263, 3]);
    const b = tf.randomNormal<tf.Rank.R4>([7, 7, 3, 64]);

    await benchmarkAndLog(
        `conv2d${getInputInfo([a, b])}`, () => tf.conv2d(a, b, 1, 'same'));
  });

  fit('relu', async () => {
    const a = tf.randomNormal([1, 129, 129, 64]);

    await benchmarkAndLog(`relu${getInputInfo([a])}`, () => tf.relu(a));
  });

  fit('pad', async () => {
    const a = tf.randomNormal([1, 129, 129, 64]);

    await benchmarkAndLog(
        `pad${getInputInfo([a])}`,
        () => tf.pad(a, [[0, 1], [0, 1], [0, 1], [0, 1]]));
  });

  fit('maxpool', async () => {
    const a = tf.randomNormal<tf.Rank.R4>([1, 131, 131, 64]);

    await benchmarkAndLog(
        `maxPool${getInputInfo([a])}`, () => tf.maxPool(a, 2, 1, 'same'));
  });
});
