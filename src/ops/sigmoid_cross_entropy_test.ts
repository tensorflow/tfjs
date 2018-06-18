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
import {describeWithFlags} from '../jasmine_util';
import {ALL_ENVS, expectArraysClose} from '../test_util';

describeWithFlags('sigmoidCrossEntropyWithLogits', ALL_ENVS, () => {
  const sigmoid = (val: number) => 1 / (1 + Math.exp(-val));

  const getExpectedVal = (labelValues: number[], logitValues: number[]) => {
    const expected = [];

    for (let i = 0; i < labelValues.length; i++) {
      expected[i] = labelValues[i] * -1 * Math.log(sigmoid(logitValues[i])) +
          ((1 - labelValues[i]) * -1 * Math.log(1 - sigmoid(logitValues[i])));
    }

    return expected;
  };

  it('1D', () => {
    const logitValues = [1, 2, 3];
    const labelValues = [0.3, -0.6, 0.1];

    const logits = tf.tensor1d(logitValues);
    const label = tf.tensor1d(labelValues);

    const expected = getExpectedVal(labelValues, logitValues);

    const y = tf.sigmoidCrossEntropyWithLogits(label, logits);

    expect(y.shape).toEqual([3]);
    expectArraysClose(y, expected);
  });

  it('2D', () => {
    const logitValues = [1, 2, 3, 4, 5, 6];
    const labelValues = [0.3, 0.6, 0.1, 0.2, 0.3, 0.5];

    const logits = tf.tensor2d(logitValues, [2, 3]);
    const label = tf.tensor2d(labelValues, [2, 3]);

    const y = tf.sigmoidCrossEntropyWithLogits(label, logits);

    const expected = getExpectedVal(labelValues, logitValues);

    expect(y.shape).toEqual([2, 3]);
    expectArraysClose(y, expected);
  });

  it('Propagates NaNs', () => {
    const logitValues = [1, 2, NaN];
    const labelValues = [0.3, -0.6, 0.1];

    const logits = tf.tensor1d(logitValues);
    const label = tf.tensor1d(labelValues);

    const expected = getExpectedVal(labelValues, logitValues);

    const y = tf.sigmoidCrossEntropyWithLogits(label, logits);

    expect(y.shape).toEqual([3]);
    expectArraysClose(y, expected);
  });

  it('throws when passed labels as a non-tensor', () => {
    const e =  // tslint:disable-next-line:max-line-length
        /Argument 'labels' passed to 'sigmoidCrossEntropyWithLogits' must be a Tensor/;
    expect(() => tf.sigmoidCrossEntropyWithLogits({} as tf.Tensor, tf.tensor1d([
      1
    ]))).toThrowError(e);
  });

  it('throws when passed logits as a non-tensor', () => {
    const e =  // tslint:disable-next-line:max-line-length
        /Argument 'logits' passed to 'sigmoidCrossEntropyWithLogits' must be a Tensor/;
    expect(
        () =>
            tf.sigmoidCrossEntropyWithLogits(tf.tensor1d([1]), {} as tf.Tensor))
        .toThrowError(e);
  });
});
