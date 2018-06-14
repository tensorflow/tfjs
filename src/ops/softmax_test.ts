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
import {describeWithFlags} from '../jasmine_util';
// tslint:disable-next-line:max-line-length
import {ALL_ENVS, expectArraysClose, expectNumbersClose} from '../test_util';

describeWithFlags('softmax', ALL_ENVS, () => {
  it('regular test', () => {
    const y = tf.softmax(tf.tensor1d([2, 1, 3]));

    expectArraysClose(y, [0.24472847, 0.09003057, 0.66524095]);
    expectNumbersClose(y.get(0) + y.get(1) + y.get(2), 1);
  });

  it('overflow', () => {
    const y = tf.softmax(tf.tensor1d([100, 100]));

    expectArraysClose(y, [0.5, 0.5]);
  });

  it('underflow', () => {
    const y = tf.softmax(tf.tensor1d([-100, -100]));

    expectArraysClose(y, [0.5, 0.5]);
  });

  it('Huge difference between probabilities', () => {
    const y = tf.softmax(tf.tensor1d([-1000, +1000]));

    expectArraysClose(y, [0, 1]);
  });

  it('Propagates NaNs', () => {
    const a = tf.tensor1d([2, 1, NaN]);
    const y = tf.softmax(a);
    expectArraysClose(y, [NaN, NaN, NaN]);
  });

  it('2D, dim=1', () => {
    const y = tf.softmax(tf.tensor2d([[2, 1, 3], [1, 3, 2]], [2, 3]), 1);
    const expected = [
      0.24472847, 0.09003057, 0.66524095, 0.09003057, 0.66524095, 0.24472847
    ];
    expect(y.rank).toBe(2);
    expectArraysClose(y, expected);
  });

  it('2D, implicit dim=1', () => {
    const y = tf.softmax(tf.tensor2d([[2, 1, 3], [1, 3, 2]], [2, 3]));
    const expected = [
      0.24472847, 0.09003057, 0.66524095, 0.09003057, 0.66524095, 0.24472847
    ];
    expect(y.rank).toBe(2);
    expectArraysClose(y, expected);
  });

  it('2D, dim=0 throws error', () => {
    const f = () => {
      tf.softmax(tf.tensor2d([[2, 1, 3], [1, 3, 2]], [2, 3]), 0);
    };
    expect(f).toThrowError();
  });

  it('1D gradient', () => {
    const x = tf.tensor1d([10, 0, -1]);
    const y = tf.softmax(x);
    const dy = tf.tensor1d([1, 2, 3]);
    const dx = tf.grad((x) => x.softmax())(x, dy);

    const totalSum = tf.sum(tf.mul(dy, y));

    expect(dx.shape).toEqual(x.shape);
    expectArraysClose(dx, [
      (dy.get(0) - totalSum.get()) * y.get(0),
      (dy.get(1) - totalSum.get()) * y.get(1),
      (dy.get(2) - totalSum.get()) * y.get(2)
    ]);
  });

  it('2D gradient', () => {
    const x = tf.tensor2d([10, 0, -1, 5, 4, 3], [2, 3]);
    const y = tf.softmax(x);
    const dy = tf.tensor2d([3, 2, 1, 1, 2, 3], [2, 3]);
    const dx = tf.grad((x) => x.softmax())(x, dy);

    const axis = -1;
    const totalSum = tf.sum(tf.mulStrict(dy, y), axis);

    expect(dx.shape).toEqual(x.shape);
    expectArraysClose(dx, [
      (dy.get(0, 0) - totalSum.get(0)) * y.get(0, 0),
      (dy.get(0, 1) - totalSum.get(0)) * y.get(0, 1),
      (dy.get(0, 2) - totalSum.get(0)) * y.get(0, 2),
      (dy.get(1, 0) - totalSum.get(1)) * y.get(1, 0),
      (dy.get(1, 1) - totalSum.get(1)) * y.get(1, 1),
      (dy.get(1, 2) - totalSum.get(1)) * y.get(1, 2)
    ]);
  });

  it('throws when passed a non-tensor', () => {
    expect(() => tf.softmax({} as tf.Tensor))
        .toThrowError(/Argument 'logits' passed to 'softmax' must be a Tensor/);
  });
});

describeWithFlags('softmaxCrossEntropy', ALL_ENVS, () => {
  it('1D', () => {
    const logits = tf.tensor1d([1, 2, 3]);
    const label = tf.tensor1d([0.3, 0.6, 0.1]);
    const softmaxLogits = tf.softmax(logits);

    const y = tf.losses.softmaxCrossEntropy(label, logits);

    expect(y.shape).toEqual([]);
    expectNumbersClose(
        y.get(),
        -Math.log(softmaxLogits.get(0)) * label.get(0) +
            -Math.log(softmaxLogits.get(1)) * label.get(1) +
            -Math.log(softmaxLogits.get(2)) * label.get(2));
  });

  it('2D implicit dim', () => {
    const logits = tf.tensor2d([1, 2, 3, 4, 5, 6], [2, 3]);
    const label = tf.tensor2d([0.3, 0.6, 0.1, 0.2, 0.3, 0.5], [2, 3]);
    const softmaxLogits = tf.softmax(logits);

    const y = tf.losses.softmaxCrossEntropy(label, logits);

    expect(y.shape).toEqual([2]);
    expectArraysClose(y, [
      -Math.log(softmaxLogits.get(0, 0)) * label.get(0, 0) +
          -Math.log(softmaxLogits.get(0, 1)) * label.get(0, 1) +
          -Math.log(softmaxLogits.get(0, 2)) * label.get(0, 2),
      -Math.log(softmaxLogits.get(1, 0)) * label.get(1, 0) +
          -Math.log(softmaxLogits.get(1, 1)) * label.get(1, 1) +
          -Math.log(softmaxLogits.get(1, 2)) * label.get(1, 2)
    ]);
  });

  it('2D, dim=1', () => {
    const logits = tf.tensor2d([1, 2, 3, 4, 5, 6], [2, 3]);
    const label = tf.tensor2d([0.3, 0.6, 0.1, 0.2, 0.3, 0.5], [2, 3]);
    const dim = 1;
    const softmaxLogits = tf.softmax(logits, dim);

    const y = tf.losses.softmaxCrossEntropy(label, logits, dim);

    expect(y.shape).toEqual([2]);
    expectArraysClose(y, [
      -Math.log(softmaxLogits.get(0, 0)) * label.get(0, 0) +
          -Math.log(softmaxLogits.get(0, 1)) * label.get(0, 1) +
          -Math.log(softmaxLogits.get(0, 2)) * label.get(0, 2),
      -Math.log(softmaxLogits.get(1, 0)) * label.get(1, 0) +
          -Math.log(softmaxLogits.get(1, 1)) * label.get(1, 1) +
          -Math.log(softmaxLogits.get(1, 2)) * label.get(1, 2)
    ]);
  });

  it('2D, dim=0 throws error', () => {
    const logits = tf.tensor2d([1, 2, 3, 4, 5, 6], [2, 3]);
    const label = tf.tensor2d([0.3, 0.6, 0.1, 0.2, 0.3, 0.5], [2, 3]);
    const dim = 0;

    expect(() => tf.losses.softmaxCrossEntropy(label, logits, dim))
        .toThrowError();
  });

  it('Propagates NaNs', () => {
    const logits = tf.tensor1d([1, 2, NaN]);
    const label = tf.tensor1d([0.3, 0.6, 0.1]);

    const y = tf.losses.softmaxCrossEntropy(label, logits);

    expect(y.shape).toEqual([]);
    expectArraysClose(y, [NaN]);
  });

  it('1D gradient', () => {
    const logits = tf.tensor1d([1, 2, 3]);
    const labels = tf.tensor1d([0.3, 0.6, 0.1]);
    const softmaxLogits = tf.softmax(logits);
    const dy = tf.scalar(2);

    const grads = tf.grads(
        (labels, logits) => tf.losses.softmaxCrossEntropy(labels, logits));
    const [dlabels, dlogits] = grads([labels, logits], dy);

    expect(dlogits.shape).toEqual(logits.shape);
    expectArraysClose(dlogits, [
      dy.get() * (softmaxLogits.get(0) - labels.get(0)),
      dy.get() * (softmaxLogits.get(1) - labels.get(1)),
      dy.get() * (softmaxLogits.get(2) - labels.get(2))
    ]);

    expect(dlabels.shape).toEqual(labels.shape);
    expectArraysClose(dlabels, [
      dy.get() * (labels.get(0) - softmaxLogits.get(0)),
      dy.get() * (labels.get(1) - softmaxLogits.get(1)),
      dy.get() * (labels.get(2) - softmaxLogits.get(2))
    ]);
  });

  it('2D gradient', () => {
    const logits = tf.tensor2d([1, 2, 3, 4, 5, 6], [2, 3]);
    const labels = tf.tensor2d([0.3, 0.6, 0.1, .2, .3, .5], [2, 3]);
    const softmaxLogits = tf.softmax(logits);
    const dy = tf.tensor1d([2, 4]);

    const dlogits = tf.grad(
        logits => tf.losses.softmaxCrossEntropy(labels, logits))(logits, dy);

    expect(dlogits.shape).toEqual(logits.shape);

    expectArraysClose(dlogits, [
      dy.get(0) * (softmaxLogits.get(0, 0) - labels.get(0, 0)),
      dy.get(0) * (softmaxLogits.get(0, 1) - labels.get(0, 1)),
      dy.get(0) * (softmaxLogits.get(0, 2) - labels.get(0, 2)),
      dy.get(1) * (softmaxLogits.get(1, 0) - labels.get(1, 0)),
      dy.get(1) * (softmaxLogits.get(1, 1) - labels.get(1, 1)),
      dy.get(1) * (softmaxLogits.get(1, 2) - labels.get(1, 2))
    ]);
  });

  it('throws when passed labels as a non-tensor', () => {
    const e =
        /Argument 'labels' passed to 'softmaxCrossEntropy' must be a Tensor/;
    expect(() => tf.losses.softmaxCrossEntropy({} as tf.Tensor, tf.tensor1d([
      1
    ]))).toThrowError(e);
  });

  it('throws when passed logits as a non-tensor', () => {
    const e =
        /Argument 'logits' passed to 'softmaxCrossEntropy' must be a Tensor/;
    expect(
        () => tf.losses.softmaxCrossEntropy(tf.tensor1d([1]), {} as tf.Tensor))
        .toThrowError(e);
  });
});
