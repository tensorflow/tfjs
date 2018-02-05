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

import * as dl from '../index';
import * as test_util from '../test_util';
import {MathTests} from '../test_util';

// softmax
{
  const tests: MathTests = it => {
    it('regular test', math => {
      const y = math.softmax(dl.tensor1d([2, 1, 3]));

      test_util.expectArraysClose(y, [0.24472847, 0.09003057, 0.66524095]);
      test_util.expectNumbersClose(y.get(0) + y.get(1) + y.get(2), 1);
    });

    it('overflow', math => {
      const y = math.softmax(dl.tensor1d([1000, 1000]));

      test_util.expectArraysClose(y, [0.5, 0.5]);
    });

    it('underflow', math => {
      const y = math.softmax(dl.tensor1d([-1000, -1000]));

      test_util.expectArraysClose(y, [0.5, 0.5]);
    });

    it('Huge difference between probabilities', math => {
      const y = math.softmax(dl.tensor1d([-1000, +1000]));

      test_util.expectArraysClose(y, [0, 1]);
    });

    it('Propagates NaNs', math => {
      const a = dl.tensor1d([2, 1, NaN]);
      const y = math.softmax(a);
      test_util.expectArraysClose(y, [NaN, NaN, NaN]);
    });

    it('2D, dim=1', math => {
      const y = math.softmax(dl.tensor2d([[2, 1, 3], [1, 3, 2]], [2, 3]), 1);
      const expected = [
        0.24472847, 0.09003057, 0.66524095, 0.09003057, 0.66524095, 0.24472847
      ];
      expect(y.rank).toBe(2);
      test_util.expectArraysClose(y, expected);
    });

    it('2D, implicit dim=1', math => {
      const y = math.softmax(dl.tensor2d([[2, 1, 3], [1, 3, 2]], [2, 3]));
      const expected = [
        0.24472847, 0.09003057, 0.66524095, 0.09003057, 0.66524095, 0.24472847
      ];
      expect(y.rank).toBe(2);
      test_util.expectArraysClose(y, expected);
    });

    it('2D, dim=0 throws error', math => {
      const f = () => {
        math.softmax(dl.tensor2d([[2, 1, 3], [1, 3, 2]], [2, 3]), 0);
      };
      expect(f).toThrowError();
    });

    it('1D gradient', math => {
      const x = dl.tensor1d([10, 0, -1]);
      const y = math.softmax(x);
      const dy = dl.tensor1d([1, 2, 3]);
      const vjp = math.vjp(() => math.softmax(x), {x}, dy);

      const totalSum = math.sum(math.multiply(dy, y));

      expect(vjp.x.shape).toEqual(x.shape);
      test_util.expectArraysClose(vjp.x, [
        (dy.get(0) - totalSum.get()) * y.get(0),
        (dy.get(1) - totalSum.get()) * y.get(1),
        (dy.get(2) - totalSum.get()) * y.get(2)
      ]);
    });

    it('2D gradient', math => {
      const x = dl.tensor2d([10, 0, -1, 5, 4, 3], [2, 3]);
      const y = math.softmax(x);
      const dy = dl.tensor2d([3, 2, 1, 1, 2, 3], [2, 3]);
      const vjp = math.vjp(() => math.softmax(x), {x}, dy);

      const axis = -1;
      const totalSum = math.sum(math.multiplyStrict(dy, y), axis);

      expect(vjp.x.shape).toEqual(x.shape);
      test_util.expectArraysClose(vjp.x, [
        (dy.get(0, 0) - totalSum.get(0)) * y.get(0, 0),
        (dy.get(0, 1) - totalSum.get(0)) * y.get(0, 1),
        (dy.get(0, 2) - totalSum.get(0)) * y.get(0, 2),
        (dy.get(1, 0) - totalSum.get(1)) * y.get(1, 0),
        (dy.get(1, 1) - totalSum.get(1)) * y.get(1, 1),
        (dy.get(1, 2) - totalSum.get(1)) * y.get(1, 2)
      ]);
    });
  };

  test_util.describeMathCPU('softmax', [tests]);
  test_util.describeMathGPU('softmax', [tests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}
{
  const tests: MathTests = it => {
    it('1D', math => {
      const logits = dl.tensor1d([1, 2, 3]);
      const label = dl.tensor1d([0.3, 0.6, 0.1]);
      const softmaxLogits = math.softmax(logits);

      const y = math.softmaxCrossEntropy(label, logits);

      expect(y.shape).toEqual([]);
      test_util.expectNumbersClose(
          y.get(),
          -Math.log(softmaxLogits.get(0)) * label.get(0) +
              -Math.log(softmaxLogits.get(1)) * label.get(1) +
              -Math.log(softmaxLogits.get(2)) * label.get(2));
    });

    it('2D implicit dim', math => {
      const logits = dl.tensor2d([1, 2, 3, 4, 5, 6], [2, 3]);
      const label = dl.tensor2d([0.3, 0.6, 0.1, 0.2, 0.3, 0.5], [2, 3]);
      const softmaxLogits = math.softmax(logits);

      const y = math.softmaxCrossEntropy(label, logits);

      expect(y.shape).toEqual([2]);
      test_util.expectArraysClose(y, [
        -Math.log(softmaxLogits.get(0, 0)) * label.get(0, 0) +
            -Math.log(softmaxLogits.get(0, 1)) * label.get(0, 1) +
            -Math.log(softmaxLogits.get(0, 2)) * label.get(0, 2),
        -Math.log(softmaxLogits.get(1, 0)) * label.get(1, 0) +
            -Math.log(softmaxLogits.get(1, 1)) * label.get(1, 1) +
            -Math.log(softmaxLogits.get(1, 2)) * label.get(1, 2)
      ]);
    });

    it('2D, dim=1', math => {
      const logits = dl.tensor2d([1, 2, 3, 4, 5, 6], [2, 3]);
      const label = dl.tensor2d([0.3, 0.6, 0.1, 0.2, 0.3, 0.5], [2, 3]);
      const dim = 1;
      const softmaxLogits = math.softmax(logits, dim);

      const y = math.softmaxCrossEntropy(label, logits, dim);

      expect(y.shape).toEqual([2]);
      test_util.expectArraysClose(y, [
        -Math.log(softmaxLogits.get(0, 0)) * label.get(0, 0) +
            -Math.log(softmaxLogits.get(0, 1)) * label.get(0, 1) +
            -Math.log(softmaxLogits.get(0, 2)) * label.get(0, 2),
        -Math.log(softmaxLogits.get(1, 0)) * label.get(1, 0) +
            -Math.log(softmaxLogits.get(1, 1)) * label.get(1, 1) +
            -Math.log(softmaxLogits.get(1, 2)) * label.get(1, 2)
      ]);
    });

    it('2D, dim=0 throws error', math => {
      const logits = dl.tensor2d([1, 2, 3, 4, 5, 6], [2, 3]);
      const label = dl.tensor2d([0.3, 0.6, 0.1, 0.2, 0.3, 0.5], [2, 3]);
      const dim = 0;

      expect(() => math.softmaxCrossEntropy(label, logits, dim)).toThrowError();
    });

    it('Propagates NaNs', math => {
      const logits = dl.tensor1d([1, 2, NaN]);
      const label = dl.tensor1d([0.3, 0.6, 0.1]);

      const y = math.softmaxCrossEntropy(label, logits);

      expect(y.shape).toEqual([]);
      test_util.expectArraysClose(y, [NaN]);
    });

    it('1D gradient', math => {
      const logits = dl.tensor1d([1, 2, 3]);
      const labels = dl.tensor1d([0.3, 0.6, 0.1]);
      const softmaxLogits = math.softmax(logits);
      const dy = dl.scalar(2);

      const vjp = math.vjp(
          () => math.softmaxCrossEntropy(labels, logits), {labels, logits}, dy);

      expect(vjp.logits.shape).toEqual(logits.shape);
      expect(vjp.labels.shape).toEqual(labels.shape);

      test_util.expectArraysClose(vjp.logits, [
        dy.get() * (softmaxLogits.get(0) - labels.get(0)),
        dy.get() * (softmaxLogits.get(1) - labels.get(1)),
        dy.get() * (softmaxLogits.get(2) - labels.get(2))
      ]);

      test_util.expectArraysClose(vjp.labels, [
        dy.get() * (labels.get(0) - softmaxLogits.get(0)),
        dy.get() * (labels.get(1) - softmaxLogits.get(1)),
        dy.get() * (labels.get(2) - softmaxLogits.get(2))
      ]);
    });

    it('2D gradient', math => {
      const logits = dl.tensor2d([1, 2, 3, 4, 5, 6], [2, 3]);
      const labels = dl.tensor2d([0.3, 0.6, 0.1, .2, .3, .5], [2, 3]);
      const softmaxLogits = math.softmax(logits);
      const dy = dl.tensor1d([2, 4]);

      const vjp = math.vjp(
          () => math.softmaxCrossEntropy(labels, logits), {labels, logits}, dy);

      expect(vjp.logits.shape).toEqual(logits.shape);
      expect(vjp.labels.shape).toEqual(labels.shape);

      test_util.expectArraysClose(vjp.logits, [
        dy.get(0) * (softmaxLogits.get(0, 0) - labels.get(0, 0)),
        dy.get(0) * (softmaxLogits.get(0, 1) - labels.get(0, 1)),
        dy.get(0) * (softmaxLogits.get(0, 2) - labels.get(0, 2)),
        dy.get(1) * (softmaxLogits.get(1, 0) - labels.get(1, 0)),
        dy.get(1) * (softmaxLogits.get(1, 1) - labels.get(1, 1)),
        dy.get(1) * (softmaxLogits.get(1, 2) - labels.get(1, 2))
      ]);

      test_util.expectArraysClose(vjp.labels, [
        dy.get(0) * (labels.get(0, 0) - softmaxLogits.get(0, 0)),
        dy.get(0) * (labels.get(0, 1) - softmaxLogits.get(0, 1)),
        dy.get(0) * (labels.get(0, 2) - softmaxLogits.get(0, 2)),
        dy.get(1) * (labels.get(1, 0) - softmaxLogits.get(1, 0)),
        dy.get(1) * (labels.get(1, 1) - softmaxLogits.get(1, 1)),
        dy.get(1) * (labels.get(1, 2) - softmaxLogits.get(1, 2))
      ]);
    });
  };

  test_util.describeMathCPU('softmaxCrossEntropy', [tests]);
  test_util.describeMathGPU('softmaxCrossEntropy', [tests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}
