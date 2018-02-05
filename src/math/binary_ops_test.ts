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

// dl.prelu
{
  const tests: MathTests = it => {
    it('basic', () => {
      const x = dl.tensor1d([0, 1, -2, -4]);
      const a = dl.tensor1d([0.15, 0.2, 0.25, 0.15]);
      const result = dl.prelu(x, a);

      expect(result.shape).toEqual(x.shape);
      test_util.expectArraysClose(result, [0, 1, -0.5, -0.6]);
    });

    it('propagates NaN', () => {
      const x = dl.tensor1d([0, 1, NaN]);
      const a = dl.tensor1d([0.15, 0.2, 0.25]);
      const result = dl.prelu(x, a);

      expect(result.shape).toEqual(x.shape);
      test_util.expectArraysClose(result, [0, 1, NaN]);
    });

    it('derivative', () => {
      const x = dl.tensor1d([0.5, 3, -0.1, -4]);
      const a = dl.tensor1d([0.2, 0.4, 0.25, 0.15]);
      const dy = dl.tensor1d([1, 1, 1, 1]);

      const dx = dl.vjp(() => dl.prelu(x, a), x, dy);

      expect(dx.shape).toEqual(x.shape);
      expect(dx.dtype).toEqual('float32');
      test_util.expectArraysClose(dx, [1, 1, 0.25, 0.15]);
    });

    it('derivative propagates NaN', () => {
      const x = dl.tensor1d([0.5, -0.1, NaN]);
      const a = dl.tensor1d([0.2, 0.3, 0.25]);
      const dy = dl.tensor1d([5, 50, 500]);

      const dx = dl.vjp(() => dl.prelu(x, a), x, dy);

      expect(dx.shape).toEqual(x.shape);
      expect(dx.dtype).toEqual('float32');
      test_util.expectArraysClose(dx, [5, 50 * 0.3, NaN], 1e-1);
    });
  };

  test_util.describeMathCPU('prelu', [tests]);
  test_util.describeMathGPU('prelu', [tests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}

// dl.maximum
{
  const tests: MathTests = it => {
    it('float32 and float32', () => {
      const a = dl.tensor1d([0.5, 3, -0.1, -4]);
      const b = dl.tensor1d([0.2, 0.4, 0.25, 0.15]);
      const result = dl.maximum(a, b);

      expect(result.shape).toEqual(a.shape);
      test_util.expectArraysClose(result, [0.5, 3, 0.25, 0.15]);
    });

    it('int32 and int32', () => {
      const a = dl.tensor1d([1, 5, 2, 3], 'int32');
      const b = dl.tensor1d([2, 3, 1, 4], 'int32');
      const result = dl.maximum(a, b);

      expect(result.shape).toEqual(a.shape);
      expect(result.dtype).toBe('int32');
      test_util.expectArraysEqual(result, [2, 5, 2, 4]);
    });

    it('bool and bool', () => {
      const a = dl.tensor1d([true, false, false, true], 'bool');
      const b = dl.tensor1d([false, false, true, true], 'bool');
      const result = dl.maximum(a, b);

      expect(result.shape).toEqual(a.shape);
      expect(result.dtype).toBe('bool');
      test_util.expectArraysEqual(result, [true, false, true, true]);
    });

    it('different dtypes throws error', () => {
      const a = dl.tensor1d([true, false, false, true], 'float32');
      const b = dl.tensor1d([false, false, true, true], 'int32');
      // tslint:disable-next-line:no-any
      expect(() => dl.maximum(a, b as any)).toThrowError();
    });

    it('propagates NaN', () => {
      const a = dl.tensor1d([0.5, -0.1, NaN]);
      const b = dl.tensor1d([0.2, 0.3, 0.25]);
      const result = dl.maximum(a, b);

      expect(result.shape).toEqual(a.shape);
      test_util.expectArraysClose(result, [0.5, 0.3, NaN]);
    });

    it('broadcasts Tensor1D and scalar', () => {
      const a = dl.tensor1d([0.5, 3, -0.1, -4]);
      const b = dl.scalar(0.6);
      const result = dl.maximum(a, b);

      expect(result.shape).toEqual(a.shape);
      test_util.expectArraysClose(result, [0.6, 3, 0.6, 0.6]);
    });

    it('broadcasts scalar and Tensor1D', () => {
      const a = dl.scalar(0.6);
      const b = dl.tensor1d([0.5, 3, -0.1, -4]);
      const result = dl.maximum(a, b);

      expect(result.shape).toEqual(b.shape);
      test_util.expectArraysClose(result, [0.6, 3, 0.6, 0.6]);
    });

    it('broadcasts Tensor1D and Tensor2D', () => {
      const a = dl.tensor1d([0.5, 0.3]);
      const b = dl.tensor2d([0.2, 0.4, 0.6, 0.15], [2, 2]);
      const result = dl.maximum(a, b);

      expect(result.shape).toEqual(b.shape);
      test_util.expectArraysClose(result, [0.5, 0.4, 0.6, 0.3]);
    });

    it('broadcasts 2x1 Tensor2D and 2x2 Tensor2D', () => {
      const a = dl.tensor2d([0.5, 0.3], [2, 1]);
      const b = dl.tensor2d([0.2, 0.4, 0.6, 0.15], [2, 2]);
      const result = dl.maximum(a, b);

      expect(result.shape).toEqual(b.shape);
      test_util.expectArraysClose(result, [0.5, 0.5, 0.6, 0.3]);
    });

    it('gradients: Scalar', () => {
      const a = dl.scalar(5.2);
      const b = dl.scalar(0.6);
      const dy = dl.scalar(3);

      const gradients = dl.vjp(() => dl.maximum(a, b), {a, b}, dy);

      expect(gradients.a.shape).toEqual(a.shape);
      expect(gradients.b.shape).toEqual(b.shape);
      expect(gradients.a.dtype).toEqual('float32');
      expect(gradients.b.dtype).toEqual('float32');

      test_util.expectArraysClose(gradients.a, [3 * 1], 1e-1);
      test_util.expectArraysClose(gradients.b, [3 * 0], 1e-1);
    });

    it('gradients: Tensor1D', () => {
      const a = dl.tensor1d([1.1, 2.6, 3, 5.9]);
      const b = dl.tensor1d([1.0, 2.7, 3, 5.8]);
      const dy = dl.tensor1d([1, 2, 3, 4]);

      const gradients = dl.vjp(() => dl.maximum(a, b), {a, b}, dy);

      expect(gradients.a.shape).toEqual(a.shape);
      expect(gradients.b.shape).toEqual(b.shape);
      expect(gradients.a.dtype).toEqual('float32');
      expect(gradients.b.dtype).toEqual('float32');

      test_util.expectArraysClose(
          gradients.a, [1 * 1, 2 * 0, 3 * 1, 4 * 1], 1e-1);
      test_util.expectArraysClose(
          gradients.b, [1 * 0, 2 * 1, 3 * 0, 4 * 0], 1e-1);
    });

    it('gradients: Tensor2D', () => {
      const a = dl.tensor2d([0.5, 0.3, 0.7, 0.9], [2, 2]);
      const b = dl.tensor2d([0.2, 0.4, 0.7, 0.15], [2, 2]);
      const dy = dl.tensor2d([1, 2, 3, 4], [2, 2]);

      const gradients = dl.vjp(() => dl.maximum(a, b), {a, b}, dy);

      expect(gradients.a.shape).toEqual(a.shape);
      expect(gradients.b.shape).toEqual(b.shape);
      expect(gradients.a.dtype).toEqual('float32');
      expect(gradients.b.dtype).toEqual('float32');

      test_util.expectArraysClose(
          gradients.a, [1 * 1, 2 * 0, 3 * 1, 4 * 1], 1e-1);
      test_util.expectArraysClose(
          gradients.b, [1 * 0, 2 * 1, 3 * 0, 4 * 0], 1e-1);
    });
  };

  test_util.describeMathCPU('maximum', [tests]);
  test_util.describeMathGPU('maximum', [tests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}

// dl.minimum
{
  const tests: MathTests = it => {
    it('float32 and float32', () => {
      const a = dl.tensor1d([0.5, 3, -0.1, -4]);
      const b = dl.tensor1d([0.2, 0.4, 0.25, 0.15]);
      const result = dl.minimum(a, b);

      expect(result.shape).toEqual(a.shape);
      test_util.expectArraysClose(result, [0.2, 0.4, -0.1, -4]);
    });

    it('int32 and int32', () => {
      const a = dl.tensor1d([1, 5, 2, 3], 'int32');
      const b = dl.tensor1d([2, 3, 1, 4], 'int32');
      const result = dl.minimum(a, b);

      expect(result.shape).toEqual(a.shape);
      expect(result.dtype).toBe('int32');
      test_util.expectArraysEqual(result, [1, 3, 1, 3]);
    });

    it('bool and bool', () => {
      const a = dl.tensor1d([true, false, false, true], 'bool');
      const b = dl.tensor1d([false, false, true, true], 'bool');
      const result = dl.minimum(a, b);

      expect(result.shape).toEqual(a.shape);
      expect(result.dtype).toBe('bool');
      test_util.expectArraysEqual(result, [false, false, false, true]);
    });

    it('different dtypes throws error', () => {
      const a = dl.tensor1d([true, false, false, true], 'float32');
      const b = dl.tensor1d([false, false, true, true], 'int32');
      // tslint:disable-next-line:no-any
      expect(() => dl.minimum(a, b as any)).toThrowError();
    });

    it('propagates NaN', () => {
      const a = dl.tensor1d([0.5, -0.1, NaN]);
      const b = dl.tensor1d([0.2, 0.3, 0.25]);
      const result = dl.minimum(a, b);

      expect(result.shape).toEqual(a.shape);
      test_util.expectArraysClose(result, [0.2, -0.1, NaN]);
    });

    it('broadcasts Tensor1D and scalar', () => {
      const a = dl.tensor1d([0.5, 3, -0.1, -4]);
      const b = dl.scalar(0.6);
      const result = dl.minimum(a, b);

      expect(result.shape).toEqual(a.shape);
      test_util.expectArraysClose(result, [0.5, 0.6, -0.1, -4]);
    });

    it('broadcasts scalar and Tensor1D', () => {
      const a = dl.scalar(0.6);
      const b = dl.tensor1d([0.5, 3, -0.1, -4]);
      const result = dl.minimum(a, b);

      expect(result.shape).toEqual(b.shape);
      test_util.expectArraysClose(result, [0.5, 0.6, -0.1, -4]);
    });

    it('broadcasts Tensor1D and Tensor2D', () => {
      const a = dl.tensor1d([0.5, 0.3]);
      const b = dl.tensor2d([0.2, 0.4, 0.6, 0.15], [2, 2]);
      const result = dl.minimum(a, b);

      expect(result.shape).toEqual(b.shape);
      test_util.expectArraysClose(result, [0.2, 0.3, 0.5, 0.15]);
    });

    it('broadcasts 2x1 Tensor2D and 2x2 Tensor2D', () => {
      const a = dl.tensor2d([0.5, 0.3], [2, 1]);
      const b = dl.tensor2d([0.2, 0.4, 0.6, 0.15], [2, 2]);
      const result = dl.minimum(a, b);

      expect(result.shape).toEqual(b.shape);
      test_util.expectArraysClose(result, [0.2, 0.4, 0.3, 0.15]);
    });

    it('gradients: Scalar', () => {
      const a = dl.scalar(5.2);
      const b = dl.scalar(0.6);
      const dy = dl.scalar(3);

      const gradients = dl.vjp(() => dl.minimum(a, b), {a, b}, dy);

      expect(gradients.a.shape).toEqual(a.shape);
      expect(gradients.b.shape).toEqual(b.shape);
      expect(gradients.a.dtype).toEqual('float32');
      expect(gradients.b.dtype).toEqual('float32');

      test_util.expectArraysClose(gradients.a, [3 * 0], 1e-1);
      test_util.expectArraysClose(gradients.b, [3 * 1], 1e-1);
    });

    it('gradients: Tensor1D', () => {
      const a = dl.tensor1d([1.1, 2.6, 3, 5.9]);
      const b = dl.tensor1d([1.0, 2.7, 3, 5.8]);
      const dy = dl.tensor1d([1, 2, 3, 4]);

      const gradients = dl.vjp(() => dl.minimum(a, b), {a, b}, dy);

      expect(gradients.a.shape).toEqual(a.shape);
      expect(gradients.b.shape).toEqual(b.shape);
      expect(gradients.a.dtype).toEqual('float32');
      expect(gradients.b.dtype).toEqual('float32');

      test_util.expectArraysClose(
          gradients.a, [1 * 0, 2 * 1, 3 * 1, 4 * 0], 1e-1);
      test_util.expectArraysClose(
          gradients.b, [1 * 1, 2 * 0, 3 * 0, 4 * 1], 1e-1);
    });

    it('gradients: Tensor2D', () => {
      const a = dl.tensor2d([0.5, 0.3, 0.7, 0.9], [2, 2]);
      const b = dl.tensor2d([0.2, 0.4, 0.7, 0.15], [2, 2]);
      const dy = dl.tensor2d([1, 2, 3, 4], [2, 2]);

      const gradients = dl.vjp(() => dl.minimum(a, b), {a, b}, dy);

      expect(gradients.a.shape).toEqual(a.shape);
      expect(gradients.b.shape).toEqual(b.shape);
      expect(gradients.a.dtype).toEqual('float32');
      expect(gradients.b.dtype).toEqual('float32');

      test_util.expectArraysClose(
          gradients.a, [1 * 0, 2 * 1, 3 * 1, 4 * 0], 1e-1);
      test_util.expectArraysClose(
          gradients.b, [1 * 1, 2 * 0, 3 * 0, 4 * 1], 1e-1);
    });
  };

  test_util.describeMathCPU('minimum', [tests]);
  test_util.describeMathGPU('minimum', [tests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}
