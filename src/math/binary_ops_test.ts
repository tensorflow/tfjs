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
// tslint:disable-next-line:max-line-length
import {ALL_ENVS, describeWithFlags, expectArraysClose, expectArraysEqual} from '../test_util';

describeWithFlags('prelu', ALL_ENVS, () => {
  it('basic', () => {
    const x = dl.tensor1d([0, 1, -2, -4]);
    const a = dl.tensor1d([0.15, 0.2, 0.25, 0.15]);
    const result = dl.prelu(x, a);

    expect(result.shape).toEqual(x.shape);
    expectArraysClose(result, [0, 1, -0.5, -0.6]);
  });

  it('propagates NaN', () => {
    const x = dl.tensor1d([0, 1, NaN]);
    const a = dl.tensor1d([0.15, 0.2, 0.25]);
    const result = dl.prelu(x, a);

    expect(result.shape).toEqual(x.shape);
    expectArraysClose(result, [0, 1, NaN]);
  });

  it('derivative', () => {
    const x = dl.tensor1d([0.5, 3, -0.1, -4]);
    const a = dl.tensor1d([0.2, 0.4, 0.25, 0.15]);
    const dy = dl.tensor1d([1, 1, 1, 1]);

    const dx = dl.grad(x => dl.prelu(x, a))(x, dy);

    expect(dx.shape).toEqual(x.shape);
    expect(dx.dtype).toEqual('float32');
    expectArraysClose(dx, [1, 1, 0.25, 0.15]);
  });

  it('derivative propagates NaN', () => {
    const x = dl.tensor1d([0.5, -0.1, NaN]);
    const a = dl.tensor1d([0.2, 0.3, 0.25]);
    const dy = dl.tensor1d([5, 50, 500]);

    const dx = dl.grad(x => dl.prelu(x, a))(x, dy);

    expect(dx.shape).toEqual(x.shape);
    expect(dx.dtype).toEqual('float32');
    expectArraysClose(dx, [5, 50 * 0.3, NaN], 1e-1);
  });
});

describeWithFlags('maximum', ALL_ENVS, () => {
  it('float32 and float32', () => {
    const a = dl.tensor1d([0.5, 3, -0.1, -4]);
    const b = dl.tensor1d([0.2, 0.4, 0.25, 0.15]);
    const result = dl.maximum(a, b);

    expect(result.shape).toEqual(a.shape);
    expectArraysClose(result, [0.5, 3, 0.25, 0.15]);
  });

  it('int32 and int32', () => {
    const a = dl.tensor1d([1, 5, 2, 3], 'int32');
    const b = dl.tensor1d([2, 3, 1, 4], 'int32');
    const result = dl.maximum(a, b);

    expect(result.shape).toEqual(a.shape);
    expect(result.dtype).toBe('int32');
    expectArraysEqual(result, [2, 5, 2, 4]);
  });

  it('bool and bool', () => {
    const a = dl.tensor1d([true, false, false, true], 'bool');
    const b = dl.tensor1d([false, false, true, true], 'bool');
    const result = dl.maximum(a, b);

    expect(result.shape).toEqual(a.shape);
    expect(result.dtype).toBe('bool');
    expectArraysEqual(result, [true, false, true, true]);
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
    expectArraysClose(result, [0.5, 0.3, NaN]);
  });

  it('broadcasts Tensor1D and scalar', () => {
    const a = dl.tensor1d([0.5, 3, -0.1, -4]);
    const b = dl.scalar(0.6);
    const result = dl.maximum(a, b);

    expect(result.shape).toEqual(a.shape);
    expectArraysClose(result, [0.6, 3, 0.6, 0.6]);
  });

  it('broadcasts scalar and Tensor1D', () => {
    const a = dl.scalar(0.6);
    const b = dl.tensor1d([0.5, 3, -0.1, -4]);
    const result = dl.maximum(a, b);

    expect(result.shape).toEqual(b.shape);
    expectArraysClose(result, [0.6, 3, 0.6, 0.6]);
  });

  it('broadcasts Tensor1D and Tensor2D', () => {
    const a = dl.tensor1d([0.5, 0.3]);
    const b = dl.tensor2d([0.2, 0.4, 0.6, 0.15], [2, 2]);
    const result = dl.maximum(a, b);

    expect(result.shape).toEqual(b.shape);
    expectArraysClose(result, [0.5, 0.4, 0.6, 0.3]);
  });

  it('broadcasts 2x1 Tensor2D and 2x2 Tensor2D', () => {
    const a = dl.tensor2d([0.5, 0.3], [2, 1]);
    const b = dl.tensor2d([0.2, 0.4, 0.6, 0.15], [2, 2]);
    const result = dl.maximum(a, b);

    expect(result.shape).toEqual(b.shape);
    expectArraysClose(result, [0.5, 0.5, 0.6, 0.3]);
  });

  it('gradients: Scalar', () => {
    const a = dl.scalar(5.2);
    const b = dl.scalar(0.6);
    const dy = dl.scalar(3);

    const grads = dl.grads((a, b) => dl.maximum(a, b));
    const [da, db] = grads([a, b], dy);

    expect(da.shape).toEqual(a.shape);
    expect(db.shape).toEqual(b.shape);
    expect(da.dtype).toEqual('float32');
    expect(db.dtype).toEqual('float32');

    expectArraysClose(da, [3 * 1], 1e-1);
    expectArraysClose(db, [3 * 0], 1e-1);
  });

  it('gradients: Tensor1D', () => {
    const a = dl.tensor1d([1.1, 2.6, 3, 5.9]);
    const b = dl.tensor1d([1.0, 2.7, 3, 5.8]);
    const dy = dl.tensor1d([1, 2, 3, 4]);

    const grads = dl.grads((a, b) => dl.maximum(a, b));
    const [da, db] = grads([a, b], dy);

    expect(da.shape).toEqual(a.shape);
    expect(db.shape).toEqual(b.shape);
    expect(da.dtype).toEqual('float32');
    expect(db.dtype).toEqual('float32');

    expectArraysClose(da, [1 * 1, 2 * 0, 3 * 1, 4 * 1], 1e-1);
    expectArraysClose(db, [1 * 0, 2 * 1, 3 * 0, 4 * 0], 1e-1);
  });

  it('gradients: Tensor2D', () => {
    const a = dl.tensor2d([0.5, 0.3, 0.7, 0.9], [2, 2]);
    const b = dl.tensor2d([0.2, 0.4, 0.7, 0.15], [2, 2]);
    const dy = dl.tensor2d([1, 2, 3, 4], [2, 2]);

    const grads = dl.grads((a, b) => dl.maximum(a, b));
    const [da, db] = grads([a, b], dy);

    expect(da.shape).toEqual(a.shape);
    expect(db.shape).toEqual(b.shape);
    expect(da.dtype).toEqual('float32');
    expect(db.dtype).toEqual('float32');

    expectArraysClose(da, [1 * 1, 2 * 0, 3 * 1, 4 * 1], 1e-1);
    expectArraysClose(db, [1 * 0, 2 * 1, 3 * 0, 4 * 0], 1e-1);
  });
});

describeWithFlags('minimum', ALL_ENVS, () => {
  it('float32 and float32', () => {
    const a = dl.tensor1d([0.5, 3, -0.1, -4]);
    const b = dl.tensor1d([0.2, 0.4, 0.25, 0.15]);
    const result = dl.minimum(a, b);

    expect(result.shape).toEqual(a.shape);
    expectArraysClose(result, [0.2, 0.4, -0.1, -4]);
  });

  it('int32 and int32', () => {
    const a = dl.tensor1d([1, 5, 2, 3], 'int32');
    const b = dl.tensor1d([2, 3, 1, 4], 'int32');
    const result = dl.minimum(a, b);

    expect(result.shape).toEqual(a.shape);
    expect(result.dtype).toBe('int32');
    expectArraysEqual(result, [1, 3, 1, 3]);
  });

  it('bool and bool', () => {
    const a = dl.tensor1d([true, false, false, true], 'bool');
    const b = dl.tensor1d([false, false, true, true], 'bool');
    const result = dl.minimum(a, b);

    expect(result.shape).toEqual(a.shape);
    expect(result.dtype).toBe('bool');
    expectArraysEqual(result, [false, false, false, true]);
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
    expectArraysClose(result, [0.2, -0.1, NaN]);
  });

  it('broadcasts Tensor1D and scalar', () => {
    const a = dl.tensor1d([0.5, 3, -0.1, -4]);
    const b = dl.scalar(0.6);
    const result = dl.minimum(a, b);

    expect(result.shape).toEqual(a.shape);
    expectArraysClose(result, [0.5, 0.6, -0.1, -4]);
  });

  it('broadcasts scalar and Tensor1D', () => {
    const a = dl.scalar(0.6);
    const b = dl.tensor1d([0.5, 3, -0.1, -4]);
    const result = dl.minimum(a, b);

    expect(result.shape).toEqual(b.shape);
    expectArraysClose(result, [0.5, 0.6, -0.1, -4]);
  });

  it('broadcasts Tensor1D and Tensor2D', () => {
    const a = dl.tensor1d([0.5, 0.3]);
    const b = dl.tensor2d([0.2, 0.4, 0.6, 0.15], [2, 2]);
    const result = dl.minimum(a, b);

    expect(result.shape).toEqual(b.shape);
    expectArraysClose(result, [0.2, 0.3, 0.5, 0.15]);
  });

  it('broadcasts 2x1 Tensor2D and 2x2 Tensor2D', () => {
    const a = dl.tensor2d([0.5, 0.3], [2, 1]);
    const b = dl.tensor2d([0.2, 0.4, 0.6, 0.15], [2, 2]);
    const result = dl.minimum(a, b);

    expect(result.shape).toEqual(b.shape);
    expectArraysClose(result, [0.2, 0.4, 0.3, 0.15]);
  });

  it('gradients: Scalar', () => {
    const a = dl.scalar(5.2);
    const b = dl.scalar(0.6);
    const dy = dl.scalar(3);

    const grads = dl.grads((a, b) => dl.minimum(a, b));
    const [da, db] = grads([a, b], dy);

    expect(da.shape).toEqual(a.shape);
    expect(db.shape).toEqual(b.shape);
    expect(da.dtype).toEqual('float32');
    expect(db.dtype).toEqual('float32');

    expectArraysClose(da, [3 * 0], 1e-1);
    expectArraysClose(db, [3 * 1], 1e-1);
  });

  it('gradients: Tensor1D', () => {
    const a = dl.tensor1d([1.1, 2.6, 3, 5.9]);
    const b = dl.tensor1d([1.0, 2.7, 3, 5.8]);
    const dy = dl.tensor1d([1, 2, 3, 4]);

    const grads = dl.grads((a, b) => dl.minimum(a, b));
    const [da, db] = grads([a, b], dy);

    expect(da.shape).toEqual(a.shape);
    expect(db.shape).toEqual(b.shape);
    expect(da.dtype).toEqual('float32');
    expect(db.dtype).toEqual('float32');

    expectArraysClose(da, [1 * 0, 2 * 1, 3 * 1, 4 * 0], 1e-1);
    expectArraysClose(db, [1 * 1, 2 * 0, 3 * 0, 4 * 1], 1e-1);
  });

  it('gradients: Tensor2D', () => {
    const a = dl.tensor2d([0.5, 0.3, 0.7, 0.9], [2, 2]);
    const b = dl.tensor2d([0.2, 0.4, 0.7, 0.15], [2, 2]);
    const dy = dl.tensor2d([1, 2, 3, 4], [2, 2]);

    const grads = dl.grads((a, b) => dl.minimum(a, b));
    const [da, db] = grads([a, b], dy);

    expect(da.shape).toEqual(a.shape);
    expect(db.shape).toEqual(b.shape);
    expect(da.dtype).toEqual('float32');
    expect(db.dtype).toEqual('float32');

    expectArraysClose(da, [1 * 0, 2 * 1, 3 * 1, 4 * 0], 1e-1);
    expectArraysClose(db, [1 * 1, 2 * 0, 3 * 0, 4 * 1], 1e-1);
  });
});
