/**
 * @license
 * Copyright 2018 Google Inc. All Rights Reserved.
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
import {describeWithFlags} from '../jasmine_util';
import {Scalar} from '../tensor';
import {ALL_ENVS, CPU_ENVS, expectArraysClose} from '../test_util';

import {sparseToDense} from './sparse_to_dense';
import {scalar, tensor1d, tensor2d, tensor3d} from './tensor_ops';

let defaultValue: Scalar;
describeWithFlags('sparseToDense', ALL_ENVS, () => {
  beforeEach(() => defaultValue = scalar(0, 'int32'));
  it('should work for scalar indices', () => {
    const indices = scalar(2, 'int32');
    const values = scalar(100, 'int32');
    const shape = [6];
    const result = sparseToDense(indices, values, shape, defaultValue);
    expect(result.shape).toEqual(shape);
    expect(result.dtype).toEqual(values.dtype);
    expectArraysClose(result, [0, 0, 100, 0, 0, 0]);
  });
  it('should work for vector', () => {
    const indices = tensor1d([0, 2, 4], 'int32');
    const values = tensor1d([100, 101, 102], 'int32');
    const shape = [6];
    const result = sparseToDense(indices, values, shape, defaultValue);
    expect(result.shape).toEqual(shape);
    expect(result.dtype).toEqual(values.dtype);
    expectArraysClose(result, [100, 0, 101, 0, 102, 0]);
  });
  it('should work for scalar value', () => {
    const indices = tensor1d([0, 2, 4], 'int32');
    const values = scalar(10, 'int32');
    const shape = [6];
    const result = sparseToDense(indices, values, shape, defaultValue);
    expect(result.shape).toEqual(shape);
    expect(result.dtype).toEqual(values.dtype);
    expectArraysClose(result, [10, 0, 10, 0, 10, 0]);
  });
  it('should work for matrix', () => {
    const indices = tensor2d([0, 1, 1, 1], [2, 2], 'int32');
    const values = tensor1d([5, 6], 'float32');
    const shape = [2, 2];
    const result =
        sparseToDense(indices, values, shape, defaultValue.toFloat());
    expect(result.shape).toEqual(shape);
    expect(result.dtype).toEqual(values.dtype);
    expectArraysClose(result, [0, 5, 0, 6]);
  });

  it('should throw exception if default value does not match dtype', () => {
    const indices = tensor2d([0, 1, 1, 1], [2, 2], 'int32');
    const values = tensor1d([5, 6], 'float32');
    const shape = [2, 2];
    expect(() => sparseToDense(indices, values, shape, scalar(1, 'int32')))
        .toThrowError();
  });

  it('should allow setting default value', () => {
    const indices = tensor2d([0, 1, 1, 1], [2, 2], 'int32');
    const values = tensor1d([5, 6], 'float32');
    const shape = [2, 2];
    const result = sparseToDense(indices, values, shape, scalar(1));
    expect(result.shape).toEqual(shape);
    expect(result.dtype).toEqual(values.dtype);
    expectArraysClose(result, [1, 5, 1, 6]);
  });

  it('no default value passed', () => {
    const indices = tensor2d([0, 1, 1, 1], [2, 2], 'int32');
    const values = tensor1d([5, 6], 'float32');
    const shape = [2, 2];
    const result = sparseToDense(indices, values, shape);
    expect(result.shape).toEqual(shape);
    expect(result.dtype).toEqual(values.dtype);
    expectArraysClose(result, [0, 5, 0, 6]);
  });

  it('should support TensorLike inputs', () => {
    const indices = [[0, 1], [1, 1]];
    const values = [5, 6];
    const shape = [2, 2];
    const result =
        sparseToDense(indices, values, shape, defaultValue.toFloat());
    expect(result.shape).toEqual(shape);
    expect(result.dtype).toEqual('float32');
    expectArraysClose(result, [0, 5, 0, 6]);
  });

  it('should throw error when indices are not int32', () => {
    const indices = scalar(2, 'float32');
    const values = scalar(100, 'int32');
    const shape = [6];
    expect(() => sparseToDense(indices, values, shape, defaultValue)).toThrow();
  });

  it('should throw error when indices rank > 2', () => {
    const indices = tensor3d([1], [1, 1, 1], 'int32');
    const values = tensor1d([100], 'float32');
    const shape = [6];
    expect(() => sparseToDense(indices, values, shape, defaultValue)).toThrow();
  });

  it('should throw error when values has rank > 1', () => {
    const indices = tensor1d([0, 4, 2], 'int32');
    const values = tensor2d([1.0, 2.0, 3.0], [3, 1], 'float32');
    const shape = [6];
    expect(() => sparseToDense(indices, values, shape, defaultValue)).toThrow();
  });

  it('should throw error when values has wrong size', () => {
    const indices = tensor1d([0, 4, 2], 'int32');
    const values = tensor1d([1.0, 2.0, 3.0, 4.0], 'float32');
    const shape = [6];
    expect(() => sparseToDense(indices, values, shape, defaultValue)).toThrow();
  });
});

describeWithFlags('sparseToDense CPU', CPU_ENVS, () => {
  it('should throw error when index out of range', () => {
    const indices = tensor1d([0, 2, 6], 'int32');
    const values = tensor1d([100, 101, 102], 'int32');
    const shape = [6];
    expect(() => sparseToDense(indices, values, shape, defaultValue)).toThrow();
  });
});
