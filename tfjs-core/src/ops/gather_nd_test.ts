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

import {ALL_ENVS, describeWithFlags} from '../jasmine_util';
import {expectArraysClose} from '../test_util';

import {gatherND} from './gather_nd';
import {scalar, tensor1d, tensor2d, tensor3d} from './tensor_ops';

describeWithFlags('gatherND', ALL_ENVS, () => {
  it('should work for simple slice', async () => {
    const indices = tensor2d([0, 4, 8], [3, 1], 'int32');
    const input =
        tensor1d([100, 101, 102, 777, 778, 779, 1000, 1001, 1002], 'int32');
    const shape = [3];
    const result = gatherND(input, indices);
    expect(result.shape).toEqual(shape);
    expect(result.dtype).toEqual(input.dtype);
    expectArraysClose(await result.data(), [100, 778, 1002]);
  });

  it('should work for indexing 2d', async () => {
    const indices = tensor2d([0, 2], [2, 1], 'int32');
    const input = tensor2d(
        [
          5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8,
          5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8
        ],
        [8, 4], 'float32');
    const shape = [2, 4];
    const result = gatherND(input, indices);
    expect(result.shape).toEqual(shape);
    expect(result.dtype).toEqual(input.dtype);
    expectArraysClose(await result.data(), [5, 5, 5, 5, 7, 7, 7, 7]);
  });

  it('should work for indexing 3d', async () => {
    const indices = tensor2d([0, 2, 1, 1], [2, 2], 'int32');
    const input = tensor3d(
        [
          5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8,
          5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8
        ],
        [2, 4, 4], 'float32');
    const shape = [2, 4];
    const result = gatherND(input, indices);
    expect(result.shape).toEqual(shape);
    expect(result.dtype).toEqual(input.dtype);
    expectArraysClose(await result.data(), [7, 7, 7, 7, 6, 6, 6, 6]);
  });

  it('should work for batch slice', async () => {
    const indices = tensor3d([0, 4, 2], [3, 1, 1], 'int32');
    const input =
        tensor1d([100, 101, 102, 777, 778, 779, 10000, 10001, 10002], 'int32');
    const shape = [3, 1];
    const result = gatherND(input, indices);
    expect(result.shape).toEqual(shape);
    expect(result.dtype).toEqual(input.dtype);
    expectArraysClose(await result.data(), [100, 778, 102]);
  });

  it('should work for batch indexing 2d', async () => {
    const indices = tensor3d([0, 2], [2, 1, 1], 'int32');
    const input = tensor2d(
        [
          5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8,
          5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8
        ],
        [8, 4], 'float32');
    const shape = [2, 1, 4];
    const result = gatherND(input, indices);
    expect(result.shape).toEqual(shape);
    expect(result.dtype).toEqual(input.dtype);
    expectArraysClose(await result.data(), [5, 5, 5, 5, 7, 7, 7, 7]);
  });

  it('should work for batch indexing 3d', async () => {
    const indices = tensor3d([0, 2, 1, 1], [2, 1, 2], 'int32');
    const input = tensor3d(
        [
          5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8,
          5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8
        ],
        [2, 4, 4], 'float32');
    const shape = [2, 1, 4];
    const result = gatherND(input, indices);
    expect(result.shape).toEqual(shape);
    expect(result.dtype).toEqual(input.dtype);
    expectArraysClose(await result.data(), [7, 7, 7, 7, 6, 6, 6, 6]);
  });

  it('should work for TensorLike inputs', async () => {
    const indices = [[0], [4], [8]];
    const input = [100, 101, 102, 777, 778, 779, 1000, 1001, 1002];
    const shape = [3];
    const result = gatherND(input, indices);
    expect(result.shape).toEqual(shape);
    expect(result.dtype).toEqual('float32');
    expectArraysClose(await result.data(), [100, 778, 1002]);
  });

  it('should throw error when indices are not int32', () => {
    const indices = tensor1d([1], 'float32');
    const input = tensor2d(
        [100, 101, 102, 103, 777, 778, 779, 780, 10000, 10001, 10002, 10004],
        [3, 4], 'float32');
    expect(() => gatherND(input, indices)).toThrow();
  });
  it('should throw error when indices are scalar', () => {
    const indices = scalar(1, 'int32');
    const input = tensor2d(
        [100, 101, 102, 103, 777, 778, 779, 780, 10000, 10001, 10002, 10004],
        [3, 4], 'float32');
    expect(() => gatherND(input, indices)).toThrow();
  });
  it('should throw error when x is scalar', () => {
    const indices = tensor2d([0, 4, 2], [3, 1], 'int32');
    const input = scalar(1.0, 'float32');
    expect(() => gatherND(input, indices)).toThrow();
  });
  it('should throw error when indices inner dim > x shape length', () => {
    const indices = tensor2d([0, 4, 2], [1, 3], 'int32');
    const input =
        tensor2d([100, 101, 102, 10000, 10001, 10002], [3, 2], 'float32');
    expect(() => gatherND(input, indices)).toThrow();
  });
});
