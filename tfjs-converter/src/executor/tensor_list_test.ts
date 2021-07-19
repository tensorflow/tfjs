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

import {memory, Tensor, tensor2d, tensor3d, test_util} from '@tensorflow/tfjs-core';

import {reserve, scatter, split, TensorList} from './tensor_list';

let tensorList: TensorList;
let tensor: Tensor;
let tensor2: Tensor;
const DTYPE = 'int32';
const SIZE = 10;
const SHAPE = [1, 1];

describe('TensorList', () => {
  beforeEach(() => {
    tensorList = new TensorList([], SHAPE, DTYPE, SIZE);
    tensor = tensor2d([1], [1, 1], 'int32');
    tensor2 = tensor2d([2], [1, 1], 'int32');
  });

  it('should initialize', () => {
    expect(tensorList.size()).toEqual(0);
    expect(tensorList.elementDtype).toEqual(DTYPE);
    expect(tensorList.maxNumElements).toEqual(SIZE);
    expect(tensorList.elementShape).toEqual(SHAPE);
  });

  it('should allow scalar shape', () => {
    tensorList = new TensorList([], -1, DTYPE, SIZE);
    expect(tensorList.size()).toEqual(0);
    expect(tensorList.elementDtype).toEqual(DTYPE);
    expect(tensorList.maxNumElements).toEqual(SIZE);
    expect(tensorList.elementShape).toEqual(-1);
  });

  it('should not dispose keep tensors when close', () => {
    const numOfTensors = memory().numTensors;
    tensorList.pushBack(tensor);
    tensorList.pushBack(tensor2);
    const size = tensorList.size();
    const keepIds = new Set([tensor.id]);
    tensorList.clearAndClose(keepIds);
    expect(tensorList.size()).toBe(0);
    expect(tensor.isDisposed).toBeFalsy();
    expect(tensor2.isDisposed).toBeTruthy();
    // disposed the tensor in the array and idTensor of the array
    expect(memory().numTensors).toEqual(numOfTensors - size);
  });

  describe('pushBack', () => {
    it('should add new tensor', () => {
      tensorList.pushBack(tensor);
      expect(tensorList.size()).toBe(1);
    });
    it('should fail if dtype does not match', () => {
      const tensor = tensor2d([1], [1, 1], 'float32');
      expect(() => tensorList.pushBack(tensor)).toThrow();
    });
    it('should fail if shape does not match', () => {
      const tensor = tensor2d([1, 2], [2, 1], 'int32');
      expect(() => tensorList.pushBack(tensor)).toThrow();
    });
    it('should not fail for multiple push', () => {
      tensorList.pushBack(tensor);
      expect(() => tensorList.pushBack(tensor)).not.toThrow();
    });
    it('should fail if greater than array size', () => {
      tensorList.maxNumElements = 1;
      tensorList.pushBack(tensor);
      expect(() => tensorList.pushBack(tensor)).toThrow();
    });
    it('should not fail for wildcard shape', () => {
      tensorList = new TensorList([], [-1, 1], DTYPE, SIZE);
      const tensor = tensor2d([1], [1, 1], 'int32');
      tensorList.pushBack(tensor);
      expect(tensorList.size()).toBe(1);
    });
    it('should create no new tensors', () => {
      const numTensors = memory().numTensors;
      tensorList.pushBack(tensor);
      expect(memory().numTensors).toEqual(numTensors);
    });
  });

  describe('popBack', () => {
    it('should add new tensor', () => {
      tensorList.pushBack(tensor);
      tensorList.popBack(SHAPE, DTYPE);
      expect(tensorList.size()).toBe(0);
    });
    it('should fail if dtype does not match', () => {
      tensorList.pushBack(tensor);
      expect(() => tensorList.popBack(SHAPE, 'float32')).toThrow();
    });
    it('should fail if shape does not match', () => {
      expect(() => tensorList.popBack([2, 1], DTYPE)).toThrow();
    });
    it('should not fail for multiple push', () => {
      tensorList.pushBack(tensor);
      tensorList.pushBack(tensor2);
      tensorList.popBack(SHAPE, DTYPE);
      expect(() => tensorList.popBack(SHAPE, DTYPE)).not.toThrow();
    });
    it('should fail if greater than array size', () => {
      expect(() => tensorList.popBack(SHAPE, DTYPE)).toThrow();
    });
    it('should create no new tensors', () => {
      tensorList.pushBack(tensor);
      const numTensors = memory().numTensors;
      tensorList.popBack(SHAPE, DTYPE);
      // a new reshaped tensor
      expect(memory().numTensors).toEqual(numTensors + 1);
    });
    it('should not fail for wildcard shape', () => {
      tensorList = new TensorList([], [-1, 1], DTYPE, SIZE);
      const tensor = tensor2d([1], [1, 1], DTYPE);
      tensorList.pushBack(tensor);
      tensorList.popBack([-1, 1], DTYPE);
      expect(tensorList.size()).toBe(0);
    });
  });
  describe('setItem', () => {
    it('should add new tensor', () => {
      tensorList.setItem(0, tensor);
      expect(tensorList.size()).toBe(1);
    });
    it('should fail if dtype does not match', () => {
      const tensor = tensor2d([1], [1, 1], 'float32');
      expect(() => tensorList.setItem(0, tensor)).toThrow();
    });
    it('should fail if shape does not match', () => {
      const tensor = tensor2d([1, 2], [2, 1], 'int32');
      expect(() => tensorList.setItem(0, tensor)).toThrow();
    });
    it('should not fail if the index has already been written', () => {
      tensorList.setItem(0, tensor);
      expect(() => tensorList.setItem(0, tensor)).not.toThrow();
    });
    it('should fail if the index greater than array size', () => {
      expect(() => tensorList.setItem(11, tensor)).toThrow();
    });
    it('should not fail for wildcard shape', () => {
      tensorList = new TensorList([], [-1, 1], DTYPE, SIZE);
      tensorList.setItem(0, tensor);
      expect(tensorList.size()).toBe(1);
    });
    it('should create no new tensors', () => {
      const numTensors = memory().numTensors;
      tensorList.setItem(0, tensor);
      expect(memory().numTensors).toEqual(numTensors);
    });
  });

  describe('getItem', () => {
    beforeEach(() => {
      tensorList.setItem(0, tensor);
      tensorList.setItem(1, tensor2);
    });

    it('should read the correct index', async () => {
      test_util.expectArraysEqual(
          await tensorList.getItem(0, SHAPE, DTYPE).data(),
          await tensor.data());
      test_util.expectArraysEqual(
          await tensorList.getItem(1, SHAPE, DTYPE).data(),
          await tensor2.data());
    });

    it('should failed if index is out of bound', () => {
      expect(() => tensorList.getItem(3, SHAPE, DTYPE)).toThrow();
      expect(() => tensorList.getItem(-1, SHAPE, DTYPE)).toThrow();
    });
    it('should create no new tensors', () => {
      const numTensors = memory().numTensors;
      const tensor1 = tensorList.getItem(0, SHAPE, DTYPE);
      const tensor2 = tensorList.getItem(1, SHAPE, DTYPE);

      tensor1.dispose();
      tensor2.dispose();
      // 2 reshape tensors
      expect(memory().numTensors).toEqual(numTensors);
    });
    it('should not fail for wildcard shape', async () => {
      const tensor3 = tensorList.getItem(0, [-1, 1], DTYPE);
      test_util.expectArraysEqual(await tensor3.data(), await tensor.data());
    });
  });

  describe('reserve', () => {
    it('should create a tensor list', async () => {
      const tensorList = reserve([1, 1], 'float32', 10);
      expect(tensorList.maxNumElements).toEqual(10);
      expect(tensorList.elementDtype).toEqual('float32');
      expect(tensorList.elementShape).toEqual([1, 1]);
    });
    it('should not fail for wildcard shape', async () => {
      const tensorList = reserve([-1, 1], 'float32', 10);
      expect(tensorList.maxNumElements).toEqual(10);
      expect(tensorList.elementDtype).toEqual('float32');
      expect(tensorList.elementShape).toEqual([-1, 1]);
    });
  });

  describe('concat', () => {
    beforeEach(() => {
      tensorList.setItem(0, tensor);
      tensorList.setItem(1, tensor2);
    });

    it('should return default concat tensors', async () => {
      const concat = tensorList.concat(DTYPE, SHAPE);
      expect(concat.shape).toEqual([2, 1]);
      test_util.expectArraysClose(await concat.data(), [1, 2]);
    });
    it('should not fail for wildcard shape', async () => {
      const concat = tensorList.concat(DTYPE, [-1, 1]);
      expect(concat.shape).toEqual([2, 1]);
      test_util.expectArraysClose(await concat.data(), [1, 2]);
    });
    it('should fail if dtype is not matched', () => {
      expect(() => tensorList.concat('float32', SHAPE)).toThrow();
    });

    it('should create one new tensor', () => {
      const numTensors: number = memory().numTensors;
      tensorList.concat(DTYPE, SHAPE);
      expect(memory().numTensors).toEqual(numTensors + 1);
    });
  });

  describe('gather', () => {
    beforeEach(() => {
      tensorList.setItem(0, tensor);
      tensorList.setItem(1, tensor2);
    });

    it('should return packed tensors when indices is specified', async () => {
      const gathered = tensorList.gather([1, 0], DTYPE, SHAPE);
      expect(gathered.shape).toEqual([2, 1, 1]);
      test_util.expectArraysClose(await gathered.data(), [2, 1]);
    });
    it('should return when indices longer than available tensors', async () => {
      const gathered = tensorList.gather([1, 0, 2, 3], DTYPE, SHAPE);
      expect(gathered.shape).toEqual([2, 1, 1]);
      test_util.expectArraysClose(await gathered.data(), [2, 1]);
    });
    it('should fail if dtype is not matched', () => {
      expect(() => tensorList.gather([0, 1], 'float32', SHAPE)).toThrow();
    });
    it('should create one new tensor', () => {
      const numTensors: number = memory().numTensors;
      tensorList.gather([0, 1], DTYPE, SHAPE);
      expect(memory().numTensors).toEqual(numTensors + 1);
    });
    it('should not fail for wildcard shape', async () => {
      const numTensors: number = memory().numTensors;
      tensorList.gather([0, 1], DTYPE, [-1, 1]);
      expect(memory().numTensors).toEqual(numTensors + 1);
    });
  });

  describe('scatter', () => {
    it('should scatter the input tensor', () => {
      const input = tensor3d([1, 2, 3], [3, 1, 1], 'int32');
      const list = scatter(input, [0, 1, 2], [1, 1], 3);
      expect(list.size()).toEqual(3);
    });

    it('should fail if indices and tensor shapes do not matched', () => {
      const input = tensor3d([1, 2, 3], [3, 1, 1], 'int32');
      expect(() => scatter(input, [1, 2], [1, 1], 2)).toThrow();
    });

    it('should fail if max index > array max size', () => {
      const input = tensor3d([1, 2, 3], [3, 1, 1], 'int32');
      expect(() => scatter(input, [0, 1, 11], [1, 1], 3)).toThrow();
    });

    it('should create three new tensors', () => {
      const input = tensor3d([1, 2, 3], [3, 1, 1], 'int32');
      const numTensors: number = memory().numTensors;
      scatter(input, [0, 1, 2], [1, 1], 3);
      // Three tensors in the list and the idTensor
      expect(memory().numTensors).toEqual(numTensors + 3 + 1);
    });
    it('should not fail for wildcard shape', async () => {
      const input = tensor3d([1, 2, 3], [3, 1, 1], 'int32');
      const list = scatter(input, [0, 1, 2], [-1, 1], 3);
      expect(list.size()).toEqual(3);
    });
  });

  describe('split', () => {
    it('should split the input tensor', () => {
      const input =
          tensor3d([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [3, 2, 2], 'int32');
      const list = split(input, [1, 1, 1], [2, 2]);
      expect(list.size()).toEqual(3);
    });

    it('should fail if indices and tensor shapes do not matched', () => {
      const input = tensor3d([1, 2, 3], [3, 1, 1], 'int32');
      expect(() => split(input, [1, 1], [1, 1])).toThrow();
    });

    it('should create three new tensors', () => {
      const input =
          tensor3d([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [3, 2, 2], 'int32');
      const numTensors: number = memory().numTensors;
      split(input, [1, 1, 1], [2, 2]);
      // Three tensors in the list and the idTensor
      expect(memory().numTensors).toEqual(numTensors + 3 + 1);
    });
    it('should not fail for wildcard shape', async () => {
      const input =
          tensor3d([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [3, 2, 2], 'int32');
      const list = split(input, [1, 1, 1], [-1, 2]);
      expect(list.size()).toEqual(3);
    });
  });
});
