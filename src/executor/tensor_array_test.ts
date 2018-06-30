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

import {memory, Tensor, tensor2d, tensor3d} from '@tensorflow/tfjs-core';
import {test_util} from '@tensorflow/tfjs-core';

import {TensorArray} from './tensor_array';

let tensorArray: TensorArray;
let tensor: Tensor;
let tensor2: Tensor;
const NAME = 'TA1';
const DTYPE = 'int32';
const SIZE = 10;
const SHAPE = [1, 1];
const IDENTICAL_SHAPE = true;
const DYNAMIC_SIZE = false;
const CLEAR_AFTER_READ = false;

describe('TensorArray', () => {
  beforeEach(() => {
    tensorArray = new TensorArray(
        NAME, DTYPE, SIZE, SHAPE, IDENTICAL_SHAPE, DYNAMIC_SIZE,
        CLEAR_AFTER_READ);
    tensor = tensor2d([1], [1, 1], 'int32');
    tensor2 = tensor2d([2], [1, 1], 'int32');
  });
  afterEach(() => tensorArray.clearAndClose());

  it('should initialize', () => {
    expect(tensorArray.size()).toEqual(0);
    expect(tensorArray.name).toEqual('TA1');
    expect(tensorArray.dynamicSize).toBeFalsy();
    expect(tensorArray.closed).toBeFalsy();
  });

  it('should close', () => {
    const numOfTensors = memory().numTensors;
    const size = tensorArray.size();
    tensorArray.clearAndClose();
    expect(tensorArray.size()).toBe(0);
    expect(tensorArray.closed).toBeTruthy();
    expect(memory().numTensors).toEqual(numOfTensors - size);
  });

  describe('write', () => {
    it('should add new tensor', () => {
      tensorArray.write(0, tensor);
      expect(tensorArray.size()).toBe(1);
    });
    it('should add multiple tensors', () => {
      tensorArray.writeMany([0, 1], [tensor, tensor2]);
      expect(tensorArray.size()).toBe(2);
    });
    it('should fail if dtype does not match', () => {
      const tensor = tensor2d([1], [1, 1], 'float32');
      expect(() => tensorArray.write(0, tensor)).toThrow();
    });
    it('should fail if shape does not match', () => {
      const tensor = tensor2d([1, 2], [2, 1], 'int32');
      expect(() => tensorArray.write(0, tensor)).toThrow();
    });
    it('should fail if the index has already been written', () => {
      tensorArray.write(0, tensor);
      expect(() => tensorArray.write(0, tensor)).toThrow();
    });
    it('should fail if the index greater than array size', () => {
      expect(() => tensorArray.write(11, tensor)).toThrow();
    });
    it('should fail if the array is closed', () => {
      tensorArray.clearAndClose();
      expect(() => tensorArray.write(0, tensor)).toThrow();
    });
    it('should create no new tensors', () => {
      const numTensors = memory().numTensors;
      tensorArray.write(0, tensor);
      expect(memory().numTensors).toEqual(numTensors);
    });
  });

  describe('read', () => {
    beforeEach(() => {
      tensorArray.writeMany([0, 1], [tensor, tensor2]);
    });

    it('should read the correct index', () => {
      expect(tensorArray.read(0)).toBe(tensor);
      expect(tensorArray.read(1)).toBe(tensor2);
    });
    it('should read the multiple indices', () => {
      expect(tensorArray.readMany([0, 1])).toEqual([tensor, tensor2]);
    });
    it('should failed if index is out of bound', () => {
      expect(() => tensorArray.read(3)).toThrow();
      expect(() => tensorArray.read(-1)).toThrow();
    });
    it('should failed if array is closed', () => {
      tensorArray.clearAndClose();
      expect(() => tensorArray.read(0)).toThrow();
    });
    it('should create no new tensors', () => {
      const numTensors = memory().numTensors;
      tensorArray.read(0);
      tensorArray.read(1);
      expect(memory().numTensors).toEqual(numTensors);
    });
  });

  describe('gather', () => {
    beforeEach(() => {
      tensorArray.writeMany([0, 1], [tensor, tensor2]);
    });

    it('should return default packed tensors', () => {
      const gathered = tensorArray.gather();
      expect(gathered.shape).toEqual([2, 1, 1]);
      test_util.expectArraysClose(gathered, [1, 2]);
    });

    it('should return packed tensors when indices is specified', () => {
      const gathered = tensorArray.gather([1, 0]);
      expect(gathered.shape).toEqual([2, 1, 1]);
      test_util.expectArraysClose(gathered, [2, 1]);
    });
    it('should fail if dtype is not matched', () => {
      expect(() => tensorArray.gather([0, 1], 'float32')).toThrow();
    });
    it('should create one new tensor', () => {
      const numTensors: number = memory().numTensors;
      tensorArray.gather();
      expect(memory().numTensors).toEqual(numTensors + 1);
    });
  });

  describe('concat', () => {
    beforeEach(() => {
      tensorArray.writeMany([0, 1], [tensor, tensor2]);
    });

    it('should return default concat tensors', () => {
      const concat = tensorArray.concat();
      expect(concat.shape).toEqual([2, 1]);
      test_util.expectArraysClose(concat, [1, 2]);
    });

    it('should fail if dtype is not matched', () => {
      expect(() => tensorArray.concat('float32')).toThrow();
    });

    it('should create one new tensor', () => {
      const numTensors: number = memory().numTensors;
      tensorArray.concat();
      expect(memory().numTensors).toEqual(numTensors + 1);
    });
  });

  describe('scatter', () => {
    it('should scatter the input tensor', () => {
      const input = tensor3d([1, 2, 3], [3, 1, 1], 'int32');
      tensorArray.scatter([0, 1, 2], input);
      expect(tensorArray.size()).toEqual(3);
    });

    it('should fail if indices and tensor shapes do not matched', () => {
      const input = tensor3d([1, 2, 3], [3, 1, 1], 'int32');
      expect(() => tensorArray.scatter([1, 2], input)).toThrow();
    });

    it('should fail if max index > array max size', () => {
      const input = tensor3d([1, 2, 3], [3, 1, 1], 'int32');
      expect(() => tensorArray.scatter([1, 2, 11], input)).toThrow();
    });

    it('should fail if dtype is not matched', () => {
      const input = tensor3d([1, 2, 3], [3, 1, 1], 'float32');
      expect(() => tensorArray.scatter([0, 1, 2], input)).toThrow();
    });
    it('should create three new tensors', () => {
      const input = tensor3d([1, 2, 3], [3, 1, 1], 'int32');
      const numTensors: number = memory().numTensors;
      tensorArray.scatter([0, 1, 2], input);
      expect(memory().numTensors).toEqual(numTensors + 3);
    });
  });

  describe('split', () => {
    beforeEach(() => {
      tensorArray = new TensorArray(
          NAME, DTYPE, 3, [2, 2], IDENTICAL_SHAPE, DYNAMIC_SIZE,
          CLEAR_AFTER_READ);
    });

    it('should split the input tensor', () => {
      const input =
          tensor3d([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [3, 2, 2], 'int32');
      tensorArray.split([1, 1, 1], input);
      expect(tensorArray.size()).toEqual(3);
    });

    it('should fail if indices and tensor shapes do not matched', () => {
      const input = tensor3d([1, 2, 3], [3, 1, 1], 'int32');
      expect(() => tensorArray.split([1, 1], input)).toThrow();
    });

    it('should fail if length > array max size', () => {
      const input = tensor3d([1, 2, 3], [3, 1, 1], 'int32');
      expect(() => tensorArray.split([1, 1, 1, 1], input)).toThrow();
    });

    it('should fail if dtype is not matched', () => {
      const input = tensor3d([1, 2, 3], [3, 1, 1], 'float32');
      expect(() => tensorArray.split([1, 1, 1], input)).toThrow();
    });

    it('should create three new tensors', () => {
      const input =
          tensor3d([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [3, 2, 2], 'int32');
      const numTensors: number = memory().numTensors;
      tensorArray.split([1, 1, 1], input);
      expect(memory().numTensors).toEqual(numTensors + 3);
    });
  });
});
