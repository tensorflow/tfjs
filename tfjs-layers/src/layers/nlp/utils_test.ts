/**
 * @license
 * Copyright 2023 Google LLC.
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

import { ones, tensor, test_util, zeros } from '@tensorflow/tfjs-core';
import { sliceUpdate, tensorArrTo2DArr, tensorToArr } from './utils';
import { expectTensorsClose } from '../../utils/test_utils';

describe('tensor to array functions', () => {
  it('tensorToArr', () => {
    const inputStr = tensor(['these', 'are', 'strings', '.']);
    const inputNum = tensor([2, 11, 15]);

    test_util.expectArraysEqual(
      tensorToArr(inputStr) as string[], ['these', 'are', 'strings', '.']);
    test_util.expectArraysEqual(tensorToArr(inputNum) as number[], [2, 11, 15]);
  });

  it('tensorArrTo2DArr', () => {
    const inputStr = [tensor(['these', 'are']), tensor(['strings', '.'])];
    const inputNum = [tensor([2, 11]), tensor([15])];

    test_util.expectArraysEqual(
      tensorArrTo2DArr(inputStr) as string[][],
      [['these', 'are'], ['strings', '.']]
    );
    test_util.expectArraysEqual(
      tensorArrTo2DArr(inputNum) as number[][], [[2, 11], [15]]);
  });
});

describe('sliceUpdate', () => {
  it('1D', () => {
    const inputs = tensor([1, 2, 3, 4, 5]);
    const startIndices = [2];
    const updates = tensor([-1, -2]);
    const expected = tensor([1, 2, -1, -2, 5]);

    const result = sliceUpdate(inputs, startIndices, updates);

    expectTensorsClose(result, expected);
  });

  it('2D', () => {
    const inputs = zeros([2, 5]);
    const startIndices = [0, 1];
    const updates = tensor([
      [-1, -2],
      [-4, -3],
    ]);
    const expected = tensor([
      [0, -1, -2, 0, 0],
      [0, -4, -3, 0, 0]
    ]);
    const result = sliceUpdate(inputs, startIndices, updates);

    expectTensorsClose(result, expected);
  });

  it('3D', () => {
    const inputs = zeros([2, 3, 4]);
    const startIndices = [0, 0, 0];
    const updates = ones([2, 1, 4]);
    const expected = tensor([
      [[1, 1, 1, 1],
       [0, 0, 0, 0],
       [0, 0, 0, 0]],
      [[1, 1, 1, 1],
       [0, 0, 0, 0],
       [0, 0, 0, 0]],
    ]);
    const result = sliceUpdate(inputs, startIndices, updates);

    expectTensorsClose(result, expected);
  });
});
