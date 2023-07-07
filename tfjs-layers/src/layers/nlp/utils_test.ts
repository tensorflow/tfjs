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

import { tensor, test_util } from '@tensorflow/tfjs-core';
import { tensorArrTo2DArr, tensorToArr } from './utils';

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
