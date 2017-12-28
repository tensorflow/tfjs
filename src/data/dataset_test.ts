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

import {Array1D, Array2D, NDArray} from '../math/ndarray';
import * as test_util from '../test_util';

import {InMemoryDataset} from './dataset';

class StubDataset extends InMemoryDataset {
  constructor(data: NDArray[][]) {
    super(data.map(value => value[0].shape));
    this.dataset = data;
  }

  fetchData(): Promise<void> {
    return new Promise<void>((resolve, reject) => {});
  }
}

describe('Dataset', () => {
  it('normalize', () => {
    const data = [
      [
        Array2D.new([2, 3], [1, 2, 10, -1, -2, .75]),
        Array2D.new([2, 3], [2, 3, 20, -2, 2, .5]),
        Array2D.new([2, 3], [3, 4, 30, -3, -4, 0]),
        Array2D.new([2, 3], [4, 5, 40, -4, 4, 1])
      ],
      [
        Array1D.randNormal([1]), Array1D.randNormal([1]),
        Array1D.randNormal([1]), Array1D.randNormal([1])
      ]
    ];
    const dataset = new StubDataset(data);

    // Normalize only the first data index.
    const dataIndex = 0;
    dataset.normalizeWithinBounds(dataIndex, 0, 1);

    let normalizedInputs = dataset.getData()[0];

    test_util.expectArraysClose(normalizedInputs[0], [0, 0, 0, 1, .25, .75]);
    test_util.expectArraysClose(
        normalizedInputs[1], [1 / 3, 1 / 3, 1 / 3, 2 / 3, .75, .5]);
    test_util.expectArraysClose(
        normalizedInputs[2], [2 / 3, 2 / 3, 2 / 3, 1 / 3, 0, 0]);
    test_util.expectArraysClose(normalizedInputs[3], [1, 1, 1, 0, 1, 1]);

    dataset.normalizeWithinBounds(dataIndex, -1, 1);

    normalizedInputs = dataset.getData()[0];

    test_util.expectArraysClose(normalizedInputs[0], [-1, -1, -1, 1, -.5, .5]);
    test_util.expectArraysClose(
        normalizedInputs[1], [-1 / 3, -1 / 3, -1 / 3, 1 / 3, .5, .0]);
    test_util.expectArraysClose(
        normalizedInputs[2], [1 / 3, 1 / 3, 1 / 3, -1 / 3, -1, -1]);
    test_util.expectArraysClose(normalizedInputs[3], [1, 1, 1, -1, 1, 1]);

    dataset.removeNormalization(dataIndex);

    normalizedInputs = dataset.getData()[0];

    test_util.expectArraysClose(normalizedInputs[0], [1, 2, 10, -1, -2, .75]);
    test_util.expectArraysClose(normalizedInputs[1], [2, 3, 20, -2, 2, .5]);
    test_util.expectArraysClose(normalizedInputs[2], [3, 4, 30, -3, -4, 0]);
    test_util.expectArraysClose(normalizedInputs[3], [4, 5, 40, -4, 4, 1]);
  });
});
