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
import {TensorInfo} from '../kernel_registry';
import {Tensor} from '../tensor';
import {assert} from '../util';

/**
 * Prepare the split size array. When the input is a number, the axis is evenly
 * divided among the split size. When the input contains the negative value, the
 * rest of the axis is allocated toward that.
 */
export function prepareSplitSize(
    x: Tensor|TensorInfo, numOrSizeSplits: number[]|number,
    axis = 0): number[] {
  let splitSizes = [];
  if (typeof (numOrSizeSplits) === 'number') {
    assert(
        x.shape[axis] % numOrSizeSplits === 0,
        () => 'Number of splits must evenly divide the axis.');
    splitSizes =
        new Array(numOrSizeSplits).fill(x.shape[axis] / numOrSizeSplits);
  } else {
    const numOfNegs = numOrSizeSplits.reduce((count, value) => {
      if (value === -1) {
        count += 1;
      }
      return count;
    }, 0);
    assert(
        numOfNegs <= 1,
        () => 'There should be only one negative value in split array.');
    const negIndex = numOrSizeSplits.indexOf(-1);
    // Allow the number of split array to be -1, which indicates the rest
    // of dimension is allocated to that split.
    if (negIndex !== -1) {
      const total = numOrSizeSplits.reduce((a, b) => b > 0 ? a + b : a);
      numOrSizeSplits[negIndex] = x.shape[axis] - total;
    }
    assert(
        x.shape[axis] === numOrSizeSplits.reduce((a, b) => a + b),
        () => 'The sum of sizes must match the size of the axis dimension.');
    splitSizes = numOrSizeSplits;
  }

  return splitSizes;
}
