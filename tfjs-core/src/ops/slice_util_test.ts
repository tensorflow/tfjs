/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
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

import {isSliceContinous} from './slice_util';

describeWithFlags('isSliceContinous', ALL_ENVS, () => {
  it('[] => []', () => {
    const shape: number[] = [];
    const size: number[] = [];
    const begin: number[] = [];
    expect(isSliceContinous(shape, begin, size)).toBeTruthy();
  });

  it('[5] sliced to [3]', () => {
    const shape = [5];
    const size = [3];
    const begin = [1];
    expect(isSliceContinous(shape, begin, size)).toBeTruthy();
  });

  it('[5, 3] sliced to [2, 3] skipping a row', () => {
    const shape = [5, 3];
    const size = [2, 3];
    const begin = [1, 0];
    expect(isSliceContinous(shape, begin, size)).toBeTruthy();
  });

  it('[5, 3] sliced to [5, 2] skipping a column', () => {
    const shape = [5, 3];
    const size = [5, 2];
    const begin = [0, 1];
    expect(isSliceContinous(shape, begin, size)).toBeFalsy();
  });

  it('[5, 3] sliced to [1, 2] skipping a row and column', () => {
    const shape = [5, 3];
    const size = [1, 2];
    const begin = [2, 1];
    expect(isSliceContinous(shape, begin, size)).toBeTruthy();
  });

  it('[1, 5, 3] sliced to [1, 2, 3], skipping middle axis', () => {
    const shape = [1, 5, 3];
    const size = [1, 2, 3];
    const begin = [0, 2, 0];
    expect(isSliceContinous(shape, begin, size)).toBeTruthy();
  });

  it('[2, 5, 3] sliced to [2, 2, 3], skipping middle axis', () => {
    const shape = [2, 5, 3];
    const size = [2, 2, 3];
    const begin = [0, 2, 0];
    expect(isSliceContinous(shape, begin, size)).toBeFalsy();
  });
});
