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

import {computeDispatch} from './webgpu_util';

describe('webgpu util', () => {
  it('computeDispatch returns dispatch dimensions based on layout of ' +
         'output dimensions and workgroupSize.',
     () => {
       const layout = {x: [0], y: [1], z: [2, 3]};
       const outputShape = [1, 2, 3, 2];

       const workgroupSize = [2, 2, 1] as [number, number, number];

       const dispatch = computeDispatch(layout, outputShape, workgroupSize);
       expect(dispatch).toEqual([1, 1, 6]);
     });

  it('computeDispatch returns dispatch dimensions based on layout of ' +
         'output dimensions, workgroupSize, and elementsPerThread.',
     () => {
       const layout = {x: [0], y: [1], z: [2, 3]};
       const outputShape = [4, 8, 12, 2];

       const workgroupSize = [2, 1, 1] as [number, number, number];
       const elementsPerThread = [2, 2, 3] as [number, number, number];

       const dispatch = computeDispatch(
           layout, outputShape, workgroupSize, elementsPerThread);
       expect(dispatch).toEqual([1, 4, 8]);
     });
});
