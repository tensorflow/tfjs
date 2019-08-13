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

import {symbolicallyComputeStrides} from './shader_util';

describe('shader util', () => {
  it('symbolicallyComputeStrides takes in array of dimensions ' +
         'and returns GLSL to compute strides for those dimensions',
     () => {
       const layout = [0, 2, 1];
       const strides = symbolicallyComputeStrides(layout, 'output');
       expect(strides[0]).toEqual('(output[1] * output[2])');
       expect(strides[1]).toEqual('output[1]');
     });

  it('symbolicallyComputeStrides throws if given a dimension ' +
         'that cannot be accessed from a GLSL data type',
     () => {
       const layout = [0, 5, 2];
       expect(() => symbolicallyComputeStrides(layout, 'output'))
           .toThrowError();
     });
});