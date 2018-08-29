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
import * as complex_util from './complex_util';

describe('complex_util', () => {
  it('mergeRealAndImagArrays', () => {
    const real = new Float32Array([1, 2, 3]);
    const imag = new Float32Array([4, 5, 6]);
    const complex = complex_util.mergeRealAndImagArrays(real, imag);
    expect(complex).toEqual(new Float32Array([1, 4, 2, 5, 3, 6]));
  });

  it('splitRealAndImagArrays', () => {
    const complex = new Float32Array([1, 4, 2, 5, 3, 6]);
    const result = complex_util.splitRealAndImagArrays(complex);
    expect(result.real).toEqual(new Float32Array([1, 2, 3]));
    expect(result.imag).toEqual(new Float32Array([4, 5, 6]));
  });
});
