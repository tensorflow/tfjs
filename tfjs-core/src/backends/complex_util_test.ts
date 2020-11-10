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
import {ALL_ENVS, describeWithFlags} from '../jasmine_util';
import {expectArraysClose} from '../test_util';

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

  it('complexWithEvenIndex', () => {
    const complex = new Float32Array([1, 2, 3, 4, 5, 6]);
    const result = complex_util.complexWithEvenIndex(complex);
    expect(result.real).toEqual(new Float32Array([1, 5]));
    expect(result.imag).toEqual(new Float32Array([2, 6]));
  });

  it('complexWithOddIndex', () => {
    const complex = new Float32Array([1, 2, 3, 4, 5, 6]);
    const result = complex_util.complexWithOddIndex(complex);
    expect(result.real).toEqual(new Float32Array([3]));
    expect(result.imag).toEqual(new Float32Array([4]));
  });
});

describeWithFlags('complex_util exponents', ALL_ENVS, () => {
  it('exponents inverse=false', () => {
    const inverse = false;
    const result = complex_util.exponents(5, inverse);
    expectArraysClose(result.real, new Float32Array([1, 0.30901700258255005]));
    expectArraysClose(result.imag, new Float32Array([0, -0.9510565400123596]));
  });
  it('exponents inverse=true', () => {
    const inverse = true;
    const result = complex_util.exponents(5, inverse);
    expectArraysClose(result.real, new Float32Array([1, 0.30901700258255005]));
    expectArraysClose(result.imag, new Float32Array([0, 0.9510565400123596]));
  });
});

describeWithFlags('complex_util assignment', ALL_ENVS, () => {
  it('assign complex value in TypedArray', () => {
    const t = new Float32Array(4);

    complex_util.assignToTypedArray(t, 1, 2, 0);
    complex_util.assignToTypedArray(t, 3, 4, 1);

    expectArraysClose(t, new Float32Array([1, 2, 3, 4]));
  });
});
