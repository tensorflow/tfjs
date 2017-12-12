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

import * as test_util from '../test_util';
import {MPRandGauss} from './rand';

function isFloat(n: number): boolean {
  return Number(n) === n && n % 1 !== 0;
}

test_util.describeCustom('MPRandGauss', () => {
  const EPSILON = 0.05;
  const SEED = 2002;

  it('should default to float32 numbers', () => {
    const rand = new MPRandGauss(0, 1.5);
    expect(isFloat(rand.nextValue())).toBe(true);
  });

  it('should handle create a mean/stdv of float32 numbers', () => {
    const rand =
        new MPRandGauss(0, 1.5, 'float32', false /* truncated */, SEED);
    const values = [];
    const size = 10000;
    for (let i = 0; i < size; i++) {
      values.push(rand.nextValue());
    }
    test_util.expectArrayInMeanStdRange(values, 0, 1.5, EPSILON);
    test_util.jarqueBeraNormalityTest(values);
  });

  it('should handle int32 numbers', () => {
    const rand = new MPRandGauss(0, 1, 'int32');
    expect(isFloat(rand.nextValue())).toBe(false);
  });

  it('should handle create a mean/stdv of int32 numbers', () => {
    const rand = new MPRandGauss(0, 2, 'int32', false /* truncated */, SEED);
    const values = [];
    const size = 10000;
    for (let i = 0; i < size; i++) {
      values.push(rand.nextValue());
    }
    test_util.expectArrayInMeanStdRange(values, 0, 2, EPSILON);
    test_util.jarqueBeraNormalityTest(values);
  });

  it('Should not have a more than 2x std-d from mean for truncated values',
     () => {
       const stdv = 1.5;
       const rand = new MPRandGauss(0, stdv, 'float32', true /* truncated */);
       for (let i = 0; i < 1000; i++) {
         expect(Math.abs(rand.nextValue())).toBeLessThan(stdv * 2);
       }
     });
});
