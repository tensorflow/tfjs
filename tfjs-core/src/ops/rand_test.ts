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

import {expectValuesInRange} from '../test_util';
import {MPRandGauss, RandGamma, UniformRandom} from './rand';
import {expectArrayInMeanStdRange, jarqueBeraNormalityTest} from './rand_util';

function isFloat(n: number): boolean {
  return Number(n) === n && n % 1 !== 0;
}

describe('MPRandGauss', () => {
  const EPSILON = 0.05;
  const SEED = 2002;

  it('should default to float32 numbers', () => {
    const rand = new MPRandGauss(0, 1.5);
    expect(isFloat(rand.nextValue())).toBe(true);
  });

  it('should handle a mean/stdv of float32 numbers', () => {
    const rand =
        new MPRandGauss(0, 1.5, 'float32', false /* truncated */, SEED);
    const values = [];
    const size = 10000;
    for (let i = 0; i < size; i++) {
      values.push(rand.nextValue());
    }
    expectArrayInMeanStdRange(values, 0, 1.5, EPSILON);
    jarqueBeraNormalityTest(values);
  });

  it('should handle int32 numbers', () => {
    const rand = new MPRandGauss(0, 1, 'int32');
    expect(isFloat(rand.nextValue())).toBe(false);
  });

  it('should handle a mean/stdv of int32 numbers', () => {
    const rand = new MPRandGauss(0, 2, 'int32', false /* truncated */, SEED);
    const values = [];
    const size = 10000;
    for (let i = 0; i < size; i++) {
      values.push(rand.nextValue());
    }
    expectArrayInMeanStdRange(values, 0, 2, EPSILON);
    jarqueBeraNormalityTest(values);
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

describe('RandGamma', () => {
  const SEED = 2002;

  it('should default to float32 numbers', () => {
    const rand = new RandGamma(2, 2, 'float32');
    expect(isFloat(rand.nextValue())).toBe(true);
  });

  it('should handle an alpha/beta of float32 numbers', () => {
    const rand = new RandGamma(2, 2, 'float32', SEED);
    const values = [];
    const size = 10000;
    for (let i = 0; i < size; i++) {
      values.push(rand.nextValue());
    }
    expectValuesInRange(values, 0, 30);
  });

  it('should handle int32 numbers', () => {
    const rand = new RandGamma(2, 2, 'int32');
    expect(isFloat(rand.nextValue())).toBe(false);
  });

  it('should handle an alpha/beta of int32 numbers', () => {
    const rand = new RandGamma(2, 2, 'int32', SEED);
    const values = [];
    const size = 10000;
    for (let i = 0; i < size; i++) {
      values.push(rand.nextValue());
    }
    expectValuesInRange(values, 0, 30);
  });
});

describe('UniformRandom', () => {
  it('float32, no seed', () => {
    const min = 0.2;
    const max = 0.24;
    const dtype = 'float32';
    const xs: number[] = [];
    for (let i = 0; i < 10; ++i) {
      const rand = new UniformRandom(min, max, dtype);
      const x = rand.nextValue();
      xs.push(x);
    }
    expect(Math.min(...xs)).toBeGreaterThanOrEqual(min);
    expect(Math.max(...xs)).toBeLessThan(max);
  });

  it('int32, no seed', () => {
    const min = 13;
    const max = 37;
    const dtype = 'int32';
    const xs: number[] = [];
    for (let i = 0; i < 10; ++i) {
      const rand = new UniformRandom(min, max, dtype);
      const x = rand.nextValue();
      expect(Number.isInteger(x)).toEqual(true);
      xs.push(x);
    }
    expect(Math.min(...xs)).toBeGreaterThanOrEqual(min);
    expect(Math.max(...xs)).toBeLessThanOrEqual(max);
  });

  it('seed is number', () => {
    const min = -1.2;
    const max = -0.4;
    const dtype = 'float32';
    const seed = 1337;
    const xs: number[] = [];
    for (let i = 0; i < 10; ++i) {
      const rand = new UniformRandom(min, max, dtype, seed);
      const x = rand.nextValue();
      expect(x).toBeGreaterThanOrEqual(min);
      expect(x).toBeLessThan(max);
      xs.push(x);
    }
    // Assert deterministic results.
    expect(Math.min(...xs)).toEqual(Math.max(...xs));
  });

  it('seed === null', () => {
    const min = 0;
    const max = 1;
    const dtype = 'float32';
    const seed: number = null;
    const rand = new UniformRandom(min, max, dtype, seed);
    const x = rand.nextValue();
    expect(x).toBeGreaterThanOrEqual(0);
    expect(x).toBeLessThan(1);
  });

  it('seed === undefined', () => {
    const min = 0;
    const max = 1;
    const dtype = 'float32';
    const seed: number = undefined;
    const rand = new UniformRandom(min, max, dtype, seed);
    const x = rand.nextValue();
    expect(x).toBeGreaterThanOrEqual(0);
    expect(x).toBeLessThan(1);
  });
});
