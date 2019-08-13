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

import * as seedrandom from 'seedrandom';

export interface RandomBase {
  nextValue(): number;
}

export interface RandomGamma {
  nextValue(): number;
}

export interface RandNormalDataTypes {
  float32: Float32Array;
  int32: Int32Array;
}

export interface RandGammaDataTypes {
  float32: Float32Array;
  int32: Int32Array;
}

// https://en.wikipedia.org/wiki/Marsaglia_polar_method
export class MPRandGauss implements RandomBase {
  private mean: number;
  private stdDev: number;
  private nextVal: number;
  private dtype?: keyof RandNormalDataTypes;
  private truncated?: boolean;
  private upper?: number;
  private lower?: number;
  private random: seedrandom.prng;

  constructor(
      mean: number, stdDeviation: number, dtype?: keyof RandNormalDataTypes,
      truncated?: boolean, seed?: number) {
    this.mean = mean;
    this.stdDev = stdDeviation;
    this.dtype = dtype;
    this.nextVal = NaN;
    this.truncated = truncated;
    if (this.truncated) {
      this.upper = this.mean + this.stdDev * 2;
      this.lower = this.mean - this.stdDev * 2;
    }
    const seedValue = seed ? seed : Math.random();
    this.random = seedrandom.alea(seedValue.toString());
  }

  /** Returns next sample from a Gaussian distribution. */
  public nextValue(): number {
    if (!isNaN(this.nextVal)) {
      const value = this.nextVal;
      this.nextVal = NaN;
      return value;
    }

    let resultX: number, resultY: number;
    let isValid = false;
    while (!isValid) {
      let v1: number, v2: number, s: number;
      do {
        v1 = 2 * this.random() - 1;
        v2 = 2 * this.random() - 1;
        s = v1 * v1 + v2 * v2;
      } while (s >= 1 || s === 0);

      const mul = Math.sqrt(-2.0 * Math.log(s) / s);
      resultX = this.mean + this.stdDev * v1 * mul;
      resultY = this.mean + this.stdDev * v2 * mul;

      if (!this.truncated || this.isValidTruncated(resultX)) {
        isValid = true;
      }
    }

    if (!this.truncated || this.isValidTruncated(resultY)) {
      this.nextVal = this.convertValue(resultY);
    }
    return this.convertValue(resultX);
  }

  /** Handles proper rounding for non-floating-point numbers. */
  private convertValue(value: number): number {
    if (this.dtype == null || this.dtype === 'float32') {
      return value;
    }
    return Math.round(value);
  }

  /** Returns true if less than 2-standard-deviations from the mean. */
  private isValidTruncated(value: number): boolean {
    return value <= this.upper && value >= this.lower;
  }
}

// Marsaglia, George, and Wai Wan Tsang. 2000. "A Simple Method for Generating
// Gamma Variables."
export class RandGamma implements RandomGamma {
  private alpha: number;
  private beta: number;
  private d: number;
  private c: number;
  private dtype?: keyof RandGammaDataTypes;
  private randu: seedrandom.prng;
  private randn: MPRandGauss;

  constructor(
      alpha: number, beta: number, dtype: keyof RandGammaDataTypes,
      seed?: number) {
    this.alpha = alpha;
    this.beta = 1 / beta;  // convert rate to scale parameter
    this.dtype = dtype;

    const seedValue = seed ? seed : Math.random();
    this.randu = seedrandom.alea(seedValue.toString());
    this.randn = new MPRandGauss(0, 1, dtype, false, this.randu());

    if (alpha < 1) {
      this.d = alpha + (2 / 3);
    } else {
      this.d = alpha - (1 / 3);
    }
    this.c = 1 / Math.sqrt(9 * this.d);
  }

  /** Returns next sample from a gamma distribution. */
  public nextValue(): number {
    let x2: number, v0: number, v1: number, x: number, u: number, v: number;
    while (true) {
      do {
        x = this.randn.nextValue();
        v = 1 + (this.c * x);
      } while (v <= 0);
      v *= v * v;
      x2 = x * x;
      v0 = 1 - (0.331 * x2 * x2);
      v1 = (0.5 * x2) + (this.d * (1 - v + Math.log(v)));
      u = this.randu();
      if (u < v0 || Math.log(u) < v1) {
        break;
      }
    }
    v = (1 / this.beta) * this.d * v;
    if (this.alpha < 1) {
      v *= Math.pow(this.randu(), 1 / this.alpha);
    }
    return this.convertValue(v);
  }
  /** Handles proper rounding for non-floating-point numbers. */
  private convertValue(value: number): number {
    if (this.dtype === 'float32') {
      return value;
    }
    return Math.round(value);
  }
}

export class UniformRandom implements RandomBase {
  private min: number;
  private range: number;
  private random: seedrandom.prng;
  private dtype?: keyof RandNormalDataTypes;

  constructor(
      min = 0, max = 1, dtype?: keyof RandNormalDataTypes,
      seed?: string|number) {
    this.min = min;
    this.range = max - min;
    this.dtype = dtype;
    if (seed == null) {
      seed = Math.random();
    }
    if (typeof seed === 'number') {
      seed = seed.toString();
    }

    if (!this.canReturnFloat() && this.range <= 1) {
      throw new Error(
          `The difference between ${min} - ${max} <= 1 and dtype is not float`);
    }
    this.random = seedrandom.alea(seed as string);
  }

  /** Handles proper rounding for non floating point numbers. */
  private canReturnFloat = () =>
      (this.dtype == null || this.dtype === 'float32');

  private convertValue(value: number): number {
    if (this.canReturnFloat()) {
      return value;
    }
    return Math.round(value);
  }

  nextValue() {
    return this.convertValue(this.min + this.range * this.random());
  }
}
