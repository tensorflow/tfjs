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

import {NDArray} from './math/ndarray';

/**
 * Initializer interface, all initializer implement this interface.
 */
export interface Initializer {
  initialize(weightsShape: number[], inputUnits: number, outputUnits: number):
      NDArray;
}

export class VarianceScalingInitializer implements Initializer {
  constructor(
      private scale = 1.0,
      private mode: 'fan_in'|'fan_out'|'fan_avg' = 'fan_in',
      private distribution: 'uniform'|'normal' = 'normal') {}

  initialize(weightsShape: number[], inputUnits: number, outputUnits: number):
      NDArray {
    let n = 0;
    if (this.mode === 'fan_in') {
      n = inputUnits;
    } else if (this.mode === 'fan_out') {
      n = outputUnits;
    } else if (this.mode === 'fan_avg') {
      n = (inputUnits + outputUnits) / 2;
    } else {
      throw new Error(
          `Unexpected mode for variance scaling initializer: ${this.mode}`);
    }

    if (this.distribution === 'normal') {
      return NDArray.randTruncatedNormal(
          weightsShape, 0.0, Math.sqrt(this.scale / n));
    } else if (this.distribution === 'uniform') {
      return NDArray.randUniform(
          weightsShape, 0.0, Math.sqrt(3 * this.scale / n));
    } else {
      throw new Error(
          `Unexpected distribution for variance scaling initializer: ` +
          `${this.distribution}`);
    }
  }
}

export class ZerosInitializer implements Initializer {
  constructor() {}

  initialize(weightsShape: number[], inputUnits: number, outputUnits: number):
      NDArray {
    return NDArray.zeros(weightsShape);
  }
}

export class OnesInitializer implements Initializer {
  constructor() {}

  initialize(weightsShape: number[], inputUnits: number, outputUnits: number):
      NDArray {
    const values = NDArray.zeros(weightsShape);
    values.fill(1);
    return values;
  }
}

export class ConstantInitializer implements Initializer {
  constructor(private value = 0) {}

  initialize(weightsShape: number[], inputUnits: number, outputUnits: number):
      NDArray {
    const values = NDArray.zeros(weightsShape);
    values.fill(this.value);
    return values;
  }
}

export class NDArrayInitializer implements Initializer {
  constructor(private ndarray: NDArray) {}

  initialize(weightsShape: number[], inputUnits: number, outputUnits: number):
      NDArray {
    return this.ndarray;
  }
}

export class RandomNormalInitializer implements Initializer {
  constructor(private mean = 0, private stdev = .05) {}

  initialize(weightsShape: number[], inputUnits: number, outputUnits: number):
      NDArray {
    return NDArray.randNormal(weightsShape, this.mean, this.stdev);
  }
}

export class RandomTruncatedNormalInitializer implements Initializer {
  constructor(private mean = 0, private stdev = .05) {}

  initialize(weightsShape: number[], inputUnits: number, outputUnits: number):
      NDArray {
    return NDArray.randTruncatedNormal(weightsShape, this.mean, this.stdev);
  }
}

export class RandomUniformInitializer implements Initializer {
  constructor(private minval = -.05, private maxval = .05) {}

  initialize(weightsShape: number[], inputUnits: number, outputUnits: number):
      NDArray {
    return NDArray.randUniform(weightsShape, this.minval, this.maxval);
  }
}
