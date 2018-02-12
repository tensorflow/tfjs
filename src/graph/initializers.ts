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

import * as ops from '../ops/ops';
import {Tensor} from '../tensor';

/**
 * Initializer interface, all initializer implement this interface.
 */
export interface Initializer {
  initialize(weightsShape: number[], inputUnits: number, outputUnits: number):
      Tensor;
}

export class VarianceScalingInitializer implements Initializer {
  constructor(
      private scale = 1.0,
      private mode: 'fan_in'|'fan_out'|'fan_avg' = 'fan_in',
      private distribution: 'uniform'|'normal' = 'normal') {}

  initialize(weightsShape: number[], inputUnits: number, outputUnits: number):
      Tensor {
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
      return ops.truncatedNormal(weightsShape, 0.0, Math.sqrt(this.scale / n));
    } else if (this.distribution === 'uniform') {
      return ops.randomUniform(
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
      Tensor {
    return ops.zeros(weightsShape);
  }
}

export class OnesInitializer implements Initializer {
  constructor() {}

  initialize(weightsShape: number[], inputUnits: number, outputUnits: number):
      Tensor {
    return ops.ones(weightsShape);
  }
}

export class ConstantInitializer implements Initializer {
  constructor(private value = 0) {}

  initialize(weightsShape: number[], inputUnits: number, outputUnits: number):
      Tensor {
    return ops.fill(weightsShape, this.value);
  }
}

export class TensorInitializer implements Initializer {
  constructor(private tensor: Tensor) {}

  initialize(weightsShape: number[], inputUnits: number, outputUnits: number):
      Tensor {
    return this.tensor;
  }
}

export class RandomNormalInitializer implements Initializer {
  constructor(private mean = 0, private stdev = .05) {}

  initialize(weightsShape: number[], inputUnits: number, outputUnits: number):
      Tensor {
    return ops.randomNormal(weightsShape, this.mean, this.stdev);
  }
}

export class RandomTruncatedNormalInitializer implements Initializer {
  constructor(private mean = 0, private stdev = .05) {}

  initialize(weightsShape: number[], inputUnits: number, outputUnits: number):
      Tensor {
    return ops.truncatedNormal(weightsShape, this.mean, this.stdev);
  }
}

export class RandomUniformInitializer implements Initializer {
  constructor(private minval = -.05, private maxval = .05) {}

  initialize(weightsShape: number[], inputUnits: number, outputUnits: number):
      Tensor {
    return ops.randomUniform(weightsShape, this.minval, this.maxval);
  }
}
