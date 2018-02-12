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

import {tidy} from '../globals';
import {NDArrayMath} from '../math';
import {Scalar, Tensor} from '../tensor';

/** A node's activation function and its derivative. */
export interface ActivationFunction {
  output<T extends Tensor>(math: NDArrayMath, input: T): T;
  der<T extends Tensor>(math: NDArrayMath, input: T, output: T): T;
  dispose(): void;
}

export class TanHFunc implements ActivationFunction {
  private one = Scalar.new(1);

  output<T extends Tensor>(math: NDArrayMath, x: T) {
    return math.tanh(x) as T;
  }

  der<T extends Tensor>(math: NDArrayMath, x: T, y: T) {
    return tidy(() => {
      const ySquared = math.multiplyStrict(y, y);
      // 1 - y^2.
      return math.subtract(this.one, ySquared as Tensor) as T;
    });
  }

  dispose() {
    this.one.dispose();
  }
}

export class ReLUFunc implements ActivationFunction {
  output<T extends Tensor>(math: NDArrayMath, x: T) {
    return math.relu(x) as T;
  }

  der<T extends Tensor>(math: NDArrayMath, x: T, y: T) {
    return math.step(x) as T;
  }

  dispose() {}
}

export class LeakyReluFunc implements ActivationFunction {
  private alpha: number;

  constructor(alpha: number) {
    this.alpha = alpha;
  }

  output<T extends Tensor>(math: NDArrayMath, x: T) {
    return math.leakyRelu(x, this.alpha) as T;
  }

  der<T extends Tensor>(math: NDArrayMath, x: T, y: T) {
    return math.step(x, this.alpha) as T;
  }

  dispose() {}
}

export class SigmoidFunc implements ActivationFunction {
  output<T extends Tensor>(math: NDArrayMath, x: T) {
    return math.sigmoid(x) as T;
  }

  der<T extends Tensor>(math: NDArrayMath, x: T, y: T): T {
    return tidy(() => {
      // y * (1 - y) = y - y^2
      const ySquared = math.multiplyStrict(y, y);
      return math.subStrict(y, ySquared) as T;
    });
  }

  dispose() {}
}

export class SquareFunc implements ActivationFunction {
  private two = Scalar.new(2);

  output<T extends Tensor>(math: NDArrayMath, x: T) {
    return math.multiplyStrict(x, x) as T;
  }

  der<T extends Tensor>(math: NDArrayMath, x: T, y: T) {
    // dy/dx = 2*x.
    return math.multiply(this.two, x as Tensor) as T;
  }

  dispose() {
    this.two.dispose();
  }
}

export class EluFunc implements ActivationFunction {
  output<T extends Tensor>(math: NDArrayMath, x: T) {
    return math.elu(x) as T;
  }

  der<T extends Tensor>(math: NDArrayMath, x: T, y: T): T {
    throw new Error('Not implemented');
  }

  dispose() {}
}
