/* Copyright 2017 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

import {NDArrayMath} from './math';
import {NDArray, Scalar} from './ndarray';

/** A node's activation function and its derivative. */
export interface ActivationFunction {
  output<T extends NDArray>(math: NDArrayMath, input: T): T;
  der<T extends NDArray>(math: NDArrayMath, input: T, output: T): T;
}

export class TanHFunc implements ActivationFunction {
  output<T extends NDArray>(math: NDArrayMath, x: T) {
    return math.scope(() => {
      return math.tanh(x);
    });
  }

  der<T extends NDArray>(math: NDArrayMath, x: T, y: T) {
    return math.scope(() => {
      const ySquared = math.elementWiseMul(y, y);
      // 1 - y^2.
      return math.scalarMinusArray(Scalar.ONE, ySquared);
    });
  }
}

export class ReLUFunc implements ActivationFunction {
  output<T extends NDArray>(math: NDArrayMath, x: T) {
    return math.scope(() => {
      return math.relu(x);
    });
  }

  der<T extends NDArray>(math: NDArrayMath, x: T, y: T) {
    return math.scope(() => {
      return math.step(x);
    });
  }
}

export class SigmoidFunc implements ActivationFunction {
  output<T extends NDArray>(math: NDArrayMath, x: T) {
    return math.scope(() => {
      return math.sigmoid(x);
    });
  }

  der<T extends NDArray>(math: NDArrayMath, x: T, y: T) {
    return math.scope(() => {
      // y * (1 - y) = y - y^2
      const ySquared = math.elementWiseMul(y, y);
      return math.sub(y, ySquared);
    });
  }
}

export class SquareFunc implements ActivationFunction {
  output<T extends NDArray>(math: NDArrayMath, x: T) {
    return math.scope(() => {
      return math.elementWiseMul(x, x);
    });
  }

  der<T extends NDArray>(math: NDArrayMath, x: T, y: T) {
    return math.scope(() => {
      // dy/dx = 2*x.
      return math.scalarTimesArray(Scalar.TWO, x);
    });
  }
}
