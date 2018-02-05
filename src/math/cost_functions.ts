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

import {ENV} from '../environment';
import {NDArrayMath} from './math';
import * as ops from './ops';
import {Scalar, Tensor} from './tensor';

/**
 * An error function and its derivative.
 */
export interface ElementWiseCostFunction {
  cost<T extends Tensor>(math: NDArrayMath, x1: T, x2: T): T;
  der<T extends Tensor>(math: NDArrayMath, x1: T, x2: T): T;
  dispose(): void;
}

export class SquareCostFunc implements ElementWiseCostFunction {
  constructor() {
    this.halfOne = ENV.math.keep(ops.scalar(0.5));
  }

  private halfOne: Scalar;

  cost<T extends Tensor>(math: NDArrayMath, x1: T, x2: T): T {
    const diff = math.subStrict(x1, x2);
    const diffSquared = math.multiplyStrict(diff, diff);
    const result = math.multiply(this.halfOne, diffSquared);

    diff.dispose();
    diffSquared.dispose();

    return result as T;
  }

  der<T extends Tensor>(math: NDArrayMath, x1: T, x2: T): T {
    return math.subStrict(x1, x2) as T;
  }

  dispose() {
    this.halfOne.dispose();
  }
}
