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

import {ENV} from '../../environment';
import {keep, tidy} from '../../globals';
import {NDArrayMath} from '../../math/math';
import {Scalar} from '../../math/tensor';
import * as util from '../../util';
import {ElementWiseCostFunction, SquareCostFunc} from '../cost_functions';
import {SymbolicTensor} from '../graph';
import * as graph_util from '../graph_util';
import {SummedTensorArrayMap, TensorArrayMap} from '../tensor_array_map';

import {Operation} from './op';

/**
 * @hidden
 */
export class ElementWiseCost extends Operation {
  private oneOverNScalar: Scalar;

  constructor(
      protected x1Tensor: SymbolicTensor, protected x2Tensor: SymbolicTensor,
      protected yTensor: SymbolicTensor,
      protected func: ElementWiseCostFunction) {
    super();
    this.oneOverNScalar =
        ENV.math.keep(Scalar.new(1 / util.sizeFromShape(x1Tensor.shape)));
  }

  feedForward(math: NDArrayMath, inferenceArrays: TensorArrayMap) {
    const x1 = inferenceArrays.get(this.x1Tensor);
    const x2 = inferenceArrays.get(this.x2Tensor);

    tidy(() => {
      const elementWiseCost = this.func.cost(x1, x2);
      const sum = math.sum(elementWiseCost);
      const result = math.scalarTimesArray(this.oneOverNScalar, sum);
      inferenceArrays.set(this.yTensor, keep(result));
    });
  }

  backProp(
      math: NDArrayMath, inferenceArrays: TensorArrayMap,
      gradientArrays: SummedTensorArrayMap) {
    const x1 = inferenceArrays.get(this.x1Tensor);
    const x2 = inferenceArrays.get(this.x2Tensor);

    tidy(() => {
      if (graph_util.shouldBackProp(this.x1Tensor)) {
        gradientArrays.add(this.x1Tensor, this.func.der(x1, x2));
      }
      if (graph_util.shouldBackProp(this.x2Tensor)) {
        gradientArrays.add(this.x2Tensor, this.func.der(x2, x1));
      }
    });
  }

  dispose() {
    this.func.dispose();
    this.oneOverNScalar.dispose();
  }
}

/**
 * @hidden
 */
export class MeanSquaredCost extends ElementWiseCost {
  constructor(
      x1Tensor: SymbolicTensor, x2Tensor: SymbolicTensor,
      yTensor: SymbolicTensor) {
    super(x1Tensor, x2Tensor, yTensor, new SquareCostFunc());
  }
}
