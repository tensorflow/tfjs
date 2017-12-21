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
// tslint:disable-next-line:max-line-length
import {ENV} from '../../environment';
// tslint:disable-next-line:max-line-length
import {ElementWiseCostFunction, SquareCostFunc} from '../../math/cost_functions';
import {NDArrayMath} from '../../math/math';
import {Scalar} from '../../math/ndarray';
import * as util from '../../util';
import {Tensor} from '../graph';
import * as graph_util from '../graph_util';
import {SummedTensorArrayMap, TensorArrayMap} from '../tensor_array_map';
import {Operation} from './op';

/**
 * @hidden
 */
export class ElementWiseCost extends Operation {
  private oneOverNScalar: Scalar;

  constructor(
      protected x1Tensor: Tensor, protected x2Tensor: Tensor,
      protected yTensor: Tensor, protected func: ElementWiseCostFunction) {
    super();
    this.oneOverNScalar =
        ENV.math.keep(Scalar.new(1 / util.sizeFromShape(x1Tensor.shape)));
  }

  feedForward(math: NDArrayMath, inferenceArrays: TensorArrayMap) {
    const x1 = inferenceArrays.get(this.x1Tensor);
    const x2 = inferenceArrays.get(this.x2Tensor);

    math.scope((keep) => {
      const elementWiseCost = this.func.cost(math, x1, x2);
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

    math.scope(() => {
      if (graph_util.shouldBackProp(this.x1Tensor)) {
        gradientArrays.add(this.x1Tensor, this.func.der(math, x1, x2));
      }
      if (graph_util.shouldBackProp(this.x2Tensor)) {
        gradientArrays.add(this.x2Tensor, this.func.der(math, x2, x1));
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
  constructor(x1Tensor: Tensor, x2Tensor: Tensor, yTensor: Tensor) {
    super(x1Tensor, x2Tensor, yTensor, new SquareCostFunc());
  }
}
