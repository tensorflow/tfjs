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

import {NDArrayMath} from '../../math/math';
import {Scalar} from '../../math/ndarray';
import {Tensor} from '../graph';
import * as graph_util from '../graph_util';
import {SummedTensorArrayMap, TensorArrayMap} from '../tensor_array_map';
import {Operation} from './op';

/**
 * @hidden
 */
export class LinearCombination extends Operation {
  /**
   * A 2-tensor linear combination operation.
   *
   * Combines tensors x1 and x2 (of the same shape) with weights c1 & c2;
   * Computes c1*x1 + c2*x2.
   */
  constructor(
      private x1Tensor: Tensor, private x2Tensor: Tensor,
      private c1Tensor: Tensor, private c2Tensor: Tensor,
      private outTensor: Tensor) {
    super();
  }

  feedForward(math: NDArrayMath, inferenceArrays: TensorArrayMap) {
    const x1 = inferenceArrays.get(this.x1Tensor);
    const x2 = inferenceArrays.get(this.x2Tensor);
    const c1 = inferenceArrays.get(this.c1Tensor).asScalar();
    const c2 = inferenceArrays.get(this.c2Tensor).asScalar();

    math.scope((keep) => {
      inferenceArrays.set(
          this.outTensor, keep(math.scaledArrayAdd(c1, x1, c2, x2)));
    });
  }

  backProp(
      math: NDArrayMath, inferenceArrays: TensorArrayMap,
      gradientArrays: SummedTensorArrayMap) {
    const x1 = inferenceArrays.get(this.x1Tensor);
    const x2 = inferenceArrays.get(this.x2Tensor);
    const c1 = inferenceArrays.get(this.c1Tensor) as Scalar;
    const c2 = inferenceArrays.get(this.c2Tensor) as Scalar;
    const dy = gradientArrays.get(this.outTensor);

    math.scope(() => {
      if (graph_util.shouldBackProp(this.x1Tensor)) {
        gradientArrays.add(this.x1Tensor, math.scalarTimesArray(c1, dy));
      }

      if (graph_util.shouldBackProp(this.x2Tensor)) {
        gradientArrays.add(this.x2Tensor, math.scalarTimesArray(c2, dy));
      }

      if (graph_util.shouldBackProp(this.c1Tensor)) {
        const dotProduct1 = math.elementWiseMul(x1, dy);
        gradientArrays.add(this.c1Tensor, math.sum(dotProduct1));
      }

      if (graph_util.shouldBackProp(this.c2Tensor)) {
        const dotProduct2 = math.elementWiseMul(x2, dy);
        gradientArrays.add(this.c2Tensor, math.sum(dotProduct2));
      }
    });
  }
}
