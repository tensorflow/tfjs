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
import {NDArray, Scalar} from '../../math/ndarray';
import * as util from '../../util';
import {Tensor} from '../graph';
import * as graph_util from '../graph_util';
import {SummedTensorArrayMap, TensorArrayMap} from '../tensor_array_map';

import {Operation} from './op';

/**
 * @hidden
 */
export class Multiply extends Operation {
  /**
   * Element-wise multiply operation. Broadcasts if one of the tensors is
   * scalar.
   */
  constructor(
      private x1Tensor: Tensor, private x2Tensor: Tensor,
      private yTensor: Tensor) {
    super();
    util.assert(
        util.sizeFromShape(x1Tensor.shape) === 1 ||
            util.sizeFromShape(x2Tensor.shape) === 1 ||
            util.arraysEqual(x1Tensor.shape, x2Tensor.shape),
        'One of t1 or t2 must be a scalar, or t1 and t2 must have ' +
            'the same shape');
  }

  feedForward(math: NDArrayMath, inferenceArrays: TensorArrayMap) {
    const t1 = inferenceArrays.get(this.x1Tensor) as Scalar;
    const t2 = inferenceArrays.get(this.x2Tensor) as Scalar;

    math.scope((keep) => {
      let result: NDArray;
      if (util.isScalarShape(t1.shape)) {
        result = math.scalarTimesArray(t1, t2);
      } else if (util.isScalarShape(t2.shape)) {
        result = math.scalarTimesArray(t2, t1);
      } else {
        result = math.elementWiseMul(t1, t2);
      }
      inferenceArrays.set(this.yTensor, keep(result));
    });
  }

  backProp(
      math: NDArrayMath, inferenceArrays: TensorArrayMap,
      gradientArrays: SummedTensorArrayMap) {
    const x1 = inferenceArrays.get(this.x1Tensor) as Scalar;
    const x2 = inferenceArrays.get(this.x2Tensor) as Scalar;
    const dy = gradientArrays.get(this.yTensor);

    math.scope(() => {
      if (graph_util.shouldBackProp(this.x1Tensor)) {
        if (util.isScalarShape(this.x1Tensor.shape)) {
          const mul = math.elementWiseMul(dy, x2);

          gradientArrays.add(this.x1Tensor, math.sum(mul));

        } else if (util.isScalarShape(x2.shape)) {
          gradientArrays.add(this.x1Tensor, math.scalarTimesArray(x2, dy));
        } else {
          gradientArrays.add(this.x1Tensor, math.elementWiseMul(x2, dy));
        }
      }

      if (graph_util.shouldBackProp(this.x2Tensor)) {
        if (util.isScalarShape(this.x2Tensor.shape)) {
          const mul = math.elementWiseMul(dy, x1);

          gradientArrays.add(this.x2Tensor, math.sum(mul));

        } else if (util.isScalarShape(x1.shape)) {
          gradientArrays.add(this.x2Tensor, math.scalarTimesArray(x1, dy));
        } else {
          gradientArrays.add(this.x2Tensor, math.elementWiseMul(x1, dy));
        }
      }
    });
  }
}
