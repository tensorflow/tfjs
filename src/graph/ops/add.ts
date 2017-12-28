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
import {Array2D, NDArray, Scalar} from '../../math/ndarray';
import * as util from '../../util';
import {Tensor} from '../graph';
import * as graph_util from '../graph_util';
import {SummedTensorArrayMap, TensorArrayMap} from '../tensor_array_map';

import {Operation} from './op';

/**
 * @hidden
 */
export class Add extends Operation {
  private dySizeScalar: Scalar;

  /** Element-wise add operation. Broadcasts if one of the tensors is scalar. */
  constructor(
      private x1Tensor: Tensor, private x2Tensor: Tensor,
      private yTensor: Tensor) {
    super();
    util.assert(
        util.sizeFromShape(x1Tensor.shape) === 1 ||
            util.sizeFromShape(x2Tensor.shape) === 1 ||
            util.arraysEqual(x1Tensor.shape, x2Tensor.shape) ||
            (x1Tensor.shape.length === 2 && x2Tensor.shape.length === 1 &&
             x1Tensor.shape[1] === x2Tensor.shape[0]) ||
            (x1Tensor.shape.length === 1 && x2Tensor.shape.length === 2 &&
             x1Tensor.shape[0] === x2Tensor.shape[1]),
        'One of t1 or t2 must be a scalar, or t1 and t2 must have ' +
            'the same shape, ' +
            'or one of them can be broadcasted (2D and 1D).');
  }

  feedForward(math: NDArrayMath, inferenceArrays: TensorArrayMap) {
    const x1 = inferenceArrays.get(this.x1Tensor) as Scalar;
    const x2 = inferenceArrays.get(this.x2Tensor) as Scalar;

    math.scope((keep) => {
      let result: NDArray;
      if (util.isScalarShape(x1.shape)) {
        result = math.scalarPlusArray(x1, x2);
      } else if (util.isScalarShape(x2.shape)) {
        result = math.scalarPlusArray(x2, x1);
      } else {
        result = math.add(x1, x2);
      }
      inferenceArrays.set(this.yTensor, keep(result));
    });
  }

  backProp(
      math: NDArrayMath, inferenceArrays: TensorArrayMap,
      gradientArrays: SummedTensorArrayMap) {
    const dy = gradientArrays.get(this.yTensor);

    math.scope(() => {
      if (graph_util.shouldBackProp(this.x1Tensor)) {
        if (this.x1Tensor.shape.length === 1 &&
            this.x2Tensor.shape.length === 2 &&
            this.x1Tensor.shape[0] === this.x2Tensor.shape[1]) {
          const sum = math.sum(dy as Array2D, 0);
          gradientArrays.add(this.x1Tensor, sum);
        } else if (util.isScalarShape(this.x1Tensor.shape)) {
          const sum = math.sum(dy);
          gradientArrays.add(this.x1Tensor, sum);
        } else {
          gradientArrays.add(this.x1Tensor, math.clone(dy));
        }
      }

      if (graph_util.shouldBackProp(this.x2Tensor)) {
        if (this.x1Tensor.shape.length === 2 &&
            this.x2Tensor.shape.length === 1 &&
            this.x1Tensor.shape[1] === this.x2Tensor.shape[0]) {
          const sum = math.sum(dy as Array2D, 0);
          gradientArrays.add(this.x2Tensor, sum);
        } else if (util.isScalarShape(this.x2Tensor.shape)) {
          const sum = math.sum(dy);
          gradientArrays.add(this.x2Tensor, sum);
        } else {
          gradientArrays.add(this.x2Tensor, math.clone(dy));
        }
      }
    });
  }

  dispose() {
    if (this.dySizeScalar != null) {
      this.dySizeScalar.dispose();
    }
  }
}
