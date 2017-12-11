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

import {MatrixOrientation} from '../../math/backends/types/matmul';
import {NDArrayMath} from '../../math/math';
import {Array1D, Array2D} from '../../math/ndarray';
import {Tensor} from '../graph';
import * as graph_util from '../graph_util';
import {SummedTensorArrayMap, TensorArrayMap} from '../tensor_array_map';

import {Operation} from './op';

/**
 * @hidden
 */
export class MatMul extends Operation {
  constructor(
      private x1Tensor: Tensor, private x2Tensor: Tensor,
      private yTensor: Tensor) {
    super();
  }

  feedForward(math: NDArrayMath, inferenceArrays: TensorArrayMap) {
    const x1 = inferenceArrays.get(this.x1Tensor);
    const x2 = inferenceArrays.get(this.x2Tensor);

    math.scope((keep) => {
      if (x1.shape.length === 2 && x2.shape.length === 2) {
        inferenceArrays.set(
            this.yTensor, keep(math.matMul(x1 as Array2D, x2 as Array2D)));
      } else if (x1.shape.length === 2 && x2.shape.length === 1) {
        inferenceArrays.set(
            this.yTensor,
            keep(math.matrixTimesVector(x1 as Array2D, x2 as Array1D)));
      } else if (x1.shape.length === 1 && x2.shape.length === 2) {
        inferenceArrays.set(
            this.yTensor,
            keep(math.vectorTimesMatrix(x1 as Array1D, x2 as Array2D)));
      }
    });
  }

  backProp(
      math: NDArrayMath, inferenceArrays: TensorArrayMap,
      gradientArrays: SummedTensorArrayMap) {
    let x1 = inferenceArrays.get(this.x1Tensor);
    let x2 = inferenceArrays.get(this.x2Tensor);
    let dy = gradientArrays.get(this.yTensor);

    if (x1.shape.length === 1) {
      x1 = x1.reshape([1, x1.size]);
      dy = dy.reshape([1, dy.size]);
    }
    if (x2.shape.length === 1) {
      x2 = x2.reshape([x2.size, 1]);
      dy = dy.reshape([dy.size, 1]);
    }

    math.scope(() => {
      // y = x1 * x2
      // dx1 = dy * x2T
      // dx2 = x1T * dy
      if (graph_util.shouldBackProp(this.x1Tensor)) {
        const dx1 = math.matMul(
            dy as Array2D, x2 as Array2D, MatrixOrientation.REGULAR,
            MatrixOrientation.TRANSPOSED);
        gradientArrays.add(
            this.x1Tensor, this.x1Tensor.shape.length === 1 ? dx1.as1D() : dx1);
      }
      if (graph_util.shouldBackProp(this.x2Tensor)) {
        const dx2 = math.matMul(
            x1 as Array2D, dy as Array2D, MatrixOrientation.TRANSPOSED,
            MatrixOrientation.REGULAR);
        gradientArrays.add(
            this.x2Tensor, this.x2Tensor.shape.length === 1 ? dx2.as1D() : dx2);
      }
    });
  }
}
