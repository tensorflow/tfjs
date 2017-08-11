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

import {Tensor} from '../graph';
import * as graph_util from '../graph_util';
import {NDArrayMath} from '../math/math';
import {NDArray, Scalar} from '../math/ndarray';
import {TensorArrayMap} from '../tensor_array_map';
import * as util from '../util';

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
            util.arraysEqual(x1Tensor.shape, x2Tensor.shape),
        'One of t1 or t2 must be a scalar, or t1 and t2 must have ' +
            'the same shape');
  }

  feedForward(math: NDArrayMath, inferenceArrays: TensorArrayMap) {
    const x1 = inferenceArrays.get(this.x1Tensor);
    const x2 = inferenceArrays.get(this.x2Tensor);

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
      gradientArrays: TensorArrayMap) {
    const dy = gradientArrays.get(this.yTensor);

    math.scope((keep) => {
      if (graph_util.shouldBackProp(this.x1Tensor)) {
        if (util.isScalarShape(this.x1Tensor.shape)) {
          const sum = math.sum(dy);
          if (this.dySizeScalar == null) {
            this.dySizeScalar = Scalar.new(dy.size);
          }
          gradientArrays.set(
              this.x1Tensor, keep(math.divide(sum, this.dySizeScalar)));
        } else {
          gradientArrays.set(this.x1Tensor, dy);
        }
      }

      if (graph_util.shouldBackProp(this.x2Tensor)) {
        if (util.isScalarShape(this.x2Tensor.shape)) {
          const sum = math.sum(dy);
          if (this.dySizeScalar == null) {
            this.dySizeScalar = Scalar.new(dy.size);
          }
          gradientArrays.set(
              this.x2Tensor, keep(math.divide(sum, this.dySizeScalar)));
        } else {
          gradientArrays.set(this.x2Tensor, dy);
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
