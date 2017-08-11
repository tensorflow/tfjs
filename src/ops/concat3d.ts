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
import * as concat3d_util from '../math/concat3d_util';
import {NDArrayMath} from '../math/math';
import {Array3D} from '../math/ndarray';
import {TensorArrayMap} from '../tensor_array_map';

import {Operation} from './op';

/**
 * @hidden
 */
export class Concat3D extends Operation {
  /**
   * A Concat 3D operation.
   *
   * Concats two 3D tensors along an axis.
   */
  constructor(
      private x1Tensor: Tensor, private x2Tensor: Tensor, private axis: number,
      private yTensor: Tensor) {
    super();
    concat3d_util.assertConcat3DShapesMatch(
        x1Tensor.shape, x2Tensor.shape, axis);
  }

  feedForward(math: NDArrayMath, inferenceArrays: TensorArrayMap) {
    const x1 = inferenceArrays.get(this.x1Tensor) as Array3D;
    const x2 = inferenceArrays.get(this.x2Tensor) as Array3D;

    math.scope((keep) => {
      const concatResult = math.concat3D(x1, x2, this.axis);
      inferenceArrays.set(this.yTensor, keep(concatResult));
    });
  }

  backProp(
      math: NDArrayMath, inferenceArrays: TensorArrayMap,
      gradientArrays: TensorArrayMap) {
    throw new Error('Concat3D backprop not implemented.');
  }
}
