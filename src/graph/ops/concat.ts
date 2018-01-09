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

import * as concat_util from '../../math/concat_util';
import {NDArrayMath} from '../../math/math';
import {Array1D, Array2D, Array3D, Array4D} from '../../math/ndarray';
import {Tensor} from '../graph';
import {SummedTensorArrayMap, TensorArrayMap} from '../tensor_array_map';

import {Operation} from './op';

/**
 * @hidden
 */
export class Concat1D extends Operation {
  /**
   * A Concat 1D operation.
   *
   * Concats two 1D tensors along an axis.
   */
  constructor(
      private x1Tensor: Tensor, private x2Tensor: Tensor,
      private yTensor: Tensor) {
    super();
  }

  feedForward(math: NDArrayMath, inferecenArrays: TensorArrayMap) {
    const x1 = inferecenArrays.get(this.x1Tensor) as Array1D;
    const x2 = inferecenArrays.get(this.x2Tensor) as Array1D;

    math.scope((keep) => {
      const concatResult = math.concat1D(x1, x2);
      inferecenArrays.set(this.yTensor, keep(concatResult));
    });
  }

  backProp(
      math: NDArrayMath, inferenceArrays: TensorArrayMap,
      gradientArrays: SummedTensorArrayMap) {
    const x1Size = this.x1Tensor.shape[0];
    const x2Size = this.x2Tensor.shape[0];
    const dy = gradientArrays.get(this.yTensor) as Array1D;
    math.scope((keep) => {
      const slice1Result = math.slice1D(dy, 0, x1Size);
      const slice2Result = math.slice1D(dy, x1Size, x2Size);
      gradientArrays.add(this.x1Tensor, slice1Result);
      gradientArrays.add(this.x2Tensor, slice2Result);
    });
  }
}

/**
 * @hidden
 */
export class Concat2D extends Operation {
  /**
   * A Concat 2D operation.
   *
   * Concats two 2D tensors along an axis.
   */
  constructor(
      private x1Tensor: Tensor, private x2Tensor: Tensor, private axis: number,
      private yTensor: Tensor) {
    super();
    concat_util.assertParams(x1Tensor.shape, x2Tensor.shape, axis);
  }

  feedForward(math: NDArrayMath, inferecenArrays: TensorArrayMap) {
    const x1 = inferecenArrays.get(this.x1Tensor) as Array2D;
    const x2 = inferecenArrays.get(this.x2Tensor) as Array2D;

    math.scope((keep) => {
      const concatResult = math.concat2D(x1, x2, this.axis);
      inferecenArrays.set(this.yTensor, keep(concatResult));
    });
  }

  backProp(
      math: NDArrayMath, inferenceArrays: TensorArrayMap,
      gradientArrays: SummedTensorArrayMap) {
    const dy = gradientArrays.get(this.yTensor) as Array2D;

    const {x1Begin, x1Size, x2Begin, x2Size} =
        concat_util.computeGradientSliceShapes2D(
            this.x1Tensor.shape, this.yTensor.shape, this.axis);

    math.scope((keep) => {
      const slice1Result = math.slice2D(dy, x1Begin, x1Size);
      const slice2Result = math.slice2D(dy, x2Begin, x2Size);
      gradientArrays.add(this.x1Tensor, slice1Result);
      gradientArrays.add(this.x2Tensor, slice2Result);
    });
  }
}

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
    concat_util.assertParams(x1Tensor.shape, x2Tensor.shape, axis);
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
      gradientArrays: SummedTensorArrayMap) {
    const dy = gradientArrays.get(this.yTensor) as Array3D;

    const {x1Begin, x1Size, x2Begin, x2Size} =
        concat_util.computeGradientSliceShapes3D(
            this.x1Tensor.shape, this.yTensor.shape, this.axis);

    math.scope((keep) => {
      const slice1Result = math.slice3D(dy, x1Begin, x1Size);
      const slice2Result = math.slice3D(dy, x2Begin, x2Size);
      gradientArrays.add(this.x1Tensor, slice1Result);
      gradientArrays.add(this.x2Tensor, slice2Result);
    });
  }
}

/**
 * @hidden
 */
export class Concat4D extends Operation {
  /**
   * A Concat 4D operation.
   *
   * Concats two 4D tensors along an axis.
   */
  constructor(
      private x1Tensor: Tensor, private x2Tensor: Tensor, private axis: number,
      private yTensor: Tensor) {
    super();
    concat_util.assertParams(x1Tensor.shape, x2Tensor.shape, axis);
  }

  feedForward(math: NDArrayMath, inferecenArrays: TensorArrayMap) {
    const x1 = inferecenArrays.get(this.x1Tensor) as Array4D;
    const x2 = inferecenArrays.get(this.x2Tensor) as Array4D;

    math.scope((keep) => {
      const concatResult = math.concat4D(x1, x2, this.axis);
      inferecenArrays.set(this.yTensor, keep(concatResult));
    });
  }

  backProp(
      math: NDArrayMath, inferenceArrays: TensorArrayMap,
      gradientArrays: SummedTensorArrayMap) {
    const dy = gradientArrays.get(this.yTensor) as Array4D;

    const {x1Begin, x1Size, x2Begin, x2Size} =
        concat_util.computeGradientSliceShapes4D(
            this.x1Tensor.shape, this.yTensor.shape, this.axis);

    math.scope((keep) => {
      const slice1Result = math.slice4D(dy, x1Begin, x1Size);
      const slice2Result = math.slice4D(dy, x2Begin, x2Size);
      gradientArrays.add(this.x1Tensor, slice1Result);
      gradientArrays.add(this.x2Tensor, slice2Result);
    });
  }
}
