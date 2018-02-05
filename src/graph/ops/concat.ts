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

import {keep, tidy} from '../../globals';
import * as concat_util from '../../math/concat_util';
import {NDArrayMath} from '../../math/math';
import {Tensor1D, Tensor2D, Tensor3D, Tensor4D} from '../../math/tensor';
import * as util from '../../util';
import {SymbolicTensor} from '../graph';
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
      private x1Tensor: SymbolicTensor, private x2Tensor: SymbolicTensor,
      private yTensor: SymbolicTensor) {
    super();
  }

  feedForward(math: NDArrayMath, inferecenArrays: TensorArrayMap) {
    const x1 = inferecenArrays.get(this.x1Tensor) as Tensor1D;
    const x2 = inferecenArrays.get(this.x2Tensor) as Tensor1D;

    tidy(() => {
      const concatResult = math.concat1D(x1, x2);
      inferecenArrays.set(this.yTensor, keep(concatResult));
    });
  }

  backProp(
      math: NDArrayMath, inferenceArrays: TensorArrayMap,
      gradientArrays: SummedTensorArrayMap) {
    tidy(() => {
      concatBackProp(
          math, this.x1Tensor, this.x2Tensor, this.yTensor, 0, gradientArrays,
          inferenceArrays);
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
      private x1Tensor: SymbolicTensor, private x2Tensor: SymbolicTensor,
      private axis: number, private yTensor: SymbolicTensor) {
    super();
    concat_util.assertParams(x1Tensor.shape, x2Tensor.shape, axis);
  }

  feedForward(math: NDArrayMath, inferecenArrays: TensorArrayMap) {
    const x1 = inferecenArrays.get(this.x1Tensor) as Tensor2D;
    const x2 = inferecenArrays.get(this.x2Tensor) as Tensor2D;

    tidy(() => {
      const concatResult = math.concat2D(x1, x2, this.axis);
      inferecenArrays.set(this.yTensor, keep(concatResult));
    });
  }

  backProp(
      math: NDArrayMath, inferenceArrays: TensorArrayMap,
      gradientArrays: SummedTensorArrayMap) {
    tidy(() => {
      concatBackProp(
          math, this.x1Tensor, this.x2Tensor, this.yTensor, this.axis,
          gradientArrays, inferenceArrays);
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
      private x1Tensor: SymbolicTensor, private x2Tensor: SymbolicTensor,
      private axis: number, private yTensor: SymbolicTensor) {
    super();
    concat_util.assertParams(x1Tensor.shape, x2Tensor.shape, axis);
  }

  feedForward(math: NDArrayMath, inferenceArrays: TensorArrayMap) {
    const x1 = inferenceArrays.get(this.x1Tensor) as Tensor3D;
    const x2 = inferenceArrays.get(this.x2Tensor) as Tensor3D;
    tidy(() => {
      const concatResult = math.concat3D(x1, x2, this.axis);
      inferenceArrays.set(this.yTensor, keep(concatResult));
    });
  }

  backProp(
      math: NDArrayMath, inferenceArrays: TensorArrayMap,
      gradientArrays: SummedTensorArrayMap) {
    tidy(() => {
      concatBackProp(
          math, this.x1Tensor, this.x2Tensor, this.yTensor, this.axis,
          gradientArrays, inferenceArrays);
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
      private x1Tensor: SymbolicTensor, private x2Tensor: SymbolicTensor,
      private axis: number, private yTensor: SymbolicTensor) {
    super();
    concat_util.assertParams(x1Tensor.shape, x2Tensor.shape, axis);
  }

  feedForward(math: NDArrayMath, inferecenArrays: TensorArrayMap) {
    const x1 = inferecenArrays.get(this.x1Tensor) as Tensor4D;
    const x2 = inferecenArrays.get(this.x2Tensor) as Tensor4D;

    tidy(() => {
      const concatResult = math.concat4D(x1, x2, this.axis);
      inferecenArrays.set(this.yTensor, keep(concatResult));
    });
  }

  backProp(
      math: NDArrayMath, inferenceArrays: TensorArrayMap,
      gradientArrays: SummedTensorArrayMap) {
    tidy(() => {
      concatBackProp(
          math, this.x1Tensor, this.x2Tensor, this.yTensor, this.axis,
          gradientArrays, inferenceArrays);
    });
  }
}

function concatBackProp(
    math: NDArrayMath, aTensor: SymbolicTensor, bTensor: SymbolicTensor,
    yTensor: SymbolicTensor, axis: number, gradArrays: SummedTensorArrayMap,
    infArrays: TensorArrayMap) {
  const dy = gradArrays.get(yTensor);
  const a = infArrays.get(aTensor);
  const b = infArrays.get(bTensor);
  const a2D = a.as2D(-1, util.sizeFromShape(a.shape.slice(axis)));
  const b2D = b.as2D(-1, util.sizeFromShape(b.shape.slice(axis)));
  const {aBegin, aSize, bBegin, bSize} = concat_util.computeGradientSliceShapes(
      a2D.shape as [number, number], b2D.shape as [number, number]);
  const dy2D = dy.as2D(-1, a2D.shape[1] + b2D.shape[1]);

  const slice1Result = math.slice2D(dy2D, aBegin, aSize).reshapeAs(a);
  const slice2Result = math.slice2D(dy2D, bBegin, bSize).reshapeAs(b);
  gradArrays.add(aTensor, slice1Result);
  gradArrays.add(bTensor, slice2Result);
}
