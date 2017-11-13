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
import {NDArray} from '../../math/ndarray';
import * as util from '../../util';
import {Tensor} from '../graph';
import {SummedTensorArrayMap, TensorArrayMap} from '../tensor_array_map';

import {Operation} from './op';

export class Reshape<T1 extends NDArray, T2 extends NDArray> extends Operation {
  constructor(private xTensor: Tensor, private yTensor: Tensor) {
    super();
    const xSize = util.sizeFromShape(xTensor.shape);
    const ySize = util.sizeFromShape(yTensor.shape);
    util.assert(
        xSize === ySize,
        `The input size (${xSize}) and output size (${ySize}) must match`);
  }

  feedForward(math: NDArrayMath, inferenceArrays: TensorArrayMap) {
    const x = inferenceArrays.get(this.xTensor) as T1;

    const clone = math.clone(x);

    math.scope(keep => {
      inferenceArrays.set(
          this.yTensor, keep(clone.reshape(this.yTensor.shape)));
    });
  }

  backProp(
      math: NDArrayMath, inferenceArrays: TensorArrayMap,
      gradientArrays: SummedTensorArrayMap) {
    const dy = gradientArrays.get(this.yTensor) as T2;

    const clone = math.clone(dy);

    math.scope(() => {
      gradientArrays.add(this.xTensor, clone.reshape(this.xTensor.shape));
    });
  }
}
