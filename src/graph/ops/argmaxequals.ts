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
import {Tensor} from '../graph';
import {SummedTensorArrayMap, TensorArrayMap} from '../tensor_array_map';

import {Operation} from './op';

/**
 * @hidden
 */
export class ArgMaxEquals extends Operation {
  /**
   * An ArgMaxEquals operation.
   */
  constructor(
      private x1Tensor: Tensor, private x2Tensor: Tensor,
      private yTensor: Tensor) {
    super();
  }

  feedForward(math: NDArrayMath, inferenceArrays: TensorArrayMap) {
    const x1 = inferenceArrays.get(this.x1Tensor);
    const x2 = inferenceArrays.get(this.x2Tensor);
    math.scope((keep) => {
      inferenceArrays.set(this.yTensor, keep(math.argMaxEquals(x1, x2)));
    });
  }

  backProp(
      math: NDArrayMath, inferenceArrays: TensorArrayMap,
      gradientArrays: SummedTensorArrayMap) {
    throw new Error('ArgMaxEquals backprop unimplemented');
  }
}
