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

import {keep, tidy} from '../../math/backends/tracking';
import * as conv_util from '../../math/conv_util';
import {NDArrayMath} from '../../math/math';
import {Tensor3D} from '../../math/tensor';
import * as util from '../../util';
import {SymbolicTensor} from '../graph';
import {SummedTensorArrayMap, TensorArrayMap} from '../tensor_array_map';
import {Operation} from './op';

/**
 * @hidden
 */
export class MaxPool extends Operation {
  private pad: number;

  constructor(
      private xTensor: SymbolicTensor, private yTensor: SymbolicTensor,
      private fieldSize: number, private stride = 1, pad?: number) {
    super();

    if (pad != null) {
      this.pad = pad;
    } else {
      this.pad = conv_util.computeDefaultPad(
          xTensor.shape as [number, number, number], this.fieldSize,
          this.stride);
    }

    util.assert(
        util.isInt(this.pad),
        `The zero padding (${this.pad}) must be an integer. Change the ` +
            `stride and/or zero pad parameters`);
  }

  feedForward(math: NDArrayMath, inferenceArrays: TensorArrayMap) {
    const x = inferenceArrays.get(this.xTensor) as Tensor3D;
    tidy(() => {
      inferenceArrays.set(
          this.yTensor,
          keep(math.maxPool(x, this.fieldSize, this.stride, this.pad)));
    });
  }

  backProp(
      math: NDArrayMath, inferenceArrays: TensorArrayMap,
      gradientArrays: SummedTensorArrayMap) {
    const x = inferenceArrays.get(this.xTensor) as Tensor3D;
    const dy = gradientArrays.get(this.yTensor) as Tensor3D;

    tidy(() => {
      gradientArrays.add(
          this.xTensor,
          math.maxPoolBackprop(dy, x, this.fieldSize, this.stride, this.pad));
    });
  }
}
