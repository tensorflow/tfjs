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

import {ENV} from '../../environment';
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
export class ReduceSum extends Operation {
  /** Element-wise add operation. Broadcasts if one of the tensors is scalar. */
  constructor(private x: Tensor, private outTensor: Tensor) {
    super();
    util.assertShapesMatch(outTensor.shape, []);
    this.ones = ENV.math.keep(NDArray.ones(x.shape));
  }

  private ones: NDArray;

  feedForward(math: NDArrayMath, inferenceArrays: TensorArrayMap) {
    const x = inferenceArrays.get(this.x);

    math.scope((keep) => {
      inferenceArrays.set(this.outTensor, keep(math.sum(x)));
    });
  }

  backProp(
      math: NDArrayMath, inferenceArrays: TensorArrayMap,
      gradientArrays: SummedTensorArrayMap) {
    if (!graph_util.shouldBackProp(this.x)) {
      return;
    }

    math.scope(() => {
      const dy = gradientArrays.get(this.outTensor) as Scalar;
      gradientArrays.add(this.x, math.scalarTimesArray(dy, this.ones));
    });
  }

  dispose() {
    this.ones.dispose();
  }
}
