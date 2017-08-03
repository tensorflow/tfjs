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
 * Split ops are used to accumulate backprop derivatives when a node's output
 * tensor is consumed by multiple nodes.
 */
export class Split extends Operation {
  constructor(private input: Tensor, private outputs: Tensor[]) {
    super();
    outputs.forEach(output => {
      util.assertShapesMatch(input.shape, output.shape);
    });
  }

  feedForward(math: NDArrayMath, inferenceArrays: TensorArrayMap) {
    const inputArray = inferenceArrays.get(this.input);
    this.outputs.forEach(output => {
      inferenceArrays.set(output, inputArray);
    });
  }

  backProp(
      math: NDArrayMath, inferenceArrays: TensorArrayMap,
      gradientArrays: TensorArrayMap) {
    if (!graph_util.shouldBackProp(this.input)) {
      return;
    }

    math.scope((keep) => {
      let dx = math.add(
          gradientArrays.get(this.outputs[0]),
          gradientArrays.get(this.outputs[1]));
      // Sum across all the derivatives of the consumers of this node.
      this.outputs.slice(2).forEach(output => {
        dx = math.add(dx, gradientArrays.get(output));
      });
      gradientArrays.set(this.input, keep(dx));
    });
  }
}
