/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
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

import {KernelBackend} from '../backends/backend';
import {ENGINE} from '../engine';
import {BroadcastTo, BroadCastToAttrs, BroadcastToInputs} from '../kernel_names';
import {NamedAttrMap} from '../kernel_registry';
import {Tensor} from '../tensor';
import {NamedTensorMap} from '../tensor_types';
import {convertToTensor} from '../tensor_util_env';
import {Rank, ShapeMap, TensorLike} from '../types';

import {op} from './operation';

/**
 * Broadcast an array to a compatible shape NumPy-style.
 *
 * The tensor's shape is compared to the broadcast shape from end to beginning.
 * Ones are prepended to the tensor's shape until is has the same length as
 * the broadcast shape. If input.shape[i]==shape[i], the (i+1)-th axis is
 * already broadcast-compatible. If input.shape[i]==1 and shape[i]==N, then
 * the input tensor is tiled N times along that axis (using tf.tile).
 *
 * @param input The tensor that is to be broadcasted.
 * @param shape The input is to be broadcast to this shape.
 */
/** @doc {heading: 'Tensors', subheading: 'Transformations'} */
function broadcastTo_<R extends Rank>(
    x: Tensor|TensorLike, shape: ShapeMap[R]): Tensor<R> {
  let input = convertToTensor(x, 'broadcastTo', 'x');
  const xShape = input.shape;

  if (shape.some(d => !(d > 0) || d % 1 !== 0)) {
    throw new Error(`broadcastTo(): Invalid broadcast shape [${shape}].`);
  }

  if (shape.length < input.rank) {
    throw new Error(`broadcastTo(): shape.length=${shape.length} < input.rank=${
        input.rank}.`);
  }

  if (shape.length > input.rank) {
    const newShape = input.shape.slice();
    while (newShape.length < shape.length) {
      newShape.unshift(1);
    }
    input = input.reshape(newShape);
  }

  const inputShape = input.shape;
  const reps: number[] = Array.from(shape);
  for (let i = shape.length - 1; i >= 0; i--) {
    if (inputShape[i] === shape[i]) {
      reps[i] = 1;
    } else if (input.shape[i] !== 1) {
      throw new Error(
          `broadcastTo(): [${xShape}] cannot be broadcast to [${shape}].`);
    }
  }
  const axes = reps.map((n, i) => n > 1 ? i : -1).filter(i => i >= 0);

  if (axes.length === 0) {
    return input.clone() as Tensor<R>;
  }

  const forward = (backend: KernelBackend) => backend.tile(input, reps);
  const keepDims = true;
  const backward = (dy: Tensor) => ({x: () => dy.sum(axes, keepDims)});

  const inputs: BroadcastToInputs = {x: input};
  const attrs: BroadCastToAttrs = {shape, inputShape};

  return ENGINE.runKernelFunc(
             forward, inputs as unknown as NamedTensorMap, backward,
             BroadcastTo, attrs as unknown as NamedAttrMap) as Tensor<R>;
}

export const broadcastTo = op({broadcastTo_});
