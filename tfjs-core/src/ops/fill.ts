/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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

import {ENGINE} from '../engine';
import {Fill, FillAttrs} from '../kernel_names';
import {NamedAttrMap} from '../kernel_registry';
import {Tensor} from '../tensor';
import {DataType, Rank, ShapeMap} from '../types';

/**
 * Creates a `tf.Tensor` filled with a scalar value.
 *
 * ```js
 * tf.fill([2, 2], 4).print();
 * ```
 *
 * @param shape An array of integers defining the output tensor shape.
 * @param value The scalar value to fill the tensor with.
 * @param dtype The type of an element in the resulting tensor. Defaults to
 * 'float'.
 */
/** @doc {heading: 'Tensors', subheading: 'Creation'} */
function fill<R extends Rank>(
    shape: ShapeMap[R], value: number|string, dtype?: DataType): Tensor<R> {
  const attrs: FillAttrs = {shape, value, dtype};

  return ENGINE.runKernelFunc(
      backend => backend.fill(shape, value, dtype), {}, null, Fill,
      attrs as {} as NamedAttrMap);
}

export {fill};
