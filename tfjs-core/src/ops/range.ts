/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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
import {Range, RangeAttrs} from '../kernel_names';
import {NamedAttrMap} from '../kernel_registry';
import {Tensor1D} from '../tensor';

/**
 * Creates a new `tf.Tensor1D` filled with the numbers in the range provided.
 *
 * The tensor is a is half-open interval meaning it includes start, but
 * excludes stop. Decrementing ranges and negative step values are also
 * supported.sv
 *
 *
 * ```js
 * tf.range(0, 9, 2).print();
 * ```
 *
 * @param start An integer start value
 * @param stop An integer stop value
 * @param step An integer increment (will default to 1 or -1)
 * @param dtype The data type of the output tensor. Defaults to 'float32'.
 *
 * @doc {heading: 'Tensors', subheading: 'Creation'}
 */
export function range(
    start: number, stop: number, step = 1,
    dtype: 'float32'|'int32' = 'float32'): Tensor1D {
  if (step === 0) {
    throw new Error('Cannot have a step of zero');
  }

  const attrs: RangeAttrs = {start, stop, step, dtype};

  return ENGINE.runKernel(Range, {} /* inputs */, attrs as {} as NamedAttrMap);
}
