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

import {ENGINE, ForwardFunc} from '../engine';
import {Range, RangeAttrs} from '../kernel_names';
import {NamedAttrMap} from '../kernel_registry';
import {Tensor, Tensor1D} from '../tensor';
import {makeZerosTypedArray} from '../util';

import {tensor1d} from './tensor1d';
import {zeros} from './zeros';

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

  const forward: ForwardFunc<Tensor> = () => {
    const sameStartStop = start === stop;
    const increasingRangeNegativeStep = start < stop && step < 0;
    const decreasingRangePositiveStep = stop < start && step > 1;

    if (sameStartStop || increasingRangeNegativeStep ||
        decreasingRangePositiveStep) {
      return zeros([0], dtype);
    }

    const numElements = Math.abs(Math.ceil((stop - start) / step));
    const values = makeZerosTypedArray(numElements, dtype);

    if (stop < start && step === 1) {
      // Auto adjust the step's sign if it hasn't been set
      // (or was set to 1)
      step = -1;
    }

    values[0] = start;
    for (let i = 1; i < values.length; i++) {
      values[i] = values[i - 1] + step;
    }

    return tensor1d(values, dtype);
  };

  const attrs: RangeAttrs = {start, stop, step, dtype};

  return ENGINE.runKernelFunc(
             forward, {} /* inputs */, null /* grad */, Range,
             attrs as {} as NamedAttrMap) as Tensor1D;
}
