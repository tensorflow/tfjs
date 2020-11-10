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
import {LinSpace, LinSpaceAttrs} from '../kernel_names';
import {NamedAttrMap} from '../kernel_registry';
import {Tensor1D} from '../tensor';

/**
 * Return an evenly spaced sequence of numbers over the given interval.
 *
 * ```js
 * tf.linspace(0, 9, 10).print();
 * ```
 * @param start The start value of the sequence.
 * @param stop The end value of the sequence.
 * @param num The number of values to generate.
 *
 * @doc {heading: 'Tensors', subheading: 'Creation'}
 */
export function linspace(start: number, stop: number, num: number): Tensor1D {
  if (num <= 0) {
    throw new Error('The number of values should be positive.');
  }

  const attrs: LinSpaceAttrs = {start, stop, num};
  return ENGINE.runKernelFunc(
      backend => backend.linspace(start, stop, num), {} /* inputs */,
      null /* grad */, LinSpace, attrs as {} as NamedAttrMap);
}
