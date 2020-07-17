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
import {Tensor} from '../tensor';
import {TensorLike, TypedArray} from '../types';
import {DataType} from '../types';
import {assert, assertNonNegativeIntegerDimensions, flatten, inferDtype, isTypedArray, sizeFromShape, toTypedArray} from '../util';

/** This is shared code across all tensor creation methods. */
export function makeTensor(
    values: TensorLike, shape: number[], inferredShape: number[],
    dtype?: DataType): Tensor {
  if (dtype == null) {
    dtype = inferDtype(values);
  }
  if (dtype === 'complex64') {
    throw new Error(
        `Cannot construct a complex64 tensor directly. ` +
        `Please use tf.complex(real, imag).`);
  }
  if (!isTypedArray(values) && !Array.isArray(values) &&
      typeof values !== 'number' && typeof values !== 'boolean' &&
      typeof values !== 'string') {
    throw new Error(
        'values passed to tensor(values) must be a number/boolean/string or ' +
        'an array of numbers/booleans/strings, or a TypedArray');
  }
  if (shape != null) {
    assertNonNegativeIntegerDimensions(shape);

    const providedSize = sizeFromShape(shape);
    const inferredSize = sizeFromShape(inferredShape);
    assert(
        providedSize === inferredSize,
        () =>
            `Based on the provided shape, [${shape}], the tensor should have ` +
            `${providedSize} values but has ${inferredSize}`);

    for (let i = 0; i < inferredShape.length; ++i) {
      const inferred = inferredShape[i];
      const flatDimsDontMatch = i === inferredShape.length - 1 ?
          inferred !== sizeFromShape(shape.slice(i)) :
          true;
      assert(
          inferredShape[i] === shape[i] || !flatDimsDontMatch,
          () => `Error creating a new Tensor. Inferred shape ` +
              `(${inferredShape}) does not match the provided ` +
              `shape (${shape}). `);
    }
  }

  if (!isTypedArray(values) && !Array.isArray(values)) {
    values = [values] as number[];
  }

  shape = shape || inferredShape;
  values = dtype !== 'string' ?
      toTypedArray(values, dtype) :
      flatten(values as string[], [], true) as string[];
  return ENGINE.makeTensor(values as TypedArray, shape, dtype);
}
