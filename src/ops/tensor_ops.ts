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

import {ENV} from '../environment';
import {Scalar, Tensor, Tensor1D, Tensor2D, Tensor3D, Tensor4D, Tensor5D, Tensor6D} from '../tensor';
import {convertToTensor, inferShape} from '../tensor_util_env';
import {TensorLike, TensorLike1D, TensorLike2D, TensorLike3D, TensorLike4D, TensorLike5D, TensorLike6D} from '../types';
import {DataType, Rank, ShapeMap} from '../types';
import {assert, assertNonNull, assertNonNegativeIntegerDimensions, flatten, inferDtype, isTypedArray, makeOnesTypedArray, makeZerosTypedArray, sizeFromShape, toTypedArray} from '../util';

import {complex, real, imag} from './complex_ops';
import {op} from './operation';

/**
 * Creates a `tf.Tensor` with the provided values, shape and dtype.
 *
 * ```js
 * // Pass an array of values to create a vector.
 * tf.tensor([1, 2, 3, 4]).print();
 * ```
 *
 * ```js
 * // Pass a nested array of values to make a matrix or a higher
 * // dimensional tensor.
 * tf.tensor([[1, 2], [3, 4]]).print();
 * ```
 *
 * ```js
 * // Pass a flat array and specify a shape yourself.
 * tf.tensor([1, 2, 3, 4], [2, 2]).print();
 * ```
 *
 * @param values The values of the tensor. Can be nested array of numbers,
 *     or a flat array, or a `TypedArray`.
 * @param shape The shape of the tensor. Optional. If not provided,
 *   it is inferred from `values`.
 * @param dtype The data type.
 */
/** @doc {heading: 'Tensors', subheading: 'Creation'} */
function tensor<R extends Rank>(
    values: TensorLike, shape?: ShapeMap[R], dtype?: DataType): Tensor<R> {
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
  const inferredShape = inferShape(values);
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
  values = dtype !== 'string' ? toTypedArray(values, dtype, ENV.get('DEBUG')) :
                                flatten(values as string[]) as string[];
  return Tensor.make(shape, {values}, dtype);
}

/**
 * Creates rank-0 `tf.Tensor` (scalar) with the provided value and dtype.
 *
 * The same functionality can be achieved with `tf.tensor`, but in general
 * we recommend using `tf.scalar` as it makes the code more readable.
 *
 * ```js
 * tf.scalar(3.14).print();
 * ```
 *
 * @param value The value of the scalar.
 * @param dtype The data type.
 */
/** @doc {heading: 'Tensors', subheading: 'Creation'} */
function scalar(value: number|boolean|string, dtype?: DataType): Scalar {
  if ((isTypedArray(value) || Array.isArray(value)) && dtype !== 'complex64') {
    throw new Error(
        'Error creating a new Scalar: value must be a primitive ' +
        '(number|boolean|string)');
  }
  return tensor(value, [], dtype);
}

/**
 * Creates rank-1 `tf.Tensor` with the provided values, shape and dtype.
 *
 * The same functionality can be achieved with `tf.tensor`, but in general
 * we recommend using `tf.tensor1d` as it makes the code more readable.
 *
 * ```js
 * tf.tensor1d([1, 2, 3]).print();
 * ```
 *
 * @param values The values of the tensor. Can be array of numbers,
 *     or a `TypedArray`.
 * @param dtype The data type.
 */
/** @doc {heading: 'Tensors', subheading: 'Creation'} */
function tensor1d(values: TensorLike1D, dtype?: DataType): Tensor1D {
  assertNonNull(values);
  const inferredShape = inferShape(values);
  if (inferredShape.length !== 1) {
    throw new Error('tensor1d() requires values to be a flat/TypedArray');
  }
  return tensor(values, inferredShape as [number], dtype);
}

/**
 * Creates rank-2 `tf.Tensor` with the provided values, shape and dtype.
 *
 * The same functionality can be achieved with `tf.tensor`, but in general
 * we recommend using `tf.tensor2d` as it makes the code more readable.
 *
 *  ```js
 * // Pass a nested array.
 * tf.tensor2d([[1, 2], [3, 4]]).print();
 * ```
 * ```js
 * // Pass a flat array and specify a shape.
 * tf.tensor2d([1, 2, 3, 4], [2, 2]).print();
 * ```
 *
 * @param values The values of the tensor. Can be nested array of numbers,
 *     or a flat array, or a `TypedArray`.
 * @param shape The shape of the tensor. If not provided, it is inferred from
 *     `values`.
 * @param dtype The data type.
 */
/** @doc {heading: 'Tensors', subheading: 'Creation'} */
function tensor2d(
    values: TensorLike2D, shape?: [number, number],
    dtype?: DataType): Tensor2D {
  assertNonNull(values);
  if (shape != null && shape.length !== 2) {
    throw new Error('tensor2d() requires shape to have two numbers');
  }
  const inferredShape = inferShape(values);
  if (inferredShape.length !== 2 && inferredShape.length !== 1) {
    throw new Error(
        'tensor2d() requires values to be number[][] or flat/TypedArray');
  }
  if (inferredShape.length === 1 && shape == null) {
    throw new Error(
        'tensor2d() requires shape to be provided when `values` ' +
        'are a flat/TypedArray');
  }
  shape = shape || inferredShape as [number, number];
  return tensor(values, shape, dtype);
}

/**
 * Creates rank-3 `tf.Tensor` with the provided values, shape and dtype.
 *
 * The same functionality can be achieved with `tf.tensor`, but in general
 * we recommend using `tf.tensor3d` as it makes the code more readable.
 *
 *  ```js
 * // Pass a nested array.
 * tf.tensor3d([[[1], [2]], [[3], [4]]]).print();
 * ```
 * ```js
 * // Pass a flat array and specify a shape.
 * tf.tensor3d([1, 2, 3, 4], [2, 2, 1]).print();
 * ```
 *
 * @param values The values of the tensor. Can be nested array of numbers,
 *     or a flat array, or a `TypedArray`.
 * @param shape The shape of the tensor. If not provided,  it is inferred from
 *     `values`.
 * @param dtype The data type.
 */
/** @doc {heading: 'Tensors', subheading: 'Creation'} */
function tensor3d(
    values: TensorLike3D, shape?: [number, number, number],
    dtype?: DataType): Tensor3D {
  assertNonNull(values);
  if (shape != null && shape.length !== 3) {
    throw new Error('tensor3d() requires shape to have three numbers');
  }
  const inferredShape = inferShape(values);
  if (inferredShape.length !== 3 && inferredShape.length !== 1) {
    throw new Error(
        'tensor3d() requires values to be number[][][] or flat/TypedArray');
  }
  if (inferredShape.length === 1 && shape == null) {
    throw new Error(
        'tensor3d() requires shape to be provided when `values` ' +
        'are a flat array');
  }
  shape = shape || inferredShape as [number, number, number];
  return tensor(values, shape, dtype);
}

/**
 * Creates rank-4 `tf.Tensor` with the provided values, shape and dtype.
 *
 * The same functionality can be achieved with `tf.tensor`, but in general
 * we recommend using `tf.tensor4d` as it makes the code more readable.
 *
 *  ```js
 * // Pass a nested array.
 * tf.tensor4d([[[[1], [2]], [[3], [4]]]]).print();
 * ```
 * ```js
 * // Pass a flat array and specify a shape.
 * tf.tensor4d([1, 2, 3, 4], [1, 2, 2, 1]).print();
 * ```
 *
 * @param values The values of the tensor. Can be nested array of numbers,
 *     or a flat array, or a `TypedArray`.
 * @param shape The shape of the tensor. Optional. If not provided,
 *   it is inferred from `values`.
 * @param dtype The data type.
 */
/** @doc {heading: 'Tensors', subheading: 'Creation'} */
function tensor4d(
    values: TensorLike4D, shape?: [number, number, number, number],
    dtype?: DataType): Tensor4D {
  assertNonNull(values);
  if (shape != null && shape.length !== 4) {
    throw new Error('tensor4d() requires shape to have four numbers');
  }
  const inferredShape = inferShape(values);
  if (inferredShape.length !== 4 && inferredShape.length !== 1) {
    throw new Error(
        'tensor4d() requires values to be number[][][][] or flat/TypedArray');
  }
  if (inferredShape.length === 1 && shape == null) {
    throw new Error(
        'tensor4d() requires shape to be provided when `values` ' +
        'are a flat array');
  }
  shape = shape || inferredShape as [number, number, number, number];
  return tensor(values, shape, dtype);
}

/**
 * Creates rank-5 `tf.Tensor` with the provided values, shape and dtype.
 *
 * The same functionality can be achieved with `tf.tensor`, but in general
 * we recommend using `tf.tensor5d` as it makes the code more readable.
 *
 *  ```js
 * // Pass a nested array.
 * tf.tensor5d([[[[[1], [2]], [[3], [4]]]]]).print();
 * ```
 * ```js
 * // Pass a flat array and specify a shape.
 * tf.tensor5d([1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 2, 2, 1]).print();
 * ```
 *
 * @param values The values of the tensor. Can be nested array of numbers,
 *     or a flat array, or a `TypedArray`.
 * @param shape The shape of the tensor. Optional. If not provided,
 *   it is inferred from `values`.
 * @param dtype The data type.
 */
/** @doc {heading: 'Tensors', subheading: 'Creation'} */
function tensor5d(
    values: TensorLike5D, shape?: [number, number, number, number, number],
    dtype?: DataType): Tensor5D {
  assertNonNull(values);
  if (shape != null && shape.length !== 5) {
    throw new Error('tensor5d() requires shape to have five numbers');
  }
  const inferredShape = inferShape(values);
  if (inferredShape.length !== 5 && inferredShape.length !== 1) {
    throw new Error(
        'tensor5d() requires values to be ' +
        'number[][][][][] or flat/TypedArray');
  }
  if (inferredShape.length === 1 && shape == null) {
    throw new Error(
        'tensor5d() requires shape to be provided when `values` ' +
        'are a flat array');
  }
  shape = shape || inferredShape as [number, number, number, number, number];
  return tensor(values, shape, dtype);
}

/**
 * Creates rank-6 `tf.Tensor` with the provided values, shape and dtype.
 *
 * The same functionality can be achieved with `tf.tensor`, but in general
 * we recommend using `tf.tensor6d` as it makes the code more readable.
 *
 *  ```js
 * // Pass a nested array.
 * tf.tensor6d([[[[[[1],[2]],[[3],[4]]],[[[5],[6]],[[7],[8]]]]]]).print();
 * ```
 * ```js
 * // Pass a flat array and specify a shape.
 * tf.tensor6d([1, 2, 3, 4, 5, 6, 7, 8], [1, 1, 2, 2, 2, 1]).print();
 * ```
 *
 * @param values The values of the tensor. Can be nested array of numbers,
 *     or a flat array, or a `TypedArray`.
 * @param shape The shape of the tensor. Optional. If not provided,
 *   it is inferred from `values`.
 * @param dtype The data type.
 */
/** @doc {heading: 'Tensors', subheading: 'Creation'} */
function tensor6d(
    values: TensorLike6D,
    shape?: [number, number, number, number, number, number],
    dtype?: DataType): Tensor6D {
  assertNonNull(values);
  if (shape != null && shape.length !== 6) {
    throw new Error('tensor6d() requires shape to have six numbers');
  }
  const inferredShape = inferShape(values);
  if (inferredShape.length !== 6 && inferredShape.length !== 1) {
    throw new Error(
        'tensor6d() requires values to be number[][][][][][] or ' +
        'flat/TypedArray');
  }
  if (inferredShape.length === 1 && shape == null) {
    throw new Error(
        'tensor6d() requires shape to be provided when `values` ' +
        'are a flat array');
  }
  shape = shape ||
      inferredShape as [number, number, number, number, number, number];
  return tensor(values, shape, dtype);
}

/**
 * Creates a `tf.Tensor` with all elements set to 1.
 *
 * ```js
 * tf.ones([2, 2]).print();
 * ```
 *
 * @param shape An array of integers defining the output tensor shape.
 * @param dtype The type of an element in the resulting tensor. Defaults to
 *     'float'.
 */
/** @doc {heading: 'Tensors', subheading: 'Creation'} */
function ones<R extends Rank>(
    shape: ShapeMap[R], dtype: DataType = 'float32'): Tensor<R> {
  if (dtype === 'complex64') {
    const real = ones(shape, 'float32');
    const imag = ones(shape, 'float32');
    return complex(real, imag);
  }
  const values = makeOnesTypedArray(sizeFromShape(shape), dtype);
  return Tensor.make(shape, {values}, dtype);
}

/**
 * Creates a `tf.Tensor` with all elements set to 0.
 *
 * ```js
 * tf.zeros([2, 2]).print();
 * ```
 *
 * @param shape An array of integers defining the output tensor shape.
 * @param dtype The type of an element in the resulting tensor. Can
 *     be 'float32', 'int32' or 'bool'. Defaults to 'float'.
 */
/** @doc {heading: 'Tensors', subheading: 'Creation'} */
function zeros<R extends Rank>(
    shape: ShapeMap[R], dtype: DataType = 'float32'): Tensor<R> {
  if (dtype === 'complex64') {
    const real = zeros(shape, 'float32');
    const imag = zeros(shape, 'float32');
    return complex(real, imag);
  }
  const values = makeZerosTypedArray(sizeFromShape(shape), dtype);
  return Tensor.make(shape, {values}, dtype);
}

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
  return ENV.engine.runKernel(backend =>
    backend.fill(shape, value, dtype), {});
}

/**
 * Creates a `tf.Tensor` with all elements set to 1 with the same shape as the
 * given tensor.
 *
 * ```js
 * const x = tf.tensor([1, 2]);
 * tf.onesLike(x).print();
 * ```
 * @param x A tensor.
 */
/** @doc {heading: 'Tensors', subheading: 'Creation'} */
function onesLike_<T extends Tensor>(x: T|TensorLike): T {
  const $x = convertToTensor(x, 'x', 'onesLike');
  if ($x.dtype === 'complex64') {
    const r = onesLike(real($x));
    const i = zerosLike(imag($x));
    return complex(r, i);
  }
  return ENV.engine.runKernel(backend => backend.onesLike($x), {$x}, null) as T;
}

/**
 * Creates a `tf.Tensor` with all elements set to 0 with the same shape as the
 * given tensor.
 *
 * ```js
 * const x = tf.tensor([1, 2]);
 * tf.zerosLike(x).print();
 * ```
 *
 * @param x The tensor of required shape.
 */
/** @doc {heading: 'Tensors', subheading: 'Creation'} */
function zerosLike_<T extends Tensor>(x: T|TensorLike): T {
  const $x = convertToTensor(x, 'x', 'zerosLike');
  return ENV.engine.runKernel(backend => backend.zerosLike($x), {$x}, null) as
      T;
}

/**
 * Return an evenly spaced sequence of numbers over the given interval.
 *
 * ```js
 * tf.linspace(0, 9, 10).print();
 * ```
 * @param start The start value of the sequence.
 * @param stop The end value of the sequence.
 * @param num The number of values to generate.
 */
/** @doc {heading: 'Tensors', subheading: 'Creation'} */
function linspace(start: number, stop: number, num: number): Tensor1D {
  if (num === 0) {
    throw new Error('Cannot request zero samples');
  }

  const step = (stop - start) / (num - 1);

  const values = makeZerosTypedArray(num, 'float32');
  values[0] = start;
  for (let i = 1; i < values.length; i++) {
    values[i] = values[i - 1] + step;
  }

  return tensor1d(values, 'float32');
}

/**
 * Creates a new `tf.Tensor1D` filled with the numbers in the range provided.
 *
 * The tensor is a is half-open interval meaning it includes start, but
 * excludes stop. Decrementing ranges and negative step values are also
 * supported.
 *
 * ```js
 * tf.range(0, 9, 2).print();
 * ```
 *
 * @param start An integer start value
 * @param stop An integer stop value
 * @param step An integer increment (will default to 1 or -1)
 * @param dtype The data type of the output tensor. Defaults to 'float32'.
 */
/** @doc {heading: 'Tensors', subheading: 'Creation'} */
function range(
    start: number, stop: number, step = 1,
    dtype: 'float32'|'int32' = 'float32'): Tensor1D {
  if (step === 0) {
    throw new Error('Cannot have a step of zero');
  }

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
}

export {
  fill,
  linspace,
  ones,
  range,
  scalar,
  tensor,
  tensor1d,
  tensor2d,
  tensor3d,
  tensor4d,
  tensor5d,
  tensor6d,
  zeros
};

export const onesLike = op({onesLike_});
export const zerosLike = op({zerosLike_});
