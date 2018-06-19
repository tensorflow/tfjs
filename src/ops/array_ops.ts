/**
 * @license
 * Copyright 2018 Google Inc. All Rights Reserved.
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

import {doc} from '../doc';
// import {ForwardFunc} from '../engine';
import {ENV} from '../environment';
// tslint:disable-next-line:max-line-length
import {Scalar, Tensor, Tensor1D, Tensor2D, Tensor3D, Tensor4D, Tensor5D, Tensor6D, TensorBuffer} from '../tensor';
import * as tensor_util from '../tensor_util';
// tslint:disable-next-line:max-line-length
import {ArrayData, DataType, DataTypeMap, Rank, ShapeMap, TensorLike, TensorLike1D, TensorLike2D, TensorLike3D, TensorLike4D, TensorLike5D, TensorLike6D, TypedArray} from '../types';
import * as util from '../util';
import * as axis_util from './axis_util';
// tslint:disable-next-line:max-line-length
import {getAxesPermutation, getInnerMostAxes, parseAxisParam} from './axis_util';
import {ConcatOps} from './concat';
import {operation} from './operation';
import {MPRandGauss} from './rand';
import {SegmentOps} from './segment_ops';

export class ArrayOps {
  /**
   * Creates a `Tensor` with the provided values, shape and dtype.
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
  @doc({heading: 'Tensors', subheading: 'Creation'})
  static tensor<R extends Rank>(
      values: TensorLike, shape?: ShapeMap[R], dtype: DataType = 'float32'):
      Tensor<R> {
    const inferredShape = util.inferShape(values);
    if (shape != null && inferredShape.length !== 1) {
      util.assertShapesMatch(
          shape, inferredShape,
          `Error creating a new Tensor. ` +
              `Inferred shape (${inferredShape}) does not match the ` +
              `provided shape (${shape}). `);
    }
    if (!util.isTypedArray(values) && !Array.isArray(values)) {
      values = [values] as number[];
    }
    shape = shape || inferredShape;
    return Tensor.make(
        shape, {values: toTypedArray(values as ArrayData<DataType>, dtype)},
        dtype);
  }

  /**
   * Creates rank-0 `Tensor` (scalar) with the provided value and dtype.
   *
   * The same functionality can be achieved with `tensor`, but in general
   * we recommend using `scalar` as it makes the code more readable.
   *
   * ```js
   * tf.scalar(3.14).print();
   * ```
   *
   * @param value The value of the scalar.
   * @param dtype The data type.
   */
  @doc({heading: 'Tensors', subheading: 'Creation'})
  static scalar(value: number|boolean, dtype: DataType = 'float32'): Scalar {
    if (util.isTypedArray(value) || Array.isArray(value)) {
      throw new Error(
          'Error creating a new Scalar: value must be a primitive ' +
          '(number|boolean)');
    }
    return ArrayOps.tensor(value, [], dtype);
  }

  /**
   * Creates rank-1 `Tensor` with the provided values, shape and dtype.
   *
   * The same functionality can be achieved with `tensor`, but in general
   * we recommend using `tensor1d` as it makes the code more readable.
   *
   * ```js
   * tf.tensor1d([1, 2, 3]).print();
   * ```
   *
   * @param values The values of the tensor. Can be array of numbers,
   *     or a `TypedArray`.
   * @param dtype The data type.
   */
  @doc({heading: 'Tensors', subheading: 'Creation'})
  static tensor1d(values: TensorLike1D, dtype: DataType = 'float32'): Tensor1D {
    const inferredShape = util.inferShape(values);
    if (inferredShape.length !== 1) {
      throw new Error('tensor1d() requires values to be a flat/TypedArray');
    }
    return ArrayOps.tensor(values, inferredShape as [number], dtype);
  }

  /**
   * Creates rank-2 `Tensor` with the provided values, shape and dtype.
   *
   * The same functionality can be achieved with `tensor`, but in general
   * we recommend using `tensor2d` as it makes the code more readable.
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
  @doc({heading: 'Tensors', subheading: 'Creation'})
  static tensor2d(
      values: TensorLike2D, shape?: [number, number],
      dtype: DataType = 'float32'): Tensor2D {
    if (shape != null && shape.length !== 2) {
      throw new Error('tensor2d() requires shape to have two numbers');
    }
    const inferredShape = util.inferShape(values);
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
    return ArrayOps.tensor(values, shape, dtype);
  }

  /**
   * Creates rank-3 `Tensor` with the provided values, shape and dtype.
   *
   * The same functionality can be achieved with `tensor`, but in general
   * we recommend using `tensor3d` as it makes the code more readable.
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
  @doc({heading: 'Tensors', subheading: 'Creation'})
  static tensor3d(
      values: TensorLike3D, shape?: [number, number, number],
      dtype: DataType = 'float32'): Tensor3D {
    if (shape != null && shape.length !== 3) {
      throw new Error('tensor3d() requires shape to have three numbers');
    }
    const inferredShape = util.inferShape(values);
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
    return ArrayOps.tensor(values, shape, dtype);
  }

  /**
   * Creates rank-4 `Tensor` with the provided values, shape and dtype.
   *
   * The same functionality can be achieved with `tensor`, but in general
   * we recommend using `tensor4d` as it makes the code more readable.
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
  @doc({heading: 'Tensors', subheading: 'Creation'})
  static tensor4d(
      values: TensorLike4D, shape?: [number, number, number, number],
      dtype: DataType = 'float32'): Tensor4D {
    if (shape != null && shape.length !== 4) {
      throw new Error('tensor4d() requires shape to have four numbers');
    }
    const inferredShape = util.inferShape(values);
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
    return ArrayOps.tensor(values, shape, dtype);
  }

  /**
   * Creates rank-5 `Tensor` with the provided values, shape and dtype.
   *
   * The same functionality can be achieved with `tensor`, but in general
   * we recommend using `tensor5d` as it makes the code more readable.
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
  @doc({heading: 'Tensors', subheading: 'Creation'})
  static tensor5d(
      values: TensorLike5D, shape?: [number, number, number, number, number],
      dtype: DataType = 'float32'): Tensor5D {
    if (shape != null && shape.length !== 5) {
      throw new Error('tensor5d() requires shape to have five numbers');
    }
    const inferredShape = util.inferShape(values);
    if (inferredShape.length !== 5 && inferredShape.length !== 1) {
      throw new Error('tensor5d() requires values to be \
           number[][][][][] or flat/TypedArray');
    }
    if (inferredShape.length === 1 && shape == null) {
      throw new Error(
          'tensor5d() requires shape to be provided when `values` ' +
          'are a flat array');
    }
    shape = shape || inferredShape as [number, number, number, number, number];
    return ArrayOps.tensor(values, shape, dtype);
  }

  /**
   * Creates rank-6 `Tensor` with the provided values, shape and dtype.
   *
   * The same functionality can be achieved with `tensor`, but in general
   * we recommend using `tensor6d` as it makes the code more readable.
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
  @doc({heading: 'Tensors', subheading: 'Creation'})
  static tensor6d(
      values: TensorLike6D,
      shape?: [number, number, number, number, number, number],
      dtype: DataType = 'float32'): Tensor6D {
    if (shape != null && shape.length !== 6) {
      throw new Error('tensor6d() requires shape to have six numbers');
    }
    const inferredShape = util.inferShape(values);
    if (inferredShape.length !== 6 && inferredShape.length !== 1) {
      throw new Error(
          'tensor6d() requires values to be number[][][][] or flat/TypedArray');
    }
    if (inferredShape.length === 1 && shape == null) {
      throw new Error(
          'tensor6d() requires shape to be provided when `values` ' +
          'are a flat array');
    }
    shape = shape ||
        inferredShape as [number, number, number, number, number, number];
    return ArrayOps.tensor(values, shape, dtype);
  }

  /**
   * Creates a `Tensor` with all elements set to 1.
   *
   * ```js
   * tf.ones([2, 2]).print();
   * ```
   *
   * @param shape An array of integers defining the output tensor shape.
   * @param dtype The type of an element in the resulting tensor. Defaults to
   *     'float'.
   */
  @doc({heading: 'Tensors', subheading: 'Creation'})
  @operation
  static ones<R extends Rank>(shape: ShapeMap[R], dtype: DataType = 'float32'):
      Tensor<R> {
    const values = makeOnesTypedArray(util.sizeFromShape(shape), dtype);
    return Tensor.make(shape, {values}, dtype);
  }

  /**
   * Creates a `Tensor` with all elements set to 0.
   *
   * ```js
   * tf.zeros([2, 2]).print();
   * ```
   *
   * @param shape An array of integers defining the output tensor shape.
   * @param dtype The type of an element in the resulting tensor. Can
   *     be 'float32', 'int32' or 'bool'. Defaults to 'float'.
   */
  @doc({heading: 'Tensors', subheading: 'Creation'})
  @operation
  static zeros<R extends Rank>(shape: ShapeMap[R], dtype: DataType = 'float32'):
      Tensor<R> {
    const values = makeZerosTypedArray(util.sizeFromShape(shape), dtype);
    return Tensor.make(shape, {values}, dtype);
  }

  /**
   * Creates a `Tensor` filled with a scalar value.
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
  @doc({heading: 'Tensors', subheading: 'Creation'})
  @operation
  static fill<R extends Rank>(
      shape: ShapeMap[R], value: number, dtype: DataType = 'float32'):
      Tensor<R> {
    const values =
        util.getTypedArrayFromDType(dtype, util.sizeFromShape(shape));
    values.fill(value);
    return Tensor.make(shape, {values}, dtype);
  }

  /**
   * Creates a `Tensor` with all elements set to 1 with the same shape as the
   * given tensor.
   *
   * ```js
   * const x = tf.tensor([1, 2]);
   * tf.onesLike(x).print();
   * ```
   * @param x A tensor.
   */
  @doc({heading: 'Tensors', subheading: 'Creation'})
  @operation
  static onesLike<T extends Tensor>(x: T): T {
    util.assertArgumentsAreTensors({x}, 'onesLike');
    return ArrayOps.ones(x.shape, x.dtype) as T;
  }

  /**
   * Creates a `Tensor` with all elements set to 0 with the same shape as the
   * given tensor.
   *
   * ```js
   * const x = tf.tensor([1, 2]);
   * tf.zerosLike(x).print();
   * ```
   *
   * @param x The tensor of required shape.
   */
  @doc({heading: 'Tensors', subheading: 'Creation'})
  @operation
  static zerosLike<T extends Tensor>(x: T): T {
    util.assertArgumentsAreTensors({x}, 'zerosLike');
    return ArrayOps.zeros(x.shape, x.dtype) as T;
  }

  /**
   * Creates a new tensor with the same values and shape as the specified
   * tensor.
   *
   * ```js
   * const x = tf.tensor([1, 2]);
   *
   * x.clone().print();
   * ```
   *
   * @param x The tensor to clone.
   */
  @doc({heading: 'Tensors', subheading: 'Creation'})
  @operation
  static clone<T extends Tensor>(x: T): T {
    util.assertArgumentsAreTensors({x}, 'clone');
    const der = (dy: T) => {
      return {x: () => dy.toFloat()};
    };

    return ENV.engine.runKernel(
               backend =>
                   Tensor.make(x.shape, {dataId: x.dataId}, x.dtype) as T,
               {x}, der) as T;
  }

  /**
   * Create an identity matrix.
   *
   * @param numRows Number of rows.
   * @param numColumns Number of columns. Defaults to `numRows`.
   * @param batchShape If provided, will add the batch shape to the beginning
   *   of the shape of the returned `Tensor` by repeating the identity
   *   matrix.
   * @param dtype Data type.
   * @returns Identity matrix of the specified size and data type, possibly
   *   with batch repetition if `batchShape` is specified.
   */
  @doc({heading: 'Tensors', subheading: 'Creation'})
  @operation
  static eye(
      numRows: number, numColumns?: number,
      batchShape?: [number]|[number, number],
      dtype: DataType = 'float32'): Tensor2D {
    if (numColumns == null) {
      numColumns = numRows;
    }
    const buffer = ArrayOps.buffer([numRows, numColumns], dtype);
    const n = numRows <= numColumns ? numRows : numColumns;
    for (let i = 0; i < n; ++i) {
      buffer.set(1, i, i);
    }
    const out = buffer.toTensor().as2D(numRows, numColumns);
    if (batchShape == null) {
      return out;
    } else {
      if (batchShape.length === 1) {
        return ArrayOps.tile(
            ArrayOps.expandDims(out, 0), [batchShape[0], 1, 1]);
      } else if (batchShape.length === 2) {
        return ArrayOps.tile(
            ArrayOps.expandDims(ArrayOps.expandDims(out, 0), 0),
            [batchShape[0], batchShape[1], 1, 1]);
      } else {
        // TODO(cais): Add support for length-3 once Tensor5D is available.
        throw new Error(
            `eye() currently supports only 1D and 2D ` +
            // tslint:disable-next-line:no-any
            `batchShapes, but received ${(batchShape as any).length}D.`);
      }
    }
  }

  /**
   * Creates a `Tensor` with values sampled from a normal distribution.
   *
   * ```js
   * tf.randomNormal([2, 2]).print();
   * ```
   *
   * @param shape An array of integers defining the output tensor shape.
   * @param mean The mean of the normal distribution.
   * @param stdDev The standard deviation of the normal distribution.
   * @param dtype The data type of the output.
   * @param seed The seed for the random number generator.
   */
  @doc({heading: 'Tensors', subheading: 'Creation'})
  @operation
  static randomNormal<R extends Rank>(
      shape: ShapeMap[R], mean = 0, stdDev = 1, dtype?: 'float32'|'int32',
      seed?: number): Tensor<R> {
    if (dtype != null && (dtype as DataType) === 'bool') {
      throw new Error(`Unsupported data type ${dtype}`);
    }
    const randGauss =
        new MPRandGauss(mean, stdDev, dtype, false /* truncated */, seed);
    const res = ArrayOps.buffer(shape, dtype);
    for (let i = 0; i < res.values.length; i++) {
      res.values[i] = randGauss.nextValue();
    }
    return res.toTensor();
  }

  /**
   * Creates a `Tensor` with values sampled from a truncated normal
   * distribution.
   *
   * ```js
   * tf.truncatedNormal([2, 2]).print();
   * ```
   *
   * The generated values follow a normal distribution with specified mean and
   * standard deviation, except that values whose magnitude is more than 2
   * standard deviations from the mean are dropped and re-picked.
   *
   * @param shape An array of integers defining the output tensor shape.
   * @param mean The mean of the normal distribution.
   * @param stdDev The standard deviation of the normal distribution.
   * @param dtype The data type of the output tensor.
   * @param seed The seed for the random number generator.
   */
  @doc({heading: 'Tensors', subheading: 'Creation'})
  @operation
  static truncatedNormal<R extends Rank>(
      shape: ShapeMap[R], mean = 0, stdDev = 1, dtype?: 'float32'|'int32',
      seed?: number): Tensor<R> {
    if (dtype != null && (dtype as DataType) === 'bool') {
      throw new Error(`Unsupported data type ${dtype}`);
    }
    const randGauss =
        new MPRandGauss(mean, stdDev, dtype, true /* truncated */, seed);
    const res = ArrayOps.buffer(shape, dtype);
    for (let i = 0; i < res.values.length; i++) {
      res.values[i] = randGauss.nextValue();
    }
    return res.toTensor();
  }

  /**
   * Creates a `Tensor` with values sampled from a uniform distribution.
   *
   * The generated values follow a uniform distribution in the range [minval,
   * maxval). The lower bound minval is included in the range, while the upper
   * bound maxval is excluded.
   *
   * ```js
   * tf.randomUniform([2, 2]).print();
   * ```
   *
   * @param shape An array of integers defining the output tensor shape.
   * @param minval The lower bound on the range of random values to generate.
   *   Defaults to 0.
   * @param maxval The upper bound on the range of random values to generate.
   *   Defaults to 1.
   * @param dtype The data type of the output tensor. Defaults to 'float32'.
   */
  @doc({heading: 'Tensors', subheading: 'Creation'})
  @operation
  static randomUniform<R extends Rank>(
      shape: ShapeMap[R], minval = 0, maxval = 1, dtype: DataType = 'float32'):
      Tensor<R> {
    const res = ArrayOps.buffer(shape, dtype);
    for (let i = 0; i < res.values.length; i++) {
      res.values[i] = util.randUniform(minval, maxval);
    }
    return res.toTensor();
  }

  /**
   * Creates a `Tensor` with values sampled from a random number generator
   * function defined by the user.
   *
   * @param shape An array of integers defining the output tensor shape.
   * @param randFunction A random number generator function which is called
   * for each element in the output tensor.
   * @param dtype The data type of the output tensor. Defaults to 'float32'.
   */
  @operation
  static rand<R extends Rank>(
      shape: ShapeMap[R], randFunction: () => number, dtype?: DataType):
      Tensor<R> {
    const size = util.sizeFromShape(shape);

    let values = null;
    if (dtype == null || dtype === 'float32') {
      values = new Float32Array(size);
    } else if (dtype === 'int32') {
      values = new Int32Array(size);
    } else if (dtype === 'bool') {
      values = new Uint8Array(size);
    } else {
      throw new Error(`Unknown data type ${dtype}`);
    }

    for (let i = 0; i < size; i++) {
      values[i] = randFunction();
    }
    return Tensor.make(shape, {values}, dtype);
  }

  /**
   * Creates a `Tensor` with values drawn from a multinomial distribution.
   *
   * ```js
   * const probs = tf.tensor([.75, .25]);
   * tf.multinomial(probs, 3).print();
   * ```
   *
   * @param logits 1D array with unnormalized log-probabilities, or
   *     2D array of shape `[batchSize, numOutcomes]`. See the `normalized`
   *     parameter.
   * @param numSamples Number of samples to draw for each row slice.
   * @param seed The seed number.
   * @param normalized Whether the provided `logits` are normalized true
   *     probabilities (sum to 1). Defaults to false.
   * @return 1D array of shape `[numSamples]`, or 2D array of shape
   *     `[batchSize, numSamples]`, depending on the rank of the input.
   */
  @operation
  static multinomial(
      logits: Tensor1D|Tensor2D, numSamples: number, seed?: number,
      normalized = false): Tensor1D|Tensor2D {
    util.assertArgumentsAreTensors({logits}, 'multinomial');
    const numOutcomes = logits.size;
    const origRank = logits.rank;
    if (numOutcomes < 2) {
      throw new Error(
          `Error in multinomial: you need at least 2 outcomes, but got ` +
          `${numOutcomes}.`);
    }
    if (origRank > 2) {
      throw new Error(
          `Rank of probabilities must be 1 or 2, but is ${origRank}`);
    }
    seed = seed || Math.random();
    const logits2D = origRank === 1 ? logits.as2D(1, -1) : logits as Tensor2D;
    const res = ENV.engine.runKernel(
        backend => backend.multinomial(logits2D, normalized, numSamples, seed),
        {logits2D});

    return origRank === 1 ? res.as1D() : res;
  }

  /**
   * Creates a one-hot `Tensor`. The locations represented by `indices` take
   * value `onValue` (defaults to 1), while all other locations take value
   * `offValue` (defaults to 0).
   *
   * ```js
   * tf.oneHot(tf.tensor1d([0, 1], 'int32'), 3).print();
   * ```
   *
   * @param indices `Tensor1D` of indices with dtype `int32`.
   * @param depth The depth of the one hot dimension.
   * @param onValue A number used to fill in output when the index matches
   * the location.
   * @param offValue A number used to fill in the output when the index does
   *     not match the location.
   */
  @doc({heading: 'Tensors', subheading: 'Creation'})
  @operation
  static oneHot(indices: Tensor1D, depth: number, onValue = 1, offValue = 0):
      Tensor2D {
    util.assert(indices.dtype === 'int32', 'Indices must be of dtype `int32`');
    if (depth < 2) {
      throw new Error(`Error in oneHot: depth must be >=2, but it is ${depth}`);
    }
    return ENV.engine.runKernel(
        backend => backend.oneHot(indices, depth, onValue, offValue),
        {indices});
  }

  /**
   * Creates a `Tensor` from an image.
   *
   * ```js
   * const image = new ImageData(1, 1);
   * image.data[0] = 100;
   * image.data[1] = 150;
   * image.data[2] = 200;
   * image.data[3] = 255;
   *
   * tf.fromPixels(image).print();
   * ```
   *
   * @param pixels The input image to construct the tensor from. The
   * supported image types are all 4-channel.
   * @param numChannels The number of channels of the output tensor. A
   * numChannels value less than 4 allows you to ignore channels. Defaults to
   * 3 (ignores alpha channel of input image).
   */
  @doc({heading: 'Tensors', subheading: 'Creation'})
  @operation
  static fromPixels(
      pixels: ImageData|HTMLImageElement|HTMLCanvasElement|HTMLVideoElement,
      numChannels = 3): Tensor3D {
    if (numChannels > 4) {
      throw new Error(
          'Cannot construct Tensor with more than 4 channels from pixels.');
    }
    return ENV.engine.fromPixels(pixels, numChannels);
  }

  /**
   * Draws a `Tensor` of pixel values to a byte array or optionally a
   * canvas.
   *
   * When the dtype of the input is 'float32', we assume values in the range
   * [0-1]. Otherwise, when input is 'int32', we assume values in the range
   * [0-255].
   *
   * Returns a promise that resolves when the canvas has been drawn to.
   *
   * @param img A rank-2 or rank-3 tensor. If rank-2, draws grayscale. If
   *     rank-3, must have depth of 1, 3 or 4. When depth of 1, draws
   * grayscale. When depth of 3, we draw with the first three components of
   * the depth dimension corresponding to r, g, b and alpha = 1. When depth of
   * 4, all four components of the depth dimension correspond to r, g, b, a.
   * @param canvas The canvas to draw to.
   */
  @doc({heading: 'Visualization'})
  static async toPixels(img: Tensor2D|Tensor3D, canvas?: HTMLCanvasElement):
      Promise<Uint8ClampedArray> {
    util.assertArgumentsAreTensors({img}, 'toPixels');

    if (img.rank !== 2 && img.rank !== 3) {
      throw new Error(
          `toPixels only supports rank 2 or 3 tensors, got rank ${img.rank}.`);
    }
    const [height, width] = img.shape.slice(0, 2);
    const depth = img.rank === 2 ? 1 : img.shape[2];

    if (depth > 4 || depth === 2) {
      throw new Error(
          `toPixels only supports depth of size ` +
          `1, 3 or 4 but got ${depth}`);
    }

    const minTensor = img.min();
    const maxTensor = img.max();
    const min = (await minTensor.data())[0];
    const max = (await maxTensor.data())[0];
    minTensor.dispose();
    maxTensor.dispose();
    if (img.dtype === 'float32') {
      if (min < 0 || max > 1) {
        throw new Error(
            `Tensor values for a float32 Tensor must be in the ` +
            `range [0 - 1] but got range [${min} - ${max}].`);
      }
    } else if (img.dtype === 'int32') {
      if (min < 0 || max > 255) {
        throw new Error(
            `Tensor values for a int32 Tensor must be in the ` +
            `range [0 - 255] but got range [${min} - ${max}].`);
      }
    } else {
      throw new Error(
          `Unsupported type for toPixels: ${img.dtype}.` +
          ` Please use float32 or int32 tensors.`);
    }

    const data = await img.data();
    const multiplier = img.dtype === 'float32' ? 255 : 1;
    const bytes = new Uint8ClampedArray(width * height * 4);

    for (let i = 0; i < height * width; ++i) {
      let r, g, b, a;
      if (depth === 1) {
        r = data[i] * multiplier;
        g = data[i] * multiplier;
        b = data[i] * multiplier;
        a = 255;
      } else if (depth === 3) {
        r = data[i * 3] * multiplier;
        g = data[i * 3 + 1] * multiplier;
        b = data[i * 3 + 2] * multiplier;
        a = 255;
      } else if (depth === 4) {
        r = data[i * 4] * multiplier;
        g = data[i * 4 + 1] * multiplier;
        b = data[i * 4 + 2] * multiplier;
        a = data[i * 4 + 3] * multiplier;
      }

      const j = i * 4;
      bytes[j + 0] = Math.round(r);
      bytes[j + 1] = Math.round(g);
      bytes[j + 2] = Math.round(b);
      bytes[j + 3] = Math.round(a);
    }

    if (canvas != null) {
      canvas.width = width;
      canvas.height = height;
      const ctx = canvas.getContext('2d');
      const imageData = new ImageData(bytes, width, height);
      ctx.putImageData(imageData, 0, 0);
    }

    return bytes;
  }

  /**
   * Reshapes a `Tensor` to a given shape.
   *
   * Given a input tensor, returns a new tensor with the same values as the
   * input tensor with shape `shape`.
   *
   * If one component of shape is the special value -1, the size of that
   * dimension is computed so that the total size remains constant. In
   * particular, a shape of [-1] flattens into 1-D. At most one component of
   * shape can be -1.
   *
   * If shape is 1-D or higher, then the operation returns a tensor with shape
   * shape filled with the values of tensor. In this case, the number of
   * elements implied by shape must be the same as the number of elements in
   * tensor.
   *
   * ```js
   * const x = tf.tensor1d([1, 2, 3, 4]);
   * x.reshape([2, 2]).print();
   * ```
   *
   * @param x The input tensor to be reshaped.
   * @param shape An array of integers defining the output tensor shape.
   */
  @doc({heading: 'Tensors', subheading: 'Transformations'})
  @operation
  static reshape<R2 extends Rank>(x: Tensor, shape: ShapeMap[R2]): Tensor<R2> {
    util.assertArgumentsAreTensors({x}, 'reshape');

    shape = util.inferFromImplicitShape(shape, x.size);
    util.assert(
        x.size === util.sizeFromShape(shape),
        'new shape and old shape must have the same number of elements.');

    const grad = (dy: Tensor<R2>) => {
      return {x: () => dy.reshape(x.shape)};
    };
    return ENV.engine.runKernel(
        backend => backend.reshape(x, shape), {x}, grad);
  }

  /**
   * Removes dimensions of size 1 from the shape of a `Tensor`.
   *
   * ```js
   * const x = tf.tensor([1, 2, 3, 4], [1, 1, 4]);
   * x.squeeze().print();
   * ```
   *
   * @param x The input tensor to be squeezed.
   * @param axis An optional list of numbers. If specified, only
   *     squeezes the dimensions listed. The dimension index starts at 0. It
   * is an error to squeeze a dimension that is not 1.
   */
  @doc({heading: 'Tensors', subheading: 'Transformations'})
  static squeeze<T extends Tensor>(x: Tensor, axis?: number[]): T {
    util.assertArgumentsAreTensors({x}, 'squeeze');
    return ArrayOps.reshape(x, util.squeezeShape(x.shape, axis).newShape) as T;
  }

  /**
   * Casts a `Tensor` to a new dtype.
   *
   * ```js
   * const x = tf.tensor1d([1.5, 2.5, 3]);
   * tf.cast(x, 'int32').print();
   * ```
   * @param x The input tensor to be casted.
   * @param dtype The dtype to cast the input tensor to.
   */
  @doc({heading: 'Tensors', subheading: 'Transformations'})
  @operation
  static cast<T extends Tensor>(x: T, dtype: DataType): T {
    util.assertArgumentsAreTensors({x}, 'cast');

    const grad = (dy: T) => {
      return {x: () => dy.clone()};
    };
    return ENV.engine.runKernel(backend => backend.cast(x, dtype), {x}, grad) as
        T;
  }

  /**
   * Construct an tensor by repeating it the number of times given by reps.
   *
   * This operation creates a new tensor by replicating `input` `reps`
   * times. The output tensor's i'th dimension has `input.shape[i] *
   * reps[i]` elements, and the values of `input` are replicated
   * `reps[i]` times along the i'th dimension. For example, tiling
   * `[a, b, c, d]` by `[2]` produces `[a, b, c, d, a, b, c, d]`.
   *
   * ```js
   * const a = tf.tensor1d([1, 2]);
   *
   * a.tile([2]).print();    // or a.tile([2])
   * ```
   *
   * ```js
   * const a = tf.tensor2d([1, 2, 3, 4], [2, 2]);
   *
   * a.tile([1, 2]).print();  // or a.tile([1, 2])
   * ```
   * @param x The tensor to tile.
   * @param reps Determines the number of replications per dimension.
   */
  @doc({heading: 'Tensors', subheading: 'Slicing and Joining'})
  @operation
  static tile<T extends Tensor>(x: T, reps: number[]): T {
    util.assertArgumentsAreTensors({x}, 'tile');

    util.assert(
        x.rank === reps.length,
        `Error in transpose: rank of input ${x.rank} ` +
            `must match length of reps ${reps}.`);
    const grad = (dy: T) => {
      const derX = () => {
        let xGrad = ArrayOps.zerosLike(x);
        // TODO(cais): Maybe reduce memory footprint by avoiding repeated
        // slicing.
        if (x.rank === 1) {
          for (let i = 0; i < reps[0]; ++i) {
            xGrad = xGrad.add(dy.slice([i * x.shape[0]], [x.shape[0]]));
          }
        } else if (x.rank === 2) {
          for (let i = 0; i < reps[0]; ++i) {
            for (let j = 0; j < reps[1]; ++j) {
              xGrad = xGrad.add(dy.slice(
                  [i * x.shape[0], j * x.shape[1]], [x.shape[0], x.shape[1]]));
            }
          }
        } else if (x.rank === 3) {
          for (let i = 0; i < reps[0]; ++i) {
            for (let j = 0; j < reps[1]; ++j) {
              for (let k = 0; k < reps[2]; ++k) {
                xGrad = xGrad.add(dy.slice(
                    [i * x.shape[0], j * x.shape[1], k * x.shape[2]],
                    [x.shape[0], x.shape[1], x.shape[2]]));
              }
            }
          }
        } else if (x.rank === 4) {
          for (let i = 0; i < reps[0]; ++i) {
            for (let j = 0; j < reps[1]; ++j) {
              for (let k = 0; k < reps[2]; ++k) {
                for (let l = 0; l < reps[3]; ++l) {
                  xGrad = xGrad.add(dy.slice(
                      [
                        i * x.shape[0], j * x.shape[1], k * x.shape[2],
                        l * x.shape[3]
                      ],
                      [x.shape[0], x.shape[1], x.shape[2], x.shape[3]]));
                }
              }
            }
          }
        } else {
          throw new Error(
              `Gradient for tile operation is not implemented for rank-` +
              `${x.rank} tensors yet.`);
        }
        return xGrad;
      };
      return {x: derX};
    };
    return ENV.engine.runKernel(backend => backend.tile(x, reps), {x}, grad);
  }

  /**
   * Gather slices from tensor `x`'s axis `axis` according to `indices`.
   *
   * ```js
   * const x = tf.tensor1d([1, 2, 3, 4]);
   * const indices = tf.tensor1d([1, 3, 3], 'int32');
   *
   * x.gather(indices).print();
   * ```
   *
   * ```js
   * const x = tf.tensor2d([1, 2, 3, 4], [2, 2]);
   * const indices = tf.tensor1d([1, 1, 0], 'int32');
   *
   * x.gather(indices).print();
   * ```
   * @param x The input tensor whose slices to be gathered.
   * @param indices The indices of the values to extract.
   * @param axis The axis over which to select values. Defaults to 0.
   */
  @doc({heading: 'Tensors', subheading: 'Slicing and Joining'})
  @operation
  static gather<T extends Tensor>(x: T, indices: Tensor1D, axis = 0): T {
    util.assertArgumentsAreTensors({x, indices}, 'gather');

    util.assert(indices.dtype === 'int32', 'Indices must be of dtype `int32`');
    axis = parseAxisParam(axis, x.shape)[0];
    const grad = (dy: T) => {
      const derX = () => {
        if (axis === 0) {
          return SegmentOps.unsortedSegmentSum(dy, indices, x.shape[axis]);
        }
        const paramsShape = x.shape;
        const indicesSize = indices.size;

        const outerShape = paramsShape.slice(0, axis);
        const outerDims = outerShape.length;
        const innerShape = paramsShape.slice(axis, paramsShape.length).slice(1);
        const innerDims = innerShape.length;

        const outerAxesIndices = arrayRange(0, outerDims);
        const innerAxesIndices =
            arrayRange(outerDims + 1, outerDims + 1 + innerDims);

        const valuesShape =
            arrayConcat([outerShape, [indicesSize], innerShape]);

        const values = dy.reshape(valuesShape);
        const reshapedIndices = indices.reshape([indicesSize]);

        const transposeDims =
            arrayConcat([[outerDims], outerAxesIndices, innerAxesIndices]);
        const valuesTranspose = values.transpose(transposeDims);

        let paramsGrad = SegmentOps.unsortedSegmentSum(
            valuesTranspose, reshapedIndices as Tensor1D, x.shape[axis]);

        const invertTransposeDims =
            axis_util.getUndoAxesPermutation(transposeDims);
        paramsGrad = paramsGrad.transpose(invertTransposeDims);

        return paramsGrad as T;
      };
      return {x: derX};
    };
    return ENV.engine.runKernel(
        backend => backend.gather(x, indices, axis), {x}, grad);
  }

  /**
   * Pads a `Tensor1D` with a given value and paddings. See `pad` for details.
   */
  static pad1d(x: Tensor1D, paddings: [number, number], constantValue = 0):
      Tensor1D {
    util.assert(
        paddings.length === 2,
        'Invalid number of paddings. Must be length of 2.');
    return ArrayOps.pad(x, [paddings], constantValue);
  }

  /**
   * Pads a `Tensor2D` with a given value and paddings. See `pad` for details.
   */
  static pad2d(
      x: Tensor2D, paddings: [[number, number], [number, number]],
      constantValue = 0): Tensor2D {
    util.assert(
        paddings.length === 2 && paddings[0].length === 2 &&
            paddings[1].length === 2,
        'Invalid number of paddings. Must be length of 2 each.');
    return ArrayOps.pad(x, paddings, constantValue);
  }

  /**
   * Pads a `Tensor3D` with a given value and paddings. See `pad` for details.
   */
  static pad3d(
      x: Tensor3D,
      paddings: [[number, number], [number, number], [number, number]],
      constantValue = 0): Tensor3D {
    util.assert(
        paddings.length === 3 && paddings[0].length === 2 &&
            paddings[1].length === 2 && paddings[2].length === 2,
        'Invalid number of paddings. Must be length of 2 each.');
    return ArrayOps.pad(x, paddings, constantValue);
  }

  /**
   * Pads a `Tensor4D` with a given value and paddings. See `pad` for details.
   */
  static pad4d(
      x: Tensor4D,
      paddings:
          [
            [number, number], [number, number], [number, number],
            [number, number]
          ],
      constantValue = 0): Tensor4D {
    util.assert(
        paddings.length === 4 && paddings[0].length === 2 &&
            paddings[1].length === 2 && paddings[2].length === 2 &&
            paddings[3].length === 2,
        'Invalid number of paddings. Must be length of 2 each.');
    return ArrayOps.pad(x, paddings, constantValue);
  }

  /**
   * Pads a `Tensor` with a given value and paddings.
   *
   * This operation currently only implements the `CONSTANT` mode.
   *
   * Also available are stricter rank-specific methods with the same signature
   * as this method that assert that `paddings` is of given length.
   *   - `tf.pad1d`
   *   - `tf.pad2d`
   *   - `tf.pad3d`
   *   - `tf.pad4d`
   *
   * ```js
   * const x = tf.tensor1d([1, 2, 3, 4]);
   * x.pad([[1, 2]]).print();
   * ```
   * @param x The tensor to pad.
   * @param paddings An array of length `R` (the rank of the tensor), where
   * each element is a length-2 tuple of ints `[padBefore, padAfter]`,
   * specifying how much to pad along each dimension of the tensor.
   * @param constantValue The pad value to use. Defaults to 0.
   */
  @doc({heading: 'Tensors', subheading: 'Transformations'})
  @operation
  static pad<T extends Tensor>(
      x: T, paddings: Array<[number, number]>, constantValue = 0): T {
    util.assertArgumentsAreTensors({x}, 'pad');

    if (x.rank === 0) {
      throw new Error('pad(scalar) is not defined. Pass non-scalar to pad');
    }
    // Pad introduces values around the original tensor, so the gradient
    // slices the original shape out of the gradient.
    const begin = paddings.map(p => p[0]);
    const grad = (dy: T) => {
      return {x: () => dy.slice(begin, x.shape)};
    };
    return ENV.engine.runKernel(
               backend => backend.pad(x, paddings, constantValue), {x}, grad) as
        T;
  }

  /**
   * Stacks a list of rank-`R` `Tensor`s into one rank-`(R+1)` `Tensor`.
   *
   * ```js
   * const a = tf.tensor1d([1, 2]);
   * const b = tf.tensor1d([3, 4]);
   * const c = tf.tensor1d([5, 6]);
   * tf.stack([a, b, c]).print();
   * ```
   *
   * @param tensors A list of tensor objects with the same shape and dtype.
   * @param axis The axis to stack along. Defaults to 0 (the first dim).
   */
  @doc({heading: 'Tensors', subheading: 'Slicing and Joining'})
  @operation
  static stack<T extends Tensor>(tensors: T[], axis = 0): Tensor {
    util.assertArgumentsAreTensors({tensors}, 'stack');

    util.assert(tensors.length >= 1, 'Pass at least one tensor to tf.stack');
    if (tensors.length === 1) {
      return tensors[0].expandDims(axis);
    }
    const rank = tensors[0].rank;
    const shape = tensors[0].shape;
    const dtype = tensors[0].dtype;

    util.assert(axis <= rank, 'Axis must be <= rank of the tensor');

    tensors.forEach(t => {
      util.assertShapesMatch(
          shape, t.shape,
          'All tensors passed to stack must have matching shapes');
    });

    tensors.forEach(t => {
      util.assert(
          dtype === t.dtype,
          'All tensors passed to stack must have matching dtypes');
    });
    const expandedTensors = tensors.map(t => t.expandDims(axis));
    return ConcatOps.concat(expandedTensors, axis);
  }

  /**
   * Unstacks a `Tensor` of rank-`R` into a list of rank-`(R-1)` `Tensor`s.
   *
   * ```js
   * const a = tf.tensor2d([1, 2, 3, 4], [2, 2]);
   *
   * tf.unstack(a).forEach(tensor => tensor.print());
   * ```
   *
   * @param value A tensor object.
   * @param axis The axis to unstack along. Defaults to 0 (the first dim).
   */
  @doc({heading: 'Tensors', subheading: 'Slicing and Joining'})
  @operation
  static unstack<T extends Tensor>(value: T, axis = 0): Tensor[] {
    const num = value.shape[axis];
    const outputShape: number[] = Array(value.rank - 1).fill(0);
    let outIndex = 0;
    for (let i = 0; i < value.rank; i++) {
      if (i !== axis) {
        outputShape[outIndex] = value.shape[i];
        outIndex++;
      }
    }

    let splitSizes: number[];
    splitSizes = Array(num).fill(1);
    const begin = Array(value.rank).fill(0);
    const size = value.shape.slice();
    return splitSizes.map(s => {
      size[axis] = s;
      const slice = value.slice(begin, size);
      begin[axis] += s;
      return slice.reshape(outputShape);
    });
  }

  /**
   * Splits a `Tensor` into sub tensors.
   *
   * If `numOrSizeSplits` is a number, splits `x` along dimension `axis`
   * into `numOrSizeSplits` smaller tensors.
   * Requires that `numOrSizeSplits` evenly divides `x.shape[axis]`.
   *
   * If `numOrSizeSplits` is a number array, splits `x` into
   * `(numOrSizeSplits.length` pieces. The shape of the `i`-th piece has the
   * same size as `x` except along dimension `axis` where the size is
   * `numOrSizeSplits[i]`.
   *
   * ```js
   * const x = tf.tensor2d([1, 2, 3, 4, 5, 6, 7, 8], [2, 4]);
   * const [a, b] = tf.split(x, 2, 1);
   * a.print();
   * b.print();
   *
   * const [c, d, e] = tf.split(x, [1, 2, 1], 1);
   * c.print();
   * d.print();
   * e.print();
   * ```
   *
   * @param x The input tensor to split.
   * @param numOrSizeSplits Either an integer indicating the number of
   * splits along the axis or an array of integers containing the sizes of
   * each output tensor along the axis. If a number then it must evenly divide
   * `x.shape[axis]`; otherwise the sum of sizes must match `x.shape[axis]`.
   * @param axis The dimension along which to split. Defaults to 0 (the first
   * dim).
   */
  @doc({heading: 'Tensors', subheading: 'Slicing and Joining'})
  @operation
  static split<T extends Tensor>(
      x: T, numOrSizeSplits: number[]|number, axis = 0): T[] {
    util.assertArgumentsAreTensors({x}, 'split');

    axis = parseAxisParam(axis, x.shape)[0];
    let splitSizes: number[];
    if (typeof (numOrSizeSplits) === 'number') {
      util.assert(
          x.shape[axis] % numOrSizeSplits === 0,
          'Number of splits must evenly divide the axis.');
      splitSizes = Array(numOrSizeSplits).fill(x.shape[axis] / numOrSizeSplits);
    } else {
      util.assert(
          x.shape[axis] === numOrSizeSplits.reduce((a, b) => a + b),
          'The sum of sizes must match the size of the axis dimension.');
      splitSizes = numOrSizeSplits;
    }
    const begin = Array(x.rank).fill(0);
    const size = x.shape.slice();
    return splitSizes.map(s => {
      size[axis] = s;
      const slice = x.slice(begin, size);
      begin[axis] += s;
      return slice;
    });
  }

  /**
   * Computes the cumulative sum of a `Tensor` along `axis`.
   *
   * ```js
   * const x = tf.tensor([1, 2, 3, 4]);
   * x.cumsum().print();
   * ```
   * ```js
   * const x = tf.tensor([[1, 2], [3, 4]]);
   * x.cumsum().print();
   * ```
   *
   * @param x The input tensor to be summed.
   * @param axis The axis along which to sum. Optional. Defaults to 0.
   * @param exclusive Whether to perform exclusive cumulative sum. Optional.
   *     Defaults to false. If set to true then the sum of each tensor entry
   *     does not include its own value, but only the values previous to it
   *     along the specified axis.
   * @param reverse Whether to sum in the opposite direction. Optional.
   *     Defaults to false.
   */
  @doc({heading: 'Operations', subheading: 'Scan'})
  static cumsum<T extends Tensor>(
      x: Tensor, axis = 0, exclusive = false, reverse = false): T {
    util.assertArgumentsAreTensors({x}, 'cumsum');

    axis = axis | 0;
    const permutation = getAxesPermutation([axis], x.rank);
    let permutedX = x;
    if (permutation != null) {
      permutedX = x.transpose(permutation);
    }
    const permutedAxis = getInnerMostAxes(1, x.rank)[0];

    const grad = (dy: T) => {
      return {permutedX: () => dy.cumsum(axis, exclusive, !reverse)};
    };
    let value = ENV.engine.runKernel(
                    backend => backend.cumsum(
                        permutedX, permutedAxis, exclusive, reverse),
                    {permutedX}, grad) as T;

    if (permutation != null) {
      value = value.transpose(permutation);
    }
    return value;
  }

  /**
   * Returns a `Tensor` that has expanded rank, by inserting a dimension
   * into the tensor's shape.
   *
   * ```js
   * const x = tf.tensor1d([1, 2, 3, 4]);
   * const axis = 1;
   * x.expandDims(axis).print();
   * ```
   *
   * @param x The input tensor whose dimensions to be expanded.
   * @param axis The dimension index at which to insert shape of `1`. Defaults
   *     to 0 (the first dimension).
   */
  @doc({heading: 'Tensors', subheading: 'Transformations'})
  @operation
  static expandDims<R2 extends Rank>(x: Tensor, axis = 0): Tensor<R2> {
    util.assertArgumentsAreTensors({x}, 'expandDims');

    util.assert(axis <= x.rank, 'Axis must be <= rank of the tensor');
    const newShape = x.shape.slice();
    newShape.splice(axis, 0, 1);
    return ArrayOps.reshape(x, newShape);
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
  @operation
  @doc({heading: 'Tensors', subheading: 'Creation'})
  static linspace(start: number, stop: number, num: number): Tensor1D {
    if (num === 0) {
      throw new Error('Cannot request zero samples');
    }

    const step = (stop - start) / (num - 1);

    const values = makeZerosTypedArray(num, 'float32');
    values[0] = start;
    for (let i = 1; i < values.length; i++) {
      values[i] = values[i - 1] + step;
    }

    return ArrayOps.tensor1d(values, 'float32');
  }

  /**
   * Creates a new `Tensor1D` filled with the numbers in the range provided.
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
  @operation
  @doc({heading: 'Tensors', subheading: 'Creation'})
  static range(
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
      return ArrayOps.zeros([0], dtype);
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

    return ArrayOps.tensor1d(values, dtype);
  }

  /**
   * Creates an empty `TensorBuffer` with the specified `shape` and `dtype`.
   *
   * The values are stored in cpu as `TypedArray`. Fill the buffer using
   * `buffer.set()`, or by modifying directly `buffer.values`.
   *
   * When done, call `buffer.toTensor()` to get an immutable `Tensor` with
   * those values.
   *
   * ```js
   * // Create a buffer and set values at particular indices.
   * const buffer = tf.buffer([2, 2]);
   * buffer.set(3, 0, 0);
   * buffer.set(5, 1, 0);
   *
   * // Convert the buffer back to a tensor.
   * buffer.toTensor().print();
   * ```
   *
   * @param shape An array of integers defining the output tensor shape.
   * @param dtype The dtype of the buffer. Defaults to 'float32'.
   * @param values The values of the buffer as `TypedArray`. Defaults to
   * zeros.
   */
  @doc({heading: 'Tensors', subheading: 'Creation'})
  static buffer<R extends Rank>(
      shape: ShapeMap[R], dtype: DataType = 'float32', values?: TypedArray):
      TensorBuffer<R> {
    return new TensorBuffer<R>(shape, dtype, values);
  }

  /**
   * Prints information about the `Tensor` including its data.
   *
   * ```js
   * const verbose = true;
   * tf.tensor2d([1, 2, 3, 4], [2, 2]).print(verbose);
   * ```
   * @param x The tensor to be printed.
   * @param verbose Whether to print verbose information about the ` Tensor`,
   * including dtype and size.
   */
  @doc({heading: 'Tensors', subheading: 'Creation'})
  static print<T extends Tensor>(x: T, verbose = false): void {
    console.log(tensor_util.tensorToString(x, verbose));
  }
}

function makeZerosTypedArray<D extends DataType>(
    size: number, dtype: D): DataTypeMap[D] {
  if (dtype == null || dtype === 'float32') {
    return new Float32Array(size);
  } else if (dtype === 'int32') {
    return new Int32Array(size);
  } else if (dtype === 'bool') {
    return new Uint8Array(size);
  } else {
    throw new Error(`Unknown data type $ {dtype}`);
  }
}

function makeOnesTypedArray<D extends DataType>(
    size: number, dtype: D): DataTypeMap[D] {
  const array = makeZerosTypedArray(size, dtype);
  for (let i = 0; i < array.length; i++) {
    array[i] = 1;
  }
  return array;
}

function toTypedArray<D extends DataType>(
    a: ArrayData<D>, dtype: D): DataTypeMap[D] {
  if (noConversionNeeded(a, dtype)) {
    return a as DataTypeMap[D];
  }
  if (Array.isArray(a)) {
    a = util.flatten(a as number[]);
  }
  return util.copyTypedArray(a, dtype);
}

function noConversionNeeded<D extends DataType>(
    a: ArrayData<D>, dtype: D): boolean {
  return (a instanceof Float32Array && dtype === 'float32') ||
      (a instanceof Int32Array && dtype === 'int32') ||
      (a instanceof Uint8Array && dtype === 'bool');
}

function arrayRange(start: number, stop: number): number[] {
  const result = [];
  for (let i = start; i < stop; ++i) {
    result.push(i);
  }
  return result;
}

function arrayConcat(arrays: number[][]): number[] {
  const result = [];
  for (let i = 0; i < arrays.length; ++i) {
    for (let j = 0; j < arrays[i].length; ++j) {
      result.push(arrays[i][j]);
    }
  }
  return result;
}
