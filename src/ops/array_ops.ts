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
import {ForwardFunc} from '../engine';
import {ENV} from '../environment';
// tslint:disable-next-line:max-line-length
import {Scalar, Tensor, Tensor1D, Tensor2D, Tensor3D, Tensor4D, TensorBuffer} from '../tensor';
import * as tensor_util from '../tensor_util';
// tslint:disable-next-line:max-line-length
import {ArrayData, DataType, DataTypeMap, Rank, ShapeMap, TensorLike, TensorLike1D, TensorLike2D, TensorLike3D, TensorLike4D, TypedArray} from '../types';
import * as util from '../util';
import {parseAxisParam} from './axis_util';
import {ConcatOps} from './concat';
import {operation} from './operation';
import {MPRandGauss} from './rand';

export class ArrayOps {
  /**
   * Creates a `Tensor` with the provided values, shape and dtype.
   *
   * ```js
   * // Pass an array of values to create a vector.
   * dl.tensor([1, 2, 3, 4]).print();
   * ```
   *
   * ```js
   * // Pass a nested array of values to make a matrix or a higher
   * // dimensional tensor.
   * dl.tensor([[1, 2], [3, 4]]).print();
   * ```
   *
   * ```js
   * // Pass a flat array and specify a shape yourself.
   * dl.tensor([1, 2, 3, 4], [2, 2]).print();
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
   * This method is mainly for self documentation and TypeScript typings as the
   * same functionality can be achieved with `tensor`. In general, we recommend
   * using this method as it makes code more readable.
   *
   * ```js
   * dl.scalar(3.14).print();
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
   * This method is mainly for self documentation and TypeScript typings as the
   * same functionality can be achieved with `tensor`. In general, we recommend
   * using this method as it makes code more readable.
   *
   * ```js
   * dl.tensor1d([1, 2, 3]).print();
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
      throw new Error(
          'Error creating a new Tensor1D: values must be a flat/TypedArray');
    }
    return ArrayOps.tensor(values, inferredShape as [number], dtype);
  }

  /**
   * Creates rank-2 `Tensor` with the provided values, shape and dtype.
   *
   * This method is mainly for self documentation and TypeScript typings as the
   * same functionality can be achieved with `tensor`. In general, we recommend
   * using this method as it makes code more readable.
   *
   *  ```js
   * // Pass a nested array.
   * dl.tensor2d([[1, 2], [3, 4]]).print();
   * ```
   * ```js
   * // Pass a flat array and specify a shape.
   * dl.tensor2d([1, 2, 3, 4], [2, 2]).print();
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
    const inferredShape = util.inferShape(values);
    if (inferredShape.length !== 2 && inferredShape.length !== 1) {
      throw new Error(
          'Error creating a new Tensor2D: values must be number[][] ' +
          'or flat/TypedArray');
    }
    shape = shape || inferredShape as [number, number];
    return ArrayOps.tensor(values, shape, dtype);
  }

  /**
   * Creates rank-3 `Tensor` with the provided values, shape and dtype.
   *
   * This method is mainly for self documentation and TypeScript typings as
   * the same functionality can be achieved with `tensor`. In general, we
   * recommend using this method as it makes code more readable.
   *
   *  ```js
   * // Pass a nested array.
   * dl.tensor3d([[[1], [2]], [[3], [4]]]).print();
   * ```
   * ```js
   * // Pass a flat array and specify a shape.
   * dl.tensor3d([1, 2, 3, 4], [2, 2, 1]).print();
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
    const inferredShape = util.inferShape(values);
    if (inferredShape.length !== 3 && inferredShape.length !== 1) {
      throw new Error(
          'Error creating a new Tensor3D: values must be number[][][]' +
          'or flat/TypedArray');
    }
    shape = shape || inferredShape as [number, number, number];
    return ArrayOps.tensor(values, shape, dtype);
  }

  /**
   * Creates rank-4 `Tensor` with the provided values, shape and dtype.
   *  ```js
   * // Pass a nested array.
   * dl.tensor4d([[[[1], [2]], [[3], [4]]]]).print();
   * ```
   * ```js
   * // Pass a flat array and specify a shape.
   * dl.tensor4d([1, 2, 3, 4], [1, 2, 2, 1]).print();
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
    const inferredShape = util.inferShape(values);
    if (inferredShape.length !== 4 && inferredShape.length !== 1) {
      throw new Error(
          'Error creating a new Tensor4D: values must be number[][][][]' +
          'or flat/TypedArray');
    }
    shape = shape || inferredShape as [number, number, number, number];
    return ArrayOps.tensor(values, shape, dtype);
  }

  /**
   * Creates a `Tensor` with all elements set to 1.
   *
   * ```js
   * dl.ones([2, 2]).print();
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
   * dl.zeros([2, 2]).print();
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
   * dl.fill([2, 2], 4).print();
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
   * const x = dl.tensor([1, 2]);
   * dl.onesLike(x).print();
   * ```
   * @param x A tensor.
   */
  @doc({heading: 'Tensors', subheading: 'Creation'})
  @operation
  static onesLike<T extends Tensor>(x: T): T {
    return ArrayOps.ones(x.shape, x.dtype) as T;
  }

  /**
   * Creates a `Tensor` with all elements set to 0 with the same shape as the
   * given tensor.
   *
   * ```js
   * const x = dl.tensor([1, 2]);
   * dl.zerosLike(x).print();
   * ```
   *
   * @param x A tensor.
   */
  @doc({heading: 'Tensors', subheading: 'Creation'})
  @operation
  static zerosLike<T extends Tensor>(x: T): T {
    return ArrayOps.zeros(x.shape, x.dtype) as T;
  }

  /**
   * Creates a new tensor with the same values and shape as the specified
   * tensor.
   *
   * ```js
   * const x = dl.tensor([1, 2]);
   * x.clone().print();
   * ```
   *
   * @param x The tensor to clone.
   */
  @doc({heading: 'Tensors', subheading: 'Creation'})
  @operation
  static clone<T extends Tensor>(x: T): T {
    return Tensor.make(x.shape, {dataId: x.dataId}, x.dtype) as T;
  }

  /**
   * Creates a `Tensor` with values sampled from a normal distribution.
   *
   * ```js
   * dl.randomNormal([2, 2]).print();
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
    return Tensor.rand(shape, () => randGauss.nextValue(), dtype);
  }

  /**
   * Creates a `Tensor` with values sampled from a truncated normal
   * distribution.
   *
   * ```js
   * dl.truncatedNormal([2, 2]).print();
   * ```
   *
   * The generated values follow a normal distribution with specified mean and
   * standard deviation, except that values whose magnitude is more than 2
   * standard deviations from the mean are dropped and re-picked.
   *
   * @param shape An array of integers defining the output tensor shape.
   * @param mean The mean of the normal distribution.
   * @param stdDev The standard deviation of the normal distribution.
   * @param dtype The data type of the output.
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
    return Tensor.rand(shape, () => randGauss.nextValue(), dtype);
  }

  /**
   * Creates a `Tensor` with values sampled from a uniform distribution.
   *
   * The generated values follow a uniform distribution in the range [minval,
   * maxval). The lower bound minval is included in the range, while the upper
   * bound maxval is excluded.
   *
   * ```js
   * dl.randomUniform([2, 2]).print();
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
    return Tensor.rand(shape, () => util.randUniform(minval, maxval), dtype);
  }

  /**
   * Creates a `Tensor` with values sampled from a random number generator
   * function defined by the user.
   *
   * @param shape An array of integers defining the output tensor shape.
   * @param randFunction A random number generator function which is called for
   * each element in the output tensor.
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
   * const probs = dl.tensor([.75, .25]);
   * dl.multinomial(probs, 3).print();
   * ```
   *
   * @param probabilities 1D array with normalized outcome probabilities, or
   *     2D array of shape `[batchSize, numOutcomes]`.
   * @param numSamples Number of samples to draw for each row slice.
   * @param seed The seed number.
   * @return 1D array of shape `[numSamples]`, or 2D array of shape
   *     `[batchSize, numSamples]`, depending on the rank of the input.
   */
  @operation
  static multinomial(
      probabilities: Tensor1D|Tensor2D, numSamples: number, seed?: number):
      Tensor1D|Tensor2D {
    const numOutcomes = probabilities.size;
    const origRank = probabilities.rank;
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

    const prob2D =
        origRank === 1 ? probabilities.as2D(1, -1) : probabilities as Tensor2D;
    const res = ENV.engine.runKernel(
        backend => backend.multinomial(prob2D, numSamples, seed), {prob2D});

    return origRank === 1 ? res.as1D() : res;
  }

  /**
   * Creates a one-hot `Tensor`. The locations represented by `indices` take
   * value `onValue` (defaults to 1), while all other locations take value
   * `offValue` (defaults to 0).
   *
   * ```js
   * dl.oneHot(dl.tensor1d([0, 1]), 3).print();
   * ```
   *
   * @param indices 1D Array of indices.
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
   * dl.fromPixels(image).print();
   * ```
   *
   * @param pixels The input image to construct the tensor from. Accepts image
   * of type `ImageData`, `HTMLImageElement`, `HTMLCanvasElement`, or
   * `HTMLVideoElement`.
   * @param numChannels The number of channels of the output tensor. The
   * supported image types are all 4-channel by default, a numChannels value
   * less than 4 allows you to ignore channels.
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
   * const x = dl.tensor1d([1, 2, 3, 4]);
   * x.reshape([2, 2]).print();
   * ```
   *
   * @param x A tensor.
   * @param shape An array of integers defining the output tensor shape.
   */
  @doc({heading: 'Tensors', subheading: 'Transformations'})
  @operation
  static reshape<R2 extends Rank>(x: Tensor, shape: ShapeMap[R2]): Tensor<R2> {
    shape = util.inferFromImplicitShape(shape, x.size);
    util.assert(
        x.size === util.sizeFromShape(shape),
        'new shape and old shape must have the same number of elements.');

    const grad = (dy: Tensor<R2>) => {
      return {x: () => dy.reshape(x.shape)};
    };
    return ENV.engine.runKernel(
        backend => Tensor.make(shape, {dataId: x.dataId}, x.dtype), {x}, grad);
  }

  /**
   * Removes dimensions of size 1 from the shape of a `Tensor`.
   *
   * ```js
   * const x = dl.tensor([1, 2, 3, 4], [1, 1, 4]);
   * x.squeeze().print();
   * ```
   * @param axis An optional list of numbers. If specified, only
   *     squeezes the dimensions listed. The dimension index starts at 0. It is
   *     an error to squeeze a dimension that is not 1.
   */
  @doc({heading: 'Tensors', subheading: 'Transformations'})
  static squeeze<T extends Tensor>(x: Tensor, axis?: number[]): T {
    return ArrayOps.reshape(x, util.squeezeShape(x.shape, axis).newShape) as T;
  }

  /**
   * Casts a tensor to a new dtype.
   *
   * ```js
   * const x = dl.tensor1d([1.5, 2.5, 3]);
   * dl.cast(x, 'int32').print();
   * ```
   * @param x A tensor.
   * @param dtype The dtype to cast the input tensor to.
   */
  @doc({heading: 'Tensors', subheading: 'Transformations'})
  @operation
  static cast<T extends Tensor>(x: T, dtype: DataType): T {
    const forw: ForwardFunc<T> = backend => {
      if (!util.hasEncodingLoss(x.dtype, dtype)) {
        // We don't change the underlying data, since we cast to higher
        // precision.
        return Tensor.make(x.shape, {dataId: x.dataId}, dtype) as T;
      }
      if (dtype === 'int32') {
        return backend.int(x);
      } else if (dtype === 'bool') {
        return backend.notEqual(x, ArrayOps.scalar(0, x.dtype)) as T;
      } else {
        throw new Error(`Error in Cast: unknown dtype argument (${dtype})`);
      }
    };
    const grad = (dy: T) => {
      return {x: () => dy.clone()};
    };
    return ENV.engine.runKernel(forw, {x}, grad) as T;
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
   * const a = dl.tensor1d([1, 2]);
   *
   * a.tile([2]).print();    // or a.tile([2])
   * ```
   *
   * ```js
   * const a = dl.tensor2d([1, 2, 3, 4], [2, 2]);
   *
   * a.tile([1, 2]).print();  // or a.tile([1, 2])
   * ```
   * @param x The tensor to transpose.
   * @param reps Determines the number of replications per dimension.
   */
  @doc({heading: 'Tensors', subheading: 'Slicing and Joining'})
  @operation
  static tile<T extends Tensor>(x: T, reps: number[]): T {
    util.assert(
        x.rank === reps.length,
        `Error in transpose: rank of input ${x.rank} ` +
            `must match length of reps ${reps}.`);
    return ENV.engine.runKernel(backend => backend.tile(x, reps), {x});
  }

  /**
   * Gather slices from tensor `x`'s axis `axis` according to `indices`.
   *
   * ```js
   * const x = dl.tensor1d([1, 2, 3, 4]);
   * const indices = dl.tensor1d([1, 3, 3]);
   *
   * x.gather(indices).print();
   * ```
   *
   * ```js
   * const x = dl.tensor2d([1, 2, 3, 4], [2, 2]);
   * const indices = dl.tensor1d([1, 1, 0]);
   *
   * x.gather(indices).print();
   * ```
   * @param x The input tensor.
   * @param indices The indices of the values to extract.
   * @param axis The axis over which to select values. Defaults to 0.
   */
  @doc({heading: 'Tensors', subheading: 'Slicing and Joining'})
  @operation
  static gather<T extends Tensor>(x: T, indices: Tensor1D, axis = 0): T {
    const axes = parseAxisParam(axis, x.shape);
    return ENV.engine.runKernel(
        backend => backend.gather(x, indices, axes[0]), {x, indices});
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
   * This operation currently only implements the `CONSTANT` mode from
   * Tensorflow's `pad` operation.
   *
   * ```js
   * const x = dl.tensor1d([1, 2, 3, 4]);
   * x.pad([[1, 2]]).print();
   * ```
   * @param x The tensor to pad.
   * @param paddings An array of length `R` (the rank of the tensor), where each
   *     element is a length-2 tuple of ints `[padBefore, padAfter]`, specifying
   *     how much to pad along each dimension of the tensor.
   * @param constantValue The pad value to use. Defaults to 0.
   */
  @doc({heading: 'Tensors', subheading: 'Transformations'})
  @operation
  static pad<T extends Tensor>(
      x: T, paddings: Array<[number, number]>, constantValue = 0): T {
    if (x.rank === 0) {
      throw new Error('pad(scalar) is not defined. Pass non-scalar to pad');
    }
    // Pad introduces values around the original tensor, so the gradient slices
    // the original shape out of the gradient.
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
   * const a = dl.tensor1d([1, 2]);
   * const b = dl.tensor1d([3, 4]);
   * const c = dl.tensor1d([5, 6]);
   * dl.stack([a, b, c]).print();
   * ```
   *
   * @param tensors A list of tensor objects with the same shape and dtype.
   * @param axis The axis to stack along. Defaults to 0 (the first dim).
   */
  @doc({heading: 'Tensors', subheading: 'Slicing and Joining'})
  @operation
  static stack<T extends Tensor>(tensors: T[], axis = 0): Tensor {
    util.assert(tensors.length >= 2, 'Pass at least two tensors to dl.stack');
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
   * Returns a `Tensor` that has expanded rank, by inserting a dimension
   * into the tensor's shape.
   *
   * ```js
   * const x = dl.tensor1d([1, 2, 3, 4]);
   * const axis = 1;
   * x.expandDims(axis).print();
   * ```
   *
   * @param axis The dimension index at which to insert shape of `1`. Defaults
   *     to 0 (the first dimension).
   */
  @doc({heading: 'Tensors', subheading: 'Transformations'})
  @operation
  static expandDims<R2 extends Rank>(x: Tensor, axis = 0): Tensor<R2> {
    util.assert(axis <= x.rank, 'Axis must be <= rank of the tensor');
    const newShape = x.shape.slice();
    newShape.splice(axis, 0, 1);
    return ArrayOps.reshape(x, newShape);
  }

  /**
   * Return an evenly spaced sequence of numbers over the given interval.
   *
   * ```js
   * dl.linspace(0, 9, 10).print();
   * ```
   * @param start The start value of the sequence
   * @param stop The end value of the sequence
   * @param num The number of values to generate
   * @param endpoint Determines whether stop is included in the
   * sequence. Defaults to true.
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

    return Tensor1D.new(values, 'float32');
  }

  /**
   * Creates a new `Tensor1D` filled with the numbers in the range provided.
   *
   * The tensor is a is half-open interval meaning it includes start, but
   * excludes stop. Decrementing ranges and negative step values are also
   * supported.
   *
   * ```js
   * dl.range(0, 9, 2).print();
   * ```
   *
   * @param start An integer start value
   * @param stop An integer stop value
   * @param step An integer increment (will default to 1 or -1)
   * @param dtype
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
   * `buffer.set()`, or by modifying directly `buffer.values`. When done,
   * call `buffer.toTensor()` to get an immutable `Tensor` with those values.
   *
   * When done, call `buffer.toTensor()` to get an immutable `Tensor` with those
   * values.
   *
   * ```js
   * // Create a buffer and set values at particular indices.
   * const buffer = dl.buffer([2, 2]);
   * buffer.set(3, 0, 0);
   * buffer.set(5, 1, 0);
   *
   * // Convert the buffer back to a tensor.
   * buffer.toTensor().print();
   * ```
   *
   * @param shape An array of integers defining the output tensor shape.
   * @param dtype The dtype of the buffer. Defaults to 'float32'.
   * @param values The values of the buffer as `TypedArray`. Defaults to zeros.
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
   * dl.tensor2d([1, 2, 3, 4], [2, 2]).print(verbose);
   * ```
   *
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
