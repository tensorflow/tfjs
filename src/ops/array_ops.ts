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

import {ENGINE} from '../engine';
import {Tensor, Tensor1D, Tensor2D, Tensor3D, Tensor4D, TensorBuffer} from '../tensor';
import {convertToTensor, convertToTensorArray} from '../tensor_util_env';
import {DataType, DataTypeMap, Rank, ShapeMap, TensorLike, TensorLike4D} from '../types';
import * as util from '../util';
import {getAxesPermutation, getInnerMostAxes} from './axis_util';
import {concat} from './concat_split';
import {op} from './operation';
import {MPRandGauss, UniformRandom} from './rand';
import {zeros, zerosLike} from './tensor_ops';

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
/** @doc {heading: 'Tensors', subheading: 'Creation'} */
function clone_<T extends Tensor>(x: T|TensorLike): T {
  const $x = convertToTensor(x, 'x', 'clone', null);
  const der = (dy: T) => {
    return {$x: () => dy.toFloat()};
  };

  return ENGINE.runKernel(
             backend =>
                 Tensor.make($x.shape, {dataId: $x.dataId}, $x.dtype) as T,
             {$x}, der) as T;
}

/**
 * Create an identity matrix.
 *
 * @param numRows Number of rows.
 * @param numColumns Number of columns. Defaults to `numRows`.
 * @param batchShape If provided, will add the batch shape to the beginning
 *   of the shape of the returned `tf.Tensor` by repeating the identity
 *   matrix.
 * @param dtype Data type.
 * @returns Identity matrix of the specified size and data type, possibly
 *   with batch repetition if `batchShape` is specified.
 */
/** @doc {heading: 'Tensors', subheading: 'Creation'} */
function eye_(
    numRows: number, numColumns?: number,
    batchShape?:
        [
          number
        ]|[number,
           number]|[number, number, number]|[number, number, number, number],
    dtype: DataType = 'float32'): Tensor2D {
  if (numColumns == null) {
    numColumns = numRows;
  }
  const buff = buffer([numRows, numColumns], dtype);
  const n = numRows <= numColumns ? numRows : numColumns;
  for (let i = 0; i < n; ++i) {
    buff.set(1, i, i);
  }
  const out = buff.toTensor().as2D(numRows, numColumns);
  if (batchShape == null) {
    return out;
  } else {
    if (batchShape.length === 1) {
      return tile(expandDims(out, 0), [batchShape[0], 1, 1]);
    } else if (batchShape.length === 2) {
      return tile(
          expandDims(expandDims(out, 0), 0),
          [batchShape[0], batchShape[1], 1, 1]);
    } else if (batchShape.length === 3) {
      return tile(
          expandDims(expandDims(expandDims(out, 0), 0), 0),
          [batchShape[0], batchShape[1], batchShape[2], 1, 1]);
    } else {
      throw new Error(
          `eye() currently supports only 1D and 2D ` +
          // tslint:disable-next-line:no-any
          `batchShapes, but received ${(batchShape as any).length}D.`);
    }
  }
}

/**
 * Creates a `tf.Tensor` with values sampled from a normal distribution.
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
/** @doc {heading: 'Tensors', subheading: 'Random'} */
function randomNormal_<R extends Rank>(
    shape: ShapeMap[R], mean = 0, stdDev = 1, dtype?: 'float32'|'int32',
    seed?: number): Tensor<R> {
  if (dtype != null && (dtype as DataType) === 'bool') {
    throw new Error(`Unsupported data type ${dtype}`);
  }
  const randGauss =
      new MPRandGauss(mean, stdDev, dtype, false /* truncated */, seed);
  const res = buffer(shape, dtype);
  for (let i = 0; i < res.values.length; i++) {
    res.values[i] = randGauss.nextValue();
  }
  return res.toTensor();
}

/**
 * Creates a `tf.Tensor` with values sampled from a truncated normal
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
/** @doc {heading: 'Tensors', subheading: 'Creation'} */
function truncatedNormal_<R extends Rank>(
    shape: ShapeMap[R], mean = 0, stdDev = 1, dtype?: 'float32'|'int32',
    seed?: number): Tensor<R> {
  if (dtype != null && (dtype as DataType) === 'bool') {
    throw new Error(`Unsupported data type ${dtype}`);
  }
  const randGauss =
      new MPRandGauss(mean, stdDev, dtype, true /* truncated */, seed);
  const res = buffer(shape, dtype);
  for (let i = 0; i < res.values.length; i++) {
    res.values[i] = randGauss.nextValue();
  }
  return res.toTensor();
}

/**
 * Creates a `tf.Tensor` with values sampled from a uniform distribution.
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
/** @doc {heading: 'Tensors', subheading: 'Random'} */
function randomUniform_<R extends Rank>(
    shape: ShapeMap[R], minval = 0, maxval = 1, dtype: DataType = 'float32',
    seed?: number|string): Tensor<R> {
  const res = buffer(shape, dtype);
  const random = new UniformRandom(minval, maxval, null, seed);
  for (let i = 0; i < res.values.length; i++) {
    res.values[i] = random.nextValue();
  }
  return res.toTensor();
}

/**
 * Creates a `tf.Tensor` with values sampled from a random number generator
 * function defined by the user.
 *
 * @param shape An array of integers defining the output tensor shape.
 * @param randFunction A random number generator function which is called
 * for each element in the output tensor.
 * @param dtype The data type of the output tensor. Defaults to 'float32'.
 */
function rand_<R extends Rank>(
    shape: ShapeMap[R], randFunction: () => number,
    dtype?: DataType): Tensor<R> {
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
 * Creates a `tf.Tensor` with values drawn from a multinomial distribution.
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
/** @doc {heading: 'Tensors', subheading: 'Random'} */
function multinomial_(
    logits: Tensor1D|Tensor2D|TensorLike, numSamples: number, seed?: number,
    normalized = false): Tensor1D|Tensor2D {
  const $logits = convertToTensor(logits, 'logits', 'multinomial');
  const numOutcomes = $logits.size;
  const origRank = $logits.rank;
  if (numOutcomes < 2) {
    throw new Error(
        `Error in multinomial: you need at least 2 outcomes, but got ` +
        `${numOutcomes}.`);
  }
  if (origRank > 2) {
    throw new Error(`Rank of probabilities must be 1 or 2, but is ${origRank}`);
  }
  seed = seed || Math.random();
  const logits2D = origRank === 1 ? $logits.as2D(1, -1) : $logits as Tensor2D;
  const res = ENGINE.runKernel(
      backend => backend.multinomial(logits2D, normalized, numSamples, seed),
      {logits2D});

  return origRank === 1 ? res.as1D() : res;
}

/**
 * Creates a one-hot `tf.Tensor`. The locations represented by `indices` take
 * value `onValue` (defaults to 1), while all other locations take value
 * `offValue` (defaults to 0). If `indices` is rank `R`, the output has rank
 * `R+1` with the last axis of size `depth`.
 *
 * ```js
 * tf.oneHot(tf.tensor1d([0, 1], 'int32'), 3).print();
 * ```
 *
 * @param indices `tf.Tensor` of indices with dtype `int32`.
 * @param depth The depth of the one hot dimension.
 * @param onValue A number used to fill in the output when the index matches
 * the location.
 * @param offValue A number used to fill in the output when the index does
 *     not match the location.
 */
/** @doc {heading: 'Tensors', subheading: 'Creation'} */
function oneHot_(
    indices: Tensor|TensorLike, depth: number, onValue = 1,
    offValue = 0): Tensor {
  if (depth < 2) {
    throw new Error(`Error in oneHot: depth must be >=2, but it is ${depth}`);
  }
  let $indices = convertToTensor(indices, 'indices', 'oneHot', 'int32');
  const outShape = [...$indices.shape, depth];
  $indices = $indices.flatten();

  const grad = (dy: Tensor2D) => {
    return {$indices: () => zeros($indices.shape, 'float32')};
  };
  const result = ENGINE.runKernel(
      backend => backend.oneHot($indices as Tensor1D, depth, onValue, offValue),
      {$indices}, grad);
  return result.reshape(outShape);
}

/**
 * Reshapes a `tf.Tensor` to a given shape.
 *
 * Given an input tensor, returns a new tensor with the same values as the
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
/** @doc {heading: 'Tensors', subheading: 'Transformations'} */
function reshape_<R2 extends Rank>(
    x: Tensor|TensorLike, shape: ShapeMap[R2]): Tensor<R2> {
  const $x = convertToTensor(x, 'x', 'reshape', null);
  shape = util.inferFromImplicitShape(shape, $x.size) as ShapeMap[R2];
  util.assert(
      $x.size === util.sizeFromShape(shape),
      () => 'new shape and old shape must have the same number of elements.');

  const grad = (dy: Tensor<R2>) => {
    return {$x: () => dy.reshape($x.shape)};
  };
  return ENGINE.runKernel(backend => backend.reshape($x, shape), {$x}, grad);
}

/**
 * Removes dimensions of size 1 from the shape of a `tf.Tensor`.
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
/** @doc {heading: 'Tensors', subheading: 'Transformations'} */
function squeeze_<T extends Tensor>(x: Tensor|TensorLike, axis?: number[]): T {
  const $x = convertToTensor(x, 'x', 'squeeze');
  return reshape($x, util.squeezeShape($x.shape, axis).newShape) as T;
}

/**
 * Casts a `tf.Tensor` to a new dtype.
 *
 * ```js
 * const x = tf.tensor1d([1.5, 2.5, 3]);
 * tf.cast(x, 'int32').print();
 * ```
 * @param x The input tensor to be casted.
 * @param dtype The dtype to cast the input tensor to.
 */
/** @doc {heading: 'Tensors', subheading: 'Transformations'} */
function cast_<T extends Tensor>(x: T|TensorLike, dtype: DataType): T {
  const $x = convertToTensor(x, 'x', 'cast');

  // Sanity checks.
  if (!util.isValidDtype(dtype)) {
    throw new Error(`Failed to cast to unknown dtype ${dtype}`);
  }
  if (dtype === 'string' && $x.dtype !== 'string' ||
      dtype !== 'string' && $x.dtype === 'string') {
    throw new Error('Only strings can be casted to strings');
  }

  const grad = (dy: T) => {
    return {$x: () => dy.clone()};
  };
  return ENGINE.runKernel(backend => backend.cast($x, dtype), {$x}, grad) as T;
}

/**
 * Construct a tensor by repeating it the number of times given by reps.
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
/** @doc {heading: 'Tensors', subheading: 'Slicing and Joining'} */
function tile_<T extends Tensor>(x: T|TensorLike, reps: number[]): T {
  const parseAs: DataType = null;
  const $x = convertToTensor(x, 'x', 'tile', parseAs);

  util.assert(
      $x.rank === reps.length,
      () => `Error in transpose: rank of input ${$x.rank} ` +
          `must match length of reps ${reps}.`);
  const grad = (dy: T, saved: Tensor[]) => {
    const [$x] = saved;
    const derX = () => {
      let xGrad = zerosLike($x);
      // TODO(cais): Maybe reduce memory footprint by avoiding repeated
      // slicing.
      if ($x.rank === 1) {
        for (let i = 0; i < reps[0]; ++i) {
          xGrad = xGrad.add(dy.slice([i * $x.shape[0]], [$x.shape[0]]));
        }
      } else if ($x.rank === 2) {
        for (let i = 0; i < reps[0]; ++i) {
          for (let j = 0; j < reps[1]; ++j) {
            xGrad = xGrad.add(dy.slice(
                [i * $x.shape[0], j * $x.shape[1]],
                [$x.shape[0], $x.shape[1]]));
          }
        }
      } else if ($x.rank === 3) {
        for (let i = 0; i < reps[0]; ++i) {
          for (let j = 0; j < reps[1]; ++j) {
            for (let k = 0; k < reps[2]; ++k) {
              xGrad = xGrad.add(dy.slice(
                  [i * $x.shape[0], j * $x.shape[1], k * $x.shape[2]],
                  [$x.shape[0], $x.shape[1], $x.shape[2]]));
            }
          }
        }
      } else if ($x.rank === 4) {
        for (let i = 0; i < reps[0]; ++i) {
          for (let j = 0; j < reps[1]; ++j) {
            for (let k = 0; k < reps[2]; ++k) {
              for (let l = 0; l < reps[3]; ++l) {
                xGrad = xGrad.add(dy.slice(
                    [
                      i * $x.shape[0], j * $x.shape[1], k * $x.shape[2],
                      l * $x.shape[3]
                    ],
                    [$x.shape[0], $x.shape[1], $x.shape[2], $x.shape[3]]));
              }
            }
          }
        }
      } else {
        throw new Error(
            `Gradient for tile operation is not implemented for rank-` +
            `${$x.rank} tensors yet.`);
      }
      return xGrad as T;
    };
    return {$x: derX};
  };
  return ENGINE.runKernel((backend, save) => {
    const res = backend.tile($x, reps);
    save([$x]);
    return res;
  }, {$x}, grad);
}

/**
 * Pads a `tf.Tensor1D` with a given value and paddings. See `pad` for details.
 */
function pad1d_(
    x: Tensor1D|TensorLike, paddings: [number, number],
    constantValue = 0): Tensor1D {
  util.assert(
      paddings.length === 2,
      () => 'Invalid number of paddings. Must be length of 2.');
  return pad(x, [paddings], constantValue);
}

/**
 * Pads a `tf.Tensor2D` with a given value and paddings. See `pad` for details.
 */
function pad2d_(
    x: Tensor2D|TensorLike, paddings: [[number, number], [number, number]],
    constantValue = 0): Tensor2D {
  util.assert(
      paddings.length === 2 && paddings[0].length === 2 &&
          paddings[1].length === 2,
      () => 'Invalid number of paddings. Must be length of 2 each.');
  return pad(x, paddings, constantValue);
}

/**
 * Pads a `tf.Tensor3D` with a given value and paddings. See `pad` for details.
 */
function pad3d_(
    x: Tensor3D|TensorLike,
    paddings: [[number, number], [number, number], [number, number]],
    constantValue = 0): Tensor3D {
  util.assert(
      paddings.length === 3 && paddings[0].length === 2 &&
          paddings[1].length === 2 && paddings[2].length === 2,
      () => 'Invalid number of paddings. Must be length of 2 each.');
  return pad(x, paddings, constantValue);
}

/**
 * Pads a `tf.Tensor4D` with a given value and paddings. See `pad` for details.
 */
function pad4d_(
    x: Tensor4D|TensorLike,
    paddings:
        [
          [number, number], [number, number], [number, number], [number, number]
        ],
    constantValue = 0): Tensor4D {
  util.assert(
      paddings.length === 4 && paddings[0].length === 2 &&
          paddings[1].length === 2 && paddings[2].length === 2 &&
          paddings[3].length === 2,
      () => 'Invalid number of paddings. Must be length of 2 each.');
  return pad(x, paddings, constantValue);
}

/**
 * Pads a `tf.Tensor` with a given value and paddings.
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
/** @doc {heading: 'Tensors', subheading: 'Transformations'} */
function pad_<T extends Tensor>(
    x: T|TensorLike, paddings: Array<[number, number]>, constantValue = 0): T {
  const $x = convertToTensor(x, 'x', 'pad');

  if ($x.rank === 0) {
    throw new Error('pad(scalar) is not defined. Pass non-scalar to pad');
  }
  // Pad introduces values around the original tensor, so the gradient
  // slices the original shape out of the gradient.
  const begin = paddings.map(p => p[0]);
  const grad = (dy: T) => {
    return {$x: () => dy.slice(begin, $x.shape)};
  };
  return ENGINE.runKernel(
             backend => backend.pad($x, paddings, constantValue), {$x}, grad) as
      T;
}

/**
 * Stacks a list of rank-`R` `tf.Tensor`s into one rank-`(R+1)` `tf.Tensor`.
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
/** @doc {heading: 'Tensors', subheading: 'Slicing and Joining'} */
function stack_<T extends Tensor>(
    tensors: Array<T|TensorLike>, axis = 0): Tensor {
  const $tensors = convertToTensorArray(tensors, 'tensors', 'stack');

  util.assert(
      $tensors.length >= 1, () => 'Pass at least one tensor to tf.stack');
  if ($tensors.length === 1) {
    return $tensors[0].expandDims(axis);
  }
  const rank = $tensors[0].rank;
  const shape = $tensors[0].shape;
  const dtype = $tensors[0].dtype;

  util.assert(axis <= rank, () => 'Axis must be <= rank of the tensor');

  $tensors.forEach(t => {
    util.assertShapesMatch(
        shape, t.shape,
        'All tensors passed to stack must have matching shapes');
  });

  $tensors.forEach(t => {
    util.assert(
        dtype === t.dtype,
        () => 'All tensors passed to stack must have matching dtypes');
  });
  const expandedTensors = $tensors.map(t => t.expandDims(axis));
  return concat(expandedTensors, axis);
}

/**
 * This operation reshapes the "batch" dimension 0 into `M + 1` dimensions of
 * shape `blockShape + [batch]`, interleaves these blocks back into the grid
 * defined by the spatial dimensions `[1, ..., M]`, to obtain a result with
 * the same rank as the input. The spatial dimensions of this intermediate
 * result are then optionally cropped according to `crops` to produce the
 * output. This is the reverse of `tf.spaceToBatchND`. See below for a precise
 * description.
 *
 * ```js
 * const x = tf.tensor4d([1, 2, 3, 4], [4, 1, 1, 1]);
 * const blockShape = [2, 2];
 * const crops = [[0, 0], [0, 0]];
 *
 * x.batchToSpaceND(blockShape, crops).print();
 * ```
 *
 * @param x A `tf.Tensor`. N-D with `x.shape` = `[batch] + spatialShape +
 * remainingShape`, where spatialShape has `M` dimensions.
 * @param blockShape A 1-D array. Must have shape `[M]`, all values must
 * be >= 1.
 * @param crops A 2-D array.  Must have shape `[M, 2]`, all values must be >= 0.
 * `crops[i] = [cropStart, cropEnd]` specifies the amount to crop from input
 * dimension `i + 1`, which corresponds to spatial dimension `i`. It is required
 * that `cropStart[i] + cropEnd[i] <= blockShape[i] * inputShape[i + 1]`
 *
 * This operation is equivalent to the following steps:
 *
 * 1. Reshape `x` to `reshaped` of shape: `[blockShape[0], ...,
 * blockShape[M-1], batch / prod(blockShape), x.shape[1], ...,
 * x.shape[N-1]]`
 *
 * 2. Permute dimensions of `reshaped`to produce `permuted` of shape `[batch /
 * prod(blockShape),x.shape[1], blockShape[0], ..., x.shape[M],
 * blockShape[M-1],x.shape[M+1], ..., x.shape[N-1]]`
 *
 * 3. Reshape `permuted` to produce `reshapedPermuted` of shape `[batch /
 * prod(blockShape),x.shape[1] * blockShape[0], ..., x.shape[M] *
 * blockShape[M-1],x.shape[M+1], ..., x.shape[N-1]]`
 *
 * 4. Crop the start and end of dimensions `[1, ..., M]` of `reshapedPermuted`
 * according to `crops` to produce the output of shape: `[batch /
 * prod(blockShape),x.shape[1] * blockShape[0] - crops[0,0] - crops[0,1],
 * ..., x.shape[M] * blockShape[M-1] - crops[M-1,0] -
 * crops[M-1,1],x.shape[M+1], ..., x.shape[N-1]]`
 */
/** @doc {heading: 'Tensors', subheading: 'Transformations'} */
function batchToSpaceND_<T extends Tensor>(
    x: T|TensorLike, blockShape: number[], crops: number[][]): T {
  const $x = convertToTensor(x, 'x', 'batchToSpaceND');
  const prod = blockShape.reduce((a, b) => a * b);

  util.assert(
      $x.rank >= 1 + blockShape.length,
      () => `input rank is ${$x.rank} but should be > than blockShape.length ${
          blockShape.length}`);

  util.assert(
      crops.length === blockShape.length,
      () => `crops.length is ${
          crops.length} but should be equal to blockShape.length  ${
          blockShape.length}`);

  util.assert(
      $x.shape[0] % prod === 0,
      () => `input tensor batch is ${
                $x.shape[0]} but is not divisible by the product of ` +
          `the elements of blockShape ${blockShape.join(' * ')} === ${prod}`);

  const grad = (dy: T) => {
    return {$x: () => dy.spaceToBatchND(blockShape, crops)};
  };

  return ENGINE.runKernel(
      backend => backend.batchToSpaceND($x, blockShape, crops), {$x}, grad);
}

/**
 * This operation divides "spatial" dimensions `[1, ..., M]` of the input into
 * a grid of blocks of shape `blockShape`, and interleaves these blocks with
 * the "batch" dimension (0) such that in the output, the spatial
 * dimensions `[1, ..., M]` correspond to the position within the grid,
 * and the batch dimension combines both the position within a spatial block
 * and the original batch position. Prior to division into blocks,
 * the spatial dimensions of the input are optionally zero padded
 * according to `paddings`. See below for a precise description.
 *
 * ```js
 * const x = tf.tensor4d([1, 2, 3, 4], [1, 2, 2, 1]);
 * const blockShape = [2, 2];
 * const paddings = [[0, 0], [0, 0]];
 *
 * x.spaceToBatchND(blockShape, paddings).print();
 * ```
 *
 * @param x A `tf.Tensor`. N-D with `x.shape` = `[batch] + spatialShape +
 * remainingShape`, where spatialShape has `M` dimensions.
 * @param blockShape A 1-D array. Must have shape `[M]`, all values must
 * be >= 1.
 * @param paddings A 2-D array. Must have shape `[M, 2]`, all values must be >=
 *     0. `paddings[i] = [padStart, padEnd]` specifies the amount to zero-pad
 * from input dimension `i + 1`, which corresponds to spatial dimension `i`. It
 * is required that
 * `(inputShape[i + 1] + padStart + padEnd) % blockShape[i] === 0`
 *
 * This operation is equivalent to the following steps:
 *
 * 1. Zero-pad the start and end of dimensions `[1, ..., M]` of the input
 * according to `paddings` to produce `padded` of shape paddedShape.
 *
 * 2. Reshape `padded` to `reshapedPadded` of shape:
 * `[batch] + [paddedShape[1] / blockShape[0], blockShape[0], ...,
 * paddedShape[M] / blockShape[M-1], blockShape[M-1]] + remainingShape`
 *
 * 3. Permute dimensions of `reshapedPadded` to produce `permutedReshapedPadded`
 * of shape: `blockShape + [batch] + [paddedShape[1] / blockShape[0], ...,
 * paddedShape[M] / blockShape[M-1]] + remainingShape`
 *
 * 4. Reshape `permutedReshapedPadded` to flatten `blockShape` into the
 * batch dimension, producing an output tensor of shape:
 * `[batch * prod(blockShape)] + [paddedShape[1] / blockShape[0], ...,
 * paddedShape[M] / blockShape[M-1]] + remainingShape`
 */
/** @doc {heading: 'Tensors', subheading: 'Transformations'} */
function spaceToBatchND_<T extends Tensor>(
    x: T|TensorLike, blockShape: number[], paddings: number[][]): T {
  const $x = convertToTensor(x, 'x', 'spaceToBatchND');

  util.assert(
      $x.rank >= 1 + blockShape.length,
      () => `input rank ${$x.rank} should be > than [blockShape] ${
          blockShape.length}`);

  util.assert(
      paddings.length === blockShape.length,
      () => `paddings.shape[0] ${
          paddings.length} must be equal to [blockShape] ${blockShape.length}`);

  util.assert(
      $x.shape.reduce(
          (a, b, i) => {
            if (i > 0 && i <= blockShape.length) {
              return a &&
                  ((b + paddings[i - 1][0] + paddings[i - 1][1]) %
                       blockShape[i - 1] ===
                   0);
            }
            return a;
          },
          true),
      () => `input spatial dimensions ${$x.shape.slice(1)} with paddings ${
          paddings.toString()} must be divisible by blockShapes ${
          blockShape.toString()}`);

  const grad = (dy: T) => {
    return {$x: () => dy.batchToSpaceND(blockShape, paddings)};
  };

  return ENGINE.runKernel(
      backend => backend.spaceToBatchND($x, blockShape, paddings), {$x}, grad);
}

/**
 * Unstacks a `tf.Tensor` of rank-`R` into a list of rank-`(R-1)` `tf.Tensor`s.
 *
 * ```js
 * const a = tf.tensor2d([1, 2, 3, 4], [2, 2]);
 *
 * tf.unstack(a).forEach(tensor => tensor.print());
 * ```
 *
 * @param x A tensor object.
 * @param axis The axis to unstack along. Defaults to 0 (the first dim).
 */
/** @doc {heading: 'Tensors', subheading: 'Slicing and Joining'} */
function unstack_(x: Tensor|TensorLike, axis = 0): Tensor[] {
  axis = axis || 0;
  const $x = convertToTensor(x, 'x', 'unstack');
  util.assert(
      axis >= -$x.shape.length && axis < $x.shape.length,
      () =>
          `Axis = ${axis} is not in [-${$x.shape.length}, ${$x.shape.length})`);
  if (axis < 0) {
    axis += $x.shape.length;
  }
  const grad = (dy: Tensor[]) => {
    return {$x: () => stack(dy, axis)};
  };
  return ENGINE.runKernel(backend => backend.unstack($x, axis), {$x}, grad);
}

/**
 * Computes the cumulative sum of a `tf.Tensor` along `axis`.
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
/** @doc {heading: 'Operations', subheading: 'Scan'} */
function cumsum_<T extends Tensor>(
    x: Tensor|TensorLike, axis = 0, exclusive = false, reverse = false): T {
  const $x = convertToTensor(x, 'x', 'cumsum');

  axis = axis | 0;
  const permutation = getAxesPermutation([axis], $x.rank);
  let permutedX = $x;
  if (permutation != null) {
    permutedX = $x.transpose(permutation);
  }
  const permutedAxis = getInnerMostAxes(1, $x.rank)[0];

  const grad = (dy: T) => {
    return {permutedX: () => dy.cumsum(axis, exclusive, !reverse)};
  };
  let value = ENGINE.runKernel(
                  backend => backend.cumsum(
                      permutedX, permutedAxis, exclusive, reverse),
                  {permutedX}, grad) as T;

  if (permutation != null) {
    value = value.transpose(permutation);
  }
  return value;
}

/**
 * Returns a `tf.Tensor` that has expanded rank, by inserting a dimension
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
/** @doc {heading: 'Tensors', subheading: 'Transformations'} */
function expandDims_<R2 extends Rank>(
    x: Tensor|TensorLike, axis = 0): Tensor<R2> {
  const parseAs: DataType = null;
  const $x = convertToTensor(x, 'x', 'expandDims', parseAs);

  util.assert(axis <= $x.rank, () => 'Axis must be <= rank of the tensor');
  const newShape = $x.shape.slice();
  if (axis < 0) {
    // Negative value is counted from the tail of rank.
    util.assert(
        -($x.rank + 1) <= axis,
        () => `Axis must be in the interval [${- ($x.rank + 1)}, ${$x.rank}]`);
    axis = $x.rank + axis + 1;
  }
  newShape.splice(axis, 0, 1);
  return reshape($x, newShape as ShapeMap[R2]);
}

/**
 * Rearranges data from depth into blocks of spatial data. More specifically,
 * this op outputs a copy of the input tensor where values from the `depth`
 * dimension are moved in spatial blocks to the `height` and `width` dimensions.
 * The attr `blockSize` indicates the input block size and how the data is
 * moved.
 *
 *  - Chunks of data of size `blockSize * blockSize` from depth are rearranged
 * into non-overlapping blocks of size `blockSize x blockSize`
 *
 *  - The width the output tensor is `inputWidth * blockSize`, whereas the
 * height is `inputHeight * blockSize`
 *
 *  - The Y, X coordinates within each block of the output image are determined
 * by the high order component of the input channel index
 *
 *  - The depth of the input tensor must be divisible by `blockSize *
 * blockSize`
 *
 * The `dataFormat` attr specifies the layout of the input and output tensors
 * with the following options: "NHWC": [ `batch, height, width, channels` ]
 * "NCHW": [ `batch, channels, height, width` ]
 *
 * ```js
 * const x = tf.tensor4d([1, 2, 3, 4], [1, 1, 1, 4]);
 * const blockSize = 2;
 * const dataFormat = "NHWC";
 *
 * tf.depthToSpace(x, blockSize, dataFormat).print();
 * ```
 *
 * @param x The input tensor of rank 4
 * @param blockSIze  An `int` that is `>= 2`. The size of the spatial block
 * @param dataFormat An optional string from: "NHWC", "NCHW". Defaults to "NHWC"
 */
/** @doc {heading: 'Tensors', subheading: 'Transformations'} */
function depthToSpace_(
    x: Tensor4D|TensorLike4D, blockSize: number,
    dataFormat: 'NHWC'|'NCHW' = 'NHWC'): Tensor4D {
  const $x = convertToTensor(x, 'x', 'depthToSpace') as Tensor4D;

  const inputHeight = (dataFormat === 'NHWC') ? $x.shape[1] : $x.shape[2];
  const inputWidth = (dataFormat === 'NHWC') ? $x.shape[2] : $x.shape[3];
  const inputDepth = (dataFormat === 'NHWC') ? $x.shape[3] : $x.shape[1];

  util.assert(
      inputHeight * blockSize >= 0,
      () => `Negative dimension size caused by overflow when multiplying
      ${inputHeight} and ${blockSize}  for depthToSpace with input shape
      ${$x.shape}`);

  util.assert(
      inputWidth * blockSize >= 0,
      () => `Negative dimension size caused by overflow when multiplying
      ${inputWidth} and ${blockSize} for depthToSpace with input shape
          ${$x.shape}`);

  util.assert(
      (inputDepth % (blockSize * blockSize) === 0),
      () => `Dimension size must be evenly divisible by ${
          blockSize * blockSize} but is ${
          inputDepth} for depthToSpace with input shape ${$x.shape}`);

  return ENGINE.runKernel(
      backend => backend.depthToSpace($x, blockSize, dataFormat), {$x});
}

/**
 * Computes the difference between two lists of numbers.
 *
 * Given a Tensor `x` and a Tensor `y`, this operation returns a Tensor `out`
 * that represents all values that are in `x` but not in `y`. The returned
 * Tensor `out` is sorted in the same order that the numbers appear in `x`
 * (duplicates are preserved). This operation also returns a Tensor indices that
 * represents the position of each out element in `x`. In other words:
 *
 * `out[i] = x[idx[i]] for i in [0, 1, ..., out.length - 1]`
 *
 * ```js
 * const x = [1, 2, 3, 4, 5, 6];
 * const y = [1, 3, 5];
 *
 * const [out, indices] = await tf.setdiff1dAsync(x, y);
 * out.print(); // [2, 4, 6]
 * indices.print(); // [1, 3, 5]
 * ```
 *
 * @param x 1-D Tensor. Values to keep.
 * @param y 1-D Tensor. Must have the same type as x. Values to exclude in the
 *     output.
 * @returns Promise of Tensor tuple [out, indices].
 *  out: Tensor with the same type as x.
 *  indices: A Tensor of type int32.
 */
/** @doc {heading: 'Tensors', subheading: 'Transformations'} */
async function setdiff1dAsync_(
    x: Tensor|TensorLike, y: Tensor|TensorLike): Promise<[Tensor, Tensor]> {
  const $x = convertToTensor(x, 'x', 'setdiff1d');
  const $y = convertToTensor(y, 'y', 'setdiff1d');

  util.assert(
      $x.dtype === $y.dtype,
      () => `x and y should have the same dtype, but got x (${
          $x.dtype}) and y (${$y.dtype}).`);

  util.assert(
      $x.rank === 1, () => `x should be 1D tensor, but got x (${$x.shape}).`);

  util.assert(
      $y.rank === 1, () => `y should be 1D tensor, but got y (${$y.shape}).`);

  const xVals = await $x.data();
  const yVals = await $y.data();
  const ySet = new Set(yVals);

  let outputSize = 0;
  for (let i = 0; i < xVals.length; i++) {
    if (!ySet.has(xVals[i])) {
      outputSize++;
    }
  }

  const buffer = new TensorBuffer([outputSize], $x.dtype);
  const indices = new TensorBuffer([outputSize], 'int32');
  for (let i = 0, p = 0; i < xVals.length; i++) {
    if (!ySet.has(xVals[i])) {
      buffer.values[p] = xVals[i];
      indices.values[p] = i;
      p++;
    }
  }
  return [buffer.toTensor(), indices.toTensor()];
}

/**
 * Creates an empty `tf.TensorBuffer` with the specified `shape` and `dtype`.
 *
 * The values are stored in CPU as `TypedArray`. Fill the buffer using
 * `buffer.set()`, or by modifying directly `buffer.values`.
 *
 * When done, call `buffer.toTensor()` to get an immutable `tf.Tensor` with
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
/** @doc {heading: 'Tensors', subheading: 'Creation'} */
function buffer<R extends Rank, D extends DataType = 'float32'>(
    shape: ShapeMap[R], dtype: D = 'float32' as D,
    values?: DataTypeMap[D]): TensorBuffer<R, D> {
  dtype = dtype || 'float32' as D;
  util.assertNonNegativeIntegerDimensions(shape);
  return new TensorBuffer<R, D>(shape, dtype, values);
}

/**
 * Prints information about the `tf.Tensor` including its data.
 *
 * ```js
 * const verbose = true;
 * tf.tensor2d([1, 2, 3, 4], [2, 2]).print(verbose);
 * ```
 * @param x The tensor to be printed.
 * @param verbose Whether to print verbose information about the ` Tensor`,
 * including dtype and size.
 */
/** @doc {heading: 'Tensors', subheading: 'Creation'} */
function print<T extends Tensor>(x: T, verbose = false): void {
  console.log(x.toString(verbose));
}

export {
  buffer,  // Not wrapped in op() since no tensors.
  print    // Not wrapped in op() since no need to increase stack trace.
};

export const batchToSpaceND = op({batchToSpaceND_});
export const cast = op({cast_});
export const clone = op({clone_});
export const cumsum = op({cumsum_});
export const depthToSpace = op({depthToSpace_});
export const expandDims = op({expandDims_});
export const eye = op({eye_});
export const multinomial = op({multinomial_});
export const oneHot = op({oneHot_});
export const pad = op({pad_});
export const pad1d = op({pad1d_});
export const pad2d = op({pad2d_});
export const pad3d = op({pad3d_});
export const pad4d = op({pad4d_});
export const rand = op({rand_});
export const randomNormal = op({randomNormal_});
export const randomUniform = op({randomUniform_});
export const reshape = op({reshape_});
export const spaceToBatchND = op({spaceToBatchND_});
export const squeeze = op({squeeze_});
export const stack = op({stack_});
export const tile = op({tile_});
export const truncatedNormal = op({truncatedNormal_});
export const unstack = op({unstack_});
export const setdiff1dAsync = setdiff1dAsync_;
