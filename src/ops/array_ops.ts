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
import {ENV} from '../environment';
import {Tensor, Tensor1D, Tensor2D, Tensor3D, Tensor4D, TensorBuffer} from '../tensor';
import {convertToTensor, convertToTensorArray} from '../tensor_util_env';
import {DataType, Rank, ShapeMap, TensorLike, TensorLike1D, TypedArray} from '../types';
import * as util from '../util';
import {getAxesPermutation, getInnerMostAxes, parseAxisParam} from './axis_util';
import {concat} from './concat';
import {op} from './operation';
import {MPRandGauss} from './rand';
import {zerosLike} from './tensor_ops';

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
  const $x = convertToTensor(x, 'x', 'clone');
  const der = (dy: T) => {
    return {$x: () => dy.toFloat()};
  };

  return ENV.engine.runKernel(
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
 *   of the shape of the returned `Tensor` by repeating the identity
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
/** @doc {heading: 'Tensors', subheading: 'Random'} */
function randomUniform_<R extends Rank>(
    shape: ShapeMap[R], minval = 0, maxval = 1,
    dtype: DataType = 'float32'): Tensor<R> {
  const res = buffer(shape, dtype);
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
/** @doc {heading: 'Tensors', subheading: 'Creation'} */
function oneHot_(
    indices: Tensor1D|TensorLike1D, depth: number, onValue = 1,
    offValue = 0): Tensor2D {
  const $indices = convertToTensor(indices, 'indices', 'oneHot', 'int32');
  util.assert($indices.dtype === 'int32', 'Indices must be of dtype `int32`');

  if (depth < 2) {
    throw new Error(`Error in oneHot: depth must be >=2, but it is ${depth}`);
  }
  return ENV.engine.runKernel(
      backend => backend.oneHot($indices, depth, onValue, offValue),
      {$indices});
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
/** @doc {heading: 'Tensors', subheading: 'Creation'} */
function fromPixels_(
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
/** @doc {heading: 'Visualization'} */
async function toPixels(
    img: Tensor2D|Tensor3D|TensorLike,
    canvas?: HTMLCanvasElement): Promise<Uint8ClampedArray> {
  const $img = convertToTensor(img, 'img', 'toPixels', 'int32');
  if ($img.rank !== 2 && $img.rank !== 3) {
    throw new Error(
        `toPixels only supports rank 2 or 3 tensors, got rank ${$img.rank}.`);
  }
  const [height, width] = $img.shape.slice(0, 2);
  const depth = $img.rank === 2 ? 1 : $img.shape[2];

  if (depth > 4 || depth === 2) {
    throw new Error(
        `toPixels only supports depth of size ` +
        `1, 3 or 4 but got ${depth}`);
  }

  const minTensor = $img.min();
  const maxTensor = $img.max();
  const min = (await minTensor.data())[0];
  const max = (await maxTensor.data())[0];
  minTensor.dispose();
  maxTensor.dispose();
  if ($img.dtype === 'float32') {
    if (min < 0 || max > 1) {
      throw new Error(
          `Tensor values for a float32 Tensor must be in the ` +
          `range [0 - 1] but got range [${min} - ${max}].`);
    }
  } else if ($img.dtype === 'int32') {
    if (min < 0 || max > 255) {
      throw new Error(
          `Tensor values for a int32 Tensor must be in the ` +
          `range [0 - 255] but got range [${min} - ${max}].`);
    }
  } else {
    throw new Error(
        `Unsupported type for toPixels: ${$img.dtype}.` +
        ` Please use float32 or int32 tensors.`);
  }

  const data = await $img.data();
  const multiplier = $img.dtype === 'float32' ? 255 : 1;
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
  if ($img !== img) {
    $img.dispose();
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
/** @doc {heading: 'Tensors', subheading: 'Transformations'} */
function reshape_<R2 extends Rank>(
    x: Tensor|TensorLike, shape: ShapeMap[R2]): Tensor<R2> {
  const $x = convertToTensor(x, 'x', 'reshape');
  shape = util.inferFromImplicitShape(shape, $x.size);
  util.assert(
      $x.size === util.sizeFromShape(shape),
      'new shape and old shape must have the same number of elements.');

  const grad = (dy: Tensor<R2>) => {
    return {$x: () => dy.reshape($x.shape)};
  };
  return ENV.engine.runKernel(
      backend => backend.reshape($x, shape), {$x}, grad);
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
/** @doc {heading: 'Tensors', subheading: 'Transformations'} */
function squeeze_<T extends Tensor>(x: Tensor|TensorLike, axis?: number[]): T {
  const $x = convertToTensor(x, 'x', 'squeeze');
  return reshape($x, util.squeezeShape($x.shape, axis).newShape) as T;
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
/** @doc {heading: 'Tensors', subheading: 'Transformations'} */
function cast_<T extends Tensor>(x: T|TensorLike, dtype: DataType): T {
  const $x = convertToTensor(x, 'x', 'cast');

  const grad = (dy: T) => {
    return {$x: () => dy.clone()};
  };
  return ENV.engine.runKernel(backend => backend.cast($x, dtype), {$x}, grad) as
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
/** @doc {heading: 'Tensors', subheading: 'Slicing and Joining'} */
function tile_<T extends Tensor>(x: T|TensorLike, reps: number[]): T {
  const $x = convertToTensor(x, 'x', 'tile');

  util.assert(
      $x.rank === reps.length,
      `Error in transpose: rank of input ${$x.rank} ` +
          `must match length of reps ${reps}.`);
  const grad = (dy: T) => {
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
      return xGrad;
    };
    return {$x: derX};
  };
  return ENV.engine.runKernel(backend => backend.tile($x, reps), {$x}, grad);
}

/**
 * Pads a `Tensor1D` with a given value and paddings. See `pad` for details.
 */
function pad1d_(
    x: Tensor1D|TensorLike, paddings: [number, number],
    constantValue = 0): Tensor1D {
  util.assert(
      paddings.length === 2,
      'Invalid number of paddings. Must be length of 2.');
  return pad(x, [paddings], constantValue);
}

/**
 * Pads a `Tensor2D` with a given value and paddings. See `pad` for details.
 */
function pad2d_(
    x: Tensor2D|TensorLike, paddings: [[number, number], [number, number]],
    constantValue = 0): Tensor2D {
  util.assert(
      paddings.length === 2 && paddings[0].length === 2 &&
          paddings[1].length === 2,
      'Invalid number of paddings. Must be length of 2 each.');
  return pad(x, paddings, constantValue);
}

/**
 * Pads a `Tensor3D` with a given value and paddings. See `pad` for details.
 */
function pad3d_(
    x: Tensor3D|TensorLike,
    paddings: [[number, number], [number, number], [number, number]],
    constantValue = 0): Tensor3D {
  util.assert(
      paddings.length === 3 && paddings[0].length === 2 &&
          paddings[1].length === 2 && paddings[2].length === 2,
      'Invalid number of paddings. Must be length of 2 each.');
  return pad(x, paddings, constantValue);
}

/**
 * Pads a `Tensor4D` with a given value and paddings. See `pad` for details.
 */
function pad4d_(
    x: Tensor4D|TensorLike,
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
  return pad(x, paddings, constantValue);
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
  return ENV.engine.runKernel(
             backend => backend.pad($x, paddings, constantValue), {$x}, grad) as
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
/** @doc {heading: 'Tensors', subheading: 'Slicing and Joining'} */
function stack_<T extends Tensor>(tensors: T[]|TensorLike[], axis = 0): Tensor {
  const $tensors = convertToTensorArray(tensors, 'tensors', 'stack');

  util.assert($tensors.length >= 1, 'Pass at least one tensor to tf.stack');
  if ($tensors.length === 1) {
    return $tensors[0].expandDims(axis);
  }
  const rank = $tensors[0].rank;
  const shape = $tensors[0].shape;
  const dtype = $tensors[0].dtype;

  util.assert(axis <= rank, 'Axis must be <= rank of the tensor');

  $tensors.forEach(t => {
    util.assertShapesMatch(
        shape, t.shape,
        'All tensors passed to stack must have matching shapes');
  });

  $tensors.forEach(t => {
    util.assert(
        dtype === t.dtype,
        'All tensors passed to stack must have matching dtypes');
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
 * output. This is the reverse of `spaceToBatchND`. See below for a precise
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
 * @param x A `Tensor`. N-D with `x.shape` = `[batch] + spatialShape +
 * remainingShape`, where spatialShape has `M` dimensions.
 * @param blockShape A 1-D array. Must be one of the following types: `int32`,
 * `int64`. Must have shape `[M]`, all values must be >= 1.
 * @param crops A 2-D array.  Must be one of the following types: `int32`,
 * `int64`. Must have shape `[M, 2]`, all values must be >= 0. `crops[i] =
 * [cropStart, cropEnd]` specifies the amount to crop from input dimension `i
 * + 1`, which corresponds to spatial dimension `i`. It is required that
 * `cropStart[i] + cropEnd[i] <= blockShape[i] * inputShape[i + 1]`
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
      `input rank should be > than [blockShape] but got ${$x.rank}`);

  util.assert(
      crops.length === blockShape.length,
      `crops.shape[0] must be equal to [blockShape] but got ${crops.length}`);

  util.assert(
      $x.shape[0] % prod === 0,
      `input tensor batch must be divisible by prod( blockShape )`);
  return ENV.engine.runKernel(
      backend => backend.batchToSpaceND($x, blockShape, crops), {});
}

/**
 * This operation divides "spatial" dimensions [1, ..., M] of the input into
 * a grid of blocks of shape block_shape, and interleaves these blocks with
 * the "batch" dimension (0) such that in the output, the spatial
 * dimensions [1, ..., M] correspond to the position within the grid,
 * and the batch dimension combines both the position within a spatial block
 * and the original batch position. Prior to division into blocks,
 * the spatial dimensions of the input are optionally zero padded
 * according to paddings. See below for a precise description.
 *
 * ```js
 * const x = tf.tensor4d([1, 2, 3, 4], [4, 1, 1, 1]);
 * const blockShape = [2, 2];
 * const paddings = [[0, 0], [0, 0]];
 *
 * x.spaceToBatchND(blockShape, paddings).print();
 * ```
 *
 * @param x A `Tensor`. N-D with `x.shape` = `[batch] + spatialShape +
 * remainingShape`, where spatialShape has `M` dimensions.
 * @param blockShape A 1-D array. Must be one of the following types: `int32`,
 * `int64`. Must have shape `[M]`, all values must be >= 1.
 * @param paddings A 2-D array.  Must be one of the following types: `int32`,
 * `int64`. Must have shape `[M, 2]`, all values must be >= 0. `paddings[i] =
 * [padStart, padEnd]` specifies the amount to zero-pad from input dimension
 * `i + 1`, which corresponds to spatial dimension `i`.
 * It is required that
 * `(inputShape[i + 1] + padStart + padEnd) % blockShape[i] === 0`
 *
 * This operation is equivalent to the following steps:
 *
 * 1. Zero-pad the start and end of dimensions [1, ..., M] of the input
 * according to paddings to produce padded of shape padded_shape.
 *
 * 2. Reshape padded to reshaped_padded of shape:
 * [batch] + [padded_shape[1] / block_shape[0], block_shape[0], ...,
 * padded_shape[M] / block_shape[M-1], block_shape[M-1]] + remaining_shape
 *
 * 3. Permute dimensions of reshaped_padded to produce permuted_
 * reshaped_padded of shape:
 * block_shape + [batch] + [padded_shape[1] / block_shape[0], ...,
 * padded_shape[M] / block_shape[M-1]] + remaining_shape
 *
 * 4. Reshape permuted_reshaped_padded to flatten block_shape into the
 * batch dimension, producing an output tensor of shape:
 * [batch * prod(block_shape)] + [padded_shape[1] / block_shape[0], ...,
 * padded_shape[M] / block_shape[M-1]] + remaining_shape
 */
/** @doc {heading: 'Tensors', subheading: 'Transformations'} */
function spaceToBatchND_<T extends Tensor>(
    x: T|TensorLike, blockShape: number[], paddings: number[][]): T {
  const $x = convertToTensor(x, 'x', 'spaceToBatchND');

  util.assert(
      $x.rank >= 1 + blockShape.length,
      `input rank should be > than [blockShape] but got ${$x.rank}`);

  util.assert(
      paddings.length === blockShape.length,
      `paddings.shape[0] must be equal to [blockShape], got ${
          paddings.length}`);

  util.assert($x.shape.reduce((a, b, i) => {
    if (i > 0 && i <= blockShape.length) {
      return a && (b % blockShape[i - 1] === 0);
    }
    return a;
  }, true), `input spatial dimensions must be divisible by blockShapes`);
  return ENV.engine.runKernel(
      backend => backend.spaceToBatchND($x, blockShape, paddings), {});
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
 * @param x A tensor object.
 * @param axis The axis to unstack along. Defaults to 0 (the first dim).
 */
/** @doc {heading: 'Tensors', subheading: 'Slicing and Joining'} */
function unstack_<T extends Tensor>(x: T|TensorLike, axis = 0): Tensor[] {
  const $x = convertToTensor(x, 'x', 'unstack');
  const num = $x.shape[axis];
  const outputShape: number[] = Array($x.rank - 1).fill(0);
  let outIndex = 0;
  for (let i = 0; i < $x.rank; i++) {
    if (i !== axis) {
      outputShape[outIndex] = $x.shape[i];
      outIndex++;
    }
  }

  let splitSizes: number[];
  splitSizes = Array(num).fill(1);
  const begin = Array($x.rank).fill(0);
  const size = $x.shape.slice();
  return splitSizes.map(s => {
    size[axis] = s;
    const slice = $x.slice(begin, size);
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
/** @doc {heading: 'Tensors', subheading: 'Slicing and Joining'} */
function split_<T extends Tensor>(
    x: T|TensorLike, numOrSizeSplits: number[]|number, axis = 0): T[] {
  const $x = convertToTensor(x, 'x', 'split');

  axis = parseAxisParam(axis, $x.shape)[0];
  let splitSizes: number[];
  if (typeof (numOrSizeSplits) === 'number') {
    util.assert(
        $x.shape[axis] % numOrSizeSplits === 0,
        'Number of splits must evenly divide the axis.');
    splitSizes = Array(numOrSizeSplits).fill($x.shape[axis] / numOrSizeSplits);
  } else {
    util.assert(
        $x.shape[axis] === numOrSizeSplits.reduce((a, b) => a + b),
        'The sum of sizes must match the size of the axis dimension.');
    splitSizes = numOrSizeSplits;
  }
  const begin = Array($x.rank).fill(0);
  const size = $x.shape.slice();
  return splitSizes.map(s => {
    size[axis] = s;
    const slice = $x.slice(begin, size);
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
/** @doc {heading: 'Tensors', subheading: 'Transformations'} */
function expandDims_<R2 extends Rank>(
    x: Tensor|TensorLike, axis = 0): Tensor<R2> {
  const $x = convertToTensor(x, 'x', 'expandDims');

  util.assert(axis <= $x.rank, 'Axis must be <= rank of the tensor');
  const newShape = $x.shape.slice();
  if (axis < 0) {
    // Negative value is counted from the tail of rank.
    util.assert(
        -($x.rank + 1) <= axis,
        `Axis must be in the interval [${- ($x.rank + 1)}, ${$x.rank}]`);
    axis = $x.rank + axis + 1;
  }
  newShape.splice(axis, 0, 1);
  return reshape($x, newShape);
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
/** @doc {heading: 'Tensors', subheading: 'Creation'} */
function buffer<R extends Rank>(
    shape: ShapeMap[R], dtype: DataType = 'float32',
    values?: TypedArray): TensorBuffer<R> {
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
/** @doc {heading: 'Tensors', subheading: 'Creation'} */
function print<T extends Tensor>(x: T, verbose = false): void {
  console.log(x.toString(verbose));
}

export {
  buffer,    // Not wrapped in op() since no tensors.
  toPixels,  // Not wrapped in op() since async.
  print      // Not wrapped in op() since no need to increase stack trace.
};

export const cast = op({cast_});
export const clone = op({clone_});
export const cumsum = op({cumsum_});
export const expandDims = op({expandDims_});
export const eye = op({eye_});
export const fromPixels = op({fromPixels_});
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
export const split = op({split_});
export const squeeze = op({squeeze_});
export const stack = op({stack_});
export const tile = op({tile_});
export const truncatedNormal = op({truncatedNormal_});
export const unstack = op({unstack_});
export const batchToSpaceND = op({batchToSpaceND_});
export const spaceToBatchND = op({spaceToBatchND_});
