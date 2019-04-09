/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

/**
 * deeplearn.js backend.
 */

import * as tfc from '@tensorflow/tfjs-core';
import {onesLike as coreOnesLike, scalar, Tensor, Tensor1D, tensor1d, Tensor2D, Tensor3D, Tensor4D, tidy, util, where, zerosLike as coreZerosLike} from '@tensorflow/tfjs-core';
import {checkDataFormat} from '../common';
import {NotImplementedError, ValueError} from '../errors';
import {DataFormat, Shape} from '../keras_format/common';
import {HasShape} from '../types';
import * as math_utils from '../utils/math_utils';

import {imageDataFormat} from './common';

// tslint:enable

/* Setting and getting backend from deeplearn.js. */

// Default deeplearn.js backend is WebGL (GPU).
let backend: 'cpu'|'webgl' = 'webgl';

export function setBackend(requestedBackend: 'cpu'|'webgl') {
  tfc.setBackend(requestedBackend);
  backend = requestedBackend;
}

export function getBackend(): 'cpu'|'webgl' {
  return backend;
}

/**
 * Indicates whether the backend is operating symbolically.
 *
 * This function will be used to determine how to interpret user code. If
 * it returns true, calls to the backend construct a symbolic graph; if
 * it returns false, calls to the backend execute immediately.
 */
export function isBackendSymbolic(): boolean {
  return false;
}

/**
 * Get the number of elements in a Tensor.
 * @param x The Tensor.
 * @return Number of elements in `x`.
 */
export function countParams(x: HasShape): number {
  const shape = x.shape;
  if (shape.length > 0) {
    return shape.reduce((a: number, b: number) => a * b);
  } else {
    // Scalar.
    return 1;
  }
}

/**
 * Casts a tensor to a different dtype and returns it.
 * @param x Input tensor.
 * @param dtype String: 'float32'|'int32'|'bool'.
 * @returns Tensor of the specified `dtype`.
 */
export function cast(x: Tensor, dtype: tfc.DataType): Tensor {
  return x.asType(dtype);
}

/**
 * Adds a 1-sized dimension at index "axis".
 * @param x Input tensor.
 * @param axis Position where to add the new axis.
 * @returns Result of the dimension expansion.
 */
export function expandDims(x: Tensor, axis = -1): Tensor {
  const outShape = x.shape.slice();
  if (axis < 0) {
    axis = outShape.length + axis + 1;
  }
  outShape.splice(axis, 0, 1);
  return x.reshape(outShape);
}

/**
 * Repeats a 2D tensor.
 *
 * If `x` has shape `[samples, dim]` and `n` is 2, for example, the output
 * will have shape `[samples, 2, dim]`.
 *
 * @param x Input tensor.
 * @param n Integer, number of times to repeat.
 * @returns The result of the repeat operation.
 * @throws ValueError: If input tensor is not 2D.
 */
export function repeat(x: Tensor, n: number): Tensor {
  return tidy(() => {
    if (x.shape.length !== 2) {
      throw new ValueError(
          `repeat() expects a rank-2 tensor, but received a ` +
          `rank-${x.shape.length} tensor.`);
    }
    const y = expandDims(x, 1);
    return tile(y, [1, n, 1]);
  });
}

/**
 * Flatten an Tensor into 1D.
 * @param x Input tensor.
 * @return The result of the flattening `x`.
 */
export function flatten(x: Tensor): Tensor {
  const newShape = [math_utils.arrayProd(x.shape)];
  return x.reshape(newShape);
}

/**
 * Turn a nD tensor into a 2D tensor with same 0th dimension.
 * In other words, it flattens each data samples of a batch.
 *
 * @param x The tensor to flatten. The rank of this tensor is required to be 2
 *   or higher.
 * @return The result of the flattening.
 */
export function batchFlatten(x: Tensor): Tensor {
  if (x.rank <= 1) {
    throw new ValueError(
        `batchFlatten requires a minimum rank of 2. Got rank: ${x.rank}.`);
  }
  const newShape = [x.shape[0], math_utils.arrayProd(x.shape, 1)];
  return x.reshape(newShape);
}

/**
 * Do slicing along the first axis.
 * @param array input `tf.Tensor`.
 * @param start starting index, inclusive.
 * @param size size of the slice along the first axis.
 * @returns result of the slicing.
 * @throws ValueError: If `array` is of an unsupported subtype of `tf.Tensor`.
 */
export function sliceAlongFirstAxis(
    array: Tensor, start: number, size: number): Tensor {
  return tidy(() => {
    switch (array.rank) {
      case 1:
        return tfc.slice1d(array as Tensor1D, start, size);
      case 2:
        return tfc.slice2d(
            array as Tensor2D, [start, 0], [size, array.shape[1]]);
      case 3:
        return tfc.slice3d(
            array as Tensor3D, [start, 0, 0],
            [size, array.shape[1], array.shape[2]]);
      case 4:
        return tfc.slice4d(
            array as Tensor4D, [start, 0, 0, 0],
            [size, array.shape[1], array.shape[2], array.shape[3]]);
      default:
        throw new ValueError(
            `sliceAlongFirstAxis() received an unsupported tensor rank: ` +
            `${array.rank}`);
    }
  });
}

/**
 * Do slicing along the last axis.
 * @param array input `tf.Tensor`.
 * @param start starting index, inclusive.
 * @param size size of the slice along the last axis.
 * @returns result of the slicing.
 * @throws ValueError: If `array` is of an unsupported subtype of `tf.Tensor`.
 */
export function sliceAlongLastAxis(
    array: Tensor, start: number, size: number): Tensor {
  return tidy(() => {
    switch (array.rank) {
      case 1:
        return tfc.slice1d(array as Tensor1D, start, size);
      case 2:
        return tfc.slice2d(
            array as Tensor2D, [0, start], [array.shape[0], size]);
      case 3:
        return tfc.slice3d(
            array as Tensor3D, [0, 0, start],
            [array.shape[0], array.shape[1], size]);
      case 4:
        return tfc.slice4d(
            array as Tensor4D, [0, 0, 0, start],
            [array.shape[0], array.shape[1], array.shape[2], size]);
      default:
        throw new ValueError(
            `sliceAlongLastAxis() received an unsupported tensor rank: ` +
            `${array.rank}`);
    }
  });
}

/**
 * Do slicing along the sepcified axis.
 * @param array input `tf.Tensor`.
 * @param start starting index, inclusive.
 * @param size of the slice along the chosen axis.
 * @param choose an axis.
 * @returns result of the slicing.
 * @throws ValueError: If `array` is of an unsupported subtype of `tf.Tensor`.
 */
export function sliceAlongAxis(
    array: Tensor, start: number, size: number, axis: number): Tensor {
  return tidy(() => {
    switch (array.rank) {
      case 1:
        return tfc.slice1d(array as Tensor1D, start, size);
      case 2:
        switch (axis) {
          case 1:
            return sliceAlongFirstAxis(array, start, size);
          case 2:
            return sliceAlongLastAxis(array, start, size);
          default:
            throw new ValueError(
                `The axis is not within the rank of the tensor ` +
                `${axis}`);
        }
      case 3:
        switch (axis) {
          case 1:
            return sliceAlongFirstAxis(array, start, size);
          case 2:
            return tfc.slice3d(
                array as Tensor3D, [0, start, 0],
                [array.shape[0], size, array.shape[2]]);
          case 3:
            return sliceAlongLastAxis(array, start, size);
          default:
            throw new ValueError(
                `The axis is not within the rank of the tensor ` +
                `${axis}`);
        }
      case 4:
        switch (axis) {
          case 1:
            return sliceAlongFirstAxis(array, start, size);
          case 2:
            return tfc.slice4d(
                array as Tensor4D, [0, start, 0, 0],
                [array.shape[0], size, array.shape[2], array.shape[3]]);
          case 3:
            return tfc.slice4d(
                array as Tensor4D, [0, 0, start, 0],
                [array.shape[0], array.shape[1], size, array.shape[3]]);
          case 4:
            return sliceAlongLastAxis(array, start, size);
          default:
            throw new ValueError(
                `The axis is not within the rank of the tensor ` +
                `${axis}`);
        }
      default:
        throw new ValueError(
            `sliceAlongLastAxis() received an unsupported tensor rank: ` +
            `${array.rank}`);
    }
  });
}

/**
 * Concatenates a list of tensors alongside the specified axis.
 * @param tensors `Array` of tensors to concatenate.
 * @param axis Concatenation axis.
 * @returns The result of the concatenation.
 */
export function concatenate(tensors: Tensor[], axis = -1): Tensor {
  let rank: number;
  if (axis < 0) {
    rank = tensors[0].rank;
    if (rank !== 0) {
      axis = rank;
    } else {
      axis = 0;
    }
  }
  if (axis === tensors[0].rank) {
    // Porting Note: This is necessary because tfc.concat() requires axis to be
    //   in the interval [-rank, rank).
    axis = -1;
  }
  // Porting Note: Sparse concat is not supported yet.
  return tfc.concat(tensors, axis);
}

/**
 * Concatenate two arrays along the first dimension.
 * @param a The 1st `tf.Tensor` to concatenate.
 * @param b The 2nd `tf.Tensor` to concatenate.
 * @returns Result of the concatenation.
 * @throws ValueError: If `a` is of an unsupported subtype of `tf.Tensor`.
 */
export function concatAlongFirstAxis(a: Tensor, b: Tensor): Tensor {
  switch (a.rank) {
    case 1:
      return tfc.concat1d([a as Tensor1D, b as Tensor1D]);
    case 2:
      return tfc.concat2d([a as Tensor2D, b as Tensor2D], 0);
    case 3:
      return tfc.concat3d([a as Tensor3D, b as Tensor3D], 0);
    case 4:
      return tfc.concat4d([a as Tensor4D, b as Tensor4D], 0);
    default:
      throw new ValueError(
          'concatAlongFirstAxis() received an unsupported tensor rank: ' +
          a.rank);
  }
}

/**
 * Creates a tensor by tiling `x` by `n`.
 * @param x A tensor.
 * @param n An Array of integers or a single integer. If an Array, the length
 *   must be the same as the number of dimensions in `x`. If a single integer,
 *   it will be treated as an Array of length 1.
 */
export function tile(x: Tensor, n: number|number[]): Tensor {
  if (!Array.isArray(n)) {
    n = [n];
  }
  if (x.rank !== n.length) {
    throw new ValueError(
        `The length of input n (${n.length}) does not match ` +
        `the number of dimensions in input x (${x.rank})`);
  }
  return tfc.tile(x, n);
}

/* Creation of random tensors. */


/**
 * Get a tensor with normal distribution of values.
 *
 * @param shape Shape of the tensor.
 * @param mean mean value of the normal distribution.
 * @param stddev standard deviation of the normal distribution.
 * @param dtype
 * @param seed
 * @return The normal tensor.
 */
export function randomNormal(
    shape: Shape, mean = 0.0, stddev = 1.0, dtype?: 'float32'|'int32',
    seed?: number): Tensor {
  return tfc.randomNormal(shape, mean, stddev, dtype, seed);
}

/* Linear Algebra */

/**
 * Multiply two tensors and returns the result as a tensor.
 *
 * For 2D tensors, this is equivalent to matrix multiplication (matMul).
 * For tensors of higher ranks, it follows the Theano behavior,
 * (e.g. `(2, 3) * (4, 3, 5) -> (2, 4, 5)`).  From the Theano documentation:
 *
 * For N dimensions it is a sum product over the last axis of x and the
 * second-to-last of y:
 *
 * @param x A tensor of at least rank 2.
 * @param y A tensor of at least rank 2.
 * @param fusedActivation (optional) A string identifying the activation
 *   function.
 * @return Result of the dot operation.
 */
export function dot(
    x: Tensor, y: Tensor, fusedActivation?: tfc.fused.Activation,
    bias?: Tensor): Tensor {
  if ((x.rank < 2) || (y.rank < 2)) {
    throw new NotImplementedError(
        `dot requires both inputs to be rank >= 2` +
        ` but got x shape = ${x.shape} and y shape = ${y.shape}`);
  }
  if (y.rank >= 3) {
    const xLastDim = x.shape.slice(-1)[0];
    const ySecondLastDim = y.shape.slice(-2)[0];
    if (xLastDim !== ySecondLastDim) {
      throw new NotImplementedError(
          `If rank y >= 3, then the second last dim` +
          ` of y must equal the last dim of x but got x shape = ${
              x.shape} and ` +
          ` y shape = ${y.shape}`);
    }
  }
  // Handle basic 2D x 2D case.
  if ((x.rank === 2) && (y.rank === 2)) {
    const transposeX = false;
    const transposeY = false;
    // tfc.fused.matMul only fuses certain activation functions. Unsupported
    // activation functions are treated as 'linear' activations, which is
    // equivalent to a no-op.
    return tfc.fused.matMul(
        x as Tensor2D, y as Tensor2D, transposeX, transposeY,
        bias ? reshapeBias(x.rank, bias, imageDataFormat()) : null,
        fusedActivation);
  } else {
    // Reshape x into the analogous 2D Tensor.
    const xFirstDims = x.shape.slice();  // Holds all but the last dim of x.
    const xLastDim = xFirstDims.pop();
    x = x.reshape([-1, xLastDim]);

    // Reshape y into the analogous 2D Tensor, and keep track of the
    // required dimensions to reproduce the output shape.
    const yShape = y.shape.slice();
    const yLastDim = yShape.pop();
    const ySecondLastDim = yShape.pop();
    const yOtherDims = [...yShape, yLastDim];
    // permutation should be like [r-2, 0, 1, 2, ... r-4, r-3, r-1]
    // where r is the rank of y.
    const perm = Array.from({length: y.rank}, (_, i) => {
      if (i === 0) {
        return y.rank - 2;
      } else if (i <= y.rank - 2) {
        return i - 1;
      }
      return i;
    });
    y = y.transpose(perm).reshape([ySecondLastDim, -1]);

    // Multiply x and y as 2D Tensors, and then reshape back to original.
    const outputShape = [...xFirstDims, ...yOtherDims];
    const transposeX = false;
    const transposeY = false;
    return tfc.fused
        .matMul(
            x as Tensor2D, y as Tensor2D, transposeX, transposeY,
            bias ? reshapeBias(x.rank, bias, imageDataFormat()) : null,
            fusedActivation)
        .reshape(outputShape);
  }
}

/**
 * Compute the sign Tensor of an input Tensor.
 *
 * Elements of the input `tf.Tensor` that are === 0 are mapped to 0.
 * Elements of the input `tf.Tensor` that are > 0 are mapped to 1.
 * Elements of the input `tf.Tensor` that are < 0 are mapped to -1.
 *
 * @param x Input `tf.Tensor`.
 * @return The sign `tf.Tensor`.
 */
export function sign(x: Tensor): Tensor {
  // TODO(cais): Move to the core.
  return tidy(() => {
    const zerosLikeX = coreZerosLike(x);
    const onesLikeX = coreOnesLike(x);
    return where(
        tfc.equal(x, zerosLikeX), zerosLikeX,
        where(
            tfc.greater(x, coreZerosLike(x)), onesLikeX,
            tfc.mul(-1, onesLikeX)));
  });
}

/**
 * Computes the one-hot representation of an integer tensor.
 * @param indices nD integer tensor of shape
 *   `(batch_size, dim1, dim2, ... dim(n-1))`
 * @param numClasses Integer, number of classes to consider.
 * @returns (n + 1)D one hot representation of the input
 *   with shape `(batch_size, dim1, dim2, ... dim(n-1), num_classes)`
 */
export function oneHot(indices: Tensor, numClasses: number): Tensor {
  return tidy(() => {
    if (indices.rank !== 1) {
      throw new Error(
          'Only 1D one-hot tensors are supported in the ' +
          'deeplearn backend, at present.');
    }
    indices = indices.toInt();
    return tfc.oneHot(indices as Tensor1D, numClasses).toFloat();
  });
}

/* Elementary math functions. */

/**
 * Retrieves the elements of indices `indices` in the tensor `reference`.
 * @param reference A tensor.
 * @param indices An integer tensor of indices or an `Array` of integers.
 * @param axis Axis along which to perform the gather operation.
 * @returns The result of the gathering as a tensor.
 */
export function gather(
    reference: Tensor, indices: number[]|Tensor1D, axis?: number): Tensor {
  return tidy(() => {
    if (Array.isArray(indices)) {
      indices = tensor1d(indices, 'int32');
    } else {
      indices = indices.toInt();
    }
    return tfc.gather(reference, indices, axis);
  });
}

/**
 * Element-wise square.
 * @param x Input tensor.
 * @return element-wise x^2
 */
export function square(x: Tensor): Tensor {
  return tfc.mulStrict(x, x);
}

/**
 * Element-wise exponentiation.
 *
 * Porting Note: In PyKeras, `a` (the exponent) is a Python integer, which
 *   takes advatnage of the backend's (e.g., TensorFlow's) automatic conversion
 *   to tensor. Here we allow `a` to be either a number or a tensor.
 *
 * @param x The base tensor.
 * @param a The exponent, tensor or number. If a number, it is rounded to the
 *   nearest integer and converted to a tensor.
 * @returns A tensor of the same shape as `x`.
 */
export function pow(x: Tensor, a: Tensor|number): Tensor {
  return tidy(() => {
    if (typeof (a) === 'number') {
      a = scalar(Math.round(a), 'int32');
    }
    if (a.dtype !== 'int32') {
      throw new NotImplementedError(
          `Non-int32 dtype (${a.dtype}) is not supported by pow() yet`);
    }
    return tfc.pow(x, a as Tensor);
  });
}

/**
 * Reshapes bias tensor according to rank of x.
 */
function reshapeBias(xRank: number, bias: Tensor, dataFormat: string) {
  const biasShape = bias.shape;

  if (bias.rank !== 1 && bias.rank !== xRank) {
    throw new ValueError(
        'Unexpected bias dimensions: ' + bias.rank +
        '; expected it to be 1 or ' + xRank);
  }

  if (xRank === 5) {
    if (dataFormat === 'channelsFirst') {
      if (biasShape.length === 1) {
        return bias.reshape([1, biasShape[0], 1, 1, 1]);
      } else {
        return bias.reshape(
            [1, biasShape[3], biasShape[0], biasShape[1], biasShape[2]]);
      }
    } else if (dataFormat === 'channelsLast') {
      if (biasShape.length === 1) {
        return bias.reshape([1, 1, 1, 1, biasShape[0]]);
      } else {
        return bias.reshape([1].concat(biasShape));
      }
    }
  } else if (xRank === 4) {
    if (dataFormat === 'channelsFirst') {
      if (biasShape.length === 1) {
        return bias.reshape([1, biasShape[0], 1, 1]);
      } else {
        return bias.reshape([1, biasShape[2], biasShape[0], biasShape[1]]);
      }
    } else if (dataFormat === 'channelsLast') {
      if (biasShape.length === 1) {
        return bias.reshape([1, 1, 1, biasShape[0]]);
      } else {
        return bias.reshape([1].concat(biasShape));
      }
    }
  } else if (xRank === 3) {
    if (dataFormat === 'channelsFirst') {
      if (biasShape.length === 1) {
        return bias.reshape([1, biasShape[0], 1]);
      } else {
        return bias.reshape([1, biasShape[1], biasShape[0]]);
      }
    } else if (dataFormat === 'channelsLast') {
      if (biasShape.length === 1) {
        return bias.reshape([1, 1, biasShape[0]]);
      } else {
        return bias.reshape([1].concat(biasShape));
      }
    }
  } else if (xRank < 3) {
    return bias;
  }
  throw new ValueError(`Unsupported input rank by biasAdd: ${bias.rank}`);
}

/* Neural-network operations. */

/**
 * Add a bias to a tensor.
 *
 * @param x The tensor to add the bias to.
 * @param bias The bias to add to `x`. Must be 1D or the same rank as `x`.
 * @return Result of the bias adding.
 * @throws ValueError: If the rank of `bias` is incorrect.
 */
export function biasAdd(
    x: Tensor, bias: Tensor, dataFormat?: DataFormat): Tensor {
  return tidy(() => {
    if (dataFormat == null) {
      dataFormat = imageDataFormat();
    }
    checkDataFormat(dataFormat);

    return x.add(reshapeBias(x.rank, bias, dataFormat));
  });
}

/**
 * Exponential linear unit (ELU).
 * @param x A tensor or variable to compute the activation function for.
 * @param alpha: A scalar, a scaling factor for the negative section.
 * @return Output of the ELU operation.
 */
export function elu(x: Tensor, alpha = 1): Tensor {
  // TODO(cais): Add support for alpha values other than 1.
  if (alpha !== 1) {
    throw new NotImplementedError(
        `Support for alpha values other than 1 (${alpha}) is not implemented ` +
        `yet.`);
  }
  return tfc.elu(x);
}

/**
 * Softsign of a tensor.
 *
 * Defined as x / (abs(x) + 1), element-wise.
 *
 * @param x: Input.
 * @returns Output.
 */
export function softsign(x: Tensor): Tensor {
  return tidy(() => tfc.div(x, tfc.abs(x).add(1)));
}

/**
 * Sets entries in `x` to zero at random, while scaling the entire tensor.
 *
 * @param x input tensor.
 * @param level fraction of the entries in the tensor that will be set to 0.
 * @param noiseShape shape of randomly generated keep/drop flags, must be
 *   broadcastable to the shape of `x`.
 * @param seed random seed to ensure determinism.
 * @returns Result of the dropout operation.
 */
export function dropout(
    x: Tensor, level: number, noiseShape?: number[], seed?: number): Tensor {
  return tidy(() => {
    // TODO(cais): Switch to deeplearn.js implementation of dropout when it
    //   becomes avaialable.
    if (noiseShape != null && !util.arraysEqual(x.shape, noiseShape)) {
      throw new NotImplementedError(
          'Non-default noise shape is not implemented yet: ' +
          JSON.stringify(noiseShape));
    }
    if (seed != null) {
      throw new NotImplementedError('seed is not implemented for dropout yet.');
    }
    let multiplier =
        tfc.step(tfc.add(-level, tfc.randomUniform(x.shape, 0, 1, 'float32')));
    // Scale the kept elements, so the expected sum is unchanged.
    multiplier = tfc.mul(1 / (1 - level), multiplier);
    return tfc.mul(x, multiplier);
  });
}

/**
 * Element-wise, segment-wise linear approximation of sigmoid.
 *
 * Returns `0.` if `x < -2.5`, `1.` if `x > 2.5`.
 * In `-2.5 <= x <= 2.5`, returns `0.2 * x + 0.5`.
 *
 * @param x Input tensor.
 * @returns Output tensor.
 */
export function hardSigmoid(x: Tensor): Tensor {
  return tidy(() => {
    const y = tfc.add(.5, tfc.mul(.2, x));
    return tfc.clipByValue(y, 0, 1);
  });
}

/**
 * Invoke `x` in the training phase, and `alt` otherwise.
 *
 * Porting Note: We do not create placeholder tensors for the `training` boolean
 *   flag here, because there is no such thing in the TF.js imperative backend.
 *
 * @param x The function to invoke iff `training` is `true`.
 * @param alt The function to invoke iff `training` is `false`.
 * @param training Boolean flag for whether training phase is active.
 * @returns The return value of `x()` if `training` is `true`, or the return
 *   value of `alt()` if `training` is `false`.
 */
export function inTrainPhase<T>(x: () => T, alt: () => T, training = false): T {
  return training ? x() : alt();
}
