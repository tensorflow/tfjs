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

// tslint:disable:max-line-length
import * as tfc from '@tensorflow/tfjs-core';
import {DataType, dispose, onesLike as coreOnesLike, Scalar, scalar, Tensor, Tensor1D, tensor1d, Tensor2D, tensor2d, Tensor3D, Tensor4D, tidy, util, variableGrads, where, zerosLike as coreZerosLike} from '@tensorflow/tfjs-core';

import {checkDataFormat, DataFormat, nameScope as commonNameScope} from '../common';
import {NotImplementedError, ValueError} from '../errors';
import {Shape, SymbolicTensor} from '../types';
import * as math_utils from '../utils/math_utils';
import {LayerVariable} from '../variables';
import {epsilon as common_epsilon} from './common';
import {imageDataFormat} from './common';

// tslint:enable

/* Setting and getting backend from deeplearn.js. */

// Default deeplearn.js backend is WebGL (GPU).
let backend: 'cpu'|'webgl' = 'webgl';

const DEFAULT_DTYPE: DataType = 'float32';

export function disposeScalarCache() {
  for (const typeKey in scalarCache) {
    for (const key in scalarCache[typeKey]) {
      scalarCache[typeKey][key].dispose();
      delete scalarCache[typeKey][key];
    }
  }
}

export function setBackend(requestedBackend: 'cpu'|'webgl') {
  tfc.setBackend(requestedBackend);
  backend = requestedBackend;
  disposeScalarCache();
}

export function getBackend(): 'cpu'|'webgl' {
  return backend;
}

const scalarCache: {[typeKey: string]: {[key: number]: Scalar}} = {
  float32: {},
  int32: {}
};

/**
 * Get scalar, with caching.
 */
export function getScalar(value: number, dtype?: DataType): Scalar {
  if (dtype === undefined) {
    dtype = DEFAULT_DTYPE;
  }
  if (scalarCache[dtype][value] == null) {
    scalarCache[dtype][value] = scalar(value, dtype);
    tfc.keep(scalarCache[dtype][value]);
  }
  return scalarCache[dtype][value];
}

/** Returns the value of the fuzz factor used in numeric expressions. */
export const epsilon = common_epsilon;

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

/* Shapes. */

/**
 * Get the shape of a tensor.
 *
 * @param x The tensor.
 * @return Shape of the tensor.
 */
export function shape(x: Tensor|SymbolicTensor): Shape {
  return x.shape;
}

/**
 * Get the shape of a tensor as an array of numbers.
 *
 * @param x The tensor.
 * @return Shape of the tensor as number[].
 */
export function intShape(x: Tensor|SymbolicTensor): number[] {
  return x.shape;
}

/**
 * Get the number of dimensions (axes).
 *
 * @param x The tensor.
 * @return Number of dimensions of `x`.
 */
export function ndim(x: Tensor|SymbolicTensor): number {
  return x.shape.length;
}

/**
 * Returns the dtype of a tensor or variable.
 *
 * @param x The tensor.
 */
export function dtype(x: Tensor|SymbolicTensor): DataType {
  // TODO(michaelterry): Update if additional data types are available.
  return (x instanceof Tensor) ? DEFAULT_DTYPE : (x as SymbolicTensor).dtype;
}

/**
 * Get the number of elements in a Tensor.
 * @param x The Tensor.
 * @return Number of elements in `x`.
 */
export function countParams(x: Tensor|SymbolicTensor): number {
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
export function cast(x: Tensor, dtype: 'float32'|'int32'|'bool'): Tensor {
  return x.asType(dtype);
}

/**
 * Adds a 1-sized dimension at index "axis".
 * @param x Input tensor.
 * @param axis Position where to add the new axis.
 * @returns Result of the dimension expansion.
 */
export function expandDims(x: Tensor, axis = -1): Tensor {
  const outShape = shape(x).slice();
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
  if (ndim(x) <= 1) {
    throw new ValueError(
        `batchFlatten requires a minimum rank of 2. Got rank: ${ndim(x)}.`);
  }
  const newShape = [x.shape[0], math_utils.arrayProd(x.shape, 1)];
  return x.reshape(newShape);
}

/**
 * Do slicing along the first axis.
 * @param array input `Tensor`.
 * @param start starting index, inclusive.
 * @param size size of the slice along the first axis.
 * @returns result of the slicing.
 * @throws ValueError: If `array` is of an unsupported subtype of `Tensor`.
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
 * @param array input `Tensor`.
 * @param start starting index, inclusive.
 * @param size size of the slice along the last axis.
 * @returns result of the slicing.
 * @throws ValueError: If `array` is of an unsupported subtype of `Tensor`.
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
 * @param array input `Tensor`.
 * @param start starting index, inclusive.
 * @param size of the slice along the chosen axis.
 * @param choose an axis.
 * @returns result of the slicing.
 * @throws ValueError: If `array` is of an unsupported subtype of `Tensor`.
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
 * Non-broadcasting batch normalization for use in training (not inference).
 *
 * The input is normalized to zero mean and unit variance along the
 * `reductionAxes`, followed by scaling with `gamma` and shifted by `beta`.
 * The result of that is returned as the first element
 * of the returned `Array`. The other two elements are the mean and variance,
 * respectively.
 *
 * @param x Input tensor to be normalized.
 * @param gamma Tensor by which to scale the input.
 * @param beta Tensor by which to center the input.
 * @param reductionAxes Axes over which to normalize.
 * @param epsilon Fuzz factor.
 * @returns An `Array` of three `Tensors`:
 *   [normalized tensor, mean of input, variance of input].
 */
function regularNormalizeBatchInTraining(
    x: Tensor, gamma: Tensor, beta: Tensor, reductionAxes: number[],
    epsilon = 1e-3): [Tensor, Tensor, Tensor] {
  return tidy(() => {
           const meanAndVariance = tfc.moments(x, reductionAxes);
           const mean = meanAndVariance.mean;
           const variance = meanAndVariance.variance;
           const normed =
               batchNormalization(x, mean, variance, beta, gamma, epsilon);
           return [normed, mean, variance];
         }) as [Tensor, Tensor, Tensor];
}

/**
 * Broadcasting batch normalization for use in training (not inference).
 *
 * The input is normalized to zero mean and unit variance along the
 * `reductionAxes`, followed by scaling with `gamma` and shifted by `beta`.
 * The result of that is returned as the first element
 * of the returned `Array`. The other two elements are the mean and variance,
 * respectively.
 *
 * @param x Input tensor to be normalized.
 * @param gamma Tensor by which to scale the input.
 * @param beta Tensor by which to center the input.
 * @param reductionAxes Axes over which to normalize.
 * @param epsilon Fuzz factor.
 * @returns An `Array` of three `Tensors`:
 *   [normalized tensor, mean of input, variance of input].
 */
function broadcastNormalizeBatchInTraining(
    x: Tensor, gamma: Tensor, beta: Tensor, reductionAxes: number[],
    epsilon = 1e-3): [Tensor, Tensor, Tensor] {
  return tidy(() => {
           const meanAndVariance = tfc.moments(x, reductionAxes);
           const mean = meanAndVariance.mean;
           const variance = meanAndVariance.variance;
           const targetShape: number[] = [];
           for (const axis of math_utils.range(0, ndim(x))) {
             if (reductionAxes.indexOf(axis) !== -1) {
               targetShape.push(1);
             } else {
               targetShape.push(x.shape[axis]);
             }
           }
           const broadcastMean = mean.reshape(targetShape);
           const broadcastVariance = variance.reshape(targetShape);
           const broadcastGamma =
               gamma == null ? null : gamma.reshape(targetShape);
           const broadcastBeta =
               beta == null ? null : beta.reshape(targetShape);
           const normed = batchNormalization(
               x, broadcastMean, broadcastVariance, broadcastBeta,
               broadcastGamma, epsilon);
           return [normed, mean, variance];
         }) as [Tensor, Tensor, Tensor];
}

/**
 * Batch normalization for use in training (not inference).
 *
 * @param x Input tensor to be normalized.
 * @param gamma Tensor by which to scale the input.
 * @param beta Tensor by which to center the input.
 * @param reductionAxes Axes over which to normalize.
 * @param epsilon Fuzz factor.
 * @returns An `Array` of three `Tensors`:
 *   [normalized tensor, mean of input, variance of input].
 */
export function normalizeBatchInTraining(
    x: Tensor, gamma: Tensor, beta: Tensor, reductionAxes: number[],
    epsilon = 1e-3): [Tensor, Tensor, Tensor] {
  if (util.arraysEqual(
          reductionAxes.slice().sort(), math_utils.range(0, ndim(x) - 1))) {
    return regularNormalizeBatchInTraining(
        x, gamma, beta, reductionAxes, epsilon);
  } else {
    return broadcastNormalizeBatchInTraining(
        x, gamma, beta, reductionAxes, epsilon);
  }
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
    rank = ndim(tensors[0]);
    if (rank !== 0) {
      axis = rank;
    } else {
      axis = 0;
    }
  }
  if (axis === ndim(tensors[0])) {
    // Porting Note: This is necessary because tfc.concat() requires axis to be
    //   in the interval [-rank, rank).
    axis = -1;
  }
  // Porting Note: Sparse concat is not supported yet.
  return tfc.concat(tensors, axis);
}

/**
 * Concatenate two arrays along the first dimension.
 * @param a The 1st `Tensor` to concatenate.
 * @param b The 2nd `Tensor` to concatenate.
 * @returns Result of the concatenation.
 * @throws ValueError: If `a` is of an unsupported subtype of `Tensor`.
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
  if (ndim(x) !== n.length) {
    throw new ValueError(
        `The length of input n (${n.length}) does not match ` +
        `the number of dimensions in input x (${ndim(x)})`);
  }
  return tfc.tile(x, n);
}

/* Creation and manipulation of tensors and variables */

/**
 * Create a Tensor with the same content as the input.
 *
 * @param x Input.
 * @return Identity output Tensor.
 */
export function identity(x: Tensor): Tensor {
  return x.clone();
}

/**
 * Instantiate an identity matrix and returns it, as a Variable
 *
 * @param size Number of rows/columns.
 * @param dtype Data type of returned Variable.
 * @param name Name of returned Variable.
 * @return A Variable, an identity matrix.
 */
export function eyeVariable(
    size: number, dtype?: DataType, name?: string): LayerVariable {
  return new LayerVariable(tfc.eye(size, size, null, dtype), dtype, name);
}

/**
 * Multiply a scalar with an Tensor.
 * @param c The Scalar.
 * @param x The Tensor.
 * @returns The result of the multiplication.
 */
export function scalarTimesArray(c: Scalar, x: Tensor): Tensor {
  return tfc.mul(c, x);
}

/**
 * Add a scalar to an Tensor.
 * @param c The Scalar.
 * @param x The Tensor.
 * @returns The result of the addition.
 */
export function scalarPlusArray(c: Scalar, x: Tensor): Tensor {
  return tfc.add(c, x);
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
 * (e.g. `(2, 3) * (4, 3, 5) -> (2, 4, 5)`).
 *
 * @param x A tensor of at least rank 2.
 * @param y A tensor of at least rank 2.
 * @return Result of the dot operation.
 */
export function dot(x: Tensor, y: Tensor): Tensor {
  if (ndim(y) !== 2) {
    throw new NotImplementedError(
        `dot support for y other than rank 2 is not yet implemented: ` +
        `y shape = ${shape}`);
  } else {
    if (ndim(x) === 2) {
      return tfc.matMul(x as Tensor2D, y as Tensor2D);
    } else if (ndim(x) === 3) {
      const xShape0 = x.shape[0];
      const xShape1 = x.shape[1];
      const xShape2 = x.shape[2];
      x = x.reshape([xShape0 * xShape1, xShape2]);
      return tfc.matMul(x as Tensor2D, y as Tensor2D).reshape([
        xShape0, xShape1, y.shape[1]
      ]);
    } else {
      throw new NotImplementedError(
          `dot support for x of rank ${ndim(x)} is not yet implemented: ` +
          `x shape = ${shape}`);
    }
  }
}

/**
 * Compute the sign Tensor of an input Tensor.
 *
 * Elements of the input `Tensor` that are === 0 are mapped to 0.
 * Elements of the input `Tensor` that are > 0 are mapped to 1.
 * Elements of the input `Tensor` that are < 0 are mapped to -1.
 *
 * @param x Input `Tensor`.
 * @return The sign `Tensor`.
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
            scalarTimesArray(getScalar(-1), onesLikeX)));
  });
}

/**
 * Compute QR decomposition of m-by-n matrix using Householder transformation.
 *
 * Requires `m >= n`.
 *
 * Implementation based on
 *   [http://www.cs.cornell.edu/~bindel/class/cs6210-f09/lec18.pdf]
 * (http://www.cs.cornell.edu/~bindel/class/cs6210-f09/lec18.pdf)
 *
 * @param x The 2D `Tensor` (matrix) to be QR-decomposed. Must have
 *   `x.shape[0] >= x.shape[1]`.
 * @return An `Array` of two `Tensor`s: `[Q, R]`, where `Q` is a unitary
 *   matrix of size `[x.shape[0], x.shape[0]]`. `R` has the same shape as
 *   `x`.
 * @throws ValueError if `x.shape[0] < x.shape[1]`, or if `x`'s rank is not 2.
 */
export function qr(x: Tensor2D): [Tensor, Tensor] {
  const [qOuter, rOuter]: [Tensor, Tensor] = tidy((): [Tensor, Tensor] => {
    // TODO(cais): Extend support to >2D as in `tf.qr` and move this
    // function to the core.
    if (x.shape.length !== 2) {
      throw new ValueError(
          `qr() requires a 2D Tensor, but got a ${x.shape.length}D Tensor.`);
    }
    if (x.shape[0] < x.shape[1]) {
      throw new ValueError(
          `qr() requires x.shape[0] >= x.shape[1], but got shape: [${
              x.shape}]`);
    }

    const m = x.shape[0];
    const n = x.shape[1];

    let q = tfc.eye(m) as Tensor2D;  // Orthogonal transform so far.
    let r = x.clone();               // Transformed matrix so far.

    const one2D = tensor2d([[1]], [1, 1]);
    let w: Tensor2D = one2D.clone();

    for (let j = 0; j < n; ++j) {
      // This tidy within the for-loop ensures we clean up temporary
      // tensors as soon as they are no longer needed.
      const rTemp = r;
      const wTemp = w;
      const qTemp = q;
      [w, r, q] = tidy((): [Tensor2D, Tensor2D, Tensor2D] => {
        // Find H = I - tau * w * w', to put zeros below R(j, j).
        const rjEnd1 = r.slice([j, j], [m - j, 1]);
        const normX = tfc.norm(rjEnd1);
        const rjj = r.slice([j, j], [1, 1]);
        const s = tfc.neg(sign(rjj)) as Tensor2D;
        const u1 = rjj.sub(tfc.mul(s, normX)) as Tensor2D;
        const wPre = tfc.div(rjEnd1, u1);
        if (wPre.shape[0] === 1) {
          w = one2D.clone();
        } else {
          w = one2D.concat(
                  wPre.slice([1, 0], [wPre.shape[0] - 1, wPre.shape[1]]), 0) as
              Tensor2D;
        }
        const tau = tfc.neg(tfc.div(tfc.matMul(s, u1), normX)) as Tensor2D;

        // -- R := HR, Q := QH.
        const rjEndAll = r.slice([j, 0], [m - j, n]);
        const tauTimesW = tau.mul(w) as Tensor2D;
        if (j === 0) {
          r = rjEndAll.sub(tauTimesW.matMul(w.transpose().matMul(rjEndAll)));
        } else {
          r = r.slice([0, 0], [j, n])
                  .concat(
                      rjEndAll.sub(
                          tauTimesW.matMul(w.transpose().matMul(rjEndAll))),
                      0) as Tensor2D;
        }
        const qAllJEnd = q.slice([0, j], [m, q.shape[1] - j]);
        if (j === 0) {
          q = qAllJEnd.sub(qAllJEnd.matMul(w).matMul(tauTimesW.transpose()));
        } else {
          q = q.slice([0, 0], [m, j])
                  .concat(
                      qAllJEnd.sub(
                          qAllJEnd.matMul(w).matMul(tauTimesW.transpose())),
                      1) as Tensor2D;
        }
        return [w, r, q];
      });
      dispose([rTemp, wTemp, qTemp]);
    }

    return [q, r];
  });
  return [qOuter, rOuter];
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
    if (ndim(indices) !== 1) {
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

/* Normalization operations. */

/**
 * Applies batch normalization on x given mean, var, beta and gamma.
 *
 * I.e. returns:
 *   `output = (x - mean) / (sqrt(var) + epsilon) * gamma + beta`
 *
 * @param x Input tensor.
 * @param mean Mean of batch.
 * @param variance Variance of batch.
 * @param beta Tensor with which to center the input.
 * @param gamma Tensor by which to scale the input.
 * @param epsilon Fuzz factor.
 * @returns The result of the batch normalization.
 */
export function batchNormalization(
    x: Tensor, mean: Tensor, variance: Tensor, beta?: Tensor, gamma?: Tensor,
    epsilon = 1e-3): Tensor {
  let out: Tensor;
  if (ndim(x) === 2) {
    out = tfc.batchNormalization2d(
        x as Tensor2D, mean as Tensor2D | Tensor1D,
        variance as Tensor2D | Tensor1D, epsilon, gamma as Tensor2D | Tensor1D,
        beta as Tensor2D | Tensor1D);
  } else if (ndim(x) === 3) {
    // TODO(cais): Check rank; give proper error message.
    out = tfc.batchNormalization3d(
        x as Tensor3D, mean as Tensor3D | Tensor1D,
        variance as Tensor3D | Tensor1D, epsilon, gamma as Tensor3D | Tensor1D,
        beta as Tensor3D | Tensor1D);
  } else if (ndim(x) === 4) {
    out = tfc.batchNormalization4d(
        x as Tensor4D, mean as Tensor4D | Tensor1D,
        variance as Tensor4D | Tensor1D, epsilon, gamma as Tensor4D | Tensor1D,
        beta as Tensor4D | Tensor1D);
  } else {
    throw new NotImplementedError(
        `batchNormalization is not implememnted for array of rank ${ndim(x)} ` +
        `yet`);
  }
  return out;
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

    if (ndim(bias) !== 1 && ndim(bias) !== ndim(x)) {
      throw new ValueError(
          'Unexpected bias dimensions: ' + ndim(bias) +
          '; expected it to be 1 or ' + ndim(x));
    }
    const biasShape = bias.shape;

    let y: Tensor;
    if (ndim(x) === 5) {
      if (dataFormat === 'channelsFirst') {
        if (biasShape.length === 1) {
          y = x.add(bias.reshape([1, biasShape[0], 1, 1, 1]));
        } else {
          y = x.add(bias.reshape(
              [1, biasShape[3], biasShape[0], biasShape[1], biasShape[2]]));
        }
      } else if (dataFormat === 'channelsLast') {
        if (biasShape.length === 1) {
          y = x.add(bias.reshape([1, 1, 1, 1, biasShape[0]]));
        } else {
          y = x.add(bias.reshape([1].concat(biasShape)));
        }
      }
    } else if (ndim(x) === 4) {
      if (dataFormat === 'channelsFirst') {
        if (biasShape.length === 1) {
          y = x.add(bias.reshape([1, biasShape[0], 1, 1]));
        } else {
          y = x.add(
              bias.reshape([1, biasShape[2], biasShape[0], biasShape[1]]));
        }
      } else if (dataFormat === 'channelsLast') {
        if (biasShape.length === 1) {
          y = x.add(bias.reshape([1, 1, 1, biasShape[0]]));
        } else {
          y = x.add(bias.reshape([1].concat(biasShape)));
        }
      }
    } else if (ndim(x) === 3) {
      if (dataFormat === 'channelsFirst') {
        if (biasShape.length === 1) {
          y = x.add(bias.reshape([1, biasShape[0], 1]));
        } else {
          y = x.add(bias.reshape([1, biasShape[1], biasShape[0]]));
        }
      } else if (dataFormat === 'channelsLast') {
        if (biasShape.length === 1) {
          y = x.add(bias.reshape([1, 1, biasShape[0]]));
        } else {
          y = x.add(bias.reshape([1].concat(biasShape)));
        }
      }
    } else if (ndim(x) < 3) {
      y = x.add(bias);
    } else {
      throw new ValueError(`Unsupported input rank by biasAdd: ${ndim(x)}`);
    }
    return y;
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
  return tidy(() => tfc.div(x, tfc.add(getScalar(1), tfc.abs(x))));
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
    x: Tensor, level: Scalar, noiseShape?: number[], seed?: number): Tensor {
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
    let multiplier = tfc.step(tfc.add(
        tfc.neg(level) as Scalar, tfc.randomUniform(x.shape, 0, 1, 'float32')));
    // Scale the kept elements, so the expected sum is unchanged.
    multiplier = tfc.mul(
        tfc.div(getScalar(1), tfc.sub(getScalar(1), level)) as Scalar,
        multiplier);
    return tfc.mul(x, multiplier);
  });
}

/**
 * Replacement for Keras's "with name_scope" construct.
 *
 * @param name The name to use for this name scope.
 * @param fn A function to call within this name scope.
 * @return The value of fn.
 */
export function nameScope<T>(name: string, fn: () => T): T {
  return commonNameScope(name, fn);
}

/**
 * Returns the default float type, as a DType.
 */
export function floatx(): DataType {
  return 'float32';
}

const _uidPrefixes: {[prefix: string]: number} = {};

/**
 * Provides a unique UID given a string prefix.
 *
 * @param prefix
 */
export function getUid(prefix = ''): string {
  if (!(prefix in _uidPrefixes)) {
    _uidPrefixes[prefix] = 0;
  }
  _uidPrefixes[prefix] += 1;
  return prefix + _uidPrefixes[prefix].toString();
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
    // TODO(cais): Maybe avoid creating scalar constants on each invocation by
    //   turning them into module-level constants.
    const y = scalarPlusArray(scalar(0.5), scalarTimesArray(scalar(0.2), x));
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

/**
 * Control flow.
 */

/**
 * Returns the gradients of `variables` w.r.t. the return value of `lossFn`.
 * @param lossFn A function which returns a Scalar to be used as the function
 *   value (i.e., numerator) for differentiation.
 * @param variables List of variables to be used as the independent variables
 *   (i.e., denominator) for differentiation.
 * @returns An Array of gradients tensors.
 */
export function gradients(
    lossFn: () => Scalar, variables: LayerVariable[]): Tensor[] {
  // TODO(cais): The return type signature can be simplified if deeplearn makes
  //   the corresponding type public.
  const variableList =
      variables.map(variable => variable.read() as tfc.Variable);
  const valudAndGrads = variableGrads(lossFn, variableList);
  return variables.map(variable => valudAndGrads.grads[variable.name]);
}
