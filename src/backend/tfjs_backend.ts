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
import {onesLike as coreOnesLike, Scalar, scalar, Tensor, Tensor1D, tensor1d, Tensor2D, tensor2d, Tensor3D, Tensor4D, util, variableGrads, where, zerosLike as coreZerosLike} from '@tensorflow/tfjs-core';

import {checkDataFormat, checkPaddingMode, checkPoolMode, DataFormat, nameScope as commonNameScope, PaddingMode, PoolMode} from '../common';
import {Constraint} from '../constraints';
import {NotImplementedError, ValueError} from '../errors';
import {ConcreteTensor, DType, LayerVariable, RnnStepFunction, Shape, SymbolicTensor, TensorInterface} from '../types';
import {pyNormalizeArrayIndex} from '../utils/generic_utils';
import * as math_utils from '../utils/math_utils';

import {epsilon as common_epsilon} from './common';
import {imageDataFormat} from './common';

// tslint:enable

/* Setting and getting backend from deeplearn.js. */

// Default deeplearn.js backend is WebGL (GPU).
let backend: 'cpu'|'webgl' = 'webgl';

const DEFAULT_DTYPE = DType.float32;

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

/**
 * Alias for deeplearn math keep: do not dispose a tensor's value.
 * @param x
 */
export function keep(x: Tensor): Tensor {
  return tfc.keep(x);
}

const scalarCache: {[typeKey: string]: {[key: number]: Scalar}} = {
  float32: {},
  int32: {}
};

/**
 * Get scalar, with caching.
 */
export function getScalar(value: number, dtype?: DType): Scalar {
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
export function shape(x: Tensor|TensorInterface): Shape {
  return x.shape;
}

/**
 * Get the shape of a tensor as an array of numbers.
 *
 * @param x The tensor.
 * @return Shape of the tensor as number[].
 */
export function intShape(x: Tensor|TensorInterface): number[] {
  return x.shape;
}

/**
 * Get the number of dimensions (axes).
 *
 * @param x The tensor.
 * @return Number of dimensions of `x`.
 */
export function ndim(x: Tensor|TensorInterface): number {
  return x.shape.length;
}

/**
 * Returns the dtype of a tensor or variable.
 *
 * @param x The tensor.
 */
export function dtype(x: Tensor|SymbolicTensor): DType {
  // TODO(michaelterry): Update if additional data types are available.
  return (x instanceof Tensor) ? DEFAULT_DTYPE : (x as SymbolicTensor).dtype;
}

/**
 * Normalize an axis specification, allowing negative indices (to enable
 * counting from the end).
 *
 * For example, if axis = -1 for an Tensor x with shape of [2, 2], then
 * normalizeAxis(x, -1) = 1.
 *
 * TODO(michaelterry): Remove once the following issue is resolved:
 *   https://github.com/PAIR-code/deeplearnjs/issues/477
 *
 * @param x The Tensor to normalize against.
 * @param axis One or more axis indices to normalize. If an array is passed in,
 *   all values must be non-null/defined.
 */
export function normalizeAxis(
    x: Tensor|TensorInterface, axis: number|number[]): number|number[] {
  if (axis == null) {
    return axis;
  }
  const xShape = shape(x);
  if (Array.isArray(axis)) {
    return axis.map(thisAxis => pyNormalizeArrayIndex(xShape, thisAxis));
  }
  return pyNormalizeArrayIndex(xShape, axis);
}

/**
 * Get the number of elements in a Tensor.
 * @param x The Tensor.
 * @return Number of elements in `x`.
 */
export function countParams(x: Tensor|TensorInterface): number {
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
 * Reshape tensor to specified shape.
 * @param x Input tensor.
 * @param shape Target shape.
 * @return The resultant tensor of the reshaping.
 */
export function reshape(x: Tensor, shape: Shape): Tensor {
  // TODO(cais): Should this call TensorMath.reshape instead for backprop?
  return x.reshape(shape);
}

/**
 * Generalized transpose of a tensor.
 * @param x The input tensor to tranpose.
 * @param perm Optional permutation array. If == null, will be set to
 *   `[n - 1, n - 2, ..., 0]`, n being `ndim(x)`, hence reducing to the usual
 *   transpose for matricies (`Tensor2D`s).
 * @returns The resultant tensor of the tranpose.
 */
export function transpose(x: Tensor, perm?: number[]): Tensor {
  return tfc.transpose(x, perm);
}

export const permuteDimensions = transpose;

/**
 * Reverse a tensor along the specified axis or axes.
 * @param x Tensor to reverse.
 * @param axes Integer or an `Array` of integers. Axis or axes to reverse.
 * @returns The result of the reverse operation.
 */
export function reverse(x: Tensor, axes: number|number[]): Tensor {
  return tfc.reverse(x, axes);
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
  return reshape(x, outShape);
}

/**
 * Removes a 1-dimension from the tensor at index "axis".
 * @param x Input tensor
 * @param axis which axes to remove.
 */
export function squeeze(x: Tensor, axis: number): Tensor {
  return tfc.squeeze(x, [axis]);
}

/**
 * Pads the middle dimension of a 3D tensor.
 *
 * @param x Input `Tensor` to be padded.
 * @param padding `Array` of 2 integers, how many zeros to add at the start and
 *   end of the middle dimension (i.e., dimension 1).
 * @return A padded 3D `Tensor`.
 */
export function temporalPadding(x: Tensor, padding?: [number, number]): Tensor {
  if (ndim(x) !== 3) {
    throw new ValueError(
        `temporalPadding expects input tensor to be 3-D, but received a ` +
        `${ndim(x)}-D tensor.`);
  }

  if (padding == null) {
    padding = [1, 1];
  }
  if (padding.length !== 2) {
    throw new ValueError(
        `temporalPadding expects input padding pattern to be a length-2 ` +
        `array, but received a length-${padding.length} array.`);
  }

  const pattern: Array<[number, number]> = [[0, 0], padding, [0, 0]];
  return tfc.pad(x, pattern);
}

/**
 * Pads the 2nd and 3rd dimensions of a 4D tensor.
 *
 * @param x Input `Tensor` to be padded.
 * @param padding `Array` of two `Array`s, each of which is an `Array` of two
 *   integers. The amount of padding at the beginning and end of the 2nd and 3rd
 *   dimensions, respectively.
 * @param dataFormat 'channelsLast' (default) or 'channelsFirst'.
 * @return Padded 4D `Tensor`.
 */
export function spatial2dPadding(
    x: Tensor, padding?: [[number, number], [number, number]],
    dataFormat?: DataFormat): Tensor {
  if (ndim(x) !== 4) {
    throw new ValueError(
        `temporalPadding expects input tensor to be 4-D, but received a ` +
        `${ndim(x)}-D tensor.`);
  }

  if (padding == null) {
    padding = [[1, 1], [1, 1]];
  }
  if (padding.length !== 2 || padding[0].length !== 2 ||
      padding[1].length !== 2) {
    throw new ValueError(
        'spatial2dPadding expects `padding` to be an Array of two Arrays, ' +
        'each of which is an Array of two integers.');
  }

  if (dataFormat == null) {
    dataFormat = imageDataFormat();
  }
  if (dataFormat !== 'channelsLast' && dataFormat !== 'channelsFirst') {
    throw new ValueError(
        `Unknown data format: ${dataFormat}. ` +
        `Supported data formats are 'channelsLast' and 'channelsFirst.`);
  }

  let pattern: Array<[number, number]>;
  if (dataFormat === 'channelsFirst') {
    pattern = [[0, 0], [0, 0], padding[0], padding[1]];
  } else {
    pattern = [[0, 0], padding[0], padding[1], [0, 0]];
  }

  return tfc.pad(x, pattern);
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
  if (x.shape.length !== 2) {
    throw new ValueError(
        `repeat() expects a rank-2 tensor, but received a ` +
        `rank-${x.shape.length} tensor.`);
  }
  const y = expandDims(x, 1);
  return tile(y, [1, n, 1]);
}

/**
 * Flatten an Tensor into 1D.
 * @param x Input tensor.
 * @return The result of the flattening `x`.
 */
export function flatten(x: Tensor): Tensor {
  const newShape = [math_utils.arrayProd(x.shape)];
  return reshape(x, newShape);
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
  return reshape(x, newShape);
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
  switch (array.rank) {
    case 1:
      return tfc.slice1d(array as Tensor1D, start, size);
    case 2:
      return tfc.slice2d(array as Tensor2D, [start, 0], [size, array.shape[1]]);
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
          `sliceAlongFirstAxis() received an unsupported subtype of Tensor: ` +
          `${array.constructor.name}`);
  }
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
  switch (array.rank) {
    case 1:
      return tfc.slice1d(array as Tensor1D, start, size);
    case 2:
      return tfc.slice2d(array as Tensor2D, [0, start], [array.shape[0], size]);
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
          `sliceAlongLastAxis() received an unsupported subtype of Tensor: ` +
          `${array.constructor.name}`);
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
          'concatAlongFirstAxis() received an unsupported subtype of ' +
          'Tensor: ' + a.constructor.name);
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
 * Create a Variable.
 * @param x The initial value of the `Variable`.
 * @param dtype optional, the type of the variable.
 * @param name optional, the name of the variable, default provided by
 * Variable.
 * @param constraint optional, a constraint to be applied after every update.
 * @return The newly instantiated `Variable`.
 */
export function variable(
    x: Tensor, dtype?: DType, name?: string,
    constraint?: Constraint): LayerVariable {
  return new LayerVariable(x, dtype, name, true, constraint);
}

/**
 * Get the values of an array of Variables.
 *
 * @param tensors An `Array` of `Variable`s to get the values of.
 * @return The values of the inputs, as an `Array` of `Tensor`s.
 */
export function batchGetValue(xs: LayerVariable[]): Tensor[] {
  return xs.map(x => x.read());
}

/**
 * Update the value of multiple Variables at once.
 *
 * @param variablesAndValues An `Array`, each element is of type
 *   [Variable, Tensor]. The first item is the
 *   `Variable` of which the value is to be updated. The second item
 *   carries the new value.
 */
export function batchSetValue(
    variablesAndValues: Array<[LayerVariable, Tensor]>): void {
  variablesAndValues.map((variableAndValue) => {
    const variable: LayerVariable = variableAndValue[0];
    variable.write(variableAndValue[1]);
  });
}

/**
 * Instantiates an all-zeros tensor and returns it.
 *
 * @param shape Shape of the tensor.
 * @param dtype DType of the tensor.
 * @param name Name of the tensor.
 * @return An all-zero Tensor.
 */
export function zeros(shape: Shape, dtype?: DType): Tensor {
  // TODO(cais): Implement logic for dtype.
  return tfc.zeros(shape);
}

/**
 * Instantiates an all-zeros Variable and returns it.
 *
 * @param shape Shape of the tensor.
 * @param dtype DType of the tensor.
 * @param name Name of the tensor.
 * @return An all-zero Variable.
 */
export function zerosVariable(
    shape: Shape, dtype?: DType, name?: string): LayerVariable {
  // TODO(cais): Implement logic for dtype.
  return new LayerVariable(zeros(shape), dtype, name);
}

/**
 * Instantiates an all-zeros tensor of the same shape as another tensor.
 *
 * @param x The other tensor.
 * @param dtype DType of the tensor.
 * @param name Name of the tensor.
 * @return A newly instantiated Variable.
 */
export function zerosLike(
    x: Tensor, dtype?: DType, name?: string): LayerVariable {
  return new LayerVariable(tfc.zerosLike(x), dtype, name);
}

/**
 * Instantiates an all-ones tensor and returns it.
 *
 * @param shape Shape of the tensor.
 * @param dtype DType of the tensor.
 * @param name Name of the tensor.
 * @return An all-ones tensor.
 */
export function ones(shape: Shape, dtype?: DType): Tensor {
  // TODO(cais): Implement logic for dtype.
  return tfc.ones(shape);
}

/**
 * Instantiates an all-ones tensor and returns it.
 *
 * @param shape Shape of the tensor.
 * @param dtype DType of the tensor.
 * @param name Name of the tensor.
 * @return An all-ones Variable.
 */
export function onesVariable(
    shape: Shape, dtype?: DType, name?: string): LayerVariable {
  // TODO(cais): Implement logic for dtype.
  const allocated = tfc.ones(shape);
  return new LayerVariable(allocated, dtype, name);
}

/**
 * Instantiates an all-ones tensor of the same shape as another tensor.
 *
 * @param x The other tensor.
 * @param dtype DType of the tensor.
 * @param name Name of the tensor.
 * @return A newly instantiated Variable.
 */
export function onesLike(
    x: Tensor, dtype?: DType, name?: string): LayerVariable {
  const allocated = tfc.onesLike(x);
  return new LayerVariable(allocated, dtype, name);
}


/**
 * Create a ConcreteTensor with the same content as the input.
 *
 * @param x Input.
 * @return Identity output ConcreteTensor.
 */
export function identity(x: Tensor): Tensor {
  return x.clone();
}

/**
 * Instantiate an identity matrix and returns it.
 *
 * @param size Number of rows/columns.
 * @param dtype Data type of returned Tensor.
 * @param name Name of returned Tensor.
 * @return An identity matrix.
 */
export function eye(size: number, dtype?: DType, name?: string): Tensor {
  const buffer: number[] = [];
  for (let i = 0; i < size; ++i) {
    for (let j = 0; j < size; ++j) {
      buffer.push(i === j ? 1 : 0);
    }
  }
  return tensor2d(buffer, [size, size]);
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
    size: number, dtype?: DType, name?: string): LayerVariable {
  return new LayerVariable(eye(size, dtype), dtype, name);
}

/**
 * Negates a tensor.
 * @param x Tensor to negate.
 */
export function neg(x: Tensor): Tensor {
  return tfc.neg(x);
}

/**
 * Add two tensors, element-wise, with support for broadcasting.
 * @param x First tensor to add.
 * @param y Second tensor to add.
 * @returns Result of the addition.
 */
export function add(x: Tensor, y: Tensor): Tensor {
  return tfc.add(x, y);
}

/**
 * Subtract two tensors, element-wise, with support for broadcasting.
 * @param x First tensor to subtract element-wise.
 * @param y Second tensor to subtract element-wise.
 * @returns Result of the subtraction.
 */
export function subtract(x: Tensor, y: Tensor): Tensor {
  return tfc.sub(x, y);
}

/**
 * Multiply two tensors, element-wise, with support for broadcasting.
 * @param x First tensor to multiply.
 * @param y Second tensor to multiply.
 * @returns Result of the multiplication.
 */
export function multiply(x: Tensor, y: Tensor): Tensor {
  return tfc.mul(x, y);
}

/**
 * Divide two tensors element-wise, with support for broadcasting.
 * @param x First tensor to divide element-wise.
 * @param y Second tensor to divide element-wise.
 * @returns Result of the division.
 */
export function divide(x: Tensor, y: Tensor): Tensor {
  return tfc.div(x, y);
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
 * Get a tensor with uniform distribution of values.
 * @param shape Shape of the tensor.
 * @param minval Lower bound of the uniform distribution.
 * @param maxval Upper bound of the uniform distribution.
 * @return The uniform-random tensor.
 */
export function randomUniform(
    shape: Shape, minval: number, maxval: number, dtype?: DType,
    seed?: number): Tensor {
  // TODO(cais): Implement logic for dtype and seed once they are supported
  // by deeplearn.js.
  return tfc.randomUniform(shape, minval, maxval);
}

/**
 * Get a Variable with uniform distribution of values.
 * @param shape Shape of the tensor.
 * @param minval Lower bound of the uniform distribution.
 * @param maxval Upper bound of the uniform distribution.
 * @param dtype
 * @param seed
 * @param name Optional name.
 * @return The uniform-random Variable.
 */
export function randomUniformVariable(
    shape: Shape, minval: number, maxval: number, dtype?: DType, seed?: number,
    name = 'randomUniform'): LayerVariable {
  return new LayerVariable(
      randomUniform(shape, minval, maxval, dtype, seed), dtype, name);
}

/**
 * Get a tensor with truncated normal distribution of values.
 *
 * The truncated normal distribution is the same as a normal distribution,
 * except the values that are more than two stddev from the mean are dropped
 * and re-picked.
 * TODO(cais): The above specification regarding range truncation is not
 * true yet, due to the underlying Tensor.randTruncatedNormal from
 * deeplearn.js. Get it fixed.
 *
 * @param shape Shape of the tensor.
 * @param mean mean value of the normal distribution.
 * @param stddev standard deviation of the normal distribution.
 * @param dtype
 * @param seed
 * @return The truncated-normal tensor.
 */
export function truncatedNormal(
    shape: Shape, mean = 0.0, stddev = 1.0, dtype?: DType,
    seed?: number): Tensor {
  // TODO(cais): Implement logic for dtype and seed once they are supported
  // by deeplearn.js.
  return tfc.truncatedNormal(shape, mean, stddev);
}

/**
 * Get a Variable with truncated-normal distribution of values.
 * @param shape Shape of the tensor.
 * @param mean mean value of the normal distribution.
 * @param stddev standard deviation of the normal distribution.
 * @param dtype
 * @param seed
 * @param name Optional name.
 * @return The truncated-normal-random Variable.
 */
export function truncatedNormalVariable(
    shape: Shape, mean = 0.0, stddev = 1.0, dtype?: DType, seed?: number,
    name = 'truncatedNormal'): LayerVariable {
  // TODO(cais): Implement logic for dtype and seed once they are supported
  // by deeplearn.js.
  return new LayerVariable(
      truncatedNormal(shape, mean, stddev, dtype, seed), dtype, name);
}

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
    shape: Shape, mean = 0.0, stddev = 1.0, dtype?: DType,
    seed?: number): Tensor {
  if (dtype === DType.bool) {
    throw new NotImplementedError(`randomNormal does not support dType bool.`);
  }
  const dtypeString = (dtype === DType.float32) ? 'float32' : 'int32';
  return tfc.randomNormal(shape, mean, stddev, dtypeString, seed);
}

/**
 * Get a Variable with normal distribution of values.
 * @param shape Shape of the tensor.
 * @param mean mean value of the normal distribution.
 * @param stddev standard deviation of the normal distribution.
 * @param dtype
 * @param seed
 * @param name Optional name.
 * @return The truncated-normal-random Variable.
 */
export function randomNormalVariable(
    shape: Shape, mean = 0.0, stddev = 1.0, dtype?: DType, seed?: number,
    name = 'randomNormal'): LayerVariable {
  return new LayerVariable(
      randomNormal(shape, mean, stddev, dtype, seed), dtype, name);
}

/* Updating Variables */

/**
 * Update the value of a Variable.
 * @param x The Variable to be updated.
 * @param xNew The new value to update to.
 * @return The Variable updated.
 */
export function update(x: LayerVariable, xNew: Tensor): LayerVariable {
  return x.write(xNew);
}

/**
 * Update the value of a Variable by adding an increment.
 * @param x The Variable to be updated.
 * @param increment The incrment to add to `x`.
 * @return The Variable updated.
 */
export function updateAdd(x: LayerVariable, increment: Tensor): LayerVariable {
  return x.write(tfc.add(x.read(), increment));
}

/**
 * Update the value of a Variable by subtracting a decrement.
 * @param x The Variable to be updated.
 * @param decrement The decrement to subtract from `x`.
 * @return The Variable updated.
 */
export function updateSub(x: LayerVariable, decrement: Tensor): LayerVariable {
  return x.write(tfc.sub(x.read(), decrement));
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
  const zerosLikeX = coreZerosLike(x);
  const onesLikeX = coreOnesLike(x);
  return where(
      equal(x, zerosLikeX), zerosLikeX,
      where(
          greater(x, coreZerosLike(x)), onesLikeX,
          scalarTimesArray(getScalar(-1), onesLikeX)));
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
  // TODO(cais): Extend support to >2D as in `tf.qr` and move this function to
  //   the core.
  if (x.shape.length !== 2) {
    throw new ValueError(
        `qr() requires a 2D Tensor, but got a ${x.shape.length}D Tensor.`);
  }
  if (x.shape[0] < x.shape[1]) {
    throw new ValueError(
        `qr() requires x.shape[0] >= x.shape[1], but got shape: [${x.shape}]`);
  }

  const m = x.shape[0];
  const n = x.shape[1];

  let q = eye(m) as Tensor2D;  // Orthogonal transform so far.
  let r = x;                   // Transformed matrix so far.

  const one2D = tensor2d([[1]], [1, 1]);
  for (let j = 0; j < n; ++j) {
    // Find H = I - tau * w * w', to put zeros below R(j, j).
    const rjEnd1 = r.slice([j, j], [m - j, 1]);
    const normX = tfc.norm(rjEnd1);
    const rjj = r.slice([j, j], [1, 1]);
    const s = tfc.neg(sign(rjj)) as Tensor2D;
    const u1 = rjj.sub(multiply(s, normX)) as Tensor2D;
    const wPre = divide(rjEnd1, u1);
    let w: Tensor2D;
    if (wPre.shape[0] === 1) {
      w = one2D;
    } else {
      w = one2D.concat(
              wPre.slice([1, 0], [wPre.shape[0] - 1, wPre.shape[1]]), 0) as
          Tensor2D;
    }
    const tau = tfc.neg(divide(tfc.matMul(s, u1), normX)) as Tensor2D;

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
  }

  return [q, r];
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
  if (ndim(indices) !== 1) {
    throw new Error(
        'Only 1D one-hot tensors are supported in the ' +
        'deeplearn backend, at present.');
  }
  indices = indices.toInt();
  return tfc.oneHot(indices as Tensor1D, numClasses).toFloat();
}

/* Elementary math functions. */

/**
 * Mean of a tensor, alongside the specified axis.
 * @param x: Input tensor.
 * @param axis: An integer or an Array of integers.
 * @param keepDims: A boolean, whether to keep the dimensions or not
 *   If `keepDims` is `false`, the rank of the tensor is reduced
 *   by `1 for each entry in `axis`. If `keepDims` is `true`,
 *   the reduced dimensions are retained with length 1.
 */
export function mean(
    x: Tensor, axis?: number|number[], keepDims?: boolean): Scalar|Tensor {
  // TODO(michaelterry): Remove once negative supported axes in deeplearn.js
  axis = normalizeAxis(x, axis);
  return tfc.mean(x, axis, keepDims);
}

/**
 * Returns the index of the maximum value along an axis.
 * @param x Input tensor.
 * @param axis Axis along which to perform the reduction.
 * @returns The argmax index tensor.
 */
export function argmax(x: Tensor, axis = -1): Tensor {
  return tfc.argMax(x, axis);
}

/**
 * Retrieves the elements of indices `indices` in the tensor `reference`.
 * @param reference A tensor.
 * @param indices An integer tensor of indices or an `Array` of integers.
 * @param axis Axis along which to perform the gather operation.
 * @returns The result of the gathering as a tensor.
 */
export function gather(
    reference: Tensor, indices: number[]|Tensor1D, axis?: number): Tensor {
  if (Array.isArray(indices)) {
    indices = tensor1d(indices, 'int32');
  } else {
    indices = indices.toInt();
  }
  return tfc.gather(reference, indices, axis);
}

/**
 * Maximum value in a tensor.
 * @param x Input ConcreteTensor
 * @param axis The axis or a set of axes to find maximum values.
 *  *  If axis is undefined, return the maximum value across all axes.
 * @param keepDims Whether to keep the dimensions or not.
 *   If `keepDims` is `false`, the rank of the tensor is reduced
 *   by 1 for each entry in `axis`. If `keepDims` is `True`,
 *   the reduced dimensions are retained with length 1.
 *
 * @return: A tensor with maximum values of `x`.
 */
export function max(
    x: Tensor, axis?: number|number[], keepDims?: boolean): Scalar|Tensor {
  return tfc.max(x, axis, keepDims);
}

/**
 * Minimum value in a tensor.
 * @param x Input ConcreteTensor
 * @param axis The axis or the `Array` of axes to find minimum values over.
 *  If axis is undefined, return the minimum value across all axes.
 * @param keepDims Whether to keep the dimensions or not.
 *   If `keepDims` is `false`, the rank of the tensor is reduced
 *   by 1 for each entry in `axis`. If `keepDims` is `True`,
 *   the reduced dimensions are retained with length 1.
 * @return: A tensor with minimum values of `x`.
 */
export function min(
    x: Tensor, axis?: number|number[], keepDims?: boolean): Scalar|Tensor {
  return tfc.min(x, axis, keepDims);
}

/**
 * Element-wise minimum of two tensors.
 * @param x Input ConcreteTensor
 * @param y Input ConcreteTensor with shape and type compatible with `x`.
 * @return: Tensor with the minimum values between `x` and `y`.
 */
export function minimum(x: Tensor, y: Tensor): Tensor {
  return tfc.minimum(x, y);
}
/**
 * Sum value in a tensor.
 * @param x Input ConcreteTensor
 * @param axis The axis or the `Array` of axes to sum over.
 *  If axis is undefined, return the sum of all values across all axes.
 * @param keepDims Whether to keep the dimensions or not.
 *   If `keepDims` is `false`, the rank of the tensor is reduced
 *   by 1 for each entry in `axis`. If `keepDims` is `True`,
 *   the reduced dimensions are retained with length 1.
 * @return: A tensor with the sum values of `x`.
 */
export function sum(
    x: Tensor, axis?: number|number[], keepDims?: boolean): Tensor {
  return tfc.sum(x, axis, keepDims);
}

/**
 * Element-wise absolute value.
 * @param x Input tensor.
 * @return element-wise |x|.
 */
export function abs(x: Tensor): Tensor {
  return tfc.abs(x);
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
 * Element-wise sqrt.
 * @param x Input tensor.
 * @return element-wise sqrt(x)
 */
export function sqrt(x: Tensor): Tensor {
  return tfc.sqrt(x);
}

/**
 * Element-wise exp.
 * @param x Input tensor.
 * @return element-wise exp(x)
 */
export function exp(x: Tensor): Tensor {
  return tfc.exp(x);
}

/**
 * Element-wise logarithm.
 *
 * @param x Input tensor.
 * @returns Element-wise log(x).
 */
export function log(x: Tensor): Tensor {
  return tfc.log(x);
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
  if (typeof (a) === 'number') {
    a = scalar(Math.round(a), 'int32');
  }
  if (a.dtype !== 'int32') {
    throw new NotImplementedError(
        `Non-int32 dtype (${a.dtype}) is not supported by pow() yet`);
  }
  return tfc.pow(x, a as Tensor);
}

/**
 * Clips values element-wise.
 *
 * @param x Input Tensor or Variable.
 * @param minValue The lowest allowed value in the output
 * @param maxValue The highest allowed value in the output
 * @returns Tensor with values limited to be between min_value and max_value
 */
export function clip(x: Tensor, minValue: number, maxValue: number): Tensor {
  return tfc.clipByValue(x, minValue, maxValue);
}

/**
 * Element-wise equality between two tensors.
 * @param x
 * @param y
 * @returns A tensor of the same shape as `x`, consisting of `0`(s) and `1`(s).
 */
export function equal(x: Tensor, y: Tensor): Tensor {
  return tfc.equal(x, y);
}

/**
 * Element-wise truth value of (x > y).
 * @param x
 * @param y
 * @returns A tensor of the same shape as `x`, consisting of `0`(s) and `1`(s).
 */
export function greater(x: Tensor, y: Tensor): Tensor {
  return tfc.greater(x, y);
}

/**
 * Element-wise truth value of (x >= y).
 * @param x
 * @param y
 * @returns A tensor of the same shape as `x`, consisting of `0`(s) and `1`(s).
 */
export function greaterEqual(x: Tensor, y: Tensor): Tensor {
  return tfc.greaterEqual(x, y);
}

/**
 * Element-wise maximum of two tensors.
 */
export function maximum(x: Tensor, y: Tensor): Tensor {
  return tfc.maximum(x, y);
}

/**
 * Element-wise sin.
 *
 * @param x Input Tensor or Variable.
 * @returns Element-wise sin(x).
 */
export function sin(x: ConcreteTensor): Tensor {
  return tfc.sin(x.value());
}

/**
 * Element-wise cos.
 *
 * @param x Input Tensor or Variable.
 * @returns Element-wise cos(x).
 */
export function cos(x: ConcreteTensor): Tensor {
  return tfc.cos(x.value());
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
        variance as Tensor2D | Tensor1D, epsilon);
  } else if (ndim(x) === 3) {
    // TODO(cais): Check rank; give proper error message.
    out = tfc.batchNormalization3d(
        x as Tensor3D, mean as Tensor3D | Tensor1D,
        variance as Tensor3D | Tensor1D, epsilon);
  } else if (ndim(x) === 4) {
    out = tfc.batchNormalization4d(
        x as Tensor4D, mean as Tensor4D | Tensor1D,
        variance as Tensor4D | Tensor1D, epsilon);
  } else {
    throw new NotImplementedError(
        `batchNormalization is not implememnted for array of rank ${ndim(x)} ` +
        `yet`);
  }
  if (gamma != null) {
    out = multiply(out, gamma);
  }
  if (beta != null) {
    out = add(out, beta);
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
        y = x.add(bias.reshape([1, biasShape[2], biasShape[0], biasShape[1]]));
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
 * Scaled Exponential linear unit (SELU).
 * Stable and Attracting Fixed Point (0, 1) for Normalized Weights.
 * see: https://arxiv.org/abs/1706.02515
 *
 * @param x A tensor or variable to compute the activation function for.
 * @return Output of the SELU operation.
 */
export function selu(x: Tensor): Tensor {
  return tfc.selu(x);
}

/**
 * Rectified linear unit (ReLU).
 * @param x Input Tensor.
 * @return ReLU output.
 */
export function relu(x: Tensor): Tensor {
  // TODO(cais): Add params alpha and max_value.
  return tfc.relu(x);
}

/**
 * Softplus of a tensor.
 *
 * Defined as log(exp(x) + 1), element-wise.
 *
 * @param x: Input.
 * @returns Output.
 */
export function softplus(x: Tensor): Tensor {
  return tfc.log(tfc.add(getScalar(1), tfc.exp(x)));
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
  return tfc.div(x, tfc.add(getScalar(1), tfc.abs(x)));
}

/**
 * Element-wise hyperbolic tan.
 *
 * @param x Input Tensor or Variable.
 * @returns Element-wise tanh(x).
 */
export function tanh(x: Tensor): Tensor {
  return tfc.tanh(x);
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
      neg(level) as Scalar, randomUniform(x.shape, 0, 1, DType.float32)));
  // Scale the kept elements, so the expected sum is unchanged.
  multiplier = tfc.mul(
      divide(getScalar(1), subtract(getScalar(1), level)) as Scalar,
      multiplier);
  return tfc.mul(x, multiplier);
}

/**
 * Normalizes a tensor wrt the L2 norm alongside the specified axis.
 * @param x
 * @param axis Axis along which to perform normalization.
 */
export function l2Normalize(x: Tensor, axis?: number): Tensor {
  const squareSum = sum(square(x), axis, true);
  const epsilonTensor = scalarTimesArray(scalar(epsilon()), tfc.onesLike(x));
  const norm = sqrt(maximum(squareSum, epsilonTensor));
  return divide(x, norm);
}

/**
 * Transpose and cast the input before the conv2d.
 * @param x Input image tensor.
 * @param dataFormat
 */
function preprocessConv2DInput(x: Tensor, dataFormat: DataFormat): Tensor {
  // TODO(cais): Cast type to float32 if not.
  checkDataFormat(dataFormat);
  if (dataFormat === 'channelsFirst') {
    return tfc.transpose(x, [0, 2, 3, 1]);  // NCHW -> NHWC.
  } else {
    return x;
  }
}

/**
 * 1D-convolution with bias added.
 *
 * Porting Note: This function does not exist in the Python Keras backend.
 *   It is exactly the same as `conv2d`, except the added `bias`.
 *
 * @param x Input tensor, rank-3, of shape `[batchSize, width, inChannels]`.
 * @param kernel Kernel, rank-3, of shape `[filterWidth, inDepth, outDepth]`.
 * @param bias Bias, rank-3, of shape `[outDepth]`.
 * @param strides
 * @param padding Padding mode.
 * @param dataFormat Data format.
 * @param dilationRate
 * @returns The result of the 1D convolution.
 * @throws ValueError, if `x`, `kernel` or `bias` is not of the correct rank.
 */
export function conv1dWithBias(
    x: Tensor, kernel: Tensor, bias: Tensor, strides = 1, padding = 'valid',
    dataFormat?: DataFormat, dilationRate = 1): Tensor {
  if (dataFormat == null) {
    dataFormat = imageDataFormat();
  }
  checkDataFormat(dataFormat);

  // Check the ranks of x, kernel and bias.
  if (x.shape.length !== 3) {
    throw new ValueError(
        `The input of a conv1dWithBias operation should be 3, but is ` +
        `${x.shape.length} instead.`);
  }
  if (kernel.shape.length !== 3) {
    throw new ValueError(
        `The kernel for a conv1dWithBias operation should be 3, but is ` +
        `${kernel.shape.length} instead`);
  }
  if (bias != null && bias.shape.length !== 1) {
    throw new ValueError(
        `The bias for a conv1dWithBias operation should be 1, but is ` +
        `${kernel.shape.length} instead`);
  }

  // TODO(cais): Support CASUAL padding mode.

  if (dataFormat === 'channelsFirst') {
    x = transpose(x, [0, 2, 1]);  // NCW -> NWC.
  }
  if (padding === 'casual') {
    throw new NotImplementedError(
        'The support for CASUAL padding mode in conv1dWithBias is not ' +
        'implemented yet.');
  }
  let y: Tensor = tfc.conv1d(
      x as Tensor2D | Tensor3D, kernel as Tensor3D, strides,
      padding === 'same' ? 'same' : 'valid', 'NWC', dilationRate);
  if (bias != null) {
    y = biasAdd(y, bias);
  }
  return y;
}

/**
 * 1D-convolution.
 *
 * @param x Input tensor, rank-3, of shape `[batchSize, width, inChannels]`.
 * @param kernel Kernel, rank-3, of shape `[filterWidth, inDepth, outDepth]`.s
 * @param strides
 * @param padding Padding mode.
 * @param dataFormat Data format.
 * @param dilationRate
 * @returns The result of the 1D convolution.
 * @throws ValueError, if `x`, `kernel` or `bias` is not of the correct rank.
 */
export function conv1d(
    x: Tensor, kernel: Tensor, strides = 1, padding = 'valid',
    dataFormat?: DataFormat, dilationRate = 1): Tensor {
  checkDataFormat(dataFormat);
  return conv1dWithBias(
      x, kernel, null, strides, padding, dataFormat, dilationRate);
}

/**
 * 2D Convolution
 * @param x
 * @param kernel kernel of the convolution.
 * @param strides strides array.
 * @param padding padding mode. Default to 'valid'.
 * @param dataFormat data format. Defaults to 'channelsLast'.
 * @param dilationRate dilation rate array.
 * @returns Result of the 2D pooling.
 */
export function conv2d(
    x: Tensor, kernel: Tensor, strides = [1, 1], padding = 'valid',
    dataFormat?: DataFormat, dilationRate?: [number, number]): Tensor {
  checkDataFormat(dataFormat);
  return conv2dWithBias(
      x, kernel, null, strides, padding, dataFormat, dilationRate);
}

/**
 * 2D Convolution with an added bias.
 * Note: This function does not exist in the Python Keras Backend. This function
 * is exactly the same as `conv2d`, except the added `bias`.
 */
export function conv2dWithBias(
    x: Tensor, kernel: Tensor, bias: Tensor, strides = [1, 1],
    padding = 'valid', dataFormat?: DataFormat,
    dilationRate?: [number, number]): Tensor {
  if (dataFormat == null) {
    dataFormat = imageDataFormat();
  }
  checkDataFormat(dataFormat);
  if (ndim(x) !== 3 && ndim(x) !== 4) {
    throw new ValueError(
        `conv2dWithBias expects input to be of rank 3 or 4, but received ` +
        `${ndim(x)}.`);
  }
  if (ndim(kernel) !== 3 && ndim(kernel) !== 4) {
    throw new ValueError(
        `conv2dWithBias expects kernel to be of rank 3 or 4, but received ` +
        `${ndim(x)}.`);
  }
  let y = preprocessConv2DInput(x, dataFormat);
  if (padding === 'casual') {
    throw new NotImplementedError(
        'The support for CASUAL padding mode in conv1dWithBias is not ' +
        'implemented yet.');
  }
  y = tfc.conv2d(
      y as Tensor3D | Tensor4D, kernel as Tensor4D, strides as [number, number],
      padding === 'same' ? 'same' : 'valid', 'NHWC', dilationRate);
  if (bias != null) {
    y = biasAdd(y, bias as Tensor1D);
  }
  if (dataFormat === 'channelsFirst') {
    y = tfc.transpose(y, [0, 3, 1, 2]);
  }
  return y;
}

/**
 * 2D convolution with separable filters.
 * @param x Input tensor.
 * @param depthwiseKernel Convolution kernel for depthwise convolution.
 * @param strides Strides (Array of two integers).
 * @param padding Padding model.
 * @param dataFormat Data format.
 * @param dilationRate Array of two integers, dilation rates for the separable
 *   convolution.
 * @returns Output tensor.
 * @throws ValueError If depthwiseKernel is not a 4D array.
 */
export function depthwiseConv2d(
    x: Tensor, depthwiseKernel: Tensor, strides: [number, number] = [1, 1],
    padding = 'valid', dataFormat?: DataFormat,
    dilationRate?: [number, number]): Tensor {
  if (dataFormat == null) {
    dataFormat = imageDataFormat();
  }
  checkDataFormat(dataFormat);
  let y = preprocessConv2DInput(x, dataFormat);
  if (ndim(x) !== 4) {
    throw new ValueError(
        `Input for depthwiseConv2d is required to be 4-D, but is instead ` +
        `${ndim(x)}-D`);
  }
  if (ndim(depthwiseKernel) !== 4) {
    throw new ValueError(
        `depthwiseKernel is required to be 4-D, but is instead ` +
        `${ndim(depthwiseKernel)}-D`);
  }
  y = tfc.depthwiseConv2d(
      y as Tensor4D, depthwiseKernel as Tensor4D, strides,
      padding === 'same' ? 'same' : 'valid', 'NHWC', dilationRate);
  if (dataFormat === 'channelsFirst') {
    y = tfc.transpose(y, [0, 3, 1, 2]);
  }
  return y;
}

/**
 * 2D pooling.
 * @param x
 * @param poolSize
 * @param stridesdes strides. Defaults to [1, 1].
 * @param padding padding. Defaults to 'valid'.
 * @param dataFormat data format. Defaults to 'channelsLast'.
 * @param poolMode Mode of pooling. Defaults to 'max'.
 * @returns Result of the 2D pooling.
 */
export function pool2d(
    x: Tensor, poolSize: [number, number], strides?: [number, number],
    padding?: PaddingMode, dataFormat?: DataFormat,
    poolMode?: PoolMode): Tensor {
  checkDataFormat(dataFormat);
  checkPoolMode(poolMode);
  checkPaddingMode(padding);
  if (strides == null) {
    strides = [1, 1];
  }
  if (padding == null) {
    padding = 'valid';
  }
  if (dataFormat == null) {
    dataFormat = imageDataFormat();
  }
  if (poolMode == null) {
    poolMode = 'max';
  }

  // TODO(cais): Remove the preprocessing step once deeplearn.js supports
  // dataFormat as an input argument.
  x = preprocessConv2DInput(x, dataFormat);  // x is NHWC after preprocessing.
  let y: Tensor;
  const paddingString = (padding === 'same') ? 'same' : 'valid';
  if (poolMode === 'max') {
    // TODO(cais): Rank check?
    y = tfc.maxPool(x as Tensor4D, poolSize, strides, paddingString);
  } else {  // 'avg'
    // TODO(cais): Check the dtype and rank of x and give clear error message
    //   if those are incorrect.
    y = tfc.avgPool(
        // TODO(cais): Rank check?
        x as Tensor3D | Tensor4D, poolSize, strides, paddingString);
  }
  if (dataFormat === 'channelsFirst') {
    y = tfc.transpose(y, [0, 3, 1, 2]);  // NHWC -> NCHW.
  }
  return y;
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
export function floatx(): DType {
  return DType.float32;
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
 * Computes the softmax function on an input tensor across the last dimension.
 *
 * Implemented by:
 *  shuffling the dimensions in x to put the axis last
 *  reshaping into a 2D tensor
 *  taking the softmax over the last dimension, and then reshaping back.
 *  undoing the dimension shuffle
 *
 * @param xNDA numeric tensor with 1 or more dimensions.
 *
 * @return A Tensor the same shape as x, but with softmax calculated
 * across the last dimension.
 */
export function softmax(x: Tensor, axis = -1): Tensor {
  return tfc.softmax(x, axis);
}

/**
 * Categorical crossentropy between an output tensor and a target tensor.
 *
 * @param target A tensor of the same shape as `output`.
 * @param output A tensor resulting from a softmax (unless `fromLogits` is
 *  `true`, in which case `output` is expected to be the logits).
 * @param fromLogits Boolean, whether `output` is the result of a softmax, or is
 *   a tensor of logits.
 */
export function categoricalCrossentropy(
    target: Tensor, output: Tensor, fromLogits = false): Tensor {
  if (fromLogits) {
    output = softmax(output);
  } else {
    // scale preds so that the class probabilities of each sample sum to 1.
    const outputSum = sum(output, shape(output).length - 1, true);
    output = divide(output, outputSum);
  }
  output = clip(output, epsilon(), 1 - epsilon());
  return tfc.neg(tfc.sum(
      tfc.mul(target.toFloat(), tfc.log(output)), shape(output).length - 1));
}

/**
 * Categorical crossentropy with integer targets.
 *
 * @param target An integer tensor.
 * @param output A tensor resulting from a softmax (unless `fromLogits` is
 *  `true`, in which case `output` is expected to be the logits).
 * @param fromLogits Boolean, whether `output` is the result of a softmax, or is
 *   a tensor of logits.
 */
export function sparseCategoricalCrossentropy(
    target: Tensor, output: Tensor, fromLogits = false): Tensor {
  const flatTarget = tfc.floor(flatten(target)).toInt() as Tensor1D;
  const outputShape = shape(output);
  const oneHotTarget = reshape(
      tfc.oneHot(flatTarget, outputShape[outputShape.length - 1]), outputShape);
  return categoricalCrossentropy(oneHotTarget, output, fromLogits);
}

/**
 * Binary crossentropy between an output tensor and a target tensor.
 *
 * @param target A tensor with the same shape as `output`.
 * @param output
 * @param fromLogits Whether `output` is expected to be a logits tensor. By
 *   default, we consider that `output` encodes a probability distribution.
 */
export function binaryCrossentropy(
    target: Tensor, output: Tensor, fromLogits = false): Tensor {
  let y: Tensor;
  if (!fromLogits) {
    y = clip(output, epsilon(), 1 - epsilon());
    y = log(divide(y, subtract(tfc.onesLike(y), y)));
  } else {
    y = output;
  }
  return sigmoidCrossEntropyWithLogits(target, y);
}

/**
 * From TensorFlow's implementation in nn_impl.py:
 *
 * For brevity, let `x = logits`, `z = labels`.  The logistic loss is
 *      z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
 *    = z * -log(1 / (1 + exp(-x))) + (1 - z) * -log(exp(-x) / (1 + exp(-x)))
 *    = z * log(1 + exp(-x)) + (1 - z) * (-log(exp(-x)) + log(1 + exp(-x)))
 *    = z * log(1 + exp(-x)) + (1 - z) * (x + log(1 + exp(-x))
 *    = (1 - z) * x + log(1 + exp(-x))
 *    = x - x * z + log(1 + exp(-x))
 * For x < 0, to avoid overflow in exp(-x), we reformulate the above
 *      x - x * z + log(1 + exp(-x))
 *    = log(exp(x)) - x * z + log(1 + exp(-x))
 *    = - x * z + log(1 + exp(x))
 * Hence, to ensure stability and avoid overflow, the implementation uses this
 * equivalent formulation
 *    max(x, 0) - x * z + log(1 + exp(-abs(x)))
 *
 * @param target The labels.
 * @param output The logits.
 */
export function sigmoidCrossEntropyWithLogits(
    target: Tensor, output: Tensor): Tensor {
  const maxOutput = tfc.maximum(output, tfc.zerosLike(output));
  const outputXTarget = tfc.mul(output, target);
  const sigmoidOutput =
      tfc.log(tfc.add(getScalar(1), tfc.exp(tfc.neg(tfc.abs(output)))));
  const result = tfc.add(tfc.sub(maxOutput, outputXTarget), sigmoidOutput);
  return result;
}

/**
 * Element-wise sigmoid.
 */
export function sigmoid(x: Tensor): Tensor {
  return tfc.sigmoid(x);
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
  // TODO(cais): Maybe avoid creating scalar constants on each invocation by
  //   turning them into module-level constants.
  const y = scalarPlusArray(scalar(0.5), scalarTimesArray(scalar(0.2), x));
  return clip(y, 0, 1);
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
 * Iterates over the time dimension of a tensor.
 *
 * @param stepFunction RNN step function.
 *   Parameters:
 *     inputs: tensor with shape `[samples, ...]` (no time dimension),
 *       representing input for the batch of samples at a certain time step.
 *     states: an Array of tensors.
 *   Returns:
 *     outputs: tensor with shape `[samples, outputDim]` (no time dimension).
 *     newStates: list of tensors, same length and shapes as `states`. The first
 *       state in the list must be the output tensor at the previous timestep.
 * @param inputs Tensor of temporal data of shape `[samples, time, ...]` (at
 *   least 3D).
 * @param initialStates Tensor with shape `[samples, outputDim]` (no time
 *   dimension), containing the initial values of the states used in the step
 *   function.
 * @param goBackwards If `true`, do the iteration over the time dimension in
 *   reverse order and return the reversed sequence.
 * @param mask Binary tensor with shape `[sample, time, 1]`, with a zero for
 *   every element that is masked.
 * @param constants An Array of constant values passed at each step.
 * @param unroll Whether to unroll the RNN or to use a symbolic loop. *Not*
 *   applicable to this imperative deeplearn.js backend. Its value is ignored.
 * @param inputLength Not relevant in this deeplearn.js backend.
 * @returns An Array: `[lastOutput, outputs, newStates]`.
 *   lastOutput: the lastest output of the RNN, of shape `[samples, ...]`.
 *   outputs: tensor with shape `[samples, time, ...]` where each entry
 *     `output[s, t]` is the output of the step function at time `t` for sample
 *     `s`.
 *   newStates: Array of tensors, latest states returned by the step function,
 *      of shape `(samples, ...)`.
 * @throws ValueError If input dimension is less than 3.
 */
export function rnn(
    stepFunction: RnnStepFunction, inputs: Tensor, initialStates: Tensor[],
    goBackwards = false, mask?: Tensor, constants?: Tensor[], unroll = false,
    inputLength?: number): [Tensor, Tensor, Tensor[]] {
  const ndim = inputs.shape.length;
  if (ndim < 3) {
    throw new ValueError(`Input should be at least 3D, but is ${ndim}D.`);
  }

  // Transpose to time-major, i.e., from [batch, time, ...] to [time, batch,
  // ...].
  const axes = [1, 0].concat(math_utils.range(2, ndim));
  inputs = transpose(inputs, axes);

  if (mask != null) {
    throw new NotImplementedError(
        'The rnn() function of the deeplearn.js backend does not support ' +
        'masking yet.');
  }

  if (constants != null) {
    throw new NotImplementedError(
        'The rnn() functoin of the deeplearn.js backend does not support ' +
        'constants yet.');
  }

  // Porting Note: the unroll option is ignored by the imperative backend.
  if (unroll) {
    console.warn(
        'Backend rnn(): the unroll = true option is not applicable to the ' +
        'imperative deeplearn.js backend.');
  }

  if (goBackwards) {
    inputs = reverse(inputs, 0);
  }

  // Porting Note: PyKeras with TensorFlow backend uses a symbolic loop
  //   (tf.while_loop). But for the imperative deeplearn.js backend, we just use
  //   the usual TypeScript control flow to iterate over the time steps in the
  //   inputs.
  // Porting Note: PyKeras patches a "_use_learning_phase" attribute to outputs.
  //   This is not idiomatic in TypeScript. The info regarding whether we are
  //   in a learning (i.e., training) phase for RNN is passed in a different
  //   way.
  //   TODO(cais): Determine in what exact way the learning phase information
  //     will be passed.

  let outputs: Tensor;
  let lastOutput: Tensor;
  let states = initialStates;
  const timeSteps = inputs.shape[0];
  for (let t = 0; t < timeSteps; ++t) {
    let currentInput = sliceAlongFirstAxis(inputs, t, 1);
    currentInput = reshape(currentInput, currentInput.shape.slice(1));
    const stepOutputs = stepFunction(currentInput, states);
    lastOutput = stepOutputs[0];
    if (t === 0) {
      outputs = lastOutput.reshape([1].concat(lastOutput.shape));
    } else {
      outputs = concatAlongFirstAxis(
          outputs, lastOutput.reshape([1].concat(lastOutput.shape)));
    }
    // TODO(soergel): Call K.concatenate() to perform only one concatenation at
    // the end, once the backend function is available.
    states = stepOutputs[1];
  }

  return [
    lastOutput,
    transpose(
        outputs, [1, 0].concat(math_utils.range(2, outputs.shape.length))),
    states
  ];
}

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
