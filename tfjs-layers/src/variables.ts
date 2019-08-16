/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

import * as tfc from '@tensorflow/tfjs-core';
import {DataType, Tensor, variableGrads} from '@tensorflow/tfjs-core';

import {getNextUniqueTensorId} from './backend/state';
import {getScopedTensorName, getUniqueTensorName} from './common';
import {Constraint} from './constraints';
import {NotImplementedError} from './errors';
import {Shape} from './keras_format/common';
import {HasShape} from './types';

const DEFAULT_VARIABLE_NAME_PREFIX = 'Variable';

/**
 * A `tf.layers.LayerVariable` is similar to a `tf.Tensor` in that it has a
 * dtype and shape, but its value is mutable.  The value is itself represented
 * as a`tf.Tensor`, and can be read with the `read()` method and updated with
 * the `write()` method.
 */
export class LayerVariable {
  readonly dtype: DataType;
  readonly shape: Shape;

  readonly id: number;
  // The fully scoped name of this Variable, including a unique suffix if needed
  readonly name: string;
  // The originally requested fully scoped name of this Variable, not including
  // any unique suffix.  This may be needed when restoring weights because this
  // original name is used as a key.
  readonly originalName: string;
  private trainable_: boolean;

  protected readonly val: tfc.Variable;
  readonly constraint: Constraint;
  /**
   * Construct Variable from a `tf.Tensor`.
   *
   * If not explicitly named, the Variable will be given a name with the
   * prefix 'Variable'. Variable names are unique. In the case of name
   * collision, suffixies '_<num>' will be added to the name.
   *
   * @param val Initial value of the Variable.
   * @param name Name of the variable. If `null` or `undefined` is provided, it
   *   will default a name with the prefix 'Variable'.
   * @param constraint Optional, projection function to be applied to the
   * variable after optimize updates
   * @throws ValueError if `name` is `null` or `undefined`.
   */
  constructor(
      val: Tensor, dtype: DataType = 'float32',
      name = DEFAULT_VARIABLE_NAME_PREFIX, trainable = true,
      constraint: Constraint = null) {
    this.dtype = dtype == null ? 'float32' : dtype;
    this.shape = val.shape;
    this.id = getNextUniqueTensorId();

    name = name == null ? DEFAULT_VARIABLE_NAME_PREFIX : name;
    this.originalName = getScopedTensorName(name);
    this.name = getUniqueTensorName(this.originalName);

    this.trainable_ = trainable;
    this.constraint = constraint;

    this.val = tfc.variable(val, this.trainable_, this.name, this.dtype);
  }

  /**
   * Get a snapshot of the Variable's value.
   *
   * The returned value is a snapshot of the Variable's value at the time of
   * the invocation. Future mutations in the value of the tensor will only
   * be reflected by future calls to this method.
   */
  read(): Tensor {
    this.assertNotDisposed();
    return this.val;
  }

  /**
   * Update the value of the Variable.
   *
   * @param newVal: The new value to update to. Must be consistent with the
   *   dtype and shape of the Variable.
   * @return This Variable.
   */
  write(newVal: Tensor) {
    // TODO(cais): Once  TF.js Core supports Tensor.dtype, check dtype match.
    this.assertNotDisposed();
    checkShapesMatch(this.val, newVal);
    // Skip updating if this is the exact same tensor.
    if (this.val.id !== newVal.id) {
      this.val.assign(newVal);
      if (this.constraint != null) {
        this.val.assign(this.constraint.apply(this.val));
      }
    }
    return this;
  }

  /**
   * Dispose this LayersVariable instance from memory.
   */
  dispose(): void {
    this.assertNotDisposed();
    this.val.dispose();
  }

  protected assertNotDisposed(): void {
    if (this.val.isDisposed) {
      throw new Error(`LayersVariable ${this.name} is already disposed.`);
    }
  }

  get trainable(): boolean {
    return this.trainable_;
  }

  set trainable(trainable: boolean) {
    this.trainable_ = trainable;
    this.val.trainable = trainable;
  }
}

function checkShapesMatch(x: HasShape, y: HasShape): void {
  if (x.shape.toString() !== y.shape.toString()) {
    throw new Error(
        'Shape mismatch: ' + JSON.stringify(x.shape) + ' vs. ' +
        JSON.stringify(y.shape));
  }
}

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
    x: Tensor, dtype?: DataType, name?: string,
    constraint?: Constraint): LayerVariable {
  return new LayerVariable(x, dtype, name, true, constraint);
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
    shape: Shape, dtype?: DataType, name?: string): LayerVariable {
  // TODO(cais): Implement logic for dtype.
  return new LayerVariable(tfc.zeros(shape), dtype, name);
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
    x: Tensor, dtype?: DataType, name?: string): LayerVariable {
  return new LayerVariable(tfc.zerosLike(x), dtype, name);
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
    shape: Shape, dtype?: DataType, name?: string): LayerVariable {
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
    x: Tensor, dtype?: DataType, name?: string): LayerVariable {
  const allocated = tfc.onesLike(x);
  return new LayerVariable(allocated, dtype, name);
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
  return new LayerVariable(tfc.eye(size), dtype, name);
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
    shape: Shape, minval: number, maxval: number, dtype?: DataType,
    seed?: number, name = 'randomUniform'): LayerVariable {
  return new LayerVariable(
      tfc.randomUniform(shape, minval, maxval, dtype), dtype, name);
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
    shape: Shape, mean = 0.0, stddev = 1.0, dtype?: DataType, seed?: number,
    name = 'truncatedNormal'): LayerVariable {
  // TODO(cais): Implement logic for dtype and seed once they are supported
  // by deeplearn.js.
  dtype = dtype || 'float32';
  if (dtype !== 'float32' && dtype !== 'int32') {
    throw new NotImplementedError(
        `randomNormal does not support dType ${dtype}.`);
  }
  return new LayerVariable(
      tfc.truncatedNormal(shape, mean, stddev, dtype, seed), dtype, name);
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
    shape: Shape, mean = 0.0, stddev = 1.0, dtype?: DataType, seed?: number,
    name = 'randomNormal'): LayerVariable {
  dtype = dtype || 'float32';
  if (dtype !== 'float32' && dtype !== 'int32') {
    throw new NotImplementedError(
        `randomNormalVariable does not support dType ${dtype}.`);
  }
  return new LayerVariable(
      tfc.randomNormal(shape, mean, stddev, dtype, seed), dtype, name);
}

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

/**
 * Get the values of an array of Variables.
 *
 * @param tensors An `Array` of `Variable`s to get the values of.
 * @return The values of the inputs, as an `Array` of`tf.Tensor`s.
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
  variablesAndValues.forEach(variableAndValue => {
    const variable: LayerVariable = variableAndValue[0];
    variable.write(variableAndValue[1]);
  });
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
    lossFn: () => tfc.Scalar, variables: LayerVariable[]): Tensor[] {
  // TODO(cais): The return type signature can be simplified if deeplearn makes
  //   the corresponding type public.
  const variableList =
      variables.map(variable => variable.read() as tfc.Variable);
  const valudAndGrads = variableGrads(lossFn, variableList);
  return variables.map(variable => valudAndGrads.grads[variable.name]);
}
