/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

/** Defines allowable data types for tensors. */

// tslint:disable:max-line-length
import * as tfc from '@tensorflow/tfjs-core';
import {doc, Tensor, variable} from '@tensorflow/tfjs-core';

import {getUniqueTensorName} from './common';
import {Constraint} from './constraints';
import {Layer} from './engine/topology';
// tslint:enable:max-line-length


/** @docalias 'float32'|'int32'|'bool' */
export enum DType {
  float32 = 'float32',
  int32 = 'int32',
  bool = 'bool',
}

/** @docalias number[] */
export type Shape = number[];

/**
 * Tensor interface.
 *
 * A tensor has a dtype and a shape.
 */
export interface TensorInterface {
  /**
   * Data type of the Tensor.
   *
   * E.g., float32, int32.
   */
  readonly dtype: DType;

  /**
   * Shape of the Tensor.
   *
   * E.g., [], [2, 2], [6, 28, 28].
   */
  readonly shape: Shape;
}

/**
 * An ID to track `SymbolicTensor`s and derived classes.
 * Required in different places in engine/topology.ts to identify unique
 * tensors.
 */
let _nextUniqueTensorId = 0;

/**
 * `SymbolicTensor` is a placeholder for a Tensor without any concrete value.
 *
 * They are most often encountered when building a graph of `Layer`s for a
 * a `Model` and the input data's shape, but not values are known.
 */
@doc({heading: 'Models', 'subheading': 'Classes'})
export class SymbolicTensor implements TensorInterface {
  /* A unique ID for the tensor to be able to differentiate tensors. */
  readonly id: number;
  readonly name?: string;
  /**
   * Replacement for _keras_history.
   */
  nodeIndex: number;
  /**
   * Replacement for _keras_history.
   */
  tensorIndex: number;

  /**
   *
   * @param dtype
   * @param shape
   * @param sourceLayer The Layer that produced this symbolic tensor.
   * @param inputs The inputs passed to sourceLayer's __call__() method.
   * @param nodeIndex
   * @param tensorIndex
   * @param callArgs The keyword arguments passed to the __call__() method.
   * @param name
   * @param outputTensorIndex The index of this tensor in the list of outputs
   *   returned by apply().
   */
  constructor(
      readonly dtype: DType, readonly shape: Shape, public sourceLayer: Layer,
      readonly inputs: SymbolicTensor[],
      // tslint:disable-next-line:no-any
      readonly callArgs: any, name?: string,
      readonly outputTensorIndex?: number) {
    this.id = _nextUniqueTensorId++;
    if (name != null) {
      this.name = getUniqueTensorName(name);
    }
  }
}

/**
 * A thin wrapper around a Tensor, with an optional name. The value is set once
 * during construction, and cannot be mutated afterwards.
 */
export class ConcreteTensor implements TensorInterface {
  readonly dtype: DType;
  readonly shape: Shape;

  readonly id: number;
  readonly name?: string;

  protected val: Tensor;

  /**
   * Construct a ConcreteTensor from an Tensor.
   * @param val: The value of the ConcreteTensor.
   * @param name: Optional name. If Truthy, Tensor names are unique. Name
   *   collisions will be resolved by appending suffix '_<num>'.
   */
  constructor(val: Tensor, name?: string) {
    // TODO(cais): This is faked to always give float32 for now. Implement real
    // logic once deeplearn.js supports DTypes.
    this.dtype = DType.float32;
    this.shape = val.shape;
    this.val = val;
    this.id = _nextUniqueTensorId++;

    if (name != null) {
      this.name = getUniqueTensorName(name);
    }
  }

  /**
   * Read the value of the tensor.
   * @return Tensor value of the tensor.
   */
  public value(): Tensor {
    return this.val;
  }
}

function checkShapesMatch(
    x: Tensor|TensorInterface, y: Tensor|TensorInterface): void {
  if (x.shape.toString() !== y.shape.toString()) {
    throw new Error(
        'Shape mismatch: ' + JSON.stringify(x.shape) + ' vs. ' +
        JSON.stringify(y.shape));
  }
}

function getValueTensor(val: Tensor|ConcreteTensor): Tensor {
  return val instanceof ConcreteTensor ? val.value() : val;
}

const DEFAULT_VARIABLE_NAME_PREFIX = 'Variable';

/**
 * A `LayerVariable` is similar to a `Tensor` in that it has a dtype and shape,
 * but its value is mutable.  The value is itself represented as a `Tensor`, and
 * can be read with the `read()` method and updated with the `write()` method.
 */
export class LayerVariable {
  readonly dtype: DType;
  readonly shape: Shape;

  readonly id: number;
  readonly name: string;
  readonly trainable: boolean;

  protected readonly val: tfc.Variable;
  readonly constraint: Constraint;
  /**
   * Construct Variable from a Tensor or a ConcreteTensor.
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
      val: Tensor|ConcreteTensor, dtype: DType = DType.float32,
      name = DEFAULT_VARIABLE_NAME_PREFIX, trainable = true,
      constraint: Constraint = null) {
    this.dtype = dtype == null ? DType.float32 : dtype;
    this.shape = val.shape;
    this.id = _nextUniqueTensorId++;
    this.name =
        getUniqueTensorName(name == null ? DEFAULT_VARIABLE_NAME_PREFIX : name);
    this.trainable = trainable;
    this.constraint = constraint;

    this.val =
        variable(getValueTensor(val), this.trainable, this.name, this.dtype);
  }

  /**
   * Get a snapshot of the Variable's value.
   *
   * The returned value is a snapshot of the Variable's value at the time of
   * the invocation. Future mutations in the value of the tensor will only
   * be reflected by future calls to this method.
   */
  read(): Tensor {
    return this.val;
  }

  /**
   * Update the value of the Variable.
   *
   * @param newVal: The new value to update to. Must be consistent with the
   *   dtype and shape of the Variable.
   * @return This Variable.
   */
  write(newVal: Tensor|ConcreteTensor) {
    // TODO(cais): Once deeplearn.js supports Tensor.dtype, check dtype match.
    checkShapesMatch(this.val, newVal);
    this.val.assign(getValueTensor(newVal));
    if (this.constraint != null) {
      this.val.assign(this.constraint.apply(this.val));
    }
    return this;
  }
}

/**
 * Type for loss a metric function.
 *
 * Takes a true value and a predicted value, and returns a loss or metric value.
 */
export type LossOrMetricFn = (yTrue: Tensor, yPred: Tensor) => Tensor;

/*
 * The type for an RNN step function.
 * The arguments are:
 *   - inputs: Input data, with shape `[sapmles, ...]`.
 * The return values are:
 *   - outputs: tensor with shape `[samples, outputDim]` (no time dimension).
 *   - newStates: Array of tensors. The `Array` has the same length as `states`
 *     in the input arguments. Each `Tensor` has the same shape as the
 *     corresponding element in `states`.
 */
export type RnnStepFunction =
    (inputs: Tensor, states: Tensor[]) => [Tensor, Tensor[]];

/**
 * Types to support JSON.
 *
 * Serialization/deserialization uses stringified-JSON as the storage
 * representation. Typically this should be used for materialized JSON
 * stored on disk/received over the wire.  Internally this is normally
 * converted to a ConfigDict that has renamed fields (TS naming conventions)
 * and support for Enums.
 */
export type JsonValue = boolean|number|string|null|JsonArray|JsonDict;
export interface JsonDict { [key: string]: JsonValue; }
export interface JsonArray extends Array<JsonValue> {}

/**
 * Types to support JSON-esque data structures internally.
 *
 * Internally ConfigDict's use camelCase keys and values where the
 * values are class names to be instantiated.  On the python side, these
 * will be snake_case.  Internally we allow Enums into the values for better
 * type safety, but these need to be converted to raw primitives (usually
 * strings) for round-tripping with python.
 *
 * toConfig returns the TS-friendly representation. model.toJSON() returns
 * the pythonic version as that's the portable format.  If you need to
 * python-ify a non-model level toConfig output, you'll need to use a
 * to-be-written-helper doing the inverse of models::convertPythonicToTs.
 *
 */
export type ConfigDictValue =
    boolean|number|string|null|ConfigDictArray|ConfigDict;
export interface ConfigDict { [key: string]: ConfigDictValue; }
export interface ConfigDictArray extends Array<ConfigDictValue> {}

export type NamedTensorMap = {
  [name: string]: Tensor;
};
