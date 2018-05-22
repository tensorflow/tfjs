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
import {DataType, doc, Scalar, Tensor} from '@tensorflow/tfjs-core';

import {getScopedTensorName, getUniqueTensorName} from './common';
import {Layer} from './engine/topology';
// tslint:enable:max-line-length

/** @docalias number[] */
export type Shape = number[];

/**
 * An ID to track `SymbolicTensor`s and derived classes.
 * Required in different places in engine/topology.ts to identify unique
 * tensors.
 */
let _nextUniqueTensorId = 0;

export function getNextUniqueTensorId(): number {
  return _nextUniqueTensorId++;
}

/**
 * `SymbolicTensor` is a placeholder for a Tensor without any concrete value.
 *
 * They are most often encountered when building a graph of `Layer`s for a
 * a `Model` and the input data's shape, but not values are known.
 */
@doc({heading: 'Models', 'subheading': 'Classes'})
export class SymbolicTensor {
  /* A unique ID for the tensor to be able to differentiate tensors. */
  readonly id: number;
  // The fully scoped name of this Variable, including a unique suffix if needed
  readonly name?: string;
  // The originally requested fully scoped name of this Variable, not including
  // any unique suffix.  This may be needed when restoring weights because this
  // original name is used as a key.
  readonly originalName?: string;
  /**
   * Rank/dimensionality of the tensor.
   */
  readonly rank: number;
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
      readonly dtype: DataType, readonly shape: Shape,
      public sourceLayer: Layer, readonly inputs: SymbolicTensor[],
      readonly callArgs: Kwargs, name?: string,
      readonly outputTensorIndex?: number) {
    this.id = getNextUniqueTensorId();
    if (name != null) {
      this.originalName = getScopedTensorName(name);
      this.name = getUniqueTensorName(this.originalName);
    }
    this.rank = shape.length;
  }
}

/**
 * Type for loss a metric function.
 *
 * Takes a true value and a predicted value, and returns a loss or metric value.
 */
export type LossOrMetricFn = (yTrue: Tensor, yPred: Tensor) => Tensor;

/**
 * Type for a regularizer function.
 */
export type RegularizerFn = () => Scalar;

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

export type NamedTensorMap = {
  [name: string]: Tensor;
};

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
export interface JsonDict {
  [key: string]: JsonValue;
}
export interface JsonArray extends Array<JsonValue> {}

/**
 * Type representing a loosely-typed bundle of keyword arguments.
 *
 * This is a looser type than JsonDict/serialization.ConfigDict as it
 * can contain arbitrary objects as its values.  It is most appropriate
 * for functions that pass through keyword arguments to other functions
 * without knowledge of the structure.  If the function can place type
 * restrictions on the keyword arguments, it should via the Config
 * interface convention used throughout.
 */
export type Kwargs = {
  // tslint:disable-next-line:no-any
  [key: string]: any
};
