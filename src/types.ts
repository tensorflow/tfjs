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
import {Scalar, Tensor} from '@tensorflow/tfjs-core';
// tslint:enable:max-line-length

/** @docalias number[] */
export type Shape = number[];

export type HasShape = {
  shape: Shape;
};

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
