/**
 * @license
 * Copyright 2017 Google Inc. All Rights Reserved.
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

import {Tensor, Variable} from './tensor';

export enum DType {
  float32 = 'float32',
  int32 = 'int32',
  bool = 'bool'
}

/** @docalias number[] */
export interface ShapeMap {
  R0: number[];
  R1: [number];
  R2: [number, number];
  R3: [number, number, number];
  R4: [number, number, number, number];
  R5: [number, number, number, number, number];
  R6: [number, number, number, number, number, number];
}

/** @hidden */
export interface DataTypeMap {
  float32: Float32Array;
  int32: Int32Array;
  bool: Uint8Array;
}
/** @docalias 'float32'|'int32'|'bool' */
export type DataType = keyof DataTypeMap;
export type TypedArray = DataTypeMap[DataType];

export enum Rank {
  R0 = 'R0',
  R1 = 'R1',
  R2 = 'R2',
  R3 = 'R3',
  R4 = 'R4',
  R5 = 'R5',
  R6 = 'R6'
}

/** @docalias TypedArray|Array */
export type TensorLike =
    TypedArray|number|boolean|number[]|number[][]|number[][][]|number[][][][]|
    number[][][][][]|number[][][][][][]|boolean[]|boolean[][]|boolean[][][]|
    boolean[][][][]|boolean[][][][][]|boolean[][][][][][];
/** @docalias TypedArray|Array */
export type TensorLike1D = TypedArray|number[]|boolean[];
/** @docalias TypedArray|Array */
export type TensorLike2D = TypedArray|number[]|number[][]|boolean[]|boolean[][];
/** @docalias TypedArray|Array */
export type TensorLike3D =
    TypedArray|number[]|number[][][]|boolean[]|boolean[][][];
/** @docalias TypedArray|Array */
export type TensorLike4D =
    TypedArray|number[]|number[][][][]|boolean[]|boolean[][][][];
/** @docalias TypedArray|Array */
export type TensorLike5D =
    TypedArray|number[]|number[][][][][]|boolean[]|boolean[][][][][];
/** @docalias TypedArray|Array */
export type TensorLike6D =
    TypedArray|number[]|number[][][][][][]|boolean[]|boolean[][][][][][];

export type FlatVector = boolean[]|number[]|TypedArray;
export type RegularArray<T> =
    T[]|T[][]|T[][][]|T[][][][]|T[][][][][]|T[][][][][][];
export type ArrayData<D extends DataType> =
    DataTypeMap[D]|RegularArray<number>|RegularArray<boolean>;

// tslint:disable-next-line:no-any
export interface RecursiveArray<T extends any> {
  [index: number]: T|RecursiveArray<T>;
}

/** @docalias {[name: string]: Tensor} */
export type NamedTensorMap = {
  [name: string]: Tensor
};

export type NamedVariableMap = {
  [name: string]: Variable;
};

enum UpcastInt32AndMap {
  float32 = 'float32',
  int32 = 'int32',
  bool = 'int32'
}

enum UpcastBoolAndMap {
  float32 = 'float32',
  int32 = 'int32',
  bool = 'bool'
}

enum UpcastFloat32AndMap {
  float32 = 'float32',
  int32 = 'float32',
  bool = 'float32'
}

const upcastTypeMap = {
  float32: UpcastFloat32AndMap,
  int32: UpcastInt32AndMap,
  bool: UpcastBoolAndMap
};

export function upcastType(typeA: DataType, typeB: DataType): DataType {
  return upcastTypeMap[typeA][typeB];
}

/** Returns the output type after summation. */
export function sumOutType(type: DataType) {
  return upcastType(type, 'int32');
}

/**
 * @docalias void|number|string|Tensor|Tensor[]|{[key:
 * string]:Tensor|number|string}
 */
export type TensorContainer = void|Tensor|string|number|boolean|
    TensorContainerObject|TensorContainerArray;
export interface TensorContainerObject { [x: string]: TensorContainer; }
export interface TensorContainerArray extends Array<TensorContainer> {}

export interface ModelPredictConfig {
  /**
   * Optional. Batch size (Integer). If unspecified, it will default to 32.
   */
  batchSize?: number;

  /**
   * Optional. Verbosity mode. Defaults to false.
   */
  verbose?: boolean;
}

export interface TensorInfo {
  // Name of the tensor.
  name: string;
  // Tensor shape information, Optional.
  shape?: number[];
  // Data type of the tensor.
  dtype: DataType;
}

/**
 * Common interface for a machine learning model that can do inference.
 */
export interface InferenceModel {
  /**
   * Return the array of input tensor info.
   */
  readonly inputs: TensorInfo[];

  /**
   * Return the array of output tensor info.
   */
  readonly outputs: TensorInfo[];

  /**
   * Execute the inference for the input tensors.
   *
   * @param input The input tensors, when there is single input for the model,
   * inputs param should be a Tensor. For models with mutliple inputs, inputs
   * params should be in either Tensor[] if the input order is fixed, or
   * otherwise NamedTensorMap format.
   * For batch inference execution, the tensors for each input need to be
   * concatenated together. For example with mobilenet, the required input shape
   * is [1, 244, 244, 3], which represents the [batch, height, width, channel].
   * If we are provide a batched data of 100 images, the input tensor should be
   * in the shape of [100, 244, 244, 3].
   *
   * @param config Prediction configuration for specifying the batch size.
   *
   * @returns Inference result tensors. The output would be single Tensor if
   * model has single output node, otherwise Tensor[] or NamedTensorMap[] will
   * be returned for model with multiple outputs.
   */
  predict(inputs: Tensor|Tensor[]|NamedTensorMap, config: ModelPredictConfig):
      Tensor|Tensor[]|NamedTensorMap;

  /**
   * Single Execute the inference for the input tensors and return activation
   * values for specified output node names without batching.
   *
   * @param input The input tensors, when there is single input for the model,
   * inputs param should be a Tensor. For models with mutliple inputs, inputs
   * params should be in either Tensor[] if the input order is fixed, or
   * otherwise NamedTensorMap format.
   *
   * @param outputs string|string[]. List of output node names to retrieve
   * activation from.
   *
   * @returns Activation values for the output nodes result tensors. The return
   * type matches specified parameter outputs type. The output would be single
   * Tensor if single output is specified, otherwise Tensor[] for multiple
   * outputs.
   */
  execute(inputs: Tensor|Tensor[]|NamedTensorMap, outputs: string|string[]):
      Tensor|Tensor[];
}
