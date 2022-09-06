/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
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

/** @docalias number[] */
export interface ArrayMap {
  R0: number;
  R1: number[];
  R2: number[][];
  R3: number[][][];
  R4: number[][][][];
  R5: number[][][][][];
  R6: number[][][][][][];
}

export interface DataTypeMap {
  float32: Float32Array;
  int32: Int32Array;
  bool: Uint8Array;
  complex64: Float32Array;
  string: string[];
}

export interface SingleValueMap {
  bool: boolean;
  int32: number;
  float32: number;
  complex64: number;
  string: string;
}

/** @docalias 'float32'|'int32'|'bool'|'complex64'|'string' */
export type DataType = keyof DataTypeMap;
export type NumericDataType = 'float32'|'int32'|'bool'|'complex64';
export type TypedArray = Float32Array|Int32Array|Uint8Array;
/** Tensor data used in tensor creation and user-facing API. */
export type DataValues = DataTypeMap[DataType];
/** The underlying tensor data that gets stored in a backend. */
export type BackendValues = Float32Array|Int32Array|Uint8Array|Uint8Array[];

export enum Rank {
  R0 = 'R0',
  R1 = 'R1',
  R2 = 'R2',
  R3 = 'R3',
  R4 = 'R4',
  R5 = 'R5',
  R6 = 'R6'
}

export type FlatVector = boolean[]|number[]|TypedArray;
export type RegularArray<T> =
    T[]|T[][]|T[][][]|T[][][][]|T[][][][][]|T[][][][][][];

// tslint:disable-next-line:no-any
export interface RecursiveArray<T extends any> {
  [index: number]: T|RecursiveArray<T>;
}

// Looks for upcasting types. Used, for example, in operations with mixed dtype
// inputs.
enum UpcastInt32AndMap {
  'float32' = 'float32',
  'int32' = 'int32',
  'bool' = 'int32',
  'complex64' = 'complex64'
}

enum UpcastBoolAndMap {
  'float32' = 'float32',
  'int32' = 'int32',
  'bool' = 'bool',
  'complex64' = 'complex64'
}

enum UpcastFloat32AndMap {
  'float32' = 'float32',
  'int32' = 'float32',
  'bool' = 'float32',
  'complex64' = 'complex64'
}

enum UpcastComplex64AndMap {
  'float32' = 'complex64',
  'int32' = 'complex64',
  'bool' = 'complex64',
  'complex64' = 'complex64'
}

const upcastTypeMap = {
  'float32': UpcastFloat32AndMap,
  'int32': UpcastInt32AndMap,
  'bool': UpcastBoolAndMap,
  'complex64': UpcastComplex64AndMap
};

export function upcastType(typeA: DataType, typeB: DataType): DataType {
  if (typeA === 'string' || typeB === 'string') {
    if (typeA === 'string' && typeB === 'string') {
      return 'string';
    }
    throw new Error(`Can not upcast ${typeA} with ${typeB}`);
  }
  return upcastTypeMap[typeA][typeB];
}

/** Returns the output type after summation. */
export function sumOutType(type: DataType): DataType {
  return upcastType(type, 'int32');
}

/** @docalias TypedArray|Array */
export type TensorLike =
    TypedArray|number|boolean|string|RecursiveArray<number|number[]|TypedArray>|
    RecursiveArray<boolean>|RecursiveArray<string>|Uint8Array[];
export type ScalarLike = number|boolean|string|Uint8Array;
/** @docalias TypedArray|Array */
export type TensorLike1D = TypedArray|number[]|boolean[]|string[]|Uint8Array[];
/** @docalias TypedArray|Array */
export type TensorLike2D = TypedArray|number[]|number[][]|boolean[]|boolean[][]|
    string[]|string[][]|Uint8Array[]|Uint8Array[][];
/** @docalias TypedArray|Array */
export type TensorLike3D = TypedArray|number[]|number[][][]|boolean[]|
    boolean[][][]|string[]|string[][][]|Uint8Array[]|Uint8Array[][][];
/** @docalias TypedArray|Array */
export type TensorLike4D = TypedArray|number[]|number[][][][]|boolean[]|
    boolean[][][][]|string[]|string[][][][]|Uint8Array[]|Uint8Array[][][][];
/** @docalias TypedArray|Array */
export type TensorLike5D =
    TypedArray|number[]|number[][][][][]|boolean[]|boolean[][][][][]|string[]|
    string[][][][][]|Uint8Array[]|Uint8Array[][][][][];
/** @docalias TypedArray|Array */
export type TensorLike6D =
    TypedArray|number[]|number[][][][][][]|boolean[]|boolean[][][][][][]|
    string[]|string[][][][][][]|Uint8Array[]|Uint8Array[][][][][];

/** Type for representing image data in Uint8Array type. */
export interface PixelData {
  width: number;
  height: number;
  data: Uint8Array;
}
