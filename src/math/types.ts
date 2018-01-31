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

import {NDArray, Variable} from './ndarray';

export enum DType {
  float32 = 'float32',
  int32 = 'int32',
  bool = 'bool'
}

export interface ShapeMap {
  R0: number[];
  R1: [number];
  R2: [number, number];
  R3: [number, number, number];
  R4: [number, number, number, number];
}

/** @hidden */
export interface DataTypeMap {
  float32: Float32Array;
  int32: Int32Array;
  bool: Uint8Array;
}
export type DataType = keyof DataTypeMap;
export type TypedArray = DataTypeMap[DataType];

export enum Rank {
  R0 = 'R0',
  R1 = 'R1',
  R2 = 'R2',
  R3 = 'R3',
  R4 = 'R4'
}

export type FlatVector = boolean[]|number[]|TypedArray;
export type RegularArray<T> = T[]|T[][]|T[][][]|T[][][][];
export type ArrayData<D extends DataType> =
    DataTypeMap[D]|RegularArray<number>|RegularArray<boolean>;

// tslint:disable-next-line:no-any
export interface RecursiveArray<T extends any> {
  [index: number]: T|RecursiveArray<T>;
}

export type NamedArrayMap = {
  [name: string]: NDArray
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
