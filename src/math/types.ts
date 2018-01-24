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

import {Array1D, Array2D, Array3D, Array4D, NDArray, Scalar} from './ndarray';

export enum DType {
  float32 = 'float32',
  int32 = 'int32',
  bool = 'bool'
}

export interface ShapeMap {
  0: number[];
  1: [number];
  2: [number, number];
  3: [number, number, number];
  4: [number, number, number, number];
  higher: number[];
}

/** @hidden */
export interface DataTypeMap {
  float32: Float32Array;
  int32: Int32Array;
  bool: Uint8Array;
}
export type DataType = keyof DataTypeMap;

/** @hidden */
export interface RankMap<D extends DataType> {
  0: Scalar<D>;
  1: Array1D<D>;
  2: Array2D<D>;
  3: Array3D<D>;
  4: Array4D<D>;
  higher: NDArray<D, 'higher'>;
}
export type Rank = keyof RankMap<DataType>;

export interface SumTypes {
  float32: 'float32';
  int32: 'int32';
  bool: 'int32';
}

export enum SumTypesMap {
  float32 = 'float32',
  int32 = 'int32',
  bool = 'int32'
}

export interface UpcastInt32And {
  float32: 'float32';
  int32: 'int32';
  bool: 'int32';
}

export enum UpcastInt32AndMap {
  float32 = 'float32',
  int32 = 'int32',
  bool = 'int32'
}

export interface UpcastBoolAnd {
  float32: 'float32';
  int32: 'int32';
  bool: 'bool';
}

export enum UpcastBoolAndMap {
  float32 = 'float32',
  int32 = 'int32',
  bool = 'bool'
}

export interface UpcastFloat32And {
  float32: 'float32';
  int32: 'float32';
  bool: 'float32';
}

export enum UpcastFloat32AndMap {
  float32 = 'float32',
  int32 = 'float32',
  bool = 'float32'
}

export interface UpcastType {
  float32: UpcastFloat32And;
  int32: UpcastInt32And;
  bool: UpcastBoolAnd;
}

const upcastTypeMap = {
  float32: UpcastFloat32AndMap,
  int32: UpcastInt32AndMap,
  bool: UpcastBoolAndMap
};

export function upcastType(typeA: DataType, typeB: DataType): DataType {
  return upcastTypeMap[typeA][typeB];
}
