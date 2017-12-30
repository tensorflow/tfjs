import {DataType} from './ndarray';

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
