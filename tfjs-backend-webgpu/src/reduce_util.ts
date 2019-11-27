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
import { util } from '@tensorflow/tfjs-core';

export interface DataTypeMap {
  float32: Float32Array;
  int32: Int32Array;
  bool: Uint8Array;
  complex64: Float32Array;
  string: string[];
}

/** @docalias 'float32'|'int32'|'bool'|'complex64'|'string' */
export type DataType = keyof DataTypeMap;

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

export function axesAreInnerMostDims(axes: number[], rank: number): boolean {
  for (let i = 0; i < axes.length; ++i) {
    if (axes[axes.length - i - 1] !== rank - 1 - i) {
      return false;
    }
  }
  return true;
}

export function assertAxesAreInnerMostDims(
  msg: string, axes: number[], rank: number): void {
  util.assert(
    axesAreInnerMostDims(axes, rank),
    () => `${msg} supports only inner-most axes for now. ` +
      `Got axes ${axes} and rank-${rank} input.`);
}

export function computeOutAndReduceShapes(
  aShape: number[], axes: number[]): [number[], number[]] {
  const outShape = [];
  const rank = aShape.length;
  for (let dim = 0; dim < rank; dim++) {
    if (axes.indexOf(dim) === -1) {
      outShape.push(aShape[dim]);
    }
  }
  const reduceShape = axes.map(dim => aShape[dim]);
  return [outShape, reduceShape];
}

export const PARALLELIZE_THRESHOLD = 30;

export interface ReduceInfo {
  windowSize: number;
  batchSize: number;
  inSize: number;
}

export function nearestDivisor(size: number, start: number): number {
  for (let i = start; i < size; ++i) {
    if (size % i === 0) {
      return i;
    }
  }
  return size;
}

export function computeOptimalWindowSize(inSize: number): number {
  if (inSize <= PARALLELIZE_THRESHOLD) {
    return inSize;
  }
  return nearestDivisor(inSize, Math.floor(Math.sqrt(inSize)));
}
