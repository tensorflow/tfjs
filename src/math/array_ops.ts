/**
 * @license
 * Copyright 2018 Google Inc. All Rights Reserved.
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

import {ENV} from '../environment';
import * as util from '../util';
import {operation} from './decorators';
import {Array3D, NDArray, NDArrayData} from './ndarray';
import {MPRandGauss, RandNormalDataTypes} from './rand';
import {DataType, DataTypeMap, Rank, RankMap} from './types';

export class Ops {
  /** Creates a ndarray of ones with the specified shape. */
  @operation
  static ones<D extends DataType = 'float32', R extends Rank = Rank>(
      shape: number[], dtype?: D): RankMap<D>[R] {
    const values = makeOnesTypedArray(util.sizeFromShape(shape), dtype);
    return NDArray.make(shape, {values}, dtype);
  }

  /** Creates a ndarray of zeros with the specified shape. */
  @operation
  static zeros<D extends DataType = 'float32', R extends Rank = Rank>(
      shape: number[], dtype?: D): RankMap<D>[R] {
    const values = makeZerosTypedArray(util.sizeFromShape(shape), dtype);
    return NDArray.make(shape, {values}, dtype);
  }

  /**
   * Creates a ndarray of ones with the same shape as the specified ndarray.
   */
  @operation
  static onesLike<T extends NDArray>(x: T): T {
    return Ops.ones(x.shape, x.dtype) as T;
  }

  /**
   * Creates a ndarray of zeros with the same shape as the specified ndarray.
   */
  @operation
  static zerosLike<T extends NDArray>(x: T): T {
    return Ops.zeros(x.shape, x.dtype) as T;
  }

  /** Creates a ndarray with the same values/shape as the specified ndarray. */
  @operation
  static clone<T extends NDArray>(x: T): T {
    const newValues = util.copyTypedArray(x.dataSync(), x.dtype);
    return NDArray.make(x.shape, {values: newValues}, x.dtype) as T;
  }

  @operation
  static randNormal<D extends keyof RandNormalDataTypes, R extends Rank>(
      shape: number[], mean = 0, stdDev = 1, dtype?: D, seed?: number):
      RankMap<D>[R] {
    if (dtype != null && dtype === 'bool') {
      throw new Error(`Unsupported data type ${dtype}`);
    }
    const randGauss =
        new MPRandGauss(mean, stdDev, dtype, false /* truncated */, seed);
    return NDArray.rand(shape, () => randGauss.nextValue(), dtype);
  }

  @operation
  static randTruncatedNormal<D extends keyof RandNormalDataTypes,
                                       R extends Rank>(
      shape: number[], mean = 0, stdDev = 1, dtype?: D, seed?: number):
      RankMap<D>[R] {
    if (dtype != null && dtype === 'bool') {
      throw new Error(`Unsupported data type ${dtype}`);
    }
    const randGauss =
        new MPRandGauss(mean, stdDev, dtype, true /* truncated */, seed);
    return NDArray.rand(shape, () => randGauss.nextValue(), dtype);
  }

  @operation
  static randUniform<D extends DataType, R extends Rank>(
      shape: number[], a: number, b: number, dtype?: D): RankMap<D>[R] {
    return NDArray.rand(shape, () => util.randUniform(a, b), dtype);
  }

  @operation
  static rand<D extends DataType, R extends Rank>(
      shape: number[], randFunction: () => number, dtype?: D): RankMap<D>[R] {
    const size = util.sizeFromShape(shape);

    let values = null;
    if (dtype == null || dtype === 'float32') {
      values = new Float32Array(size);
    } else if (dtype === 'int32') {
      values = new Int32Array(size);
    } else if (dtype === 'bool') {
      values = new Uint8Array(size);
    } else {
      throw new Error(`Unknown data type ${dtype}`);
    }

    for (let i = 0; i < size; i++) {
      values[i] = randFunction();
    }
    return NDArray.make(shape, {values}, dtype);
  }

  @operation
  static fromPixels(
      pixels: ImageData|HTMLImageElement|HTMLCanvasElement|HTMLVideoElement,
      numChannels = 3): Array3D<'int32'> {
    if (numChannels > 4) {
      throw new Error(
          'Cannot construct NDArray with more than 4 channels from pixels.');
    }
    const ndarrayData: NDArrayData<'int32'> = {};
    const shape: [number, number, number] =
        [pixels.height, pixels.width, numChannels];
    const res = NDArray.make(shape, ndarrayData, 'int32') as Array3D<'int32'>;
    ENV.math.writePixels(res.dataId, pixels, numChannels);
    return res;
  }
}

function makeZerosTypedArray<D extends DataType>(
    size: number, dtype: D): DataTypeMap[D] {
  if (dtype == null || dtype === 'float32') {
    return new Float32Array(size);
  } else if (dtype === 'int32') {
    return new Int32Array(size);
  } else if (dtype === 'bool') {
    return new Uint8Array(size);
  } else {
    throw new Error(`Unknown data type ${dtype}`);
  }
}

function makeOnesTypedArray<D extends DataType>(
    size: number, dtype: D): DataTypeMap[D] {
  const array = makeZerosTypedArray(size, dtype);
  for (let i = 0; i < array.length; i++) {
    array[i] = 1;
  }
  return array;
}
