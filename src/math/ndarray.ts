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

import {ENV} from '../environment';
import * as util from '../util';
import {ArrayData} from '../util';
import {NDArrayMath} from './math';
import {RandNormalDataTypes} from './rand';
import {MPRandGauss} from './rand';

export enum DType {
  float32 = 'float32',
  int32 = 'int32',
  bool = 'bool'
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

/** @hidden */
export interface NDArrayData<D extends DataType> {
  dataId?: number;
  values?: DataTypeMap[D];
}

export interface ShapeMap {
  0: number[];
  1: [number];
  2: [number, number];
  3: [number, number, number];
  4: [number, number, number, number];
  higher: number[];
}

export class NDArray<D extends DataType = DataType, R extends Rank = Rank> {
  private static nextId = 0;
  private static nextDataId = 0;

  /** Unique id of this ndarray. */
  id: number;
  /**
   * Id of the bucket holding the data for this ndarray. Multiple arrays can
   * point to the same bucket (e.g. when calling array.reshape()).
   */
  dataId: number;
  /** The shape of the ndarray. */
  shape: ShapeMap[R];
  /** Number of elements in the ndarray. */
  size: number;
  /** The data type for the array. */
  dtype: D;
  /** The rank type for the array ('0','1','2','3','4','higher'). */
  rankType: R;

  /**
   * Number of elements to skip in each dimension when indexing. See
   * https://docs.scipy.org/doc/numpy/reference/generated
   *     /numpy.ndarray.strides.html
   */
  strides: number[];

  protected math: NDArrayMath;

  protected constructor(
      shape: number[], dtype: D, values?: DataTypeMap[D], dataId?: number,
      math?: NDArrayMath) {
    this.math = math || ENV.math;
    this.size = util.sizeFromShape(shape);
    if (values != null) {
      util.assert(
          this.size === values.length,
          `Constructing ndarray of shape (${this.size}) should match the ` +
              `length of values (${values.length})`);
    }
    this.shape = shape;
    this.dtype = dtype || ('float32' as D);
    const dim = this.shape.length;

    if (dim < 2) {
      this.strides = [];
    } else {
      // Last dimension has implicit stride of 1, thus having D-1 (instead of D)
      // strides.
      this.strides = new Array(dim - 1);
      this.strides[dim - 2] = this.shape[dim - 1];
      for (let i = dim - 3; i >= 0; --i) {
        this.strides[i] = this.strides[i + 1] * this.shape[i + 1];
      }
    }
    this.dataId = dataId != null ? dataId : NDArray.nextDataId++;
    this.id = NDArray.nextId++;
    this.rankType = (this.rank < 5 ? this.rank.toString() : 'higher') as R;
    this.math.register(this);
    if (values != null) {
      this.math.write(this.dataId, values);
    }
  }

  /** Creates a ndarray of ones with the specified shape. */
  static ones<D extends DataType = DataType, R extends Rank = Rank>(
      shape: number[], dtype?: D): RankMap<D>[R] {
    const values = makeOnesTypedArray(util.sizeFromShape(shape), dtype);
    return NDArray.make(shape, {values}, dtype);
  }

  /** Creates a ndarray of zeros with the specified shape. */
  static zeros<D extends DataType = DataType, R extends Rank = Rank>(
      shape: number[], dtype?: D): RankMap<D>[R] {
    const values = makeZerosTypedArray(util.sizeFromShape(shape), dtype);
    return NDArray.make(shape, {values}, dtype);
  }

  /**
   * Creates a ndarray of ones with the same shape as the specified ndarray.
   */
  static onesLike<T extends NDArray>(another: T): T {
    return NDArray.ones(another.shape, another.dtype) as T;
  }

  /**
   * Creates a ndarray of zeros with the same shape as the specified ndarray.
   */
  static zerosLike<T extends NDArray>(another: T): T {
    return NDArray.zeros(another.shape, another.dtype) as T;
  }

  /** Creates a ndarray with the same values/shape as the specified ndarray. */
  static like<T extends NDArray>(another: T): T {
    const newValues = copyTypedArray(another.dataSync(), another.dtype);
    return NDArray.make(
               another.shape, {values: newValues}, another.dtype,
               another.math) as T;
  }

  /**
   * Makes a new ndarray with the provided shape and values. Values should be in
   * a flat array.
   */
  static make<D extends DataType = 'float32', R extends Rank = Rank>(
      shape: number[], data: NDArrayData<D>, dtype?: D,
      math?: NDArrayMath): RankMap<D>[R] {
    switch (shape.length) {
      case 0:
        return new Scalar(shape, dtype, data.values, data.dataId, math);
      case 1:
        return new Array1D(shape, dtype, data.values, data.dataId, math);
      case 2:
        return new Array2D(
            shape as [number, number], dtype, data.values, data.dataId, math);
      case 3:
        return new Array3D(
            shape as [number, number, number], dtype, data.values, data.dataId,
            math);
      case 4:
        return new Array4D(
            shape as [number, number, number, number], dtype, data.values,
            data.dataId, math);
      default:
        return new NDArray(shape, dtype, data.values, data.dataId, math) as
            RankMap<D>[R];
    }
  }

  static fromPixels(
      pixels: ImageData|HTMLImageElement|HTMLCanvasElement|HTMLVideoElement,
      numChannels = 3, math?: NDArrayMath): Array3D<'int32'> {
    if (numChannels > 4) {
      throw new Error(
          'Cannot construct NDArray with more than 4 channels from pixels.');
    }
    const ndarrayData: NDArrayData<'int32'> = {};
    const shape: [number, number, number] =
        [pixels.height, pixels.width, numChannels];
    math = math || ENV.math;
    const res =
        NDArray.make(shape, ndarrayData, 'int32', math) as Array3D<'int32'>;
    math.writePixels(res.dataId, pixels, numChannels);
    return res;
  }

  /** Reshapes the current ndarray into the provided shape. */
  reshape<R2 extends Rank>(newShape: number[]): RankMap<D>[R2] {
    this.throwIfDisposed();
    return this.math.reshape(this, newShape);
  }

  /**
   * @param axis An optional list of number. If specified, only
   * squeezes the dimensions listed. The dimension index starts at 0. It is an
   * error to squeeze a dimension that is not 1.
   */
  squeeze<T extends NDArray<D>>(axis?: number[]): T {
    return this.reshape(util.squeezeShape(this.shape, axis).newShape) as T;
  }

  /** Flatten a NDArray to a 1D array. */
  flatten(): Array1D<D> {
    this.throwIfDisposed();
    if (this instanceof Array1D) {
      return this;
    }
    return this.as1D();
  }

  asScalar(): Scalar<D> {
    this.throwIfDisposed();
    util.assert(this.size === 1, 'The array must have only 1 element.');
    return this.reshape<'0'>([]);
  }

  as1D(): Array1D<D> {
    this.throwIfDisposed();
    return this.reshape<'1'>([this.size]);
  }

  as2D(rows: number, columns: number): Array2D<D> {
    this.throwIfDisposed();
    return this.reshape<'2'>([rows, columns]);
  }

  as3D(rows: number, columns: number, depth: number): Array3D<D> {
    this.throwIfDisposed();
    return this.reshape<'3'>([rows, columns, depth]);
  }

  as4D(rows: number, columns: number, depth: number, depth2: number):
      Array4D<D> {
    this.throwIfDisposed();
    return this.reshape<'4'>([rows, columns, depth, depth2]);
  }

  asType<D2 extends DataType>(dtype: D2): NDArray<D2, R> {
    this.throwIfDisposed();
    return this.math.cast(this, dtype) as NDArray<D2, R>;
  }

  get rank(): number {
    return this.shape.length;
  }

  get(...locs: number[]) {
    let index = locs[locs.length - 1];
    for (let i = 0; i < locs.length - 1; ++i) {
      index += this.strides[i] * locs[i];
    }
    return this.dataSync()[index];
  }

  add(value: number, ...locs: number[]) {
    this.set(this.get(...locs) + value, ...locs);
  }

  set(value: number, ...locs: number[]) {
    this.throwIfDisposed();
    util.assert(
        locs.length === this.rank,
        `The number of provided coordinates (${locs.length}) must ` +
            `match the rank (${this.rank})`);
    let index = locs.length > 0 ? locs[locs.length - 1] : 0;
    for (let i = 0; i < locs.length - 1; ++i) {
      index += this.strides[i] * locs[i];
    }
    const vals = this.dataSync();
    vals[index] = value;
    this.math.write(this.dataId, vals);
  }

  async val(...locs: number[]): Promise<number> {
    this.throwIfDisposed();
    await this.data();
    return this.get(...locs);
  }

  locToIndex(locs: ShapeMap[R]): number {
    this.throwIfDisposed();
    if (locs.length === 0) {
      return 0;
    }
    let index = locs[locs.length - 1];
    for (let i = 0; i < locs.length - 1; ++i) {
      index += this.strides[i] * locs[i];
    }
    return index;
  }

  indexToLoc(index: number): ShapeMap[R] {
    this.throwIfDisposed();
    const locs: number[] = new Array(this.shape.length);
    for (let i = 0; i < locs.length - 1; ++i) {
      locs[i] = Math.floor(index / this.strides[i]);
      index -= locs[i] * this.strides[i];
    }
    locs[locs.length - 1] = index;
    return locs;
  }

  fill(value: number) {
    this.throwIfDisposed();
    const vals = this.dataSync();
    vals.fill(value);
    this.math.write(this.dataId, vals);
  }

  /** @deprecated Use dataSync() instead. */
  getValues(): DataTypeMap[D] {
    return this.dataSync();
  }

  /** @deprecated Use data() instead. */
  getValuesAsync(): Promise<DataTypeMap[D]> {
    return this.data();
  }

  /**
   * Asynchronously downloads the values from the NDArray. Returns a promise
   * that resolves when the data is ready.
   */
  async data(): Promise<DataTypeMap[D]> {
    this.throwIfDisposed();
    return this.math.read(this.dataId);
  }

  /**
   * Synchronously downloads the values from the NDArray. This blocks the UI
   * thread until the values are ready, which can cause performance issues.
   */
  dataSync(): DataTypeMap[D] {
    this.throwIfDisposed();
    return this.math.readSync(this.dataId);
  }

  dispose(): void {
    if (this.isDisposed) {
      return;
    }
    this.isDisposed = true;
    this.math.disposeData(this.dataId);
  }

  equals(t: NDArray<D, R>): boolean {
    this.throwIfDisposed();
    return this.dtype === t.dtype && util.arraysEqual(this.shape, t.shape) &&
        util.arraysEqual(this.dataSync(), t.dataSync());
  }

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

  static randNormal<D extends keyof RandNormalDataTypes, R extends Rank>(
      shape: number[], mean = 0, stdDev = 1, dtype?: D,
      seed?: number): RankMap<D>[R] {
    if (dtype != null && dtype === 'bool') {
      throw new Error(`Unsupported data type ${dtype}`);
    }
    const randGauss =
        new MPRandGauss(mean, stdDev, dtype, false /* truncated */, seed);
    return NDArray.rand(shape, () => randGauss.nextValue(), dtype);
  }

  static randTruncatedNormal<D extends keyof RandNormalDataTypes,
                                       R extends Rank>(
      shape: number[], mean = 0, stdDev = 1, dtype?: D,
      seed?: number): RankMap<D>[R] {
    if (dtype != null && dtype === 'bool') {
      throw new Error(`Unsupported data type ${dtype}`);
    }
    const randGauss =
        new MPRandGauss(mean, stdDev, dtype, true /* truncated */, seed);
    return NDArray.rand(shape, () => randGauss.nextValue(), dtype);
  }

  static randUniform<D extends DataType, R extends Rank>(
      shape: number[], a: number, b: number, dtype?: D): RankMap<D>[R] {
    return NDArray.rand(shape, () => util.randUniform(a, b), dtype);
  }

  private isDisposed = false;
  private throwIfDisposed() {
    if (this.isDisposed) {
      throw new Error(`NDArray is disposed.`);
    }
  }
}

export class Scalar<D extends DataType = DataType> extends NDArray<D, '0'> {
  static new<D extends DataType = 'float32'>(value: number|boolean, dtype?: D):
      Scalar<D> {
    const values = [value] as number[] | boolean[];
    return new Scalar([], dtype, toTypedArray(values, dtype));
  }

  get(): number {
    return this.dataSync()[0];
  }

  async val(): Promise<number> {
    await this.data();
    return this.get();
  }

  add(value: number) {
    this.dataSync()[0] += value;
  }

  asType<D2 extends DataType>(dtype: D2): Scalar<D2> {
    return super.asType(dtype);
  }

  locToIndex(loc: number[]): number {
    return 0;
  }

  indexToLoc(index: number): number[] {
    return [];
  }
}

export class Array1D<D extends DataType = DataType> extends NDArray<D, '1'> {
  static new<D extends DataType = 'float32'>(
      values: DataTypeMap[D]|number[]|boolean[], dtype?: D): Array1D<D> {
    if (!instanceofTypedArray(values)) {
      const inferredShape = util.inferShape(values as number[] | boolean[]);
      util.assert(
          inferredShape.length === 1,
          `Error constructing Array1D. Shape of values ${inferredShape} is ` +
              `not 1 dimensional.`);
    }
    return new Array1D([values.length], dtype, toTypedArray(values, dtype));
  }

  get(i: number): number {
    return this.dataSync()[i];
  }

  async val(i: number): Promise<number> {
    await this.data();
    return this.get(i);
  }

  add(value: number, i: number) {
    this.dataSync()[i] += value;
  }

  locToIndex(loc: [number]): number {
    return loc[0];
  }

  indexToLoc(index: number): [number] {
    return [index];
  }

  asType<D2 extends DataType>(dtype: D2): Array1D<D2> {
    return super.asType(dtype) as Array1D<D2>;
  }

  static ones<D extends DataType = DataType>(shape: [number], dtype?: D):
      Array1D<D> {
    return NDArray.ones<D, '1'>(shape, dtype);
  }

  static zeros<D extends DataType = DataType>(shape: [number], dtype?: D):
      Array1D<D> {
    return NDArray.zeros<D, '1'>(shape, dtype);
  }

  static randNormal<D extends keyof RandNormalDataTypes>(
      shape: [number], mean = 0, stdDev = 1, dtype?: D,
      seed?: number): Array1D<D> {
    if (dtype != null && dtype === 'bool') {
      throw new Error(`Unsupported data type ${dtype}`);
    }
    const randGauss =
        new MPRandGauss(mean, stdDev, dtype, false /* truncated */, seed);
    return NDArray.rand(shape, () => randGauss.nextValue(), dtype) as
        Array1D<D>;
  }

  static randTruncatedNormal<D extends keyof RandNormalDataTypes>(
      shape: [number], mean = 0, stdDev = 1, dtype?: D,
      seed?: number): Array1D<D> {
    if (dtype != null && dtype === 'bool') {
      throw new Error(`Unsupported data type ${dtype}`);
    }
    const randGauss =
        new MPRandGauss(mean, stdDev, dtype, true /* truncated */, seed);
    return NDArray.rand(shape, () => randGauss.nextValue(), dtype) as
        Array1D<D>;
  }

  static randUniform<D extends DataType>(
      shape: [number], a: number, b: number, dtype?: D): Array1D<D> {
    return NDArray.rand(shape, () => util.randUniform(a, b), dtype) as
        Array1D<D>;
  }
}

export class Array2D<D extends DataType = DataType> extends NDArray<D, '2'> {
  constructor(
      shape: [number, number], dtype: D, values?: DataTypeMap[D],
      dataId?: number, math?: NDArrayMath) {
    util.assert(shape.length === 2, 'Shape should be of length 2');
    super(shape, dtype, values, dataId, math);
  }

  static new<D extends DataType = 'float32'>(
      shape: [number, number],
      values: DataTypeMap[D]|number[]|number[][]|boolean[]|boolean[][],
      dtype?: D): Array2D<D> {
    if (!instanceofTypedArray(values)) {
      const inferredShape = util.inferShape(values as number[] | boolean[]);
      if (inferredShape.length > 1) {
        util.assertShapesMatch(
            shape, inferredShape,
            `Error when constructing Array2D. Shape of values ` +
                `${inferredShape} does not match the provided shape ` +
                `${shape}. `);
      }
    }
    return new Array2D(shape, dtype, toTypedArray(values, dtype));
  }

  get(i: number, j: number) {
    return this.dataSync()[this.strides[0] * i + j];
  }

  add(value: number, i: number, j: number) {
    this.dataSync()[this.strides[0] * i + j] += value;
  }

  async val(i: number, j: number): Promise<number> {
    await this.data();
    return this.get(i, j);
  }

  locToIndex(locs: [number, number]): number {
    return this.strides[0] * locs[0] + locs[1];
  }

  indexToLoc(index: number): [number, number] {
    return [Math.floor(index / this.strides[0]), index % this.strides[0]];
  }

  asType<D2 extends DataType>(dtype: D2): Array2D<D2> {
    return super.asType(dtype) as Array2D<D2>;
  }

  static ones<D extends DataType = DataType>(
      shape: [number, number], dtype?: D): Array2D<D> {
    return NDArray.ones<D, '2'>(shape, dtype);
  }

  static zeros<D extends DataType = DataType>(
      shape: [number, number], dtype?: D): Array2D<D> {
    return NDArray.zeros<D, '2'>(shape, dtype);
  }

  static randNormal<D extends keyof RandNormalDataTypes>(
      shape: [number, number], mean = 0, stdDev = 1, dtype?: D,
      seed?: number): Array2D<D> {
    if (dtype != null && dtype === 'bool') {
      throw new Error(`Unsupported data type ${dtype}`);
    }
    const randGauss =
        new MPRandGauss(mean, stdDev, dtype, false /* truncated */, seed);
    return NDArray.rand(shape, () => randGauss.nextValue(), dtype) as
        Array2D<D>;
  }

  static randTruncatedNormal<D extends keyof RandNormalDataTypes>(
      shape: [number, number], mean = 0, stdDev = 1, dtype?: D,
      seed?: number): Array2D<D> {
    if (dtype != null && dtype === 'bool') {
      throw new Error(`Unsupported data type ${dtype}`);
    }
    const randGauss =
        new MPRandGauss(mean, stdDev, dtype, true /* truncated */, seed);
    return NDArray.rand(shape, () => randGauss.nextValue(), dtype) as
        Array2D<D>;
  }

  static randUniform<D extends DataType>(
      shape: [number, number], a: number, b: number, dtype?: D): Array2D<D> {
    return NDArray.rand(shape, () => util.randUniform(a, b), dtype) as
        Array2D<D>;
  }
}

export class Array3D<D extends DataType = DataType> extends NDArray<D, '3'> {
  constructor(
      shape: [number, number, number], dtype: D, values?: DataTypeMap[D],
      dataId?: number, math?: NDArrayMath) {
    util.assert(shape.length === 3, 'Shape should be of length 3');
    super(shape, dtype, values, dataId, math);
  }

  static new<D extends DataType = 'float32'>(
      shape: [number, number, number],
      values: DataTypeMap[D]|number[]|number[][][]|boolean[]|boolean[][][],
      dtype?: D): Array3D<D> {
    if (!instanceofTypedArray(values)) {
      const inferredShape = util.inferShape(values as number[] | boolean[]);
      if (inferredShape.length > 1) {
        util.assertShapesMatch(
            shape, inferredShape,
            `Error when constructing Array3D. Shape of values ` +
                `${inferredShape} does not match the provided shape ` +
                `${shape}. `);
      }
    }
    return new Array3D(shape, dtype, toTypedArray(values, dtype));
  }

  get(i: number, j: number, k: number) {
    return this.dataSync()[this.strides[0] * i + this.strides[1] * j + k];
  }

  async val(i: number, j: number, k: number): Promise<number> {
    await this.data();
    return this.get(i, j, k);
  }

  add(value: number, i: number, j: number, k: number) {
    this.dataSync()[this.strides[0] * i + this.strides[1] * j + k] += value;
  }

  locToIndex(locs: [number, number, number]): number {
    return this.strides[0] * locs[0] + this.strides[1] * locs[1] + locs[2];
  }

  indexToLoc(index: number): [number, number, number] {
    const i = Math.floor(index / this.strides[0]);
    index -= i * this.strides[0];
    return [i, Math.floor(index / this.strides[1]), index % this.strides[1]];
  }
  static ones<D extends DataType = DataType>(
      shape: [number, number, number], dtype?: D): Array3D<D> {
    return NDArray.ones<D, '3'>(shape, dtype);
  }

  asType<D2 extends DataType>(dtype: D2): Array3D<D2> {
    return super.asType(dtype) as Array3D<D2>;
  }

  static zeros<D extends DataType = DataType>(
      shape: [number, number, number], dtype?: D): Array3D<D> {
    return NDArray.zeros<D, '3'>(shape, dtype);
  }

  static randNormal<D extends keyof RandNormalDataTypes>(
      shape: [number, number, number], mean = 0, stdDev = 1, dtype?: D,
      seed?: number): Array3D<D> {
    if (dtype != null && dtype === 'bool') {
      throw new Error(`Unsupported data type ${dtype}`);
    }
    const randGauss =
        new MPRandGauss(mean, stdDev, dtype, false /* truncated */, seed);
    return NDArray.rand(shape, () => randGauss.nextValue(), dtype) as
        Array3D<D>;
  }

  static randTruncatedNormal<D extends keyof RandNormalDataTypes>(
      shape: [number, number, number], mean = 0, stdDev = 1, dtype?: D,
      seed?: number): Array3D<D> {
    if (dtype != null && dtype === 'bool') {
      throw new Error(`Unsupported data type ${dtype}`);
    }
    const randGauss =
        new MPRandGauss(mean, stdDev, dtype, true /* truncated */, seed);
    return NDArray.rand(shape, () => randGauss.nextValue(), dtype) as
        Array3D<D>;
  }

  static randUniform<D extends DataType>(
      shape: [number, number, number], a: number, b: number,
      dtype?: D): Array3D<D> {
    return NDArray.rand(shape, () => util.randUniform(a, b), dtype) as
        Array3D<D>;
  }
}

export class Array4D<D extends DataType = DataType> extends NDArray<D, '4'> {
  constructor(
      shape: [number, number, number, number], dtype: D,
      values?: DataTypeMap[D], dataId?: number, math?: NDArrayMath) {
    util.assert(shape.length === 4, 'Shape should be of length 4');
    super(shape, dtype, values, dataId, math);
  }

  static new<D extends DataType = 'float32'>(
      shape: [number, number, number, number],
      values: DataTypeMap[D]|number[]|number[][][][]|boolean[]|boolean[][][][],
      dtype?: D): Array4D<D> {
    if (!instanceofTypedArray(values)) {
      const inferredShape = util.inferShape(values as number[] | boolean[]);
      if (inferredShape.length > 1) {
        util.assertShapesMatch(
            shape, inferredShape,
            `Error when constructing Array4D. Shape of values ` +
                `${inferredShape} does not match the provided shape ` +
                `${shape}. `);
      }
    }
    return new Array4D(shape, dtype, toTypedArray(values, dtype));
  }

  get(i: number, j: number, k: number, l: number) {
    return this.dataSync()
        [this.strides[0] * i + this.strides[1] * j + this.strides[2] * k + l];
  }

  async val(i: number, j: number, k: number, l: number): Promise<number> {
    await this.data();
    return this.get(i, j, k, l);
  }

  add(value: number, i: number, j: number, k: number, l: number) {
    this.dataSync()
        [this.strides[0] * i + this.strides[1] * j + this.strides[2] * k + l] +=
        value;
  }

  locToIndex(locs: [number, number, number, number]): number {
    return this.strides[0] * locs[0] + this.strides[1] * locs[1] +
        this.strides[2] * locs[2] + locs[3];
  }

  indexToLoc(index: number): [number, number, number, number] {
    const i = Math.floor(index / this.strides[0]);
    index -= i * this.strides[0];
    const j = Math.floor(index / this.strides[1]);
    index -= j * this.strides[1];
    return [i, j, Math.floor(index / this.strides[2]), index % this.strides[2]];
  }

  asType<D2 extends DataType>(dtype: D2): Array4D<D2> {
    return super.asType(dtype) as Array4D<D2>;
  }

  static ones<D extends DataType = DataType>(
      shape: [number, number, number, number], dtype?: D): Array4D<D> {
    return NDArray.ones<D, '4'>(shape, dtype);
  }

  static zeros<D extends DataType = DataType>(
      shape: [number, number, number, number], dtype?: D): Array4D<D> {
    return NDArray.zeros<D, '4'>(shape, dtype);
  }

  static randNormal<D extends keyof RandNormalDataTypes>(
      shape: [number, number, number, number], mean = 0, stdDev = 1, dtype?: D,
      seed?: number): Array4D<D> {
    if (dtype != null && dtype === 'bool') {
      throw new Error(`Unsupported data type ${dtype}`);
    }
    const randGauss =
        new MPRandGauss(mean, stdDev, dtype, false /* truncated */, seed);
    return NDArray.rand(shape, () => randGauss.nextValue(), dtype) as
        Array4D<D>;
  }

  static randTruncatedNormal<D extends keyof RandNormalDataTypes>(
      shape: [number, number, number, number], mean = 0, stdDev = 1, dtype?: D,
      seed?: number): Array4D<D> {
    if (dtype != null && dtype === 'bool') {
      throw new Error(`Unsupported data type ${dtype}`);
    }
    const randGauss =
        new MPRandGauss(mean, stdDev, dtype, true /* truncated */, seed);
    return NDArray.rand(shape, () => randGauss.nextValue(), dtype) as
        Array4D<D>;
  }

  static randUniform<D extends DataType>(
      shape: [number, number, number, number], a: number, b: number,
      dtype?: D): Array4D<D> {
    return NDArray.rand(shape, () => util.randUniform(a, b), dtype) as
        Array4D<D>;
  }
}

export class Variable<D extends DataType = DataType, R extends Rank = Rank>
    extends NDArray<D, R> {
  private static nextVarId = 0;
  name: string;

  /**
   * Private constructor since we can not add logic before calling super().
   * Instead, we expose static `Variable.variable` method below, which will be
   * added to global namespace.
   */
  private constructor(
      initialValue: NDArray<D, R>, public trainable = true, name?: string) {
    super(
        initialValue.shape, initialValue.dtype, null /* values */,
        initialValue.dataId);
    initialValue.dispose();
    this.name = name;
    if (this.name == null) {
      this.name = Variable.nextVarId.toString();
      Variable.nextVarId++;
    }
    this.math.registerVariable(this);
  }

  /**
   * Creates a new variable with the provided initial value.
   *
   * @param initialValue An ndarray.
   * @param trainable If true, optimizers are allowed to update it.
   * @param name Name of the variable. Defaults to a unique id.
   * @param dtype If set, initialValue will be converted to the given type.
   */
  static variable<D extends DataType, R extends Rank>(
      initialValue: NDArray<D, R>, trainable = true, name?: string,
      dtype?: D): Variable<D, R> {
    if (dtype != null && dtype !== initialValue.dtype) {
      initialValue = initialValue.asType(dtype);
    }
    return new Variable(initialValue, trainable, name);
  }

  /** Assign a new array to this variable. The old array will be disposed. */
  assign(newValue: NDArray<D, R>): void {
    if (newValue.dtype !== this.dtype) {
      throw new Error(
          `dtype of the new value (${newValue.dtype}) and ` +
          `previous value (${this.dtype}) must match`);
    }
    if (!util.arraysEqual(newValue.shape, this.shape)) {
      throw new Error(
          `shape of the new value (${newValue.shape}) and ` +
          `previous value (${this.shape}) must match`);
    }
    this.math.disposeData(this.dataId);
    this.dataId = newValue.dataId;
    this.math.register(this);
    newValue.dispose();
  }
}

const variable = Variable.variable;
export {variable};

function copyTypedArray<D extends DataType>(
    array: DataTypeMap[D]|number[]|boolean[], dtype: D): DataTypeMap[D] {
  if (dtype == null || dtype === 'float32') {
    return new Float32Array(array as number[]);
  } else if (dtype === 'int32') {
    const vals = new Int32Array(array.length);
    for (let i = 0; i < vals.length; ++i) {
      const val = array[i] as number;
      if (util.isValNaN(val, 'int32')) {
        vals[i] = util.getNaN('int32');
      } else {
        vals[i] = val;
      }
    }
    return vals;
  } else if (dtype === 'bool') {
    const bool = new Uint8Array(array.length);
    for (let i = 0; i < bool.length; ++i) {
      const val = array[i] as number;
      if (util.isValNaN(val as number, 'bool')) {
        bool[i] = util.getNaN('bool');
      } else if (Math.round(val) !== 0) {
        bool[i] = 1;
      }
    }
    return bool;
  } else {
    throw new Error(`Unknown data type ${dtype}`);
  }
}

function instanceofTypedArray(a: ArrayData): boolean {
  return a instanceof Float32Array || a instanceof Int32Array ||
      a instanceof Uint8Array;
}

function noConversionNeeded(a: ArrayData, dtype: DataType): boolean {
  return (a instanceof Float32Array && dtype === 'float32') ||
      (a instanceof Int32Array && dtype === 'int32') ||
      (a instanceof Uint8Array && dtype === 'bool');
}

function toTypedArray<D extends DataType>(
    a: ArrayData, dtype: D): DataTypeMap[D] {
  if (noConversionNeeded(a, dtype)) {
    return a as DataTypeMap[D];
  }
  if (Array.isArray(a)) {
    a = util.flatten(a) as number[];
  }
  return copyTypedArray(a, dtype);
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
