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
export interface DataTypes {
  float32: Float32Array;
  int32: Int32Array;
  bool: Uint8Array;
}

/** @hidden */
export interface NDArrayData<T extends keyof DataTypes> {
  id?: number;
  values?: DataTypes[T];
}

export class NDArray<T extends keyof DataTypes = keyof DataTypes> {
  static nextId = 0;

  id: number;
  /** The shape of the ndarray. */
  shape: number[];
  /** Number of elements in the ndarray. */
  size: number;
  /** The data type for the array. */
  dtype: T;

  /**
   * Number of elements to skip in each dimension when indexing. See
   * https://docs.scipy.org/doc/numpy/reference/generated
   *     /numpy.ndarray.strides.html
   */
  strides: number[];

  private math: NDArrayMath;

  protected constructor(
      shape: number[], dtype: T, values?: DataTypes[T], id?: number,
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
    this.dtype = dtype || ('float32' as T);
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
    this.id = id;
    if (this.id == null) {
      this.id = NDArray.nextId++;
      this.math.register(this);
      this.math.write(this.id, values, this.dtype, this.shape);
    }
  }

  /** Creates a ndarray of ones with the specified shape. */
  static ones<T extends keyof DataTypes = keyof DataTypes>(
      shape: number[], dtype?: T): NDArray<T> {
    const values = makeOnesTypedArray(util.sizeFromShape(shape), dtype);
    return NDArray.make(shape, {values}, dtype);
  }

  /** Creates a ndarray of zeros with the specified shape. */
  static zeros<T extends keyof DataTypes = keyof DataTypes>(
      shape: number[], dtype?: T): NDArray<T> {
    const values = makeZerosTypedArray(util.sizeFromShape(shape), dtype);
    return NDArray.make(shape, {values}, dtype);
  }

  /**
   * Creates a ndarray of ones with the same shape as the specified ndarray.
   */
  static onesLike<G extends keyof DataTypes, T extends NDArray<G>>(another: T):
      T {
    return NDArray.ones(another.shape, another.dtype) as T;
  }

  /**
   * Creates a ndarray of zeros with the same shape as the specified ndarray.
   */
  static zerosLike<G extends keyof DataTypes, T extends NDArray<G>>(another: T):
      T {
    return NDArray.zeros(another.shape, another.dtype) as T;
  }

  /** Creates a ndarray with the same values/shape as the specified ndarray. */
  static like<G extends keyof DataTypes, T extends NDArray<G>>(another: T): T {
    const newValues = copyTypedArray(another.getValues(), another.dtype);
    return NDArray.make(another.shape, {values: newValues}, another.dtype) as T;
  }

  /**
   * Makes a new ndarray with the provided shape and values. Values should be in
   * a flat array.
   */
  static make<T extends keyof DataTypes = keyof DataTypes>(
      shape: number[], data: NDArrayData<T>, dtype?: T,
      math?: NDArrayMath): NDArray<T> {
    switch (shape.length) {
      case 0:
        return new Scalar(shape, dtype, data.values, data.id, math);
      case 1:
        return new Array1D(shape, dtype, data.values, data.id, math);
      case 2:
        return new Array2D(
            shape as [number, number], dtype, data.values, data.id, math);
      case 3:
        return new Array3D(
            shape as [number, number, number], dtype, data.values, data.id,
            math);
      case 4:
        return new Array4D(
            shape as [number, number, number, number], dtype, data.values,
            data.id, math);
      default:
        return new NDArray(shape, dtype, data.values, data.id, math);
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
    const res = NDArray.make(shape, ndarrayData, 'int32') as Array3D<'int32'>;
    math = math || ENV.math;
    math.writePixels(res.id, pixels, numChannels);
    return res;
  }

  /** Reshapes the current ndarray into the provided shape. */
  reshape(newShape: number[]): NDArray<T> {
    this.throwIfDisposed();
    newShape = util.inferFromImplicitShape(newShape, this.size);
    if (util.arraysEqual(this.shape, newShape)) {
      // No-op.
      return this;
    }

    const data: NDArrayData<T> = {id: this.id};

    util.assert(
        this.size === util.sizeFromShape(newShape),
        'new shape and old shape must have the same number of elements.');

    return NDArray.make(newShape, data, this.dtype);
  }

  /**
   * Flatten a NDArray to a 1D array
   * @param {T1} ndarray
   * @returns {Array1D}
   */
  flatten(): Array1D<T> {
    this.throwIfDisposed();
    if (this instanceof Array1D) {
      return this;
    }
    return this.as1D();
  }

  asScalar(): Scalar<T> {
    this.throwIfDisposed();
    util.assert(this.size === 1, 'The array must have only 1 element.');
    return this.reshape([]);
  }

  as1D(): Array1D<T> {
    this.throwIfDisposed();
    return this.reshape([this.size]) as Array1D<T>;
  }

  as2D(rows: number, columns: number): Array2D<T> {
    this.throwIfDisposed();
    return this.reshape([rows, columns]) as Array2D<T>;
  }

  as3D(rows: number, columns: number, depth: number): Array3D<T> {
    this.throwIfDisposed();
    return this.reshape([rows, columns, depth]) as Array3D<T>;
  }

  as4D(rows: number, columns: number, depth: number, depth2: number):
      Array4D<T> {
    this.throwIfDisposed();
    return this.reshape([rows, columns, depth, depth2]) as Array4D<T>;
  }

  asType<G extends keyof DataTypes>(dtype: G): NDArray<G> {
    this.throwIfDisposed();
    if (this.dtype === dtype as string) {
      // No-op.
      return this as NDArray as NDArray<G>;
    }
    // TODO(dsmilkov): Migrate casting to the backend.
    const vals = this.dataSync();
    const newVals = toTypedArray(vals, dtype);
    return NDArray.make<G>(this.shape, {values: newVals}, dtype);
  }

  get rank(): number {
    return this.shape.length;
  }

  get(...locs: number[]) {
    let index = locs[locs.length - 1];
    for (let i = 0; i < locs.length - 1; ++i) {
      index += this.strides[i] * locs[i];
    }
    return this.getValues()[index];
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
    const vals = this.getValues();
    vals[index] = value;
    this.math.disposeData(this.id);
    this.math.write(this.id, vals, this.dtype, this.shape);
  }

  async val(...locs: number[]): Promise<number> {
    this.throwIfDisposed();
    await this.data();
    return this.get(...locs);
  }

  locToIndex(locs: number[]): number {
    this.throwIfDisposed();
    let index = locs[locs.length - 1];
    for (let i = 0; i < locs.length - 1; ++i) {
      index += this.strides[i] * locs[i];
    }
    return index;
  }

  indexToLoc(index: number): number[] {
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
    const vals = this.getValues();
    vals.fill(value);
    this.math.disposeData(this.id);
    this.math.write(this.id, vals, this.dtype, this.shape);
  }

  /** @deprecated Use dataSync() instead. */
  getValues(): DataTypes[T] {
    return this.dataSync();
  }

  /** @deprecated Use data() instead. */
  getValuesAsync(): Promise<DataTypes[T]> {
    return this.data();
  }

  /**
   * Asynchronously downloads the values from the NDArray. Returns a promise
   * that resolves when the data is ready.
   */
  async data(): Promise<DataTypes[T]> {
    this.throwIfDisposed();
    return this.math.read(this.id);
  }

  /**
   * Synchronously downloads the values from the NDArray. This blocks the UI
   * thread until the values are ready, which can cause performance issues.
   */
  dataSync(): DataTypes[T] {
    this.throwIfDisposed();
    return this.math.readSync(this.id);
  }

  dispose(): void {
    this.isDisposed = true;
    this.math.disposeData(this.id);
  }

  equals(t: NDArray<T>): boolean {
    this.throwIfDisposed();
    return this.dtype === t.dtype && util.arraysEqual(this.shape, t.shape) &&
        util.arraysEqual(this.getValues(), t.getValues());
  }

  static rand<T extends keyof DataTypes>(
      shape: number[], randFunction: () => number, dtype?: T): NDArray<T> {
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

  static randNormal<T extends keyof RandNormalDataTypes>(
      shape: number[], mean = 0, stdDev = 1, dtype?: T,
      seed?: number): NDArray<T> {
    if (dtype != null && dtype === 'bool') {
      throw new Error(`Unsupported data type ${dtype}`);
    }
    const randGauss =
        new MPRandGauss(mean, stdDev, dtype, false /* truncated */, seed);
    return NDArray.rand(shape, () => randGauss.nextValue(), dtype);
  }

  static randTruncatedNormal<T extends keyof RandNormalDataTypes>(
      shape: number[], mean = 0, stdDev = 1, dtype?: T,
      seed?: number): NDArray<T> {
    if (dtype != null && dtype === 'bool') {
      throw new Error(`Unsupported data type ${dtype}`);
    }
    const randGauss =
        new MPRandGauss(mean, stdDev, dtype, true /* truncated */, seed);
    return NDArray.rand(shape, () => randGauss.nextValue(), dtype);
  }

  static randUniform<T extends keyof DataTypes>(
      shape: number[], a: number, b: number, dtype?: T): NDArray<T> {
    return NDArray.rand(shape, () => util.randUniform(a, b), dtype);
  }

  private isDisposed = false;
  private throwIfDisposed() {
    if (this.isDisposed) {
      throw new Error(`NDArray is disposed.`);
    }
  }
}

export class Scalar<T extends keyof DataTypes = keyof DataTypes> extends
    NDArray<T> {
  static new<T extends keyof DataTypes = keyof DataTypes>(
      value: number|boolean, dtype?: T) {
    const values = [value] as number[] | boolean[];
    return new Scalar([], dtype, toTypedArray(values, dtype));
  }

  get(): number {
    return this.getValues()[0];
  }

  async val(): Promise<number> {
    await this.data();
    return this.get();
  }

  add(value: number) {
    this.getValues()[0] += value;
  }

  asType<G extends keyof DataTypes>(dtype: G): Scalar<G> {
    return super.asType(dtype);
  }

  locToIndex(loc: number[]): number {
    return 0;
  }

  indexToLoc(index: number): number[] {
    return [];
  }
}

export class Array1D<T extends keyof DataTypes = keyof DataTypes> extends
    NDArray<T> {
  shape: [number];

  static new<T extends keyof DataTypes = keyof DataTypes>(
      values: DataTypes[T]|number[]|boolean[], dtype?: T): Array1D<T> {
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
    return this.getValues()[i];
  }

  async val(i: number): Promise<number> {
    await this.data();
    return this.get(i);
  }

  add(value: number, i: number) {
    this.getValues()[i] += value;
  }

  locToIndex(loc: [number]): number {
    return loc[0];
  }

  indexToLoc(index: number): [number] {
    return [index];
  }

  asType<G extends keyof DataTypes>(dtype: G): Array1D<G> {
    return super.asType(dtype) as Array1D<G>;
  }

  static ones<T extends keyof DataTypes = keyof DataTypes>(
      shape: [number], dtype?: T): Array1D<T> {
    return NDArray.ones(shape, dtype) as Array1D<T>;
  }

  static zeros<T extends keyof DataTypes = keyof DataTypes>(
      shape: [number], dtype?: T): Array1D<T> {
    return NDArray.zeros(shape, dtype) as Array1D<T>;
  }

  static randNormal<T extends keyof RandNormalDataTypes>(
      shape: [number], mean = 0, stdDev = 1, dtype?: T,
      seed?: number): Array1D<T> {
    if (dtype != null && dtype === 'bool') {
      throw new Error(`Unsupported data type ${dtype}`);
    }
    const randGauss =
        new MPRandGauss(mean, stdDev, dtype, false /* truncated */, seed);
    return NDArray.rand(shape, () => randGauss.nextValue(), dtype) as
        Array1D<T>;
  }

  static randTruncatedNormal<T extends keyof RandNormalDataTypes>(
      shape: [number], mean = 0, stdDev = 1, dtype?: T,
      seed?: number): Array1D<T> {
    if (dtype != null && dtype === 'bool') {
      throw new Error(`Unsupported data type ${dtype}`);
    }
    const randGauss =
        new MPRandGauss(mean, stdDev, dtype, true /* truncated */, seed);
    return NDArray.rand(shape, () => randGauss.nextValue(), dtype) as
        Array1D<T>;
  }

  static randUniform<T extends keyof DataTypes>(
      shape: [number], a: number, b: number, dtype?: T): Array1D<T> {
    return NDArray.rand(shape, () => util.randUniform(a, b), dtype) as
        Array1D<T>;
  }
}

export class Array2D<T extends keyof DataTypes = keyof DataTypes> extends
    NDArray<T> {
  shape: [number, number];

  private stride0: number;

  constructor(
      shape: [number, number], dtype: T, values?: DataTypes[T], id?: number,
      math?: NDArrayMath) {
    util.assert(shape.length === 2, 'Shape should be of length 2');
    super(shape, dtype, values, id, math);
    this.stride0 = this.strides[0];
  }

  static new<T extends keyof DataTypes = keyof DataTypes>(
      shape: [number, number],
      values: DataTypes[T]|number[]|number[][]|boolean[]|boolean[][],
      dtype?: T): Array2D<T> {
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
    return this.getValues()[this.stride0 * i + j];
  }

  add(value: number, i: number, j: number) {
    this.getValues()[this.stride0 * i + j] += value;
  }

  async val(i: number, j: number): Promise<number> {
    await this.data();
    return this.get(i, j);
  }

  locToIndex(locs: [number, number]): number {
    return this.stride0 * locs[0] + locs[1];
  }

  indexToLoc(index: number): [number, number] {
    return [Math.floor(index / this.stride0), index % this.stride0];
  }

  asType<G extends keyof DataTypes>(dtype: G): Array2D<G> {
    return super.asType(dtype) as Array2D<G>;
  }

  static ones<T extends keyof DataTypes = keyof DataTypes>(
      shape: [number, number], dtype?: T): Array2D<T> {
    return NDArray.ones(shape, dtype) as Array2D<T>;
  }

  static zeros<T extends keyof DataTypes = keyof DataTypes>(
      shape: [number, number], dtype?: T): Array2D<T> {
    return NDArray.zeros(shape, dtype) as Array2D<T>;
  }

  static randNormal<T extends keyof RandNormalDataTypes>(
      shape: [number, number], mean = 0, stdDev = 1, dtype?: T,
      seed?: number): Array2D<T> {
    if (dtype != null && dtype === 'bool') {
      throw new Error(`Unsupported data type ${dtype}`);
    }
    const randGauss =
        new MPRandGauss(mean, stdDev, dtype, false /* truncated */, seed);
    return NDArray.rand(shape, () => randGauss.nextValue(), dtype) as
        Array2D<T>;
  }

  static randTruncatedNormal<T extends keyof RandNormalDataTypes>(
      shape: [number, number], mean = 0, stdDev = 1, dtype?: T,
      seed?: number): Array2D<T> {
    if (dtype != null && dtype === 'bool') {
      throw new Error(`Unsupported data type ${dtype}`);
    }
    const randGauss =
        new MPRandGauss(mean, stdDev, dtype, true /* truncated */, seed);
    return NDArray.rand(shape, () => randGauss.nextValue(), dtype) as
        Array2D<T>;
  }

  static randUniform<T extends keyof DataTypes>(
      shape: [number, number], a: number, b: number, dtype?: T): Array2D<T> {
    return NDArray.rand(shape, () => util.randUniform(a, b), dtype) as
        Array2D<T>;
  }
}

export class Array3D<T extends keyof DataTypes = keyof DataTypes> extends
    NDArray<T> {
  shape: [number, number, number];
  private stride0: number;
  private stride1: number;

  constructor(
      shape: [number, number, number], dtype: T, values?: DataTypes[T],
      id?: number, math?: NDArrayMath) {
    util.assert(shape.length === 3, 'Shape should be of length 3');
    super(shape, dtype, values, id, math);
    this.stride0 = this.strides[0];
    this.stride1 = this.strides[1];
  }

  static new<T extends keyof DataTypes = keyof DataTypes>(
      shape: [number, number, number],
      values: DataTypes[T]|number[]|number[][][]|boolean[]|boolean[][][],
      dtype?: T) {
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
    return this.getValues()[this.stride0 * i + this.stride1 * j + k];
  }

  async val(i: number, j: number, k: number): Promise<number> {
    await this.data();
    return this.get(i, j, k);
  }

  add(value: number, i: number, j: number, k: number) {
    this.getValues()[this.stride0 * i + this.stride1 * j + k] += value;
  }

  locToIndex(locs: [number, number, number]): number {
    return this.stride0 * locs[0] + this.stride1 * locs[1] + locs[2];
  }

  indexToLoc(index: number): [number, number, number] {
    const i = Math.floor(index / this.stride0);
    index -= i * this.stride0;
    return [i, Math.floor(index / this.stride1), index % this.stride1];
  }

  asType<G extends keyof DataTypes>(dtype: G): Array3D<G> {
    return super.asType(dtype) as Array3D<G>;
  }

  static ones<T extends keyof DataTypes = keyof DataTypes>(
      shape: [number, number, number], dtype?: T): Array3D<T> {
    return NDArray.ones(shape, dtype) as Array3D<T>;
  }

  static zeros<T extends keyof DataTypes = keyof DataTypes>(
      shape: [number, number, number], dtype?: T): Array3D<T> {
    return NDArray.zeros(shape, dtype) as Array3D<T>;
  }

  static randNormal<T extends keyof RandNormalDataTypes>(
      shape: [number, number, number], mean = 0, stdDev = 1, dtype?: T,
      seed?: number): Array3D<T> {
    if (dtype != null && dtype === 'bool') {
      throw new Error(`Unsupported data type ${dtype}`);
    }
    const randGauss =
        new MPRandGauss(mean, stdDev, dtype, false /* truncated */, seed);
    return NDArray.rand(shape, () => randGauss.nextValue(), dtype) as
        Array3D<T>;
  }

  static randTruncatedNormal<T extends keyof RandNormalDataTypes>(
      shape: [number, number, number], mean = 0, stdDev = 1, dtype?: T,
      seed?: number): Array3D<T> {
    if (dtype != null && dtype === 'bool') {
      throw new Error(`Unsupported data type ${dtype}`);
    }
    const randGauss =
        new MPRandGauss(mean, stdDev, dtype, true /* truncated */, seed);
    return NDArray.rand(shape, () => randGauss.nextValue(), dtype) as
        Array3D<T>;
  }

  static randUniform<T extends keyof DataTypes>(
      shape: [number, number, number], a: number, b: number,
      dtype?: T): Array3D<T> {
    return NDArray.rand(shape, () => util.randUniform(a, b), dtype) as
        Array3D<T>;
  }
}

export class Array4D<T extends keyof DataTypes = keyof DataTypes> extends
    NDArray<T> {
  shape: [number, number, number, number];
  private stride0: number;
  private stride1: number;
  private stride2: number;

  constructor(
      shape: [number, number, number, number], dtype: T, values?: DataTypes[T],
      id?: number, math?: NDArrayMath) {
    util.assert(shape.length === 4, 'Shape should be of length 4');
    super(shape, dtype, values, id, math);
    this.stride0 = this.strides[0];
    this.stride1 = this.strides[1];
    this.stride2 = this.strides[2];
  }

  static new<T extends keyof DataTypes = keyof DataTypes>(
      shape: [number, number, number, number],
      values: DataTypes[T]|number[]|number[][][][]|boolean[]|boolean[][][][],
      dtype?: T) {
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
    return this.getValues()
        [this.stride0 * i + this.stride1 * j + this.stride2 * k + l];
  }

  async val(i: number, j: number, k: number, l: number): Promise<number> {
    await this.data();
    return this.get(i, j, k, l);
  }

  add(value: number, i: number, j: number, k: number, l: number) {
    this.getValues()
        [this.stride0 * i + this.stride1 * j + this.stride2 * k + l] += value;
  }

  locToIndex(locs: [number, number, number, number]): number {
    return this.stride0 * locs[0] + this.stride1 * locs[1] +
        this.stride2 * locs[2] + locs[3];
  }

  indexToLoc(index: number): [number, number, number, number] {
    const i = Math.floor(index / this.stride0);
    index -= i * this.stride0;
    const j = Math.floor(index / this.stride1);
    index -= j * this.stride1;
    return [i, j, Math.floor(index / this.stride2), index % this.stride2];
  }

  asType<G extends keyof DataTypes>(dtype: G): Array4D<G> {
    return super.asType(dtype) as Array4D<G>;
  }

  static ones<T extends keyof DataTypes = keyof DataTypes>(
      shape: [number, number, number, number], dtype?: T): Array4D<T> {
    return NDArray.ones(shape, dtype) as Array4D<T>;
  }

  static zeros<T extends keyof DataTypes = keyof DataTypes>(
      shape: [number, number, number, number], dtype?: T): Array4D<T> {
    return NDArray.zeros(shape, dtype) as Array4D<T>;
  }

  static randNormal<T extends keyof RandNormalDataTypes>(
      shape: [number, number, number, number], mean = 0, stdDev = 1, dtype?: T,
      seed?: number): Array4D<T> {
    if (dtype != null && dtype === 'bool') {
      throw new Error(`Unsupported data type ${dtype}`);
    }
    const randGauss =
        new MPRandGauss(mean, stdDev, dtype, false /* truncated */, seed);
    return NDArray.rand(shape, () => randGauss.nextValue(), dtype) as
        Array4D<T>;
  }

  static randTruncatedNormal<T extends keyof RandNormalDataTypes>(
      shape: [number, number, number, number], mean = 0, stdDev = 1, dtype?: T,
      seed?: number): Array4D<T> {
    if (dtype != null && dtype === 'bool') {
      throw new Error(`Unsupported data type ${dtype}`);
    }
    const randGauss =
        new MPRandGauss(mean, stdDev, dtype, true /* truncated */, seed);
    return NDArray.rand(shape, () => randGauss.nextValue(), dtype) as
        Array4D<T>;
  }

  static randUniform<T extends keyof DataTypes>(
      shape: [number, number, number, number], a: number, b: number,
      dtype?: T): Array4D<T> {
    return NDArray.rand(shape, () => util.randUniform(a, b), dtype) as
        Array4D<T>;
  }
}

function copyTypedArray<T extends keyof DataTypes>(
    array: DataTypes[T]|number[]|boolean[], dtype: T): DataTypes[T] {
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

function noConversionNeeded(a: ArrayData, dtype: keyof DataTypes): boolean {
  return (a instanceof Float32Array && dtype === 'float32') ||
      (a instanceof Int32Array && dtype === 'int32') ||
      (a instanceof Uint8Array && dtype === 'bool');
}

function toTypedArray<T extends keyof DataTypes>(
    a: ArrayData, dtype: T): DataTypes[T] {
  if (noConversionNeeded(a, dtype)) {
    return a as DataTypes[T];
  }
  if (Array.isArray(a)) {
    a = util.flatten(a) as number[];
  }
  return copyTypedArray(a, dtype);
}

function makeZerosTypedArray<T extends keyof DataTypes>(
    size: number, dtype: T): DataTypes[T] {
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

function makeOnesTypedArray<T extends keyof DataTypes>(
    size: number, dtype: T): DataTypes[T] {
  const array = makeZerosTypedArray(size, dtype);
  for (let i = 0; i < array.length; i++) {
    array[i] = 1;
  }
  return array;
}
