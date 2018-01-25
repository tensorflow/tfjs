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
import * as array_ops from './array_ops';
import {MatrixOrientation} from './backends/types/matmul';
import * as ops from './ops';
import {RandNormalDataTypes} from './rand';
// tslint:disable-next-line:max-line-length
import {DataType, DataTypeMap, Rank, RankMap, ShapeMap, SumTypes} from './types';

/** @hidden */
export interface NDArrayData<D extends DataType> {
  dataId?: number;
  values?: DataTypeMap[D];
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

  protected constructor(
      shape: number[], dtype: D, values?: DataTypeMap[D], dataId?: number) {
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
    ENV.math.register(this);
    if (values != null) {
      ENV.math.write(this.dataId, values);
    }
  }

  /** @deprecated Please use dl.ones() */
  static ones<D extends DataType = 'float32', R extends Rank = Rank>(
      shape: number[], dtype?: D): RankMap<D>[R] {
    return array_ops.Ops.ones(shape, dtype);
  }

  /** @deprecated Please use dl.zeros() */
  static zeros<D extends DataType = 'float32', R extends Rank = Rank>(
      shape: number[], dtype?: D): RankMap<D>[R] {
    return array_ops.Ops.zeros(shape, dtype);
  }

  /** @deprecated Please use dl.onesLike() */
  static onesLike<T extends NDArray>(x: T): T {
    return array_ops.Ops.onesLike(x);
  }

  /** @deprecated Please use dl.zerosLike() */
  static zerosLike<T extends NDArray>(x: T): T {
    return array_ops.Ops.zerosLike(x);
  }

  /** @deprecated Please use dl.clone() */
  static like<D extends DataType, R extends Rank>(x: NDArray<D, R>):
      RankMap<D>[R] {
    return array_ops.Ops.clone(x);
  }

  /**
   * Makes a new ndarray with the provided shape and values. Values should be in
   * a flat array.
   */
  static make<D extends DataType = 'float32', R extends Rank = Rank>(
      shape: number[], data: NDArrayData<D>, dtype?: D): RankMap<D>[R] {
    switch (shape.length) {
      case 0:
        return new Scalar(shape, dtype, data.values, data.dataId);
      case 1:
        return new Array1D(shape, dtype, data.values, data.dataId);
      case 2:
        return new Array2D(
            shape as [number, number], dtype, data.values, data.dataId);
      case 3:
        return new Array3D(
            shape as [number, number, number], dtype, data.values, data.dataId);
      case 4:
        return new Array4D(
            shape as [number, number, number, number], dtype, data.values,
            data.dataId);
      default:
        return new NDArray(shape, dtype, data.values, data.dataId) as
            RankMap<D>[R];
    }
  }

  /** @deprecated Please use dl.fromPixels() */
  static fromPixels(
      pixels: ImageData|HTMLImageElement|HTMLCanvasElement|HTMLVideoElement,
      numChannels = 3): Array3D<'int32'> {
    return array_ops.Ops.fromPixels(pixels, numChannels);
  }

  /** @deprecated Please use dl.rand() */
  static rand<D extends DataType, R extends Rank>(
      shape: number[], randFunction: () => number, dtype?: D): RankMap<D>[R] {
    return array_ops.Ops.rand(shape, randFunction, dtype);
  }

  /** @deprecated Please use dl.randNormal() */
  static randNormal<D extends keyof RandNormalDataTypes, R extends Rank>(
      shape: number[], mean = 0, stdDev = 1, dtype?: D,
      seed?: number): RankMap<D>[R] {
    return array_ops.Ops.randNormal(shape, mean, stdDev, dtype, seed);
  }

  /** @deprecated Please use dl.randTruncatedNormal() */
  static randTruncatedNormal<D extends keyof RandNormalDataTypes,
                                       R extends Rank>(
      shape: number[], mean = 0, stdDev = 1, dtype?: D,
      seed?: number): RankMap<D>[R] {
    return array_ops.Ops.randTruncatedNormal(shape, mean, stdDev, dtype, seed);
  }

  /** @deprecated Please use dl.randUniform() */
  static randUniform<D extends DataType, R extends Rank>(
      shape: number[], a: number, b: number, dtype?: D): RankMap<D>[R] {
    return array_ops.Ops.randUniform(shape, a, b, dtype);
  }

  /** Reshapes the current ndarray into the provided shape. */
  reshape<R2 extends Rank>(newShape: ShapeMap[R2]): RankMap<D>[R2] {
    this.throwIfDisposed();
    return ENV.math.reshape(this, newShape);
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
    return ENV.math.cast(this, dtype) as NDArray<D2, R>;
  }

  get rank(): number {
    return this.shape.length;
  }

  get(...locs: number[]) {
    if (locs.length === 0) {
      locs = [0];
    }
    let index = locs[locs.length - 1];
    for (let i = 0; i < locs.length - 1; ++i) {
      index += this.strides[i] * locs[i];
    }
    return this.dataSync()[index];
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
    ENV.math.write(this.dataId, vals);
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
    ENV.math.write(this.dataId, vals);
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
    return ENV.math.read(this.dataId);
  }

  /**
   * Synchronously downloads the values from the NDArray. This blocks the UI
   * thread until the values are ready, which can cause performance issues.
   */
  dataSync(): DataTypeMap[D] {
    this.throwIfDisposed();
    return ENV.math.readSync(this.dataId);
  }

  dispose(): void {
    if (this.isDisposed) {
      return;
    }
    this.isDisposed = true;
    ENV.math.disposeData(this.dataId);
  }

  private isDisposed = false;
  private throwIfDisposed() {
    if (this.isDisposed) {
      throw new Error(`NDArray is disposed.`);
    }
  }

  /** Casts the array to type `float32` */
  toFloat() {
    return this.asType('float32');
  }

  /** Casts the array to type `int32` */
  toInt() {
    return this.asType('int32');
  }

  /** Casts the array to type `bool` */
  toBool() {
    return this.asType('bool');
  }

  // Chain API.

  matMul(
      b: Array2D<D>, aOrientation = MatrixOrientation.REGULAR,
      bOrientation = MatrixOrientation.REGULAR): Array2D<D> {
    return ops.matMul(this as Array2D<D>, b, aOrientation, bOrientation);
  }
  slice(begin: ShapeMap[R], size: ShapeMap[R]): RankMap<D>[R] {
    return ops.slice(this, begin, size);
  }
  reverse(axis: number|number[]): RankMap<D>[R] {
    return ops.reverse(this, axis);
  }
  concat(x: NDArray<D, R>, axis: number): RankMap<D>[R] {
    return ops.concat(this, x, axis);
  }
  batchNormalization(
      mean: RankMap<D>[R]|Array1D, variance: RankMap<D>[R]|Array1D,
      varianceEpsilon = .001, scale?: RankMap<D>[R]|Array1D,
      offset?: RankMap<D>[R]|Array1D): RankMap<D>[R] {
    return ops.batchNormalization(
        this, mean, variance, varianceEpsilon, scale, offset);
  }
  avgPool(
      filterSize: [number, number]|number, strides: [number, number]|number,
      pad: 'valid'|'same'|number,
      dimRoundingMode?: 'floor'|'round'|'ceil'): RankMap<'float32'>[R] {
    return ops.avgPool(
        this as NDArray<'int32'|'float32', '3'|'4'>, filterSize, strides, pad,
        dimRoundingMode);
  }
  maxPool(
      filterSize: [number, number]|number, strides: [number, number]|number,
      pad: 'valid'|'same'|number,
      dimRoundingMode?: 'floor'|'round'|'ceil'): RankMap<D>[R] {
    return ops.maxPool(
        this as NDArray<D, '3'|'4'>, filterSize, strides, pad, dimRoundingMode);
  }
  minPool(
      filterSize: [number, number]|number, strides: [number, number]|number,
      pad: 'valid'|'same'|number,
      dimRoundingMode?: 'floor'|'round'|'ceil'): RankMap<D>[R] {
    return ops.minPool(
        this as NDArray<D, '3'|'4'>, filterSize, strides, pad, dimRoundingMode);
  }
  clone(): RankMap<D>[R] {
    return ops.clone(this);
  }

  // Reduction ops.

  logSumExp<T extends NDArray<'float32'>>(
      axis: number|number[] = null, keepDims = false): T {
    return ops.logSumExp(this, axis, keepDims);
  }
  sum<T extends NDArray<SumTypes[D]>>(
      axis: number|number[] = null, keepDims = false): T {
    return ops.sum(this, axis, keepDims);
  }
  mean<T extends NDArray<'float32'>>(
      axis: number|number[] = null, keepDims = false): T {
    return ops.mean(this, axis, keepDims);
  }
  min<T extends NDArray<D>>(axis: number|number[] = null, keepDims = false): T {
    return ops.min(this, axis, keepDims);
  }
  max<T extends NDArray<D>>(axis: number|number[] = null, keepDims = false): T {
    return ops.max(this, axis, keepDims);
  }
  argMin<T extends NDArray<'int32'>>(axis: number = null): T {
    return ops.argMin(this, axis);
  }
  argMax<T extends NDArray<'int32'>>(axis: number = null): T {
    return ops.argMax(this, axis);
  }
  argMaxEquals(x: NDArray): Scalar<'bool'> {
    return ops.argMaxEquals(this, x);
  }

  // Binary ops.

  add<T extends NDArray<D>>(x: NDArray<D>): T {
    return ops.add(this, x);
  }
  addStrict(x: NDArray<D, R>): RankMap<D>[R] {
    return ops.addStrict(this, x);
  }
  sub<T extends NDArray<D>>(x: NDArray<D>): T {
    return ops.sub(this, x);
  }
  subStrict(x: NDArray<D, R>): RankMap<D>[R] {
    return ops.subStrict(this, x);
  }
  pow<T extends NDArray<D>>(exp: NDArray<'int32'>): T {
    return ops.pow(this, exp);
  }
  powStrict(exp: NDArray<'int32'>): RankMap<D>[R] {
    return ops.powStrict(this, exp);
  }
  mul<T extends NDArray<D>>(x: NDArray<D>): T {
    return ops.mul(this, x);
  }
  mulStrict(x: NDArray<D, R>): RankMap<D>[R] {
    return ops.mulStrict(this, x);
  }
  div<T extends NDArray<'float32'>>(x: NDArray<D>): T {
    return ops.div(this, x);
  }
  divStrict(x: NDArray<D, R>): RankMap<D>[R] {
    return ops.divStrict(this, x);
  }
  minimum<T extends NDArray<D>>(x: NDArray<D>): T {
    return ops.minimum(this, x);
  }
  minimumStrict(x: NDArray<D, R>): RankMap<D>[R] {
    return ops.minimumStrict(this, x);
  }
  maximum<T extends NDArray<D>>(x: NDArray<D>): T {
    return ops.maximum(this, x);
  }
  maximumStrict(x: NDArray<D, R>): RankMap<D>[R] {
    return ops.maximumStrict(this, x);
  }
  transpose(perm?: number[]): RankMap<D>[R] {
    return ops.transpose(this, perm);
  }

  // Compare ops.

  notEqual<T extends NDArray<'bool'>>(x: NDArray<D>): T {
    return ops.notEqual(this, x);
  }
  notEqualStrict(x: NDArray<D, R>): RankMap<'bool'>[R] {
    return ops.notEqualStrict(this, x);
  }
  less<T extends NDArray<'bool'>>(x: NDArray<D>): T {
    return ops.less(this, x);
  }
  lessStrict(x: NDArray<D, R>): RankMap<'bool'>[R] {
    return ops.lessStrict(this, x);
  }
  equal<T extends NDArray<'bool'>>(x: NDArray<D>): T {
    return ops.equal(this, x);
  }
  equalStrict(x: NDArray<D, R>): RankMap<'bool'>[R] {
    return ops.equalStrict(this, x);
  }
  lessEqual<T extends NDArray<'bool'>>(x: NDArray<D>): T {
    return ops.lessEqual(this, x);
  }
  lessEqualStrict(x: NDArray<D, R>): RankMap<'bool'>[R] {
    return ops.lessEqualStrict(this, x);
  }
  greater<T extends NDArray<'bool'>>(x: NDArray<D>): T {
    return ops.greater(this, x);
  }
  greaterStrict(x: NDArray<D, R>): RankMap<'bool'>[R] {
    return ops.greaterStrict(this, x);
  }
  greaterEqual<T extends NDArray<'bool'>>(x: NDArray<D>): T {
    return ops.greaterEqual(this, x);
  }
  greaterEqualStrict(x: NDArray<D, R>): RankMap<'bool'>[R] {
    return ops.greaterEqualStrict(this, x);
  }

  // Unary ops.
  neg(): RankMap<D>[R] {
    return ops.neg(this);
  }
  ceil(): RankMap<D>[R] {
    return ops.ceil(this);
  }
  floor(): RankMap<D>[R] {
    return ops.floor(this);
  }
  exp(): RankMap<D>[R] {
    return ops.exp(this);
  }
  log(): RankMap<D>[R] {
    return ops.log(this);
  }
  sqrt(): RankMap<D>[R] {
    return ops.sqrt(this);
  }
  square(): RankMap<D>[R] {
    return ops.square(this);
  }
  abs(): RankMap<D>[R] {
    return ops.abs(this);
  }
  clip(min: number, max: number): RankMap<D>[R] {
    return ops.clip(this, min, max);
  }
  relu(): RankMap<D>[R] {
    return ops.relu(this);
  }
  elu(): RankMap<D>[R] {
    return ops.elu(this);
  }
  selu(): RankMap<D>[R] {
    return ops.selu(this);
  }
  leakyRelu(alpha = 0.2): RankMap<D>[R] {
    return ops.leakyRelu(this, alpha);
  }
  prelu(alpha: NDArray<D, R>): RankMap<D>[R] {
    return ops.prelu(this, alpha);
  }
  sigmoid(): RankMap<D>[R] {
    return ops.sigmoid(this);
  }
  sin(): RankMap<D>[R] {
    return ops.sin(this);
  }
  cos(): RankMap<D>[R] {
    return ops.cos(this);
  }
  tan(): RankMap<D>[R] {
    return ops.tan(this);
  }
  asin(): RankMap<D>[R] {
    return ops.asin(this);
  }
  acos(): RankMap<D>[R] {
    return ops.acos(this);
  }
  atan(): RankMap<D>[R] {
    return ops.atan(this);
  }
  sinh(): RankMap<D>[R] {
    return ops.sinh(this);
  }
  cosh(): RankMap<D>[R] {
    return ops.cosh(this);
  }
  tanh(): RankMap<D>[R] {
    return ops.tanh(this);
  }
  step(alpha = 0.0): RankMap<D>[R] {
    return ops.step(this, alpha);
  }
}

export class Scalar<D extends DataType = DataType> extends NDArray<D, '0'> {
  static new<D extends DataType = 'float32'>(value: number|boolean, dtype?: D):
      Scalar<D> {
    const values = [value] as number[] | boolean[];
    return new Scalar([], dtype, toTypedArray(values, dtype));
  }

  async val(): Promise<number> {
    await this.data();
    return this.get();
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

  async val(i: number): Promise<number> {
    await this.data();
    return this.get(i);
  }

  locToIndex(loc: [number]): number {
    return loc[0];
  }

  indexToLoc(index: number): [number] {
    return [index];
  }

  /** @deprecated Please use dl.ones() */
  static ones<D extends DataType = 'float32'>(shape: number[], dtype?: D):
      Array1D<D> {
    return array_ops.Ops.ones<D, '1'>(shape, dtype);
  }

  /** @deprecated Please use dl.zeros() */
  static zeros<D extends DataType = 'float32'>(shape: number[], dtype?: D):
      Array1D<D> {
    return array_ops.Ops.zeros<D, '1'>(shape, dtype);
  }

  /** @deprecated Please use dl.rand() */
  static rand<D extends DataType>(
      shape: number[], randFunction: () => number, dtype?: D): Array1D<D> {
    return array_ops.Ops.rand<D, '1'>(shape, randFunction, dtype);
  }

  /** @deprecated Please use dl.randNormal() */
  static randNormal<D extends keyof RandNormalDataTypes>(
      shape: number[], mean = 0, stdDev = 1, dtype?: D,
      seed?: number): Array1D<D> {
    return array_ops.Ops.randNormal<D, '1'>(shape, mean, stdDev, dtype, seed);
  }

  /** @deprecated Please use dl.randTruncatedNormal() */
  static randTruncatedNormal<D extends keyof RandNormalDataTypes>(
      shape: number[], mean = 0, stdDev = 1, dtype?: D,
      seed?: number): Array1D<D> {
    return array_ops.Ops.randTruncatedNormal<D, '1'>(
        shape, mean, stdDev, dtype, seed);
  }

  /** @deprecated Please use dl.randUniform() */
  static randUniform<D extends DataType>(
      shape: number[], a: number, b: number, dtype?: D): Array1D<D> {
    return array_ops.Ops.randUniform<D, '1'>(shape, a, b, dtype);
  }
}

export class Array2D<D extends DataType = DataType> extends NDArray<D, '2'> {
  constructor(
      shape: [number, number], dtype: D, values?: DataTypeMap[D],
      dataId?: number) {
    util.assert(shape.length === 2, 'Shape should be of length 2');
    super(shape, dtype, values, dataId);
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

  /** @deprecated Please use dl.ones() */
  static ones<D extends DataType = 'float32'>(shape: number[], dtype?: D):
      Array2D<D> {
    return array_ops.Ops.ones<D, '2'>(shape, dtype);
  }

  /** @deprecated Please use dl.zeros() */
  static zeros<D extends DataType = 'float32'>(shape: number[], dtype?: D):
      Array2D<D> {
    return array_ops.Ops.zeros<D, '2'>(shape, dtype);
  }

  /** @deprecated Please use dl.rand() */
  static rand<D extends DataType>(
      shape: number[], randFunction: () => number, dtype?: D): Array2D<D> {
    return array_ops.Ops.rand<D, '2'>(shape, randFunction, dtype);
  }

  /** @deprecated Please use dl.randNormal() */
  static randNormal<D extends keyof RandNormalDataTypes>(
      shape: number[], mean = 0, stdDev = 1, dtype?: D,
      seed?: number): Array2D<D> {
    return array_ops.Ops.randNormal<D, '2'>(shape, mean, stdDev, dtype, seed);
  }

  /** @deprecated Please use dl.randTruncatedNormal() */
  static randTruncatedNormal<D extends keyof RandNormalDataTypes>(
      shape: number[], mean = 0, stdDev = 1, dtype?: D,
      seed?: number): Array2D<D> {
    return array_ops.Ops.randTruncatedNormal<D, '2'>(
        shape, mean, stdDev, dtype, seed);
  }

  /** @deprecated Please use dl.randUniform() */
  static randUniform<D extends DataType>(
      shape: number[], a: number, b: number, dtype?: D): Array2D<D> {
    return array_ops.Ops.randUniform<D, '2'>(shape, a, b, dtype);
  }
}

export class Array3D<D extends DataType = DataType> extends NDArray<D, '3'> {
  constructor(
      shape: [number, number, number], dtype: D, values?: DataTypeMap[D],
      dataId?: number) {
    util.assert(shape.length === 3, 'Shape should be of length 3');
    super(shape, dtype, values, dataId);
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

  async val(i: number, j: number, k: number): Promise<number> {
    await this.data();
    return this.get(i, j, k);
  }

  locToIndex(locs: [number, number, number]): number {
    return this.strides[0] * locs[0] + this.strides[1] * locs[1] + locs[2];
  }

  indexToLoc(index: number): [number, number, number] {
    const i = Math.floor(index / this.strides[0]);
    index -= i * this.strides[0];
    return [i, Math.floor(index / this.strides[1]), index % this.strides[1]];
  }

  /** @deprecated Please use dl.ones() */
  static ones<D extends DataType = 'float32'>(shape: number[], dtype?: D):
      Array3D<D> {
    return array_ops.Ops.ones<D, '3'>(shape, dtype);
  }

  /** @deprecated Please use dl.zeros() */
  static zeros<D extends DataType = 'float32'>(shape: number[], dtype?: D):
      Array3D<D> {
    return array_ops.Ops.zeros<D, '3'>(shape, dtype);
  }

  /** @deprecated Please use dl.rand() */
  static rand<D extends DataType>(
      shape: number[], randFunction: () => number, dtype?: D): Array3D<D> {
    return array_ops.Ops.rand<D, '3'>(shape, randFunction, dtype);
  }

  /** @deprecated Please use dl.randNormal() */
  static randNormal<D extends keyof RandNormalDataTypes>(
      shape: number[], mean = 0, stdDev = 1, dtype?: D,
      seed?: number): Array3D<D> {
    return array_ops.Ops.randNormal<D, '3'>(shape, mean, stdDev, dtype, seed);
  }

  /** @deprecated Please use dl.randTruncatedNormal() */
  static randTruncatedNormal<D extends keyof RandNormalDataTypes>(
      shape: number[], mean = 0, stdDev = 1, dtype?: D,
      seed?: number): Array3D<D> {
    return array_ops.Ops.randTruncatedNormal<D, '3'>(
        shape, mean, stdDev, dtype, seed);
  }

  /** @deprecated Please use dl.randUniform() */
  static randUniform<D extends DataType>(
      shape: number[], a: number, b: number, dtype?: D): Array3D<D> {
    return array_ops.Ops.randUniform<D, '3'>(shape, a, b, dtype);
  }
}

export class Array4D<D extends DataType = DataType> extends NDArray<D, '4'> {
  constructor(
      shape: [number, number, number, number], dtype: D,
      values?: DataTypeMap[D], dataId?: number) {
    util.assert(shape.length === 4, 'Shape should be of length 4');
    super(shape, dtype, values, dataId);
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

  async val(i: number, j: number, k: number, l: number): Promise<number> {
    await this.data();
    return this.get(i, j, k, l);
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

  /** @deprecated Please use dl.ones() */
  static ones<D extends DataType = 'float32'>(shape: number[], dtype?: D):
      Array4D<D> {
    return array_ops.Ops.ones<D, '4'>(shape, dtype);
  }

  /** @deprecated Please use dl.zeros() */
  static zeros<D extends DataType = 'float32'>(shape: number[], dtype?: D):
      Array4D<D> {
    return array_ops.Ops.zeros<D, '4'>(shape, dtype);
  }

  /** @deprecated Please use dl.rand() */
  static rand<D extends DataType>(
      shape: number[], randFunction: () => number, dtype?: D): Array4D<D> {
    return array_ops.Ops.rand<D, '4'>(shape, randFunction, dtype);
  }

  /** @deprecated Please use dl.randNormal() */
  static randNormal<D extends keyof RandNormalDataTypes>(
      shape: number[], mean = 0, stdDev = 1, dtype?: D,
      seed?: number): Array4D<D> {
    return array_ops.Ops.randNormal<D, '4'>(shape, mean, stdDev, dtype, seed);
  }

  /** @deprecated Please use dl.randTruncatedNormal() */
  static randTruncatedNormal<D extends keyof RandNormalDataTypes>(
      shape: number[], mean = 0, stdDev = 1, dtype?: D,
      seed?: number): Array4D<D> {
    return array_ops.Ops.randTruncatedNormal<D, '4'>(
        shape, mean, stdDev, dtype, seed);
  }

  /** @deprecated Please use dl.randUniform() */
  static randUniform<D extends DataType>(
      shape: number[], a: number, b: number, dtype?: D): Array4D<D> {
    return array_ops.Ops.randUniform<D, '4'>(shape, a, b, dtype);
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
    ENV.math.registerVariable(this);
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
      initialValue = initialValue.asType(dtype) as NDArray<D, R>;
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
    ENV.math.disposeData(this.dataId);
    this.dataId = newValue.dataId;
    ENV.math.register(this);
    newValue.dispose();
  }
}

const variable = Variable.variable;
export {variable};

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
  return util.copyTypedArray(a, dtype);
}
