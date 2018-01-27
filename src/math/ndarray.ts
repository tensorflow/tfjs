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
import {MatrixOrientation} from './backends/types/matmul';
import * as ops from './ops';
import {RandNormalDataTypes} from './rand';
// tslint:disable-next-line:max-line-length
import {ArrayData, DataType, DataTypeMap, Rank, ShapeMap, TypedArray} from './types';

/** @hidden */
export interface NDArrayData {
  dataId?: number;
  values?: TypedArray;
}

export class NDArray<R extends Rank = Rank> {
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
  dtype: DataType;
  /** The rank type for the array (see `Rank` enum). */
  rankType: R;

  /**
   * Number of elements to skip in each dimension when indexing. See
   * https://docs.scipy.org/doc/numpy/reference/generated
   *     /numpy.ndarray.strides.html
   */
  strides: number[];

  protected constructor(
      shape: ShapeMap[R], dtype: DataType, values?: TypedArray,
      dataId?: number) {
    this.size = util.sizeFromShape(shape);
    if (values != null) {
      util.assert(
          this.size === values.length,
          `Constructing ndarray of shape (${this.size}) should match the ` +
              `length of values (${values.length})`);
    }
    this.shape = shape;
    this.dtype = dtype || 'float32';
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
  static ones<R extends Rank>(shape: ShapeMap[R], dtype?: DataType):
      NDArray<R> {
    return ops.ones(shape, dtype);
  }

  /** @deprecated Please use dl.zeros() */
  static zeros<R extends Rank>(shape: ShapeMap[R], dtype?: DataType):
      NDArray<R> {
    return ops.zeros(shape, dtype);
  }

  /** @deprecated Please use dl.onesLike() */
  static onesLike<T extends NDArray>(x: T): T {
    return ops.onesLike(x);
  }

  /** @deprecated Please use dl.zerosLike() */
  static zerosLike<T extends NDArray>(x: T): T {
    return ops.zerosLike(x);
  }

  /** @deprecated Please use dl.clone() */
  static like<T extends NDArray>(x: T): T {
    return ops.clone(x);
  }

  /**
   * Makes a new ndarray with the provided shape and values. Values should be in
   * a flat array.
   */
  static make<T extends NDArray<R>, D extends DataType = 'float32',
                                              R extends Rank = Rank>(
      shape: ShapeMap[R], data: NDArrayData, dtype?: D): T {
    return new NDArray(shape, dtype, data.values, data.dataId) as T;
  }

  /** @deprecated Please use dl.fromPixels() */
  static fromPixels(
      pixels: ImageData|HTMLImageElement|HTMLCanvasElement|HTMLVideoElement,
      numChannels = 3): Array3D {
    return ops.fromPixels(pixels, numChannels);
  }

  /** @deprecated Please use dl.rand() */
  static rand<R extends Rank>(
      shape: ShapeMap[R], randFunction: () => number,
      dtype?: DataType): NDArray<R> {
    return ops.rand(shape, randFunction, dtype);
  }

  /** @deprecated Please use dl.randNormal() */
  static randNormal<R extends Rank>(
      shape: ShapeMap[R], mean = 0, stdDev = 1,
      dtype?: keyof RandNormalDataTypes, seed?: number): NDArray<R> {
    return ops.randNormal(shape, mean, stdDev, dtype, seed);
  }

  /** @deprecated Please use dl.truncatedNormal() */
  static randTruncatedNormal<R extends Rank>(
      shape: ShapeMap[R], mean = 0, stdDev = 1,
      dtype?: keyof RandNormalDataTypes, seed?: number): NDArray<R> {
    return ops.truncatedNormal(shape, mean, stdDev, dtype, seed);
  }

  /** @deprecated Please use dl.randUniform() */
  static randUniform<R extends Rank>(
      shape: ShapeMap[R], a: number, b: number, dtype?: DataType): NDArray<R> {
    return ops.randUniform(shape, a, b, dtype);
  }

  /**
   * @param axis An optional list of number. If specified, only
   * squeezes the dimensions listed. The dimension index starts at 0. It is an
   * error to squeeze a dimension that is not 1.
   */
  squeeze<T extends NDArray>(axis?: number[]): T {
    this.throwIfDisposed();
    return this.reshape(util.squeezeShape(this.shape, axis).newShape) as T;
  }

  /** Flatten a NDArray to a 1D array. */
  flatten(): Array1D {
    this.throwIfDisposed();
    return this.as1D();
  }

  asScalar(): Scalar {
    this.throwIfDisposed();
    util.assert(this.size === 1, 'The array must have only 1 element.');
    return this.reshape<Rank.R0>([]);
  }

  as1D(): Array1D {
    this.throwIfDisposed();
    return this.reshape<Rank.R1>([this.size]);
  }

  as2D(rows: number, columns: number): Array2D {
    this.throwIfDisposed();
    return this.reshape<Rank.R2>([rows, columns]);
  }

  as3D(rows: number, columns: number, depth: number): Array3D {
    this.throwIfDisposed();
    return this.reshape<Rank.R3>([rows, columns, depth]);
  }

  as4D(rows: number, columns: number, depth: number, depth2: number): Array4D {
    this.throwIfDisposed();
    return this.reshape<Rank.R4>([rows, columns, depth, depth2]);
  }

  asType<T extends this>(this: T, dtype: DataType): T {
    this.throwIfDisposed();
    return ops.cast(this, dtype);
  }

  get rank(): number {
    return this.shape.length;
  }

  get(...locs: number[]) {
    this.throwIfDisposed();
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
    if (locs.length === 0) {
      locs = [0];
    }
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
    if (locs.length === 0) {
      locs = [0];
    }
    this.throwIfDisposed();
    await this.data();
    return this.get(...locs);
  }

  locToIndex(locs: number[]): number {
    this.throwIfDisposed();
    if (this.rank === 0) {
      return 0;
    } else if (this.rank === 1) {
      return locs[0];
    }
    let index = locs[locs.length - 1];
    for (let i = 0; i < locs.length - 1; ++i) {
      index += this.strides[i] * locs[i];
    }
    return index;
  }

  indexToLoc(index: number): number[] {
    this.throwIfDisposed();
    if (this.rank === 0) {
      return [];
    } else if (this.rank === 1) {
      return [index];
    }
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
  getValues(): TypedArray {
    return this.dataSync();
  }

  /** @deprecated Use data() instead. */
  getValuesAsync(): Promise<TypedArray> {
    return this.data();
  }

  /**
   * Asynchronously downloads the values from the NDArray. Returns a promise
   * that resolves when the data is ready.
   */
  async data(): Promise<TypedArray> {
    this.throwIfDisposed();
    return ENV.math.read(this.dataId);
  }

  /**
   * Synchronously downloads the values from the NDArray. This blocks the UI
   * thread until the values are ready, which can cause performance issues.
   */
  dataSync(): TypedArray {
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
  protected throwIfDisposed() {
    if (this.isDisposed) {
      throw new Error(`NDArray is disposed.`);
    }
  }

  /** Casts the array to type `float32` */
  toFloat<T extends this>(this: T): T {
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

  /** Reshapes the current ndarray into the provided shape. */
  reshape<R2 extends Rank>(newShape: ShapeMap[R2]): NDArray<R2> {
    this.throwIfDisposed();
    return ops.reshape(this, newShape);
  }

  tile<T extends this>(this: T, reps: number[]): T {
    this.throwIfDisposed();
    return ops.tile(this, reps);
  }

  gather<T extends this>(this: T, indices: Array1D, axis = 0): T {
    this.throwIfDisposed();
    return ops.gather(this, indices);
  }

  matMul(
      b: Array2D, aOrientation = MatrixOrientation.REGULAR,
      bOrientation = MatrixOrientation.REGULAR): Array2D {
    this.throwIfDisposed();
    return ops.matMul(this as Array2D, b, aOrientation, bOrientation);
  }
  slice(begin: ShapeMap[R], size: ShapeMap[R]): NDArray<R> {
    this.throwIfDisposed();
    return ops.slice(this, begin, size);
  }
  reverse(axis: number|number[]): NDArray<R> {
    this.throwIfDisposed();
    return ops.reverse(this, axis);
  }
  concat(x: NDArray<R>, axis: number): NDArray<R> {
    this.throwIfDisposed();
    return ops.concat(this, x, axis);
  }
  batchNormalization(
      mean: NDArray<R>|Array1D, variance: NDArray<R>|Array1D,
      varianceEpsilon = .001, scale?: NDArray<R>|Array1D,
      offset?: NDArray<R>|Array1D): NDArray<R> {
    this.throwIfDisposed();
    return ops.batchNormalization(
        this, mean, variance, varianceEpsilon, scale, offset);
  }

  clone(): NDArray<R> {
    this.throwIfDisposed();
    return ops.clone(this);
  }

  // Reduction ops.

  logSumExp<T extends NDArray>(axis: number|number[] = null, keepDims = false):
      T {
    this.throwIfDisposed();
    return ops.logSumExp(this, axis, keepDims);
  }
  sum<T extends NDArray>(axis: number|number[] = null, keepDims = false): T {
    this.throwIfDisposed();
    return ops.sum(this, axis, keepDims);
  }
  mean<T extends NDArray>(axis: number|number[] = null, keepDims = false): T {
    this.throwIfDisposed();
    return ops.mean(this, axis, keepDims);
  }
  min<T extends NDArray>(axis: number|number[] = null, keepDims = false): T {
    this.throwIfDisposed();
    return ops.min(this, axis, keepDims);
  }
  max<T extends NDArray>(axis: number|number[] = null, keepDims = false): T {
    this.throwIfDisposed();
    return ops.max(this, axis, keepDims);
  }
  argMin<T extends NDArray>(axis: number = null): T {
    this.throwIfDisposed();
    return ops.argMin(this, axis);
  }
  argMax<T extends NDArray>(axis: number = null): T {
    this.throwIfDisposed();
    return ops.argMax(this, axis);
  }
  argMaxEquals(x: NDArray): Scalar {
    this.throwIfDisposed();
    return ops.argMaxEquals(this, x);
  }

  // Binary ops.

  add<T extends NDArray>(x: NDArray): T {
    this.throwIfDisposed();
    return ops.add(this, x);
  }
  addStrict<T extends this>(this: T, x: T): T {
    this.throwIfDisposed();
    return ops.addStrict(this, x);
  }
  sub<T extends NDArray>(x: NDArray): T {
    this.throwIfDisposed();
    return ops.sub(this, x);
  }
  subStrict<T extends this>(this: T, x: T): T {
    this.throwIfDisposed();
    return ops.subStrict(this, x);
  }
  pow<T extends NDArray>(exp: NDArray): T {
    this.throwIfDisposed();
    return ops.pow(this, exp);
  }
  powStrict(exp: NDArray): NDArray<R> {
    this.throwIfDisposed();
    return ops.powStrict(this, exp);
  }
  mul<T extends NDArray>(x: NDArray): T {
    this.throwIfDisposed();
    return ops.mul(this, x);
  }
  mulStrict<T extends this>(this: T, x: T): T {
    this.throwIfDisposed();
    return ops.mulStrict(this, x);
  }
  div<T extends NDArray>(x: NDArray): T {
    this.throwIfDisposed();
    return ops.div(this, x);
  }
  divStrict<T extends this>(this: T, x: T): T {
    this.throwIfDisposed();
    return ops.divStrict(this, x);
  }
  minimum<T extends NDArray>(x: NDArray): T {
    this.throwIfDisposed();
    return ops.minimum(this, x);
  }
  minimumStrict<T extends this>(this: T, x: T): T {
    this.throwIfDisposed();
    return ops.minimumStrict(this, x);
  }
  maximum<T extends NDArray>(x: NDArray): T {
    this.throwIfDisposed();
    return ops.maximum(this, x);
  }
  maximumStrict<T extends this>(this: T, x: T): T {
    this.throwIfDisposed();
    return ops.maximumStrict(this, x);
  }
  transpose(perm?: number[]): NDArray<R> {
    this.throwIfDisposed();
    return ops.transpose(this, perm);
  }

  // Compare ops.

  notEqual<T extends NDArray>(x: NDArray): T {
    this.throwIfDisposed();
    return ops.notEqual(this, x);
  }
  notEqualStrict<T extends this>(this: T, x: T): T {
    this.throwIfDisposed();
    return ops.notEqualStrict(this, x);
  }
  less<T extends NDArray>(x: NDArray): T {
    this.throwIfDisposed();
    return ops.less(this, x);
  }
  lessStrict<T extends this>(this: T, x: T): T {
    this.throwIfDisposed();
    return ops.lessStrict(this, x);
  }
  equal<T extends NDArray>(x: NDArray): T {
    this.throwIfDisposed();
    return ops.equal(this, x);
  }
  equalStrict<T extends this>(this: T, x: T): T {
    this.throwIfDisposed();
    return ops.equalStrict(this, x);
  }
  lessEqual<T extends NDArray>(x: NDArray): T {
    this.throwIfDisposed();
    return ops.lessEqual(this, x);
  }
  lessEqualStrict<T extends this>(this: T, x: T): T {
    this.throwIfDisposed();
    return ops.lessEqualStrict(this, x);
  }
  greater<T extends NDArray>(x: NDArray): T {
    this.throwIfDisposed();
    return ops.greater(this, x);
  }
  greaterStrict<T extends this>(this: T, x: T): T {
    this.throwIfDisposed();
    return ops.greaterStrict(this, x);
  }
  greaterEqual<T extends NDArray>(x: NDArray): T {
    this.throwIfDisposed();
    return ops.greaterEqual(this, x);
  }
  greaterEqualStrict<T extends this>(this: T, x: T): T {
    this.throwIfDisposed();
    return ops.greaterEqualStrict(this, x);
  }

  // Compare ops.
  logicalAnd(x: NDArray): NDArray {
    this.throwIfDisposed();
    return ops.logicalAnd(this, x);
  }
  logicalOr(x: NDArray): NDArray {
    this.throwIfDisposed();
    return ops.logicalOr(this, x);
  }
  where(condition: NDArray, x: NDArray): NDArray {
    this.throwIfDisposed();
    return ops.where(condition, this, x);
  }

  // Unary ops.
  neg(): NDArray<R> {
    this.throwIfDisposed();
    return ops.neg(this);
  }
  ceil(): NDArray<R> {
    this.throwIfDisposed();
    return ops.ceil(this);
  }
  floor(): NDArray<R> {
    this.throwIfDisposed();
    return ops.floor(this);
  }
  exp(): NDArray<R> {
    this.throwIfDisposed();
    return ops.exp(this);
  }
  log(): NDArray<R> {
    this.throwIfDisposed();
    return ops.log(this);
  }
  sqrt(): NDArray<R> {
    this.throwIfDisposed();
    return ops.sqrt(this);
  }
  square(): NDArray<R> {
    this.throwIfDisposed();
    return ops.square(this);
  }
  abs(): NDArray<R> {
    this.throwIfDisposed();
    return ops.abs(this);
  }
  clip(min: number, max: number): NDArray<R> {
    this.throwIfDisposed();
    return ops.clip(this, min, max);
  }
  relu(): NDArray<R> {
    this.throwIfDisposed();
    return ops.relu(this);
  }
  elu(): NDArray<R> {
    this.throwIfDisposed();
    return ops.elu(this);
  }
  selu(): NDArray<R> {
    this.throwIfDisposed();
    return ops.selu(this);
  }
  leakyRelu(alpha = 0.2): NDArray<R> {
    this.throwIfDisposed();
    return ops.leakyRelu(this, alpha);
  }
  prelu(alpha: NDArray<R>): NDArray<R> {
    this.throwIfDisposed();
    return ops.prelu(this, alpha);
  }
  sigmoid(): NDArray<R> {
    this.throwIfDisposed();
    return ops.sigmoid(this);
  }
  sin(): NDArray<R> {
    this.throwIfDisposed();
    return ops.sin(this);
  }
  cos(): NDArray<R> {
    this.throwIfDisposed();
    return ops.cos(this);
  }
  tan(): NDArray<R> {
    this.throwIfDisposed();
    return ops.tan(this);
  }
  asin(): NDArray<R> {
    this.throwIfDisposed();
    return ops.asin(this);
  }
  acos(): NDArray<R> {
    this.throwIfDisposed();
    return ops.acos(this);
  }
  atan(): NDArray<R> {
    this.throwIfDisposed();
    return ops.atan(this);
  }
  sinh(): NDArray<R> {
    this.throwIfDisposed();
    return ops.sinh(this);
  }
  cosh(): NDArray<R> {
    this.throwIfDisposed();
    return ops.cosh(this);
  }
  tanh(): NDArray<R> {
    this.throwIfDisposed();
    return ops.tanh(this);
  }
  step(alpha = 0.0): NDArray<R> {
    this.throwIfDisposed();
    return ops.step(this, alpha);
  }
  softmax<T extends this>(this: T, dim = -1): T {
    this.throwIfDisposed();
    return ops.softmax(this, dim);
  }

  // Image ops.
  resizeBilinear<T extends Array3D|Array4D>(
      this: T, newShape2D: [number, number], alignCorners = false): T {
    (this as NDArray).throwIfDisposed();
    return ops.image.resizeBilinear(this, newShape2D, alignCorners);
  }

  // Convolutions.
  conv1d<T extends Array2D|Array3D>(
      this: T, filter: Array3D, bias: Array1D|null, stride: number,
      pad: 'valid'|'same'|number, dimRoundingMode?: 'floor'|'round'|'ceil'): T {
    (this as NDArray).throwIfDisposed();
    return ops.conv1d(this, filter, bias, stride, pad, dimRoundingMode);
  }
  conv2d<T extends Array3D|Array4D>(
      this: T, filter: Array4D, bias: Array1D|null,
      strides: [number, number]|number, pad: 'valid'|'same'|number,
      dimRoundingMode?: 'floor'|'round'|'ceil'): T {
    (this as NDArray).throwIfDisposed();
    return ops.conv2d(this, filter, bias, strides, pad, dimRoundingMode);
  }
  conv2dTranspose<T extends Array3D|Array4D>(
      this: T, filter: Array4D,
      outputShape: [number, number, number, number]|[number, number, number],
      strides: [number, number]|number, pad: 'valid'|'same'|number,
      dimRoundingMode?: 'floor'|'round'|'ceil'): T {
    (this as NDArray).throwIfDisposed();
    return ops.conv2dTranspose(
        this, filter, outputShape, strides, pad, dimRoundingMode);
  }
  depthwiseConv2D<T extends Array3D|Array4D>(
      this: T, filter: Array4D, strides: [number, number]|number,
      pad: 'valid'|'same'|number, rates: [number, number]|number = [1, 1],
      dimRoundingMode?: 'floor'|'round'|'ceil'): T {
    (this as NDArray).throwIfDisposed();
    return ops.depthwiseConv2D(
        this, filter, strides, pad, rates, dimRoundingMode);
  }

  // Pooling.
  avgPool<T extends Array3D|Array4D>(
      this: T, filterSize: [number, number]|number,
      strides: [number, number]|number, pad: 'valid'|'same'|number,
      dimRoundingMode?: 'floor'|'round'|'ceil'): T {
    (this as NDArray).throwIfDisposed();
    return ops.avgPool(this, filterSize, strides, pad, dimRoundingMode);
  }
  maxPool<T extends Array3D|Array4D>(
      this: T, filterSize: [number, number]|number,
      strides: [number, number]|number, pad: 'valid'|'same'|number,
      dimRoundingMode?: 'floor'|'round'|'ceil'): T {
    (this as NDArray).throwIfDisposed();
    return ops.maxPool(this, filterSize, strides, pad, dimRoundingMode);
  }
  minPool<T extends Array3D|Array4D>(
      this: T, filterSize: [number, number]|number,
      strides: [number, number]|number, pad: 'valid'|'same'|number,
      dimRoundingMode?: 'floor'|'round'|'ceil'): T {
    (this as NDArray).throwIfDisposed();
    return ops.minPool(this, filterSize, strides, pad, dimRoundingMode);
  }
}

export class Scalar extends NDArray<Rank.R0> {
  static new(value: number|boolean, dtype?: DataType): Scalar {
    const values = [value] as number[] | boolean[];
    return new Scalar([], dtype, toTypedArray(values, dtype));
  }
}

export class Array1D extends NDArray<Rank.R1> {
  static new<D extends DataType = 'float32'>(
      values: DataTypeMap[D]|number[]|boolean[], dtype?: D): Array1D {
    if (!instanceofTypedArray(values)) {
      const inferredShape = util.inferShape(values as number[] | boolean[]);
      util.assert(
          inferredShape.length === 1,
          `Error constructing Array1D. Shape of values ${inferredShape} is ` +
              `not 1 dimensional.`);
    }
    return new Array1D([values.length], dtype, toTypedArray(values, dtype));
  }
}

export class Array2D extends NDArray<Rank.R2> {
  static new<D extends DataType = 'float32'>(
      shape: [number, number],
      values: DataTypeMap[D]|number[]|number[][]|boolean[]|boolean[][],
      dtype?: D): Array2D {
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
}

export class Array3D extends NDArray<Rank.R3> {
  static new<D extends DataType = 'float32'>(
      shape: [number, number, number],
      values: DataTypeMap[D]|number[]|number[][][]|boolean[]|boolean[][][],
      dtype?: D): Array3D {
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
}

export class Array4D extends NDArray<Rank.R4> {
  static new<D extends DataType = 'float32'>(
      shape: [number, number, number, number],
      values: DataTypeMap[D]|number[]|number[][][][]|boolean[]|boolean[][][][],
      dtype?: D): Array4D {
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
}

export class Variable<R extends Rank = Rank> extends NDArray<R> {
  private static nextVarId = 0;
  name: string;

  /**
   * Private constructor since we can not add logic before calling super().
   * Instead, we expose static `Variable.variable` method below, which will be
   * added to global namespace.
   */
  private constructor(
      initialValue: NDArray<R>, public trainable = true, name?: string) {
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
  static variable<R extends Rank>(
      initialValue: NDArray<R>, trainable = true, name?: string,
      dtype?: DataType): Variable<R> {
    if (dtype != null && dtype !== initialValue.dtype) {
      initialValue = initialValue.asType(dtype) as NDArray<R>;
    }
    return new Variable(initialValue, trainable, name);
  }

  /** Assign a new array to this variable. The old array will be disposed. */
  assign(newValue: NDArray<R>): void {
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

function instanceofTypedArray(a: ArrayData<DataType>): boolean {
  return a instanceof Float32Array || a instanceof Int32Array ||
      a instanceof Uint8Array;
}

function noConversionNeeded<D extends DataType>(
    a: ArrayData<D>, dtype: D): boolean {
  return (a instanceof Float32Array && dtype === 'float32') ||
      (a instanceof Int32Array && dtype === 'int32') ||
      (a instanceof Uint8Array && dtype === 'bool');
}

function toTypedArray<D extends DataType>(
    a: ArrayData<D>, dtype: D): DataTypeMap[D] {
  if (noConversionNeeded(a, dtype)) {
    return a as DataTypeMap[D];
  }
  if (Array.isArray(a)) {
    a = util.flatten(a as number[]);
  }
  return util.copyTypedArray(a, dtype);
}
