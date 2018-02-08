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
import {DataType, DataTypeMap, Rank, ShapeMap, TypedArray} from './types';

/** @hidden */
export interface TensorData {
  dataId?: DataId;
  values?: TypedArray;
}

export class TensorBuffer<R extends Rank> {
  values: TypedArray;
  private strides: number[];

  constructor(public shape: ShapeMap[R], public dtype: DataType) {
    this.values = util.getTypedArrayFromDType(dtype, util.sizeFromShape(shape));
    this.strides = computeStrides(shape);
  }

  set(value: number, ...locs: number[]) {
    if (locs.length === 0) {
      locs = [0];
    }
    util.assert(
        locs.length === this.rank,
        `The number of provided coordinates (${locs.length}) must ` +
            `match the rank (${this.rank})`);
    const index = this.locToIndex(locs);
    this.values[index] = value;
  }

  locToIndex(locs: number[]): number {
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

  get rank() {
    return this.shape.length;
  }

  toTensor(): Tensor<R> {
    return Tensor.make(this.shape, {values: this.values}, this.dtype);
  }
}

/**
 * We wrap data id since we use weak map to avoid memory leaks.
 * Since we have our own memory management, we have a reference counter
 * mapping a tensor to its data, so there is always a pointer (even if that
 * data is otherwise garbage collectable).
 * See https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/
 * Global_Objects/WeakMap
 */
export type DataId = object;  // object instead of {} to force non-primitive.

export class Tensor<R extends Rank = Rank> {
  private static nextId = 0;

  /** Unique id of this tensor. */
  readonly id: number;
  /**
   * Id of the bucket holding the data for this tensor. Multiple arrays can
   * point to the same bucket (e.g. when calling array.reshape()).
   */
  dataId: DataId;
  /** The shape of the tensor. */
  readonly shape: ShapeMap[R];
  /** Number of elements in the tensor. */
  readonly size: number;
  /** The data type for the array. */
  readonly dtype: DataType;
  /** The rank type for the array (see `Rank` enum). */
  readonly rankType: R;

  /**
   * Number of elements to skip in each dimension when indexing. See
   * https://docs.scipy.org/doc/numpy/reference/generated/\
   * numpy.ndarray.strides.html
   */
  readonly strides: number[];

  protected constructor(
      shape: ShapeMap[R], dtype: DataType, values?: TypedArray,
      dataId?: DataId) {
    this.size = util.sizeFromShape(shape);
    if (values != null) {
      util.assert(
          this.size === values.length,
          `Constructing tensor of shape (${this.size}) should match the ` +
              `length of values (${values.length})`);
    }
    this.shape = shape;
    this.dtype = dtype || 'float32';
    this.strides = computeStrides(shape);
    this.dataId = dataId != null ? dataId : {};
    this.id = Tensor.nextId++;
    this.rankType = (this.rank < 5 ? this.rank.toString() : 'higher') as R;
    ENV.engine.registerTensor(this);
    if (values != null) {
      ENV.engine.write(this.dataId, values);
    }
  }

  /** @deprecated Please use dl.ones() */
  static ones<R extends Rank>(shape: ShapeMap[R], dtype?: DataType): Tensor<R> {
    return ops.ones(shape, dtype);
  }

  /** @deprecated Please use dl.zeros() */
  static zeros<R extends Rank>(shape: ShapeMap[R], dtype?: DataType):
      Tensor<R> {
    return ops.zeros(shape, dtype);
  }

  /** @deprecated Please use dl.onesLike() */
  static onesLike<T extends Tensor>(x: T): T {
    return ops.onesLike(x);
  }

  /** @deprecated Please use dl.zerosLike() */
  static zerosLike<T extends Tensor>(x: T): T {
    return ops.zerosLike(x);
  }

  /** @deprecated Please use dl.clone() */
  static like<T extends Tensor>(x: T): T {
    return ops.clone(x);
  }

  /**
   * Makes a new tensor with the provided shape and values. Values should be in
   * a flat array.
   */
  static make<T extends Tensor<R>, D extends DataType = 'float32',
                                             R extends Rank = Rank>(
      shape: ShapeMap[R], data: TensorData, dtype?: D): T {
    return new Tensor(shape, dtype, data.values, data.dataId) as T;
  }

  /** @deprecated Please use dl.fromPixels() */
  static fromPixels(
      pixels: ImageData|HTMLImageElement|HTMLCanvasElement|HTMLVideoElement,
      numChannels = 3): Tensor3D {
    return ops.fromPixels(pixels, numChannels);
  }

  /** @deprecated Please use dl.rand() */
  static rand<R extends Rank>(
      shape: ShapeMap[R], randFunction: () => number,
      dtype?: DataType): Tensor<R> {
    return ops.rand(shape, randFunction, dtype);
  }

  /** @deprecated Please use dl.randomNormal() */
  static randNormal<R extends Rank>(
      shape: ShapeMap[R], mean = 0, stdDev = 1,
      dtype?: keyof RandNormalDataTypes, seed?: number): Tensor<R> {
    return ops.randomNormal(shape, mean, stdDev, dtype, seed);
  }

  /** @deprecated Please use dl.truncatedNormal() */
  static randTruncatedNormal<R extends Rank>(
      shape: ShapeMap[R], mean = 0, stdDev = 1,
      dtype?: keyof RandNormalDataTypes, seed?: number): Tensor<R> {
    return ops.truncatedNormal(shape, mean, stdDev, dtype, seed);
  }

  /** @deprecated Please use dl.randomUniform() */
  static randUniform<R extends Rank>(
      shape: ShapeMap[R], a: number, b: number, dtype?: DataType): Tensor<R> {
    return ops.randomUniform(shape, a, b, dtype);
  }

  /**
   * @param axis An optional list of number. If specified, only
   * squeezes the dimensions listed. The dimension index starts at 0. It is an
   * error to squeeze a dimension that is not 1.
   */
  squeeze<T extends Tensor>(axis?: number[]): T {
    this.throwIfDisposed();
    return this.reshape(util.squeezeShape(this.shape, axis).newShape) as T;
  }

  /** Flatten a Tensor to a 1D array. */
  flatten(): Tensor1D {
    this.throwIfDisposed();
    return this.as1D();
  }

  asScalar(): Scalar {
    this.throwIfDisposed();
    util.assert(this.size === 1, 'The array must have only 1 element.');
    return this.reshape<Rank.R0>([]);
  }

  as1D(): Tensor1D {
    this.throwIfDisposed();
    return this.reshape<Rank.R1>([this.size]);
  }

  as2D(rows: number, columns: number): Tensor2D {
    this.throwIfDisposed();
    return this.reshape<Rank.R2>([rows, columns]);
  }

  as3D(rows: number, columns: number, depth: number): Tensor3D {
    this.throwIfDisposed();
    return this.reshape<Rank.R3>([rows, columns, depth]);
  }

  as4D(rows: number, columns: number, depth: number, depth2: number): Tensor4D {
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

  /** @deprecated Use dataSync() instead. */
  getValues(): TypedArray {
    return this.dataSync();
  }

  /** @deprecated Use data() instead. */
  getValuesAsync(): Promise<TypedArray> {
    return this.data();
  }

  /**
   * Asynchronously downloads the values from the Tensor. Returns a promise
   * that resolves when the data is ready.
   */
  async data(): Promise<TypedArray> {
    this.throwIfDisposed();
    return ENV.engine.read(this.dataId);
  }

  /**
   * Synchronously downloads the values from the Tensor. This blocks the UI
   * thread until the values are ready, which can cause performance issues.
   */
  dataSync(): TypedArray {
    this.throwIfDisposed();
    return ENV.engine.readSync(this.dataId);
  }

  dispose(): void {
    if (this.isDisposed) {
      return;
    }
    this.isDisposed = true;
    ENV.engine.disposeTensor(this);
  }

  private isDisposed = false;
  protected throwIfDisposed() {
    if (this.isDisposed) {
      throw new Error(`Tensor is disposed.`);
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

  /** Reshapes the current tensor into the provided shape. */
  reshape<R2 extends Rank>(newShape: ShapeMap[R2]): Tensor<R2> {
    this.throwIfDisposed();
    return ops.reshape(this, newShape);
  }

  reshapeAs<T extends Tensor>(x: T): T {
    this.throwIfDisposed();
    return this.reshape(x.shape) as T;
  }

  tile<T extends this>(this: T, reps: number[]): T {
    this.throwIfDisposed();
    return ops.tile(this, reps);
  }

  gather<T extends this>(this: T, indices: Tensor1D, axis = 0): T {
    this.throwIfDisposed();
    return ops.gather(this, indices);
  }

  matMul(
      b: Tensor2D, aOrientation = MatrixOrientation.REGULAR,
      bOrientation = MatrixOrientation.REGULAR): Tensor2D {
    this.throwIfDisposed();
    return ops.matMul(this as Tensor2D, b, aOrientation, bOrientation);
  }
  slice(begin: ShapeMap[R], size: ShapeMap[R]): Tensor<R> {
    this.throwIfDisposed();
    return ops.slice(this, begin, size);
  }
  reverse(axis: number|number[]): Tensor<R> {
    this.throwIfDisposed();
    return ops.reverse(this, axis);
  }
  concat(x: Tensor<R>, axis: number): Tensor<R> {
    this.throwIfDisposed();
    return ops.concat(this, x, axis);
  }
  batchNormalization(
      mean: Tensor<R>|Tensor1D, variance: Tensor<R>|Tensor1D,
      varianceEpsilon = .001, scale?: Tensor<R>|Tensor1D,
      offset?: Tensor<R>|Tensor1D): Tensor<R> {
    this.throwIfDisposed();
    return ops.batchNormalization(
        this, mean, variance, varianceEpsilon, scale, offset);
  }

  clone(): Tensor<R> {
    this.throwIfDisposed();
    return ops.clone(this);
  }

  // Reduction ops.

  logSumExp<T extends Tensor>(axis: number|number[] = null, keepDims = false):
      T {
    this.throwIfDisposed();
    return ops.logSumExp(this, axis, keepDims);
  }
  sum<T extends Tensor>(axis: number|number[] = null, keepDims = false): T {
    this.throwIfDisposed();
    return ops.sum(this, axis, keepDims);
  }
  mean<T extends Tensor>(axis: number|number[] = null, keepDims = false): T {
    this.throwIfDisposed();
    return ops.mean(this, axis, keepDims);
  }
  min<T extends Tensor>(axis: number|number[] = null, keepDims = false): T {
    this.throwIfDisposed();
    return ops.min(this, axis, keepDims);
  }
  max<T extends Tensor>(axis: number|number[] = null, keepDims = false): T {
    this.throwIfDisposed();
    return ops.max(this, axis, keepDims);
  }
  argMin<T extends Tensor>(axis: number = null): T {
    this.throwIfDisposed();
    return ops.argMin(this, axis);
  }
  argMax<T extends Tensor>(axis: number = null): T {
    this.throwIfDisposed();
    return ops.argMax(this, axis);
  }
  argMaxEquals(x: Tensor): Scalar {
    this.throwIfDisposed();
    return ops.argMaxEquals(this, x);
  }

  // Binary ops.

  add<T extends Tensor>(x: Tensor): T {
    this.throwIfDisposed();
    return ops.add(this, x);
  }
  addStrict<T extends this>(this: T, x: T): T {
    this.throwIfDisposed();
    return ops.addStrict(this, x);
  }
  sub<T extends Tensor>(x: Tensor): T {
    this.throwIfDisposed();
    return ops.sub(this, x);
  }
  subStrict<T extends this>(this: T, x: T): T {
    this.throwIfDisposed();
    return ops.subStrict(this, x);
  }
  pow<T extends Tensor>(exp: Tensor): T {
    this.throwIfDisposed();
    return ops.pow(this, exp);
  }
  powStrict(exp: Tensor): Tensor<R> {
    this.throwIfDisposed();
    return ops.powStrict(this, exp);
  }
  mul<T extends Tensor>(x: Tensor): T {
    this.throwIfDisposed();
    return ops.mul(this, x);
  }
  mulStrict<T extends this>(this: T, x: T): T {
    this.throwIfDisposed();
    return ops.mulStrict(this, x);
  }
  div<T extends Tensor>(x: Tensor): T {
    this.throwIfDisposed();
    return ops.div(this, x);
  }
  divStrict<T extends this>(this: T, x: T): T {
    this.throwIfDisposed();
    return ops.divStrict(this, x);
  }
  minimum<T extends Tensor>(x: Tensor): T {
    this.throwIfDisposed();
    return ops.minimum(this, x);
  }
  minimumStrict<T extends this>(this: T, x: T): T {
    this.throwIfDisposed();
    return ops.minimumStrict(this, x);
  }
  maximum<T extends Tensor>(x: Tensor): T {
    this.throwIfDisposed();
    return ops.maximum(this, x);
  }
  maximumStrict<T extends this>(this: T, x: T): T {
    this.throwIfDisposed();
    return ops.maximumStrict(this, x);
  }
  transpose(perm?: number[]): Tensor<R> {
    this.throwIfDisposed();
    return ops.transpose(this, perm);
  }

  // Compare ops.

  notEqual<T extends Tensor>(x: Tensor): T {
    this.throwIfDisposed();
    return ops.notEqual(this, x);
  }
  notEqualStrict<T extends this>(this: T, x: T): T {
    this.throwIfDisposed();
    return ops.notEqualStrict(this, x);
  }
  less<T extends Tensor>(x: Tensor): T {
    this.throwIfDisposed();
    return ops.less(this, x);
  }
  lessStrict<T extends this>(this: T, x: T): T {
    this.throwIfDisposed();
    return ops.lessStrict(this, x);
  }
  equal<T extends Tensor>(x: Tensor): T {
    this.throwIfDisposed();
    return ops.equal(this, x);
  }
  equalStrict<T extends this>(this: T, x: T): T {
    this.throwIfDisposed();
    return ops.equalStrict(this, x);
  }
  lessEqual<T extends Tensor>(x: Tensor): T {
    this.throwIfDisposed();
    return ops.lessEqual(this, x);
  }
  lessEqualStrict<T extends this>(this: T, x: T): T {
    this.throwIfDisposed();
    return ops.lessEqualStrict(this, x);
  }
  greater<T extends Tensor>(x: Tensor): T {
    this.throwIfDisposed();
    return ops.greater(this, x);
  }
  greaterStrict<T extends this>(this: T, x: T): T {
    this.throwIfDisposed();
    return ops.greaterStrict(this, x);
  }
  greaterEqual<T extends Tensor>(x: Tensor): T {
    this.throwIfDisposed();
    return ops.greaterEqual(this, x);
  }
  greaterEqualStrict<T extends this>(this: T, x: T): T {
    this.throwIfDisposed();
    return ops.greaterEqualStrict(this, x);
  }

  // Compare ops.
  logicalAnd(x: Tensor): Tensor {
    this.throwIfDisposed();
    return ops.logicalAnd(this, x);
  }
  logicalOr(x: Tensor): Tensor {
    this.throwIfDisposed();
    return ops.logicalOr(this, x);
  }
  logicalXor(x: Tensor): Tensor {
    this.throwIfDisposed();
    return ops.logicalXor(this, x);
  }
  where(condition: Tensor, x: Tensor): Tensor {
    this.throwIfDisposed();
    return ops.where(condition, this, x);
  }

  // Unary ops.
  neg(): Tensor<R> {
    this.throwIfDisposed();
    return ops.neg(this);
  }
  ceil(): Tensor<R> {
    this.throwIfDisposed();
    return ops.ceil(this);
  }
  floor(): Tensor<R> {
    this.throwIfDisposed();
    return ops.floor(this);
  }
  exp(): Tensor<R> {
    this.throwIfDisposed();
    return ops.exp(this);
  }
  log(): Tensor<R> {
    this.throwIfDisposed();
    return ops.log(this);
  }
  sqrt(): Tensor<R> {
    this.throwIfDisposed();
    return ops.sqrt(this);
  }
  square(): Tensor<R> {
    this.throwIfDisposed();
    return ops.square(this);
  }
  abs(): Tensor<R> {
    this.throwIfDisposed();
    return ops.abs(this);
  }
  clip(min: number, max: number): Tensor<R> {
    this.throwIfDisposed();
    return ops.clip(this, min, max);
  }
  relu(): Tensor<R> {
    this.throwIfDisposed();
    return ops.relu(this);
  }
  elu(): Tensor<R> {
    this.throwIfDisposed();
    return ops.elu(this);
  }
  selu(): Tensor<R> {
    this.throwIfDisposed();
    return ops.selu(this);
  }
  leakyRelu(alpha = 0.2): Tensor<R> {
    this.throwIfDisposed();
    return ops.leakyRelu(this, alpha);
  }
  prelu(alpha: Tensor<R>): Tensor<R> {
    this.throwIfDisposed();
    return ops.prelu(this, alpha);
  }
  sigmoid(): Tensor<R> {
    this.throwIfDisposed();
    return ops.sigmoid(this);
  }
  sin(): Tensor<R> {
    this.throwIfDisposed();
    return ops.sin(this);
  }
  cos(): Tensor<R> {
    this.throwIfDisposed();
    return ops.cos(this);
  }
  tan(): Tensor<R> {
    this.throwIfDisposed();
    return ops.tan(this);
  }
  asin(): Tensor<R> {
    this.throwIfDisposed();
    return ops.asin(this);
  }
  acos(): Tensor<R> {
    this.throwIfDisposed();
    return ops.acos(this);
  }
  atan(): Tensor<R> {
    this.throwIfDisposed();
    return ops.atan(this);
  }
  sinh(): Tensor<R> {
    this.throwIfDisposed();
    return ops.sinh(this);
  }
  cosh(): Tensor<R> {
    this.throwIfDisposed();
    return ops.cosh(this);
  }
  tanh(): Tensor<R> {
    this.throwIfDisposed();
    return ops.tanh(this);
  }
  step(alpha = 0.0): Tensor<R> {
    this.throwIfDisposed();
    return ops.step(this, alpha);
  }
  softmax<T extends this>(this: T, dim = -1): T {
    this.throwIfDisposed();
    return ops.softmax(this, dim);
  }

  // Image ops.
  resizeBilinear<T extends Tensor3D|Tensor4D>(
      this: T, newShape2D: [number, number], alignCorners = false): T {
    (this as Tensor).throwIfDisposed();
    return ops.image.resizeBilinear(this, newShape2D, alignCorners);
  }

  // Convolutions.
  conv1d<T extends Tensor2D|Tensor3D>(
      this: T, filter: Tensor3D, bias: Tensor1D|null, stride: number,
      pad: 'valid'|'same'|number, dimRoundingMode?: 'floor'|'round'|'ceil'): T {
    (this as Tensor).throwIfDisposed();
    return ops.conv1d(this, filter, bias, stride, pad, dimRoundingMode);
  }
  conv2d<T extends Tensor3D|Tensor4D>(
      this: T, filter: Tensor4D, bias: Tensor1D|null,
      strides: [number, number]|number, pad: 'valid'|'same'|number,
      dimRoundingMode?: 'floor'|'round'|'ceil'): T {
    (this as Tensor).throwIfDisposed();
    return ops.conv2d(this, filter, bias, strides, pad, dimRoundingMode);
  }
  conv2dTranspose<T extends Tensor3D|Tensor4D>(
      this: T, filter: Tensor4D,
      outputShape: [number, number, number, number]|[number, number, number],
      strides: [number, number]|number, pad: 'valid'|'same'|number,
      dimRoundingMode?: 'floor'|'round'|'ceil'): T {
    (this as Tensor).throwIfDisposed();
    return ops.conv2dTranspose(
        this, filter, outputShape, strides, pad, dimRoundingMode);
  }
  depthwiseConv2D<T extends Tensor3D|Tensor4D>(
      this: T, filter: Tensor4D, strides: [number, number]|number,
      pad: 'valid'|'same'|number, rates: [number, number]|number = [1, 1],
      dimRoundingMode?: 'floor'|'round'|'ceil'): T {
    (this as Tensor).throwIfDisposed();
    return ops.depthwiseConv2d(
        this, filter, strides, pad, rates, dimRoundingMode);
  }

  // Pooling.
  avgPool<T extends Tensor3D|Tensor4D>(
      this: T, filterSize: [number, number]|number,
      strides: [number, number]|number, pad: 'valid'|'same'|number,
      dimRoundingMode?: 'floor'|'round'|'ceil'): T {
    (this as Tensor).throwIfDisposed();
    return ops.avgPool(this, filterSize, strides, pad, dimRoundingMode);
  }
  maxPool<T extends Tensor3D|Tensor4D>(
      this: T, filterSize: [number, number]|number,
      strides: [number, number]|number, pad: 'valid'|'same'|number,
      dimRoundingMode?: 'floor'|'round'|'ceil'): T {
    (this as Tensor).throwIfDisposed();
    return ops.maxPool(this, filterSize, strides, pad, dimRoundingMode);
  }
  minPool<T extends Tensor3D|Tensor4D>(
      this: T, filterSize: [number, number]|number,
      strides: [number, number]|number, pad: 'valid'|'same'|number,
      dimRoundingMode?: 'floor'|'round'|'ceil'): T {
    (this as Tensor).throwIfDisposed();
    return ops.minPool(this, filterSize, strides, pad, dimRoundingMode);
  }
}

export class Scalar extends Tensor<Rank.R0> {
  static new(value: number|boolean, dtype?: DataType): Scalar {
    return ops.scalar(value, dtype);
  }
}

export class Tensor1D extends Tensor<Rank.R1> {
  static new<D extends DataType = 'float32'>(
      values: DataTypeMap[D]|number[]|boolean[], dtype?: D): Tensor1D {
    return ops.tensor1d(values, dtype);
  }
}

export class Tensor2D extends Tensor<Rank.R2> {
  static new<D extends DataType = 'float32'>(
      shape: [number, number],
      values: DataTypeMap[D]|number[]|number[][]|boolean[]|boolean[][],
      dtype?: D): Tensor2D {
    return ops.tensor2d(values, shape, dtype);
  }
}

export class Tensor3D extends Tensor<Rank.R3> {
  static new<D extends DataType = 'float32'>(
      shape: [number, number, number],
      values: DataTypeMap[D]|number[]|number[][][]|boolean[]|boolean[][][],
      dtype?: D): Tensor3D {
    return ops.tensor3d(values, shape, dtype);
  }
}

export class Tensor4D extends Tensor<Rank.R4> {
  static new<D extends DataType = 'float32'>(
      shape: [number, number, number, number],
      values: DataTypeMap[D]|number[]|number[][][][]|boolean[]|boolean[][][][],
      dtype?: D): Tensor4D {
    return ops.tensor4d(values, shape, dtype);
  }
}

export class Variable<R extends Rank = Rank> extends Tensor<R> {
  private static nextVarId = 0;
  name: string;

  /**
   * Private constructor since we can not add logic before calling super().
   * Instead, we expose static `Variable.variable` method below, which will be
   * added to global namespace.
   */
  private constructor(
      initialValue: Tensor<R>, public trainable = true, name?: string) {
    super(
        initialValue.shape, initialValue.dtype, null /* values */,
        initialValue.dataId);
    initialValue.dispose();
    this.name = name;
    if (this.name == null) {
      this.name = Variable.nextVarId.toString();
      Variable.nextVarId++;
    }
    ENV.engine.registerVariable(this);
  }

  /**
   * Creates a new variable with the provided initial value.
   *
   * @param initialValue A tensor.
   * @param trainable If true, optimizers are allowed to update it.
   * @param name Name of the variable. Defaults to a unique id.
   * @param dtype If set, initialValue will be converted to the given type.
   */
  static variable<R extends Rank>(
      initialValue: Tensor<R>, trainable = true, name?: string,
      dtype?: DataType): Variable<R> {
    if (dtype != null && dtype !== initialValue.dtype) {
      initialValue = initialValue.asType(dtype) as Tensor<R>;
    }
    return new Variable(initialValue, trainable, name);
  }

  /** Assign a new array to this variable. The old array will be disposed. */
  assign(newValue: Tensor<R>): void {
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
    ENV.engine.disposeTensor(this);
    this.dataId = newValue.dataId;
    ENV.engine.registerTensor(this);
    newValue.dispose();
  }
}

const variable = Variable.variable;
export {variable};

function computeStrides(shape: number[]): number[] {
  const rank = shape.length;
  if (rank < 2) {
    return [];
  }

  // Last dimension has implicit stride of 1, thus having D-1 (instead of D)
  // strides.
  const strides = new Array(rank - 1);
  strides[rank - 2] = shape[rank - 1];
  for (let i = rank - 3; i >= 0; --i) {
    strides[i] = strides[i + 1] * shape[i + 1];
  }
  return strides;
}

// Aliases for backwards compatibility.
export {
  Tensor as NDArray,
  Tensor1D as Array1D,
  Tensor2D as Array2D,
  Tensor3D as Array3D,
  Tensor4D as Array4D
};
