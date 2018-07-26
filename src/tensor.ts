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

import {tensorToString} from './tensor_format';
import {DataType, Rank, ShapeMap, TypedArray} from './types';
import * as util from './util';
import {computeStrides} from './util';

/** @hidden */
export interface TensorData {
  dataId?: DataId;
  values?: TypedArray;
}

/**
 * A mutable object, similar to `Tensor`, that allows users to set values
 * at locations before converting to an immutable `Tensor`.
 *
 * See `buffer` for creating a tensor buffer.
 */
/** @doc {heading: 'Tensors', subheading: 'Classes'} */
export class TensorBuffer<R extends Rank> {
  size: number;
  shape: ShapeMap[R];
  strides: number[];
  values: TypedArray;

  constructor(shape: ShapeMap[R], public dtype: DataType, values: TypedArray) {
    if (values != null) {
      const n = values.length;
      const size = util.sizeFromShape(shape);
      util.assert(
          n === size,
          `Length of values '${n}' does not match the size ` +
              `inferred by the shape '${size}'`);
    }
    this.shape = shape.slice();
    this.values =
        values || util.getTypedArrayFromDType(dtype, util.sizeFromShape(shape));
    this.strides = computeStrides(shape);
    this.size = util.sizeFromShape(shape);
  }

  /**
   * Sets a value in the buffer at a given location.
   *
   * @param value The value to set.
   * @param locs  The location indices.
   */
  /** @doc {heading: 'Tensors', subheading: 'Creation'} */
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

  /**
   * Returns the value in the buffer at the provided location.
   *
   * @param locs The location indices.
   */
  /** @doc {heading: 'Tensors', subheading: 'Creation'} */
  get(...locs: number[]): number {
    if (locs.length === 0) {
      locs = [0];
    }
    let index = locs[locs.length - 1];
    for (let i = 0; i < locs.length - 1; ++i) {
      index += this.strides[i] * locs[i];
    }
    return this.values[index];
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

  /**
   * Creates an immutable `Tensor` object from the buffer.
   */
  /** @doc {heading: 'Tensors', subheading: 'Creation'} */
  toTensor(): Tensor<R> {
    return Tensor.make(this.shape, {values: this.values}, this.dtype);
  }
}

export interface TensorTracker {
  registerTensor(t: Tensor): void;
  disposeTensor(t: Tensor): void;
  write(dataId: DataId, values: TypedArray): void;
  read(dataId: DataId): Promise<TypedArray>;
  readSync(dataId: DataId): TypedArray;
  registerVariable(v: Variable): void;
}

/**
 * The Tensor class calls into this handler to delegate chaining operations.
 */
export interface OpHandler {
  cast<T extends Tensor>(x: T, dtype: DataType): T;
  buffer<R extends Rank>(
      shape: ShapeMap[R], dtype: DataType,
      values?: TypedArray): TensorBuffer<R>;
  print<T extends Tensor>(x: T, verbose: boolean): void;
  reshape<R2 extends Rank>(x: Tensor, shape: ShapeMap[R2]): Tensor<R2>;
  expandDims<R2 extends Rank>(x: Tensor, axis: number): Tensor<R2>;
  cumsum<T extends Tensor>(
      x: Tensor, axis: number, exclusive: boolean, reverse: boolean): T;
  squeeze<T extends Tensor>(x: Tensor, axis?: number[]): T;
  clone<T extends Tensor>(x: T): T;
  tile<T extends Tensor>(x: T, reps: number[]): T;
  gather<T extends Tensor>(x: T, indices: Tensor1D, axis: number): T;
  matMul(a: Tensor2D, b: Tensor2D, transposeA: boolean, transposeB: boolean):
      Tensor2D;
  dot(t1: Tensor, t2: Tensor): Tensor;
  norm(
      x: Tensor, ord: number|'euclidean'|'fro', axis: number|number[],
      keepDims: boolean): Tensor;
  slice<R extends Rank, T extends Tensor<R>>(
      x: T, begin: number|number[], size?: number|number[]): T;
  reverse<T extends Tensor>(x: T, axis?: number|number[]): T;
  concat<T extends Tensor>(tensors: T[], axis: number): T;
  stack<T extends Tensor>(tensors: T[], axis: number): Tensor;
  unstack<T extends Tensor>(value: T, axis: number): Tensor[];
  pad<T extends Tensor>(
      x: T, paddings: Array<[number, number]>, constantValue: number): T;
  batchNormalization<R extends Rank>(
      x: Tensor<R>, mean: Tensor<R>|Tensor1D, variance: Tensor<R>|Tensor1D,
      varianceEpsilon: number, scale?: Tensor<R>|Tensor1D,
      offset?: Tensor<R>|Tensor1D): Tensor<R>;
  all<T extends Tensor>(x: Tensor, axis: number|number[], keepDims: boolean): T;
  any<T extends Tensor>(x: Tensor, axis: number|number[], keepDims: boolean): T;
  logSumExp<T extends Tensor>(
      x: Tensor, axis: number|number[], keepDims: boolean): T;
  sum<T extends Tensor>(x: Tensor, axis: number|number[], keepDims: boolean): T;
  mean<T extends Tensor>(x: Tensor, axis: number|number[], keepDims: boolean):
      T;
  min<T extends Tensor>(x: Tensor, axis: number|number[], keepDims: boolean): T;
  max<T extends Tensor>(x: Tensor, axis: number|number[], keepDims: boolean): T;
  argMin<T extends Tensor>(x: Tensor, axis: number): T;
  argMax<T extends Tensor>(x: Tensor, axis: number): T;
  add<T extends Tensor>(a: Tensor, b: Tensor): T;
  addStrict<T extends Tensor>(a: T, b: T): T;
  sub<T extends Tensor>(a: Tensor, b: Tensor): T;
  subStrict<T extends Tensor>(a: T, b: T): T;
  pow<T extends Tensor>(base: T, exp: Tensor): T;
  powStrict<T extends Tensor>(base: T, exp: Tensor): T;
  mul<T extends Tensor>(a: Tensor, b: Tensor): T;
  mulStrict<T extends Tensor>(a: T, b: T): T;
  div<T extends Tensor>(a: Tensor, b: Tensor): T;
  floorDiv<T extends Tensor>(a: Tensor, b: Tensor): T;
  divStrict<T extends Tensor>(a: T, b: T): T;
  mod<T extends Tensor>(a: Tensor, b: Tensor): T;
  modStrict<T extends Tensor>(a: T, b: T): T;
  minimum<T extends Tensor>(a: Tensor, b: Tensor): T;
  minimumStrict<T extends Tensor>(a: T, b: T): T;
  maximum<T extends Tensor>(a: Tensor, b: Tensor): T;
  maximumStrict<T extends Tensor>(a: T, b: T): T;
  squaredDifference<T extends Tensor>(a: Tensor, b: Tensor): T;
  squaredDifferenceStrict<T extends Tensor>(a: T, b: T): T;
  transpose<T extends Tensor>(x: T, perm?: number[]): T;
  logicalNot<T extends Tensor>(x: T): T;
  logicalAnd<T extends Tensor>(a: Tensor, b: Tensor): T;
  logicalOr<T extends Tensor>(a: Tensor, b: Tensor): T;
  logicalXor<T extends Tensor>(a: Tensor, b: Tensor): T;
  where<T extends Tensor>(condition: Tensor, a: T, b: T): T;
  notEqual<T extends Tensor>(a: Tensor, b: Tensor): T;
  notEqualStrict<T extends Tensor>(a: T, b: T): T;
  less<T extends Tensor>(a: Tensor, b: Tensor): T;
  lessStrict<T extends Tensor>(a: T, b: T): T;
  equal<T extends Tensor>(a: Tensor, b: Tensor): T;
  equalStrict<T extends Tensor>(a: T, b: T): T;
  lessEqual<T extends Tensor>(a: Tensor, b: Tensor): T;
  lessEqualStrict<T extends Tensor>(a: T, b: T): T;
  greater<T extends Tensor>(a: Tensor, b: Tensor): T;
  greaterStrict<T extends Tensor>(a: T, b: T): T;
  greaterEqual<T extends Tensor>(a: Tensor, b: Tensor): T;
  greaterEqualStrict<T extends Tensor>(a: T, b: T): T;
  neg<T extends Tensor>(x: T): T;
  ceil<T extends Tensor>(x: T): T;
  floor<T extends Tensor>(x: T): T;
  sign<T extends Tensor>(x: T): T;
  round<T extends Tensor>(x: T): T;
  exp<T extends Tensor>(x: T): T;
  expm1<T extends Tensor>(x: T): T;
  log<T extends Tensor>(x: T): T;
  log1p<T extends Tensor>(x: T): T;
  sqrt<T extends Tensor>(x: T): T;
  rsqrt<T extends Tensor>(x: T): T;
  square<T extends Tensor>(x: T): T;
  reciprocal<T extends Tensor>(x: T): T;
  abs<T extends Tensor>(x: T): T;
  clipByValue<T extends Tensor>(
      x: T, clipValueMin: number, clipValueMax: number): T;
  sigmoid<T extends Tensor>(x: T): T;
  logSigmoid<T extends Tensor>(x: T): T;
  softplus<T extends Tensor>(x: T): T;
  sin<T extends Tensor>(x: T): T;
  cos<T extends Tensor>(x: T): T;
  tan<T extends Tensor>(x: T): T;
  asin<T extends Tensor>(x: T): T;
  acos<T extends Tensor>(x: T): T;
  atan<T extends Tensor>(x: T): T;
  sinh<T extends Tensor>(x: T): T;
  cosh<T extends Tensor>(x: T): T;
  tanh<T extends Tensor>(x: T): T;
  asinh<T extends Tensor>(x: T): T;
  acosh<T extends Tensor>(x: T): T;
  atanh<T extends Tensor>(x: T): T;
  erf<T extends Tensor>(x: T): T;
  step<T extends Tensor>(x: T, alpha: number): T;
  relu<T extends Tensor>(x: T): T;
  elu<T extends Tensor>(x: T): T;
  selu<T extends Tensor>(x: T): T;
  leakyRelu<T extends Tensor>(x: T, alpha: number): T;
  prelu<T extends Tensor>(x: T, alpha: T): T;
  softmax<T extends Tensor>(logits: T, dim: number): T;
  image: {
    resizeBilinear<T extends Tensor3D|Tensor4D>(
        images: T, size: [number, number], alignCorners: boolean): T;
    resizeNearestNeighbor<T extends Tensor3D|Tensor4D>(
        images: T, size: [number, number], alignCorners: boolean): T;
  };
  conv1d<T extends Tensor2D|Tensor3D>(
      x: T, filter: Tensor3D, stride: number, pad: 'valid'|'same'|number,
      dataFormat: 'NWC'|'NCW', dilation: number,
      dimRoundingMode?: 'floor'|'round'|'ceil'): T;
  conv2d<T extends Tensor3D|Tensor4D>(
      x: T, filter: Tensor4D, strides: [number, number]|number,
      pad: 'valid'|'same'|number, dataFormat: 'NHWC'|'NCHW',
      dilations: [number, number]|number,
      dimRoundingMode?: 'floor'|'round'|'ceil'): T;
  conv2dTranspose<T extends Tensor3D|Tensor4D>(
      x: T, filter: Tensor4D,
      outputShape: [number, number, number, number]|[number, number, number],
      strides: [number, number]|number, pad: 'valid'|'same'|number,
      dimRoundingMode?: 'floor'|'round'|'ceil'): T;
  depthwiseConv2d<T extends Tensor3D|Tensor4D>(
      x: T, filter: Tensor4D, strides: [number, number]|number,
      pad: 'valid'|'same'|number, dataFormat: 'NHWC'|'NCHW',
      dilations: [number, number]|number,
      dimRoundingMode?: 'floor'|'round'|'ceil'): T;
  maxPool<T extends Tensor3D|Tensor4D>(
      x: T, filterSize: [number, number]|number,
      strides: [number, number]|number, pad: 'valid'|'same'|number,
      dimRoundingMode?: 'floor'|'round'|'ceil'): T;
  avgPool<T extends Tensor3D|Tensor4D>(
      x: T, filterSize: [number, number]|number,
      strides: [number, number]|number, pad: 'valid'|'same'|number,
      dimRoundingMode?: 'floor'|'round'|'ceil'): T;
  localResponseNormalization<T extends Tensor3D|Tensor4D>(
      x: T, depthRadius: number, bias: number, alpha: number, beta: number): T;
  unsortedSegmentSum<T extends Tensor>(
      x: T, segmentIds: Tensor1D, numSegments: number): T;
  batchToSpaceND<T extends Tensor>(
      x: T, blockShape: number[], crops: number[][]): T;
  spaceToBatchND<T extends Tensor>(
      x: T, blockShape: number[], paddings: number[][]): T;
}

// For tracking tensor creation and disposal.
let trackerFn: () => TensorTracker = null;
// Used by chaining methods to call into ops.
let opHandler: OpHandler = null;

/**
 * An external consumer can register itself as the tensor tracker. This way
 * the Tensor class can notify the tracker for every tensor created and
 * disposed.
 */
export function setTensorTracker(fn: () => TensorTracker) {
  trackerFn = fn;
}

/**
 * An external consumer can register itself as the op handler. This way the
 * Tensor class can have chaining methods that call into ops via the op handler.
 */
export function setOpHandler(handler: OpHandler) {
  opHandler = handler;
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

/**
 * A `Tensor` object represents an immutable, multidimensional array of numbers
 * that has a shape and a data type.
 *
 * See `tensor` for details on how to create a `Tensor`.
 */
/** @doc {heading: 'Tensors', subheading: 'Classes'} */
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
    this.shape = shape.slice();
    this.dtype = dtype || 'float32';
    this.strides = computeStrides(shape);
    this.dataId = dataId != null ? dataId : {};
    this.id = Tensor.nextId++;
    this.rankType = (this.rank < 5 ? this.rank.toString() : 'higher') as R;
    trackerFn().registerTensor(this);
    if (values != null) {
      trackerFn().write(this.dataId, values);
    }
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

  /** Flatten a Tensor to a 1D array. */
  /** @doc {heading: 'Tensors', subheading: 'Classes'} */
  flatten(): Tensor1D {
    this.throwIfDisposed();
    return this.as1D();
  }

  /** Converts a size-1 `Tensor` to a `Scalar`. */
  /** @doc {heading: 'Tensors', subheading: 'Classes'} */
  asScalar(): Scalar {
    this.throwIfDisposed();
    util.assert(this.size === 1, 'The array must have only 1 element.');
    return this.reshape<Rank.R0>([]);
  }

  /** Converts a `Tensor` to a `Tensor1D`. */
  /** @doc {heading: 'Tensors', subheading: 'Classes'} */
  as1D(): Tensor1D {
    this.throwIfDisposed();
    return this.reshape<Rank.R1>([this.size]);
  }

  /**
   * Converts a `Tensor` to a `Tensor2D`.
   *
   * @param rows Number of rows in `Tensor2D`.
   * @param columns Number of columns in `Tensor2D`.
   */
  /** @doc {heading: 'Tensors', subheading: 'Classes'} */
  as2D(rows: number, columns: number): Tensor2D {
    this.throwIfDisposed();
    return this.reshape<Rank.R2>([rows, columns]);
  }

  /**
   * Converts a `Tensor` to a `Tensor3D`.
   *
   * @param rows Number of rows in `Tensor3D`.
   * @param columns Number of columns in `Tensor3D`.
   * @param depth Depth of `Tensor3D`.
   */
  /** @doc {heading: 'Tensors', subheading: 'Classes'} */
  as3D(rows: number, columns: number, depth: number): Tensor3D {
    this.throwIfDisposed();
    return this.reshape<Rank.R3>([rows, columns, depth]);
  }

  /**
   * Converts a `Tensor` to a `Tensor4D`.
   *
   * @param rows Number of rows in `Tensor4D`.
   * @param columns Number of columns in `Tensor4D`.
   * @param depth Depth of `Tensor4D`.
   * @param depth2 4th dimension of `Tensor4D`.
   */
  /** @doc {heading: 'Tensors', subheading: 'Classes'} */
  as4D(rows: number, columns: number, depth: number, depth2: number): Tensor4D {
    this.throwIfDisposed();
    return this.reshape<Rank.R4>([rows, columns, depth, depth2]);
  }

  /**
   * Casts a `Tensor` to a specified dtype.
   *
   * @param dtype Data-type to cast the tensor to.
   */
  /** @doc {heading: 'Tensors', subheading: 'Classes'} */
  asType<T extends this>(this: T, dtype: DataType): T {
    this.throwIfDisposed();
    return opHandler.cast(this, dtype) as T;
  }

  get rank(): number {
    return this.shape.length;
  }

  /**
   * Returns the value in the tensor at the provided location.
   * If using WebGL backend, this is a blocking call.
   * Prefer calling the `async data()[flatIndex]` method instead.
   *
   * @param locs The location indices.
   */
  get(...locs: number[]) {
    util.assert(
        locs.length === this.rank,
        'Number of coordinates in get() must match the rank of the tensor');
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

  /** Returns a `TensorBuffer` that holds the underlying data. */
  /** @doc {heading: 'Tensors', subheading: 'Classes'} */
  buffer(): TensorBuffer<R> {
    return opHandler.buffer(this.shape, this.dtype, this.dataSync());
  }

  /**
   * Asynchronously downloads the values from the `Tensor`. Returns a promise of
   * `TypedArray` that resolves when the computation has finished.
   */
  /** @doc {heading: 'Tensors', subheading: 'Classes'} */
  async data(): Promise<TypedArray> {
    this.throwIfDisposed();
    return trackerFn().read(this.dataId);
  }

  /**
   * Synchronously downloads the values from the `Tensor`. This blocks the UI
   * thread until the values are ready, which can cause performance issues.
   */
  /** @doc {heading: 'Tensors', subheading: 'Classes'} */
  dataSync(): TypedArray {
    this.throwIfDisposed();
    return trackerFn().readSync(this.dataId);
  }

  /**
   * Disposes `Tensor` from memory.
   */
  /** @doc {heading: 'Tensors', subheading: 'Classes'} */
  dispose(): void {
    if (this.isDisposed) {
      return;
    }
    trackerFn().disposeTensor(this);
    this.isDisposedInternal = true;
  }

  private isDisposedInternal = false;
  get isDisposed(): boolean {
    return this.isDisposedInternal;
  }

  private throwIfDisposed() {
    if (this.isDisposed) {
      throw new Error(`Tensor is disposed.`);
    }
  }

  /** Casts the array to type `float32` */
  /** @doc {heading: 'Tensors', subheading: 'Classes'} */
  toFloat<T extends this>(this: T): T {
    return this.asType('float32');
  }

  /** Casts the array to type `int32` */
  /** @doc {heading: 'Tensors', subheading: 'Classes'} */
  toInt() {
    return this.asType('int32');
  }

  /** Casts the array to type `bool` */
  /** @doc {heading: 'Tensors', subheading: 'Classes'} */
  toBool() {
    return this.asType('bool');
  }

  /**
   * Prints the `Tensor`. See `print` for details.
   *
   * @param verbose Whether to print verbose information about the tensor,
   *    including dtype and size.
   */
  /** @doc {heading: 'Tensors', subheading: 'Classes'} */
  print(verbose = false): void {
    return opHandler.print(this, verbose);
  }

  /**
   * Reshapes the tensor into the provided shape.
   * See `reshape` for more details.
   *
   * @param newShape An array of integers defining the output tensor shape.
   */
  /** @doc {heading: 'Tensors', subheading: 'Classes'} */
  reshape<R2 extends Rank>(newShape: ShapeMap[R2]): Tensor<R2> {
    this.throwIfDisposed();
    return opHandler.reshape(this, newShape);
  }

  /**
   * Reshapes the tensor into the shape of the provided tensor.
   *
   * @param x The tensor of required shape.
   */
  /** @doc {heading: 'Tensors', subheading: 'Classes'} */
  reshapeAs<T extends Tensor>(x: T): T {
    this.throwIfDisposed();
    return this.reshape(x.shape) as T;
  }

  /**
   * Returns a `Tensor` that has expanded rank, by inserting a dimension
   * into the tensor's shape. See `expandDims` for details.
   *
   * @param axis The dimension index at which to insert shape of 1. Defaults to
   *    0 (the first dimension).
   */
  /** @doc {heading: 'Tensors', subheading: 'Classes'} */
  expandDims<R2 extends Rank>(axis = 0): Tensor<R2> {
    return opHandler.expandDims(this, axis);
  }

  /**
   * Returns the cumulative sum of the `Tensor` along `axis`.
   *
   * @param axis The axis along which to sum. Optional. Defaults to 0.
   * @param exclusive Whether to perform exclusive cumulative sum. Defaults to
   *    false. If set to true then the sum of each tensor entry does not include
   *    its own value, but only the values previous to it along the specified
   *    axis.
   * @param reverse Whether to sum in the opposite direction. Defaults to
   *    false.
   */
  /** @doc {heading: 'Tensors', subheading: 'Classes'} */
  cumsum<T extends Tensor>(axis = 0, exclusive = false, reverse = false): T {
    return opHandler.cumsum(this, axis, exclusive, reverse);
  }

  /**
   * Returns a `Tensor` with dimensions of size 1 removed from the shape.
   * See `squeeze` for more details.
   *
   * @param axis A list of numbers. If specified, only squeezes the
   *    dimensions listed. The dimension index starts at 0. It is an error to
   *    squeeze a dimension that is not 1.
   */
  /** @doc {heading: 'Tensors', subheading: 'Classes'} */
  squeeze<T extends Tensor>(axis?: number[]): T {
    this.throwIfDisposed();
    return opHandler.squeeze(this, axis);
  }

  /** Returns a copy of the tensor. See `clone` for details. */
  /** @doc {heading: 'Tensors', subheading: 'Classes'} */
  clone<T extends Tensor>(this: T): T {
    this.throwIfDisposed();
    return opHandler.clone(this);
  }

  /** Returns a human-readable description of the tensor. Useful for logging. */
  /** @doc {heading: 'Tensors', subheading: 'Classes'} */
  toString(verbose = false): string {
    const vals = this.dataSync();
    return tensorToString(vals, this.shape, this.dtype, verbose);
  }

  // Below is chain API that is not exposed to docs to avoid repetition. To
  // expose a method, move it above this comment and add @doc and jsdoc.

  tile<T extends this>(this: T, reps: number[]): T {
    this.throwIfDisposed();
    return opHandler.tile(this, reps) as T;
  }

  gather<T extends this>(this: T, indices: Tensor1D, axis = 0): T {
    this.throwIfDisposed();
    return opHandler.gather(this, indices, axis) as T;
  }

  matMul(b: Tensor2D, transposeA = false, transposeB = false): Tensor2D {
    this.throwIfDisposed();
    return opHandler.matMul(this as Tensor2D, b, transposeA, transposeB);
  }
  dot(b: Tensor): Tensor {
    this.throwIfDisposed();
    return opHandler.dot(this, b);
  }
  norm(
      ord: number|'euclidean'|'fro' = 'euclidean', axis: number|number[] = null,
      keepDims = false): Tensor {
    this.throwIfDisposed();
    return opHandler.norm(this, ord, axis, keepDims);
  }
  slice<T extends Tensor<R>>(
      this: T, begin: number|number[], size?: number|number[]): T {
    this.throwIfDisposed();
    return opHandler.slice(this, begin, size);
  }
  reverse<T extends Tensor>(this: T, axis?: number|number[]): T {
    this.throwIfDisposed();
    return opHandler.reverse(this, axis);
  }
  concat<T extends Tensor>(this: T, x: T, axis = 0): T {
    this.throwIfDisposed();
    return opHandler.concat([this, x], axis);
  }
  stack(x: Tensor, axis = 0): Tensor {
    return opHandler.stack([this, x], axis);
  }
  unstack(x: Tensor, axis = 0): Tensor[] {
    return opHandler.unstack(this, axis);
  }
  pad<T extends Tensor>(
      this: T, paddings: Array<[number, number]>, constantValue = 0): T {
    return opHandler.pad(this, paddings, constantValue);
  }
  batchNormalization(
      mean: Tensor<R>|Tensor1D, variance: Tensor<R>|Tensor1D,
      varianceEpsilon = .001, scale?: Tensor<R>|Tensor1D,
      offset?: Tensor<R>|Tensor1D): Tensor<R> {
    this.throwIfDisposed();
    return opHandler.batchNormalization(
        this, mean, variance, varianceEpsilon, scale, offset);
  }

  // Reduction ops.
  all<T extends Tensor>(axis: number|number[] = null, keepDims = false): T {
    this.throwIfDisposed();
    return opHandler.all(this, axis, keepDims);
  }
  any<T extends Tensor>(axis: number|number[] = null, keepDims = false): T {
    this.throwIfDisposed();
    return opHandler.any(this, axis, keepDims);
  }
  logSumExp<T extends Tensor>(axis: number|number[] = null, keepDims = false):
      T {
    this.throwIfDisposed();
    return opHandler.logSumExp(this, axis, keepDims);
  }
  sum<T extends Tensor>(axis: number|number[] = null, keepDims = false): T {
    this.throwIfDisposed();
    return opHandler.sum(this, axis, keepDims);
  }
  mean<T extends Tensor>(axis: number|number[] = null, keepDims = false): T {
    this.throwIfDisposed();
    return opHandler.mean(this, axis, keepDims);
  }
  min<T extends Tensor>(axis: number|number[] = null, keepDims = false): T {
    this.throwIfDisposed();
    return opHandler.min(this, axis, keepDims);
  }
  max<T extends Tensor>(axis: number|number[] = null, keepDims = false): T {
    this.throwIfDisposed();
    return opHandler.max(this, axis, keepDims);
  }
  argMin<T extends Tensor>(axis: number = null): T {
    this.throwIfDisposed();
    return opHandler.argMin(this, axis);
  }
  argMax<T extends Tensor>(axis: number = null): T {
    this.throwIfDisposed();
    return opHandler.argMax(this, axis);
  }

  // Transformations
  cast<T extends this>(dtype: DataType): T {
    this.throwIfDisposed();
    return opHandler.cast(this as T, dtype) as T;
  }

  // Binary ops.

  add<T extends Tensor>(x: Tensor): T {
    this.throwIfDisposed();
    return opHandler.add(this, x);
  }
  addStrict<T extends this>(this: T, x: T): T {
    this.throwIfDisposed();
    return opHandler.addStrict(this, x) as T;
  }
  sub<T extends Tensor>(x: Tensor): T {
    this.throwIfDisposed();
    return opHandler.sub(this, x);
  }
  subStrict<T extends this>(this: T, x: T): T {
    this.throwIfDisposed();
    return opHandler.subStrict(this, x) as T;
  }
  pow<T extends Tensor>(this: T, exp: Tensor): T {
    this.throwIfDisposed();
    return opHandler.pow(this, exp);
  }
  powStrict(exp: Tensor): Tensor<R> {
    this.throwIfDisposed();
    return opHandler.powStrict(this, exp);
  }
  mul<T extends Tensor>(x: Tensor): T {
    this.throwIfDisposed();
    return opHandler.mul(this, x);
  }
  mulStrict<T extends this>(this: T, x: T): T {
    this.throwIfDisposed();
    return opHandler.mulStrict(this, x) as T;
  }
  div<T extends Tensor>(x: Tensor): T {
    this.throwIfDisposed();
    return opHandler.div(this, x);
  }
  floorDiv<T extends Tensor>(x: Tensor): T {
    this.throwIfDisposed();
    return opHandler.floorDiv(this, x);
  }
  divStrict<T extends this>(this: T, x: T): T {
    this.throwIfDisposed();
    return opHandler.divStrict(this, x) as T;
  }
  minimum<T extends Tensor>(x: Tensor): T {
    this.throwIfDisposed();
    return opHandler.minimum(this, x);
  }
  minimumStrict<T extends this>(this: T, x: T): T {
    this.throwIfDisposed();
    return opHandler.minimumStrict(this, x) as T;
  }
  maximum<T extends Tensor>(x: Tensor): T {
    this.throwIfDisposed();
    return opHandler.maximum(this, x);
  }
  maximumStrict<T extends this>(this: T, x: T): T {
    this.throwIfDisposed();
    return opHandler.maximumStrict(this, x) as T;
  }
  mod<T extends Tensor>(x: Tensor): T {
    this.throwIfDisposed();
    return opHandler.mod(this, x);
  }
  modStrict<T extends this>(this: T, x: T): T {
    this.throwIfDisposed();
    return opHandler.modStrict(this, x) as T;
  }
  squaredDifference<T extends Tensor>(x: Tensor): T {
    this.throwIfDisposed();
    return opHandler.squaredDifference(this, x);
  }
  squaredDifferenceStrict<T extends this>(this: T, x: T): T {
    this.throwIfDisposed();
    return opHandler.squaredDifferenceStrict(this, x) as T;
  }
  transpose<T extends Tensor>(this: T, perm?: number[]): T {
    this.throwIfDisposed();
    return opHandler.transpose(this, perm);
  }

  // Compare ops.

  notEqual<T extends Tensor>(x: Tensor): T {
    this.throwIfDisposed();
    return opHandler.notEqual(this, x);
  }
  notEqualStrict<T extends this>(this: T, x: T): T {
    this.throwIfDisposed();
    return opHandler.notEqualStrict(this, x) as T;
  }
  less<T extends Tensor>(x: Tensor): T {
    this.throwIfDisposed();
    return opHandler.less(this, x);
  }
  lessStrict<T extends this>(this: T, x: T): T {
    this.throwIfDisposed();
    return opHandler.lessStrict(this, x) as T;
  }
  equal<T extends Tensor>(x: Tensor): T {
    this.throwIfDisposed();
    return opHandler.equal(this, x);
  }
  equalStrict<T extends this>(this: T, x: T): T {
    this.throwIfDisposed();
    return opHandler.equalStrict(this, x) as T;
  }
  lessEqual<T extends Tensor>(x: Tensor): T {
    this.throwIfDisposed();
    return opHandler.lessEqual(this, x);
  }
  lessEqualStrict<T extends this>(this: T, x: T): T {
    this.throwIfDisposed();
    return opHandler.lessEqualStrict(this, x) as T;
  }
  greater<T extends Tensor>(x: Tensor): T {
    this.throwIfDisposed();
    return opHandler.greater(this, x);
  }
  greaterStrict<T extends this>(this: T, x: T): T {
    this.throwIfDisposed();
    return opHandler.greaterStrict(this, x) as T;
  }
  greaterEqual<T extends Tensor>(x: Tensor): T {
    this.throwIfDisposed();
    return opHandler.greaterEqual(this, x);
  }
  greaterEqualStrict<T extends this>(this: T, x: T): T {
    this.throwIfDisposed();
    return opHandler.greaterEqualStrict(this, x) as T;
  }

  // Compare ops.
  logicalAnd(x: Tensor): Tensor {
    this.throwIfDisposed();
    return opHandler.logicalAnd(this, x);
  }
  logicalOr(x: Tensor): Tensor {
    this.throwIfDisposed();
    return opHandler.logicalOr(this, x);
  }
  logicalNot<T extends Tensor>(this: T): T {
    this.throwIfDisposed();
    return opHandler.logicalNot(this);
  }
  logicalXor(x: Tensor): Tensor {
    this.throwIfDisposed();
    return opHandler.logicalXor(this, x);
  }
  where(condition: Tensor, x: Tensor): Tensor {
    this.throwIfDisposed();
    return opHandler.where(condition, this, x);
  }

  // Unary ops.
  neg<T extends Tensor>(this: T): T {
    this.throwIfDisposed();
    return opHandler.neg(this);
  }
  ceil<T extends Tensor>(this: T): T {
    this.throwIfDisposed();
    return opHandler.ceil(this);
  }
  floor<T extends Tensor>(this: T): T {
    this.throwIfDisposed();
    return opHandler.floor(this);
  }
  sign<T extends Tensor>(this: T): T {
    this.throwIfDisposed();
    return opHandler.sign(this);
  }
  exp<T extends Tensor>(this: T): T {
    this.throwIfDisposed();
    return opHandler.exp(this);
  }
  expm1<T extends Tensor>(this: T): T {
    this.throwIfDisposed();
    return opHandler.expm1(this);
  }
  log<T extends Tensor>(this: T): T {
    this.throwIfDisposed();
    return opHandler.log(this);
  }
  log1p<T extends Tensor>(this: T): T {
    this.throwIfDisposed();
    return opHandler.log1p(this);
  }
  sqrt<T extends Tensor>(this: T): T {
    this.throwIfDisposed();
    return opHandler.sqrt(this);
  }
  rsqrt<T extends Tensor>(this: T): T {
    this.throwIfDisposed();
    return opHandler.rsqrt(this);
  }
  square<T extends Tensor>(this: T): T {
    this.throwIfDisposed();
    return opHandler.square(this);
  }
  reciprocal<T extends Tensor>(this: T): T {
    this.throwIfDisposed();
    return opHandler.reciprocal(this);
  }
  abs<T extends Tensor>(this: T): T {
    this.throwIfDisposed();
    return opHandler.abs(this);
  }
  clipByValue(min: number, max: number): Tensor<R> {
    this.throwIfDisposed();
    return opHandler.clipByValue(this, min, max);
  }
  relu<T extends Tensor>(this: T): T {
    this.throwIfDisposed();
    return opHandler.relu(this);
  }
  elu<T extends Tensor>(this: T): T {
    this.throwIfDisposed();
    return opHandler.elu(this);
  }
  selu<T extends Tensor>(this: T): T {
    this.throwIfDisposed();
    return opHandler.selu(this);
  }
  leakyRelu(alpha = 0.2): Tensor<R> {
    this.throwIfDisposed();
    return opHandler.leakyRelu(this, alpha);
  }
  prelu(alpha: Tensor<R>): Tensor<R> {
    this.throwIfDisposed();
    return opHandler.prelu(this, alpha);
  }
  sigmoid<T extends Tensor>(this: T): T {
    this.throwIfDisposed();
    return opHandler.sigmoid(this);
  }
  logSigmoid<T extends Tensor>(this: T): T {
    this.throwIfDisposed();
    return opHandler.logSigmoid(this);
  }
  softplus<T extends Tensor>(this: T): T {
    this.throwIfDisposed();
    return opHandler.softplus(this);
  }
  sin<T extends Tensor>(this: T): T {
    this.throwIfDisposed();
    return opHandler.sin(this);
  }
  cos<T extends Tensor>(this: T): T {
    this.throwIfDisposed();
    return opHandler.cos(this);
  }
  tan<T extends Tensor>(this: T): T {
    this.throwIfDisposed();
    return opHandler.tan(this);
  }
  asin<T extends Tensor>(this: T): T {
    this.throwIfDisposed();
    return opHandler.asin(this);
  }
  acos<T extends Tensor>(this: T): T {
    this.throwIfDisposed();
    return opHandler.acos(this);
  }
  atan<T extends Tensor>(this: T): T {
    this.throwIfDisposed();
    return opHandler.atan(this);
  }
  sinh<T extends Tensor>(this: T): T {
    this.throwIfDisposed();
    return opHandler.sinh(this);
  }
  cosh<T extends Tensor>(this: T): T {
    this.throwIfDisposed();
    return opHandler.cosh(this);
  }
  tanh<T extends Tensor>(this: T): T {
    this.throwIfDisposed();
    return opHandler.tanh(this);
  }
  asinh<T extends Tensor>(this: T): T {
    this.throwIfDisposed();
    return opHandler.asinh(this);
  }
  acosh<T extends Tensor>(this: T): T {
    this.throwIfDisposed();
    return opHandler.acosh(this);
  }
  atanh<T extends Tensor>(this: T): T {
    this.throwIfDisposed();
    return opHandler.atanh(this);
  }
  erf<T extends Tensor>(this: T): T {
    this.throwIfDisposed();
    return opHandler.erf(this);
  }
  round<T extends Tensor>(this: T): T {
    this.throwIfDisposed();
    return opHandler.round(this);
  }
  step<T extends Tensor>(this: T, alpha = 0.0): T {
    this.throwIfDisposed();
    return opHandler.step(this, alpha);
  }
  softmax<T extends this>(this: T, dim = -1): T {
    this.throwIfDisposed();
    return opHandler.softmax(this, dim) as T;
  }

  // Image ops.
  resizeBilinear<T extends Tensor3D|Tensor4D>(
      this: T, newShape2D: [number, number], alignCorners = false): T {
    (this as Tensor).throwIfDisposed();
    return opHandler.image.resizeBilinear(this, newShape2D, alignCorners);
  }

  resizeNearestNeighbor<T extends Tensor3D|Tensor4D>(
      this: T, newShape2D: [number, number], alignCorners = false): T {
    (this as Tensor).throwIfDisposed();
    return opHandler.image.resizeNearestNeighbor(
        this, newShape2D, alignCorners);
  }

  // Convolutions.
  conv1d<T extends Tensor2D|Tensor3D>(
      this: T, filter: Tensor3D, stride: number, pad: 'valid'|'same'|number,
      dataFormat: 'NWC'|'NCW' = 'NWC', dilation = 1,
      dimRoundingMode?: 'floor'|'round'|'ceil'): T {
    (this as Tensor).throwIfDisposed();
    return opHandler.conv1d(
        this, filter, stride, pad, dataFormat, dilation, dimRoundingMode);
  }
  conv2d<T extends Tensor3D|Tensor4D>(
      this: T, filter: Tensor4D, strides: [number, number]|number,
      pad: 'valid'|'same'|number, dataFormat: 'NHWC'|'NCHW' = 'NHWC',
      dilations: [number, number]|number = [1, 1],
      dimRoundingMode?: 'floor'|'round'|'ceil'): T {
    (this as Tensor).throwIfDisposed();
    return opHandler.conv2d(
        this, filter, strides, pad, dataFormat, dilations, dimRoundingMode);
  }
  conv2dTranspose<T extends Tensor3D|Tensor4D>(
      this: T, filter: Tensor4D,
      outputShape: [number, number, number, number]|[number, number, number],
      strides: [number, number]|number, pad: 'valid'|'same'|number,
      dimRoundingMode?: 'floor'|'round'|'ceil'): T {
    (this as Tensor).throwIfDisposed();
    return opHandler.conv2dTranspose(
        this, filter, outputShape, strides, pad, dimRoundingMode);
  }
  depthwiseConv2D<T extends Tensor3D|Tensor4D>(
      this: T, filter: Tensor4D, strides: [number, number]|number,
      pad: 'valid'|'same'|number, dataFormat: 'NHWC'|'NCHW' = 'NHWC',
      dilations: [number, number]|number = [1, 1],
      dimRoundingMode?: 'floor'|'round'|'ceil'): T {
    (this as Tensor).throwIfDisposed();
    return opHandler.depthwiseConv2d(
        this, filter, strides, pad, dataFormat, dilations, dimRoundingMode);
  }

  // Pooling.
  avgPool<T extends Tensor3D|Tensor4D>(
      this: T, filterSize: [number, number]|number,
      strides: [number, number]|number, pad: 'valid'|'same'|number,
      dimRoundingMode?: 'floor'|'round'|'ceil'): T {
    (this as Tensor).throwIfDisposed();
    return opHandler.avgPool(this, filterSize, strides, pad, dimRoundingMode);
  }
  maxPool<T extends Tensor3D|Tensor4D>(
      this: T, filterSize: [number, number]|number,
      strides: [number, number]|number, pad: 'valid'|'same'|number,
      dimRoundingMode?: 'floor'|'round'|'ceil'): T {
    (this as Tensor).throwIfDisposed();
    return opHandler.maxPool(this, filterSize, strides, pad, dimRoundingMode);
  }
  localResponseNormalization<T extends Tensor3D|Tensor4D>(
      this: T, radius = 5, bias = 1, alpha = 1, beta = 0.5): T {
    return opHandler.localResponseNormalization(
        this, radius, bias, alpha, beta);
  }

  variable(trainable = true, name?: string, dtype?: DataType): Variable<R> {
    this.throwIfDisposed();
    return Variable.variable(this, trainable, name, dtype);
  }

  unsortedSegmentSum<T extends Tensor>(
      this: T, segmentIds: Tensor1D, numSegments: number): T {
    this.throwIfDisposed();
    return opHandler.unsortedSegmentSum(this, segmentIds, numSegments);
  }

  batchToSpaceND<T extends Tensor>(
      this: T, blockShape: number[], crops: number[][]): T {
    this.throwIfDisposed();
    return opHandler.batchToSpaceND(this, blockShape, crops);
  }

  spaceToBatchND<T extends Tensor>(
      this: T, blockShape: number[], paddings: number[][]): T {
    this.throwIfDisposed();
    return opHandler.spaceToBatchND(this, blockShape, paddings);
  }
}
Object.defineProperty(Tensor, Symbol.hasInstance, {
  value: (instance: Tensor) => {
    return !!instance && instance.shape != null && instance.dtype != null;
  }
});

/** @doclink Tensor */
export type Scalar = Tensor<Rank.R0>;
/** @doclink Tensor */
export type Tensor1D = Tensor<Rank.R1>;
/** @doclink Tensor */
export type Tensor2D = Tensor<Rank.R2>;
/** @doclink Tensor */
export type Tensor3D = Tensor<Rank.R3>;
/** @doclink Tensor */
export type Tensor4D = Tensor<Rank.R4>;
/** @doclink Tensor */
export type Tensor5D = Tensor<Rank.R5>;
/** @doclink Tensor */
export type Tensor6D = Tensor<Rank.R6>;

/**
 * A mutable `Tensor`, useful for persisting state, e.g. for training.
 */
/** @doc {heading: 'Tensors', subheading: 'Classes'} */
export class Variable<R extends Rank = Rank> extends Tensor<R> {
  private static nextVarId = 0;
  name: string;

  /**
   * Private constructor since we can not add logic before calling `super()`.
   * Instead, we expose static `Variable.variable` method below, which will be
   * added to global namespace.
   */
  private constructor(
      initialValue: Tensor<R>, public trainable = true, name?: string) {
    super(
        initialValue.shape, initialValue.dtype, null /* values */,
        initialValue.dataId);
    this.name = name;
    if (this.name == null) {
      this.name = Variable.nextVarId.toString();
      Variable.nextVarId++;
    }
    try {
      trackerFn().registerVariable(this);
    } catch (ex) {
      trackerFn().disposeTensor(this);
      throw ex;
    }
  }

  /**
   * Creates a new variable with the provided initial value.
   * ```js
   * const x = tf.variable(tf.tensor([1, 2, 3]));
   * x.assign(tf.tensor([4, 5, 6]));
   *
   * x.print();
   * ```
   *
   * @param initialValue Initial value for the tensor.
   * @param trainable If true, optimizers are allowed to update it.
   * @param name Name of the variable. Defaults to a unique id.
   * @param dtype If set, initialValue will be converted to the given type.
   */
  /** @doc {heading: 'Tensors', subheading: 'Creation'} */
  static variable<R extends Rank>(
      initialValue: Tensor<R>, trainable = true, name?: string,
      dtype?: DataType): Variable<R> {
    if (dtype != null && dtype !== initialValue.dtype) {
      initialValue = initialValue.asType(dtype) as Tensor<R>;
    }
    return new Variable(initialValue, trainable, name);
  }

  /**
   * Assign a new `Tensor` to this variable. The new `Tensor` must have the
   * same shape and dtype as the old `Tensor`.
   *
   * @param newValue New tensor to be assigned to this variable.
   */
  /** @doc {heading: 'Tensors', subheading: 'Classes'} */
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
    trackerFn().disposeTensor(this);
    this.dataId = newValue.dataId;
    trackerFn().registerTensor(this);
  }
}
Object.defineProperty(Variable, Symbol.hasInstance, {
  value: (instance: Variable) => {
    return instance instanceof Tensor && instance.assign != null &&
        instance.assign instanceof Function;
  }
});

const variable = Variable.variable;
export {variable};
