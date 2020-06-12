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
import {ArrayMap, BackendValues, DataType, DataTypeMap, DataValues, NumericDataType, Rank, ShapeMap, SingleValueMap, TensorLike, TensorLike1D, TypedArray} from './types';
import * as util from './util';
import {computeStrides, toNestedArray} from './util';

export interface TensorData<D extends DataType> {
  dataId?: DataId;
  values?: DataTypeMap[D];
}

// This interface mimics KernelBackend (in backend.ts), which would create a
// circular dependency if imported.
export interface Backend {}

/**
 * A mutable object, similar to `tf.Tensor`, that allows users to set values
 * at locations before converting to an immutable `tf.Tensor`.
 *
 * See `tf.buffer` for creating a tensor buffer.
 */
/** @doc {heading: 'Tensors', subheading: 'Classes'} */
export class TensorBuffer<R extends Rank, D extends DataType = 'float32'> {
  size: number;
  shape: ShapeMap[R];
  strides: number[];
  values: DataTypeMap[D];

  constructor(shape: ShapeMap[R], public dtype: D, values?: DataTypeMap[D]) {
    this.shape = shape.slice() as ShapeMap[R];
    this.size = util.sizeFromShape(shape);

    if (values != null) {
      const n = values.length;
      util.assert(
          n === this.size,
          () => `Length of values '${n}' does not match the size ` +
              `inferred by the shape '${this.size}'.`);
    }
    if (dtype === 'complex64') {
      throw new Error(
          `complex64 dtype TensorBuffers are not supported. Please create ` +
          `a TensorBuffer for the real and imaginary parts separately and ` +
          `call tf.complex(real, imag).`);
    }
    this.values = values || util.getArrayFromDType(dtype, this.size);
    this.strides = computeStrides(shape);
  }

  /**
   * Sets a value in the buffer at a given location.
   *
   * @param value The value to set.
   * @param locs  The location indices.
   */
  /** @doc {heading: 'Tensors', subheading: 'Creation'} */
  set(value: SingleValueMap[D], ...locs: number[]): void {
    if (locs.length === 0) {
      locs = [0];
    }
    util.assert(
        locs.length === this.rank,
        () => `The number of provided coordinates (${locs.length}) must ` +
            `match the rank (${this.rank})`);

    const index = this.locToIndex(locs);
    this.values[index] = value as number;
  }

  /**
   * Returns the value in the buffer at the provided location.
   *
   * @param locs The location indices.
   */
  /** @doc {heading: 'Tensors', subheading: 'Creation'} */
  get(...locs: number[]): SingleValueMap[D] {
    if (locs.length === 0) {
      locs = [0];
    }
    let i = 0;
    for (const loc of locs) {
      if (loc < 0 || loc >= this.shape[i]) {
        const msg = `Requested out of range element at ${locs}. ` +
            `  Buffer shape=${this.shape}`;
        throw new Error(msg);
      }
      i++;
    }
    let index = locs[locs.length - 1];
    for (let i = 0; i < locs.length - 1; ++i) {
      index += this.strides[i] * locs[i];
    }
    return this.values[index] as SingleValueMap[D];
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
   * Creates an immutable `tf.Tensor` object from the buffer.
   */
  /** @doc {heading: 'Tensors', subheading: 'Creation'} */
  toTensor(): Tensor<R> {
    return trackerFn().makeTensor(this.values, this.shape, this.dtype) as
        Tensor<R>;
  }
}

export interface TensorTracker {
  makeTensor(
      values: DataValues, shape: number[], dtype: DataType,
      backend?: Backend): Tensor;
  makeVariable(
      initialValue: Tensor, trainable?: boolean, name?: string,
      dtype?: DataType): Variable;
  incRef(a: Tensor, backend: Backend): void;
  disposeTensor(t: Tensor): void;
  disposeVariable(v: Variable): void;
  read(dataId: DataId): Promise<BackendValues>;
  readSync(dataId: DataId): BackendValues;
}

/**
 * The Tensor class calls into this handler to delegate chaining operations.
 */
export interface OpHandler {
  cast<T extends Tensor>(x: T, dtype: DataType): T;
  buffer<R extends Rank, D extends DataType>(
      shape: ShapeMap[R], dtype: D,
      values?: DataTypeMap[D]): TensorBuffer<R, D>;
  print<T extends Tensor>(x: T, verbose: boolean): void;
  reshape<R2 extends Rank>(x: Tensor, shape: ShapeMap[R2]): Tensor<R2>;
  expandDims<R2 extends Rank>(x: Tensor, axis: number): Tensor<R2>;
  squeeze<T extends Tensor>(x: Tensor, axis?: number[]): T;
  clone<T extends Tensor>(x: T): T;
  gather<T extends Tensor>(x: T, indices: Tensor|TensorLike, axis: number): T;
  norm(
      x: Tensor, ord: number|'euclidean'|'fro', axis: number|number[],
      keepDims: boolean): Tensor;
  slice<R extends Rank, T extends Tensor<R>>(
      x: T, begin: number|number[], size?: number|number[]): T;
  reverse<T extends Tensor>(x: T, axis?: number|number[]): T;
  stack<T extends Tensor>(tensors: Array<T|TensorLike>, axis: number): Tensor;
  unstack<T extends Tensor>(value: T, axis: number): Tensor[];
  all<T extends Tensor>(x: Tensor, axis: number|number[], keepDims: boolean): T;
  any<T extends Tensor>(x: Tensor, axis: number|number[], keepDims: boolean): T;
  logSumExp<T extends Tensor>(
      x: Tensor, axis: number|number[], keepDims: boolean): T;
  sum<T extends Tensor>(x: Tensor, axis: number|number[], keepDims: boolean): T;
  prod<T extends Tensor>(x: Tensor, axis: number|number[], keepDims: boolean):
      T;
  mean<T extends Tensor>(x: Tensor, axis: number|number[], keepDims: boolean):
      T;
  min<T extends Tensor>(x: Tensor, axis: number|number[], keepDims: boolean): T;
  argMin<T extends Tensor>(x: Tensor, axis: number): T;
  argMax<T extends Tensor>(x: Tensor, axis: number): T;
  addStrict<T extends Tensor>(a: T, b: T|TensorLike): T;
  subStrict<T extends Tensor>(a: T, b: T|TensorLike): T;
  powStrict<T extends Tensor>(base: T, exp: Tensor|TensorLike): T;
  mulStrict<T extends Tensor>(a: T, b: T|TensorLike): T;
  divStrict<T extends Tensor>(a: T, b: T|TensorLike): T;
  modStrict<T extends Tensor>(a: T, b: T|TensorLike): T;
  minimumStrict<T extends Tensor>(a: T, b: T|TensorLike): T;
  maximumStrict<T extends Tensor>(a: T, b: T|TensorLike): T;
  squaredDifferenceStrict<T extends Tensor>(a: T, b: T|TensorLike): T;
  logicalNot<T extends Tensor>(x: T): T;
  logicalAnd<T extends Tensor>(a: Tensor, b: Tensor|TensorLike): T;
  logicalOr<T extends Tensor>(a: Tensor, b: Tensor|TensorLike): T;
  logicalXor<T extends Tensor>(a: Tensor, b: Tensor|TensorLike): T;
  where<T extends Tensor>(condition: Tensor|TensorLike, a: T, b: T|TensorLike):
      T;
  notEqualStrict<T extends Tensor>(a: T, b: T|TensorLike): T;
  lessStrict<T extends Tensor>(a: T, b: T|TensorLike): T;
  equalStrict<T extends Tensor>(a: T, b: T|TensorLike): T;
  lessEqualStrict<T extends Tensor>(a: T, b: T|TensorLike): T;
  greaterStrict<T extends Tensor>(a: T, b: T|TensorLike): T;
  greaterEqualStrict<T extends Tensor>(a: T, b: T|TensorLike): T;
  neg<T extends Tensor>(x: T): T;
  ceil<T extends Tensor>(x: T): T;
  floor<T extends Tensor>(x: T): T;
  sign<T extends Tensor>(x: T): T;
  isNaN<T extends Tensor>(x: T): T;
  isInf<T extends Tensor>(x: T): T;
  isFinite<T extends Tensor>(x: T): T;
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
  zerosLike<T extends Tensor>(x: T): T;
  onesLike<T extends Tensor>(x: T): T;
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
  softmax<T extends Tensor>(logits: T, dim: number): T;
  logSoftmax<T extends Tensor>(logits: T, axis: number): T;
  image: {
    resizeBilinear<T extends Tensor3D|Tensor4D>(
        images: T, size: [number, number], alignCorners: boolean): T;
    resizeNearestNeighbor<T extends Tensor3D|Tensor4D>(
        images: T, size: [number, number], alignCorners: boolean): T;
  };
  unsortedSegmentSum<T extends Tensor>(
      x: T, segmentIds: Tensor1D|TensorLike1D, numSegments: number): T;
  topk<T extends Tensor>(x: T, k: number, sorted: boolean):
      {values: T, indices: T};
  stridedSlice(
      x: Tensor, begin: number[], end: number[], strides: number[],
      beginMask: number, endMask: number, ellipsisMask: number,
      newAxisMask: number, shrinkAxisMask: number): Tensor;
  spectral: {
    fft(x: Tensor): Tensor; ifft(x: Tensor): Tensor; rfft(x: Tensor): Tensor;
    irfft(x: Tensor): Tensor
  };
}

// For tracking tensor creation and disposal.
let trackerFn: () => TensorTracker = null;
// Used by chaining methods to call into ops.
let opHandler: OpHandler = null;
// Used to warn about deprecated methods.
let deprecationWarningFn: (msg: string) => void = null;
// This here so that we can use this method on dev branches and keep the
// functionality at master.
// tslint:disable-next-line:no-unused-expression
[deprecationWarningFn];

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
 * Tensor class can have chaining methods that call into ops via the op
 * handler.
 */
export function setOpHandler(handler: OpHandler) {
  opHandler = handler;
}

/**
 * Sets the deprecation warning function to be used by this file. This way the
 * Tensor class can be a leaf but still use the environment.
 */
export function setDeprecationWarningFn(fn: (msg: string) => void) {
  deprecationWarningFn = fn;
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

// Declare this namespace to make Tensor class augmentation work in google3.
export declare namespace Tensor {}
/**
 * A `tf.Tensor` object represents an immutable, multidimensional array of
 * numbers that has a shape and a data type.
 *
 * See `tf.tensor` for details on how to create a `tf.Tensor`.
 */
/** @doc {heading: 'Tensors', subheading: 'Classes'} */
export class Tensor<R extends Rank = Rank> {
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

  /** Whether this tensor has been globally kept. */
  kept = false;
  /** The id of the scope this tensor is being tracked in. */
  scopeId: number;

  /**
   * Number of elements to skip in each dimension when indexing. See
   * https://docs.scipy.org/doc/numpy/reference/generated/\
   * numpy.ndarray.strides.html
   */
  readonly strides: number[];

  constructor(shape: ShapeMap[R], dtype: DataType, dataId: DataId, id: number) {
    this.shape = shape.slice() as ShapeMap[R];
    this.dtype = dtype || 'float32';
    this.size = util.sizeFromShape(shape);
    this.strides = computeStrides(shape);
    this.dataId = dataId;
    this.id = id;
    this.rankType = (this.rank < 5 ? this.rank.toString() : 'higher') as R;
  }

  /** Flatten a Tensor to a 1D array. */
  /** @doc {heading: 'Tensors', subheading: 'Classes'} */
  flatten(): Tensor1D {
    this.throwIfDisposed();
    return this.as1D();
  }

  /** Converts a size-1 `tf.Tensor` to a `tf.Scalar`. */
  /** @doc {heading: 'Tensors', subheading: 'Classes'} */
  asScalar(): Scalar {
    this.throwIfDisposed();
    util.assert(this.size === 1, () => 'The array must have only 1 element.');
    return this.reshape<Rank.R0>([]);
  }

  /** Converts a `tf.Tensor` to a `tf.Tensor1D`. */
  /** @doc {heading: 'Tensors', subheading: 'Classes'} */
  as1D(): Tensor1D {
    this.throwIfDisposed();
    return this.reshape<Rank.R1>([this.size]);
  }

  /**
   * Converts a `tf.Tensor` to a `tf.Tensor2D`.
   *
   * @param rows Number of rows in `tf.Tensor2D`.
   * @param columns Number of columns in `tf.Tensor2D`.
   */
  /** @doc {heading: 'Tensors', subheading: 'Classes'} */
  as2D(rows: number, columns: number): Tensor2D {
    this.throwIfDisposed();
    return this.reshape<Rank.R2>([rows, columns]);
  }

  /**
   * Converts a `tf.Tensor` to a `tf.Tensor3D`.
   *
   * @param rows Number of rows in `tf.Tensor3D`.
   * @param columns Number of columns in `tf.Tensor3D`.
   * @param depth Depth of `tf.Tensor3D`.
   */
  /** @doc {heading: 'Tensors', subheading: 'Classes'} */
  as3D(rows: number, columns: number, depth: number): Tensor3D {
    this.throwIfDisposed();
    return this.reshape<Rank.R3>([rows, columns, depth]);
  }

  /**
   * Converts a `tf.Tensor` to a `tf.Tensor4D`.
   *
   * @param rows Number of rows in `tf.Tensor4D`.
   * @param columns Number of columns in `tf.Tensor4D`.
   * @param depth Depth of `tf.Tensor4D`.
   * @param depth2 4th dimension of `tf.Tensor4D`.
   */
  /** @doc {heading: 'Tensors', subheading: 'Classes'} */
  as4D(rows: number, columns: number, depth: number, depth2: number): Tensor4D {
    this.throwIfDisposed();
    return this.reshape<Rank.R4>([rows, columns, depth, depth2]);
  }

  /**
   * Converts a `tf.Tensor` to a `tf.Tensor5D`.
   *
   * @param rows Number of rows in `tf.Tensor5D`.
   * @param columns Number of columns in `tf.Tensor5D`.
   * @param depth Depth of `tf.Tensor5D`.
   * @param depth2 4th dimension of `tf.Tensor5D`.
   * @param depth3 5th dimension of 'tf.Tensor5D'
   */
  /** @doc {heading: 'Tensors', subheading: 'Classes'} */
  as5D(
      rows: number, columns: number, depth: number, depth2: number,
      depth3: number): Tensor5D {
    this.throwIfDisposed();
    return this.reshape<Rank.R5>([rows, columns, depth, depth2, depth3]);
  }

  /**
   * Casts a `tf.Tensor` to a specified dtype.
   *
   * @param dtype Data-type to cast the tensor to.
   */
  /** @doc {heading: 'Tensors', subheading: 'Classes'} */
  asType<T extends this>(this: T, dtype: DataType): T {
    this.throwIfDisposed();
    return opHandler.cast(this, dtype);
  }

  get rank(): number {
    return this.shape.length;
  }

  /**
   * Returns a promise of `tf.TensorBuffer` that holds the underlying data.
   */
  /** @doc {heading: 'Tensors', subheading: 'Classes'} */
  async buffer<D extends DataType = 'float32'>(): Promise<TensorBuffer<R, D>> {
    const vals = await this.data<D>();
    return opHandler.buffer(this.shape, this.dtype as D, vals);
  }

  /** Returns a `tf.TensorBuffer` that holds the underlying data. */
  /** @doc {heading: 'Tensors', subheading: 'Classes'} */
  bufferSync<D extends DataType = 'float32'>(): TensorBuffer<R, D> {
    return opHandler.buffer(this.shape, this.dtype as D, this.dataSync());
  }

  /**
   * Returns the tensor data as a nested array. The transfer of data is done
   * asynchronously.
   */
  /** @doc {heading: 'Tensors', subheading: 'Classes'} */
  async array(): Promise<ArrayMap[R]> {
    const vals = await this.data();
    return toNestedArray(this.shape, vals) as ArrayMap[R];
  }

  /**
   * Returns the tensor data as a nested array. The transfer of data is done
   * synchronously.
   */
  /** @doc {heading: 'Tensors', subheading: 'Classes'} */
  arraySync(): ArrayMap[R] {
    return toNestedArray(this.shape, this.dataSync()) as ArrayMap[R];
  }

  /**
   * Asynchronously downloads the values from the `tf.Tensor`. Returns a
   * promise of `TypedArray` that resolves when the computation has finished.
   */
  /** @doc {heading: 'Tensors', subheading: 'Classes'} */
  async data<D extends DataType = NumericDataType>(): Promise<DataTypeMap[D]> {
    this.throwIfDisposed();
    const data = trackerFn().read(this.dataId);
    if (this.dtype === 'string') {
      const bytes = await data as Uint8Array[];
      try {
        return bytes.map(b => util.decodeString(b)) as DataTypeMap[D];
      } catch {
        throw new Error(
            'Failed to decode the string bytes into utf-8. ' +
            'To get the original bytes, call tensor.bytes().');
      }
    }
    return data as Promise<DataTypeMap[D]>;
  }

  /**
   * Synchronously downloads the values from the `tf.Tensor`. This blocks the
   * UI thread until the values are ready, which can cause performance issues.
   */
  /** @doc {heading: 'Tensors', subheading: 'Classes'} */
  dataSync<D extends DataType = NumericDataType>(): DataTypeMap[D] {
    this.throwIfDisposed();
    const data = trackerFn().readSync(this.dataId);
    if (this.dtype === 'string') {
      try {
        return (data as Uint8Array[]).map(b => util.decodeString(b)) as
            DataTypeMap[D];
      } catch {
        throw new Error(
            'Failed to decode the string bytes into utf-8. ' +
            'To get the original bytes, call tensor.bytes().');
      }
    }
    return data as DataTypeMap[D];
  }

  /** Returns the underlying bytes of the tensor's data. */
  async bytes(): Promise<Uint8Array[]|Uint8Array> {
    this.throwIfDisposed();
    const data = await trackerFn().read(this.dataId);
    if (this.dtype === 'string') {
      return data as Uint8Array[];
    } else {
      return new Uint8Array((data as TypedArray).buffer);
    }
  }

  /**
   * Disposes `tf.Tensor` from memory.
   */
  /** @doc {heading: 'Tensors', subheading: 'Classes'} */
  dispose(): void {
    if (this.isDisposed) {
      return;
    }
    trackerFn().disposeTensor(this);
    this.isDisposedInternal = true;
  }

  protected isDisposedInternal = false;
  get isDisposed(): boolean {
    return this.isDisposedInternal;
  }

  throwIfDisposed() {
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
   * Prints the `tf.Tensor`. See `tf.print` for details.
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
   * See `tf.reshape` for more details.
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
   * Returns a `tf.Tensor` that has expanded rank, by inserting a dimension
   * into the tensor's shape. See `tf.expandDims` for details.
   *
   * @param axis The dimension index at which to insert shape of 1. Defaults to
   *     0 (the first dimension).
   */
  /** @doc {heading: 'Tensors', subheading: 'Classes'} */
  expandDims<R2 extends Rank>(axis = 0): Tensor<R2> {
    return opHandler.expandDims(this, axis);
  }

  /**
   * Returns a `tf.Tensor` with dimensions of size 1 removed from the shape.
   * See `tf.squeeze` for more details.
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

  /** Returns a copy of the tensor. See `tf.clone` for details. */
  /** @doc {heading: 'Tensors', subheading: 'Classes'} */
  clone<T extends Tensor>(this: T): T {
    this.throwIfDisposed();
    return opHandler.clone(this);
  }

  /**
   * Returns a human-readable description of the tensor. Useful for logging.
   */
  /** @doc {heading: 'Tensors', subheading: 'Classes'} */
  toString(verbose = false): string {
    const vals = this.dataSync();
    return tensorToString(vals, this.shape, this.dtype, verbose);
  }

  // Below is chain API that is not exposed to docs to avoid repetition. To
  // expose a method, move it above this comment and add @doc and jsdoc.

  gather<T extends this>(this: T, indices: Tensor|TensorLike, axis = 0): T {
    this.throwIfDisposed();
    return opHandler.gather(this, indices, axis);
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
  stack(x: Tensor, axis = 0): Tensor {
    return opHandler.stack([this, x], axis);
  }
  unstack(axis = 0): Tensor[] {
    return opHandler.unstack(this, axis);
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
  prod<T extends Tensor>(axis: number|number[] = null, keepDims = false): T {
    this.throwIfDisposed();
    return opHandler.prod(this, axis, keepDims);
  }
  mean<T extends Tensor>(axis: number|number[] = null, keepDims = false): T {
    this.throwIfDisposed();
    return opHandler.mean(this, axis, keepDims);
  }
  min<T extends Tensor>(axis: number|number[] = null, keepDims = false): T {
    this.throwIfDisposed();
    return opHandler.min(this, axis, keepDims);
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
    return opHandler.cast(this as T, dtype);
  }

  // Binary ops.
  /**
   * @deprecated strict variants of ops have been deprecated
   */
  addStrict<T extends this>(this: T, x: T|TensorLike): T {
    this.throwIfDisposed();
    return opHandler.addStrict(this, x);
  }
  /**
   * @deprecated strict variants of ops have been deprecated
   */
  subStrict<T extends this>(this: T, x: T|TensorLike): T {
    this.throwIfDisposed();
    return opHandler.subStrict(this, x);
  }
  /**
   * @deprecated strict variants of ops have been deprecated
   */
  powStrict(exp: Tensor|TensorLike): Tensor<R> {
    this.throwIfDisposed();
    return opHandler.powStrict(this, exp);
  }
  /**
   * @deprecated strict variants of ops have been deprecated
   */
  mulStrict<T extends this>(this: T, x: T|TensorLike): T {
    this.throwIfDisposed();
    return opHandler.mulStrict(this, x);
  }
  /**
   * @deprecated strict variants of ops have been deprecated
   */
  divStrict<T extends this>(this: T, x: T|TensorLike): T {
    this.throwIfDisposed();
    return opHandler.divStrict(this, x);
  }
  /**
   * @deprecated strict variants of ops have been deprecated
   */
  minimumStrict<T extends this>(this: T, x: T|TensorLike): T {
    this.throwIfDisposed();
    return opHandler.minimumStrict(this, x);
  }
  /**
   * @deprecated strict variants of ops have been deprecated
   */
  maximumStrict<T extends this>(this: T, x: T|TensorLike): T {
    this.throwIfDisposed();
    return opHandler.maximumStrict(this, x);
  }
  /**
   * @deprecated strict variants of ops have been deprecated
   */
  modStrict<T extends this>(this: T, x: T|TensorLike): T {
    this.throwIfDisposed();
    return opHandler.modStrict(this, x);
  }
  /**
   * @deprecated strict variants of ops have been deprecated
   */
  squaredDifferenceStrict<T extends this>(this: T, x: T|TensorLike): T {
    this.throwIfDisposed();
    return opHandler.squaredDifferenceStrict(this, x);
  }

  // Compare ops.
  /**
   * @deprecated strict variants of ops have been deprecated
   */
  notEqualStrict<T extends this>(this: T, x: T|TensorLike): T {
    this.throwIfDisposed();
    return opHandler.notEqualStrict(this, x);
  }
  /**
   * @deprecated strict variants of ops have been deprecated
   */
  lessStrict<T extends this>(this: T, x: T|TensorLike): T {
    this.throwIfDisposed();
    return opHandler.lessStrict(this, x);
  }
  /**
   * @deprecated strict variants of ops have been deprecated
   */
  equalStrict<T extends this>(this: T, x: T|TensorLike): T {
    this.throwIfDisposed();
    return opHandler.equalStrict(this, x);
  }
  /**
   * @deprecated strict variants of ops have been deprecated
   */
  lessEqualStrict<T extends this>(this: T, x: T|TensorLike): T {
    this.throwIfDisposed();
    return opHandler.lessEqualStrict(this, x);
  }
  /**
   * @deprecated strict variants of ops have been deprecated
   */
  greaterStrict<T extends this>(this: T, x: T|TensorLike): T {
    this.throwIfDisposed();
    return opHandler.greaterStrict(this, x);
  }
  /**
   * @deprecated strict variants of ops have been deprecated
   */
  greaterEqualStrict<T extends this>(this: T, x: T|TensorLike): T {
    this.throwIfDisposed();
    return opHandler.greaterEqualStrict(this, x);
  }

  // Compare ops.
  logicalAnd(x: Tensor|TensorLike): Tensor {
    this.throwIfDisposed();
    return opHandler.logicalAnd(this, x);
  }
  logicalOr(x: Tensor|TensorLike): Tensor {
    this.throwIfDisposed();
    return opHandler.logicalOr(this, x);
  }
  logicalNot<T extends Tensor>(this: T): T {
    this.throwIfDisposed();
    return opHandler.logicalNot(this);
  }
  logicalXor(x: Tensor|TensorLike): Tensor {
    this.throwIfDisposed();
    return opHandler.logicalXor(this, x);
  }
  where(condition: Tensor|TensorLike, x: Tensor|TensorLike): Tensor {
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
  isNaN<T extends Tensor>(this: T): T {
    this.throwIfDisposed();
    return opHandler.isNaN(this);
  }
  isInf<T extends Tensor>(this: T): T {
    this.throwIfDisposed();
    return opHandler.isInf(this);
  }
  isFinite<T extends Tensor>(this: T): T {
    this.throwIfDisposed();
    return opHandler.isFinite(this);
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
  zerosLike<T extends Tensor>(this: T): T {
    this.throwIfDisposed();
    return opHandler.zerosLike(this);
  }
  onesLike<T extends Tensor>(this: T): T {
    this.throwIfDisposed();
    return opHandler.onesLike(this);
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
    return opHandler.softmax(this, dim);
  }
  logSoftmax<T extends this>(this: T, axis = -1): T {
    this.throwIfDisposed();
    return opHandler.logSoftmax(this, axis);
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

  // Pooling.
  variable(trainable = true, name?: string, dtype?: DataType): Variable<R> {
    this.throwIfDisposed();
    return trackerFn().makeVariable(this, trainable, name, dtype) as
        Variable<R>;
  }

  unsortedSegmentSum<T extends Tensor>(
      this: T, segmentIds: Tensor1D|TensorLike1D, numSegments: number): T {
    this.throwIfDisposed();
    return opHandler.unsortedSegmentSum(this, segmentIds, numSegments);
  }

  topk<T extends Tensor>(this: T, k = 1, sorted = true):
      {values: T, indices: T} {
    this.throwIfDisposed();
    return opHandler.topk(this, k, sorted);
  }

  stridedSlice(
      this: Tensor, begin: number[], end: number[], strides: number[],
      beginMask = 0, endMask = 0, ellipsisMask = 0, newAxisMask = 0,
      shrinkAxisMask = 0): Tensor {
    this.throwIfDisposed();
    return opHandler.stridedSlice(
        this, begin, end, strides, beginMask, endMask, ellipsisMask,
        newAxisMask, shrinkAxisMask);
  }

  fft(this: Tensor): Tensor {
    this.throwIfDisposed();
    return opHandler.spectral.fft(this);
  }

  ifft(this: Tensor): Tensor {
    this.throwIfDisposed();
    return opHandler.spectral.ifft(this);
  }

  rfft(this: Tensor): Tensor {
    this.throwIfDisposed();
    return opHandler.spectral.rfft(this);
  }

  irfft(this: Tensor): Tensor {
    this.throwIfDisposed();
    return opHandler.spectral.irfft(this);
  }
}
Object.defineProperty(Tensor, Symbol.hasInstance, {
  value: (instance: Tensor) => {
    return !!instance && instance.dataId != null && instance.shape != null &&
        instance.dtype != null;
  }
});

export interface NumericTensor<R extends Rank = Rank> extends Tensor<R> {
  dtype: NumericDataType;
  dataSync<D extends DataType = NumericDataType>(): DataTypeMap[D];
  data<D extends DataType = NumericDataType>(): Promise<DataTypeMap[D]>;
}

export interface StringTensor<R extends Rank = Rank> extends Tensor<R> {
  dtype: 'string';
  dataSync<D extends DataType = 'string'>(): DataTypeMap[D];
  data<D extends DataType = 'string'>(): Promise<DataTypeMap[D]>;
}

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
 * A mutable `tf.Tensor`, useful for persisting state, e.g. for training.
 */
/** @doc {heading: 'Tensors', subheading: 'Classes'} */
export class Variable<R extends Rank = Rank> extends Tensor<R> {
  name: string;

  constructor(
      initialValue: Tensor<R>, public trainable: boolean, name: string,
      tensorId: number) {
    super(
        initialValue.shape, initialValue.dtype, initialValue.dataId, tensorId);
    this.name = name;
  }

  /**
   * Assign a new `tf.Tensor` to this variable. The new `tf.Tensor` must have
   * the same shape and dtype as the old `tf.Tensor`.
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
    trackerFn().incRef(this, null /* backend */);
  }

  dispose(): void {
    trackerFn().disposeVariable(this);
    this.isDisposedInternal = true;
  }
}

Object.defineProperty(Variable, Symbol.hasInstance, {
  value: (instance: Variable) => {
    return instance instanceof Tensor && instance.assign != null &&
        instance.assign instanceof Function;
  }
});
