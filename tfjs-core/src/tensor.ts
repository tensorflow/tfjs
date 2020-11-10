/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
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
import {ArrayMap, BackendValues, DataType, DataTypeMap, DataValues, NumericDataType, Rank, ShapeMap, SingleValueMap, TypedArray} from './types';
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
 *
 * @doc {heading: 'Tensors', subheading: 'Classes'}
 */
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
   *
   * @doc {heading: 'Tensors', subheading: 'Creation'}
   */
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
   *
   * @doc {heading: 'Tensors', subheading: 'Creation'}
   */
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
   *
   * @doc {heading: 'Tensors', subheading: 'Creation'}
   */
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
  clone<T extends Tensor>(x: T): T;
  // TODO(yassogba) bring reshape back?
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
 *
 * @doc {heading: 'Tensors', subheading: 'Classes'}
 */
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

  get rank(): number {
    return this.shape.length;
  }

  /**
   * Returns a promise of `tf.TensorBuffer` that holds the underlying data.
   *
   * @doc {heading: 'Tensors', subheading: 'Classes'}
   */
  async buffer<D extends DataType = 'float32'>(): Promise<TensorBuffer<R, D>> {
    const vals = await this.data<D>();
    return opHandler.buffer(this.shape, this.dtype as D, vals);
  }

  /**
   * Returns a `tf.TensorBuffer` that holds the underlying data.
   * @doc {heading: 'Tensors', subheading: 'Classes'}
   */
  bufferSync<D extends DataType = 'float32'>(): TensorBuffer<R, D> {
    return opHandler.buffer(this.shape, this.dtype as D, this.dataSync());
  }

  /**
   * Returns the tensor data as a nested array. The transfer of data is done
   * asynchronously.
   *
   * @doc {heading: 'Tensors', subheading: 'Classes'}
   */
  async array(): Promise<ArrayMap[R]> {
    const vals = await this.data();
    return toNestedArray(this.shape, vals) as ArrayMap[R];
  }

  /**
   * Returns the tensor data as a nested array. The transfer of data is done
   * synchronously.
   *
   * @doc {heading: 'Tensors', subheading: 'Classes'}
   */
  arraySync(): ArrayMap[R] {
    return toNestedArray(this.shape, this.dataSync()) as ArrayMap[R];
  }

  /**
   * Asynchronously downloads the values from the `tf.Tensor`. Returns a
   * promise of `TypedArray` that resolves when the computation has finished.
   *
   * @doc {heading: 'Tensors', subheading: 'Classes'}
   */
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
   *
   * @doc {heading: 'Tensors', subheading: 'Classes'}
   */
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
   *
   * @doc {heading: 'Tensors', subheading: 'Classes'}
   */
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

  /**
   * Prints the `tf.Tensor`. See `tf.print` for details.
   *
   * @param verbose Whether to print verbose information about the tensor,
   *    including dtype and size.
   *
   * @doc {heading: 'Tensors', subheading: 'Classes'}
   */
  print(verbose = false): void {
    return opHandler.print(this, verbose);
  }

  /**
   * Returns a copy of the tensor. See `tf.clone` for details.
   * @doc {heading: 'Tensors', subheading: 'Classes'}
   */
  clone<T extends Tensor>(this: T): T {
    this.throwIfDisposed();
    return opHandler.clone(this);
  }

  /**
   * Returns a human-readable description of the tensor. Useful for logging.
   *
   * @doc {heading: 'Tensors', subheading: 'Classes'}
   */
  toString(verbose = false): string {
    const vals = this.dataSync();
    return tensorToString(vals, this.shape, this.dtype, verbose);
  }

  cast<T extends this>(dtype: DataType): T {
    this.throwIfDisposed();
    return opHandler.cast(this as T, dtype);
  }
  variable(trainable = true, name?: string, dtype?: DataType): Variable<R> {
    this.throwIfDisposed();
    return trackerFn().makeVariable(this, trainable, name, dtype) as
        Variable<R>;
  }
}
Object.defineProperty(Tensor, Symbol.hasInstance, {
  value: (instance: Tensor) => {
    // Implementation note: we should use properties of the object that will be
    // defined before the constructor body has finished executing (methods).
    // This is because when this code is transpiled by babel, babel will call
    // classCallCheck before the constructor body is run.
    // See https://github.com/tensorflow/tfjs/issues/3384 for backstory.
    return !!instance && instance.data != null && instance.dataSync != null &&
        instance.throwIfDisposed != null;
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
 *
 * @doc {heading: 'Tensors', subheading: 'Classes'}
 */
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
   *
   * @doc {heading: 'Tensors', subheading: 'Classes'}
   */
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
