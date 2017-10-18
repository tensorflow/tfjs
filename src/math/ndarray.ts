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
import {GPGPUContext} from './webgl/gpgpu_context';
import {TextureType} from './webgl/tex_util';
import {TextureManager} from './webgl/texture_manager';
import * as webgl_util from './webgl/webgl_util';

// These global variables need to be initialized to null so that closure knows
// not to seal them.
/** @hidden */
export let GPGPU: GPGPUContext = null;
/** @hidden */
export let TEXTURE_MANAGER: TextureManager = null;

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
  values?: DataTypes[T];
  texture?: WebGLTexture;
  /** [rows, columns] shape of the texture. */
  textureShapeRC?: [number, number];
  textureType?: TextureType;
}

/** @hidden */
export function initializeGPU(
    gpgpu: GPGPUContext, textureManager: TextureManager) {
  GPGPU = gpgpu;
  TEXTURE_MANAGER = textureManager;
}

function throwIfGPUNotInitialized() {
  if (GPGPU == null || TEXTURE_MANAGER == null) {
    throw new Error('GPU not intialized.');
  }
}

export class NDArray<T extends keyof DataTypes = keyof DataTypes> {
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
  protected strides: number[];

  private ndarrayData: NDArrayData<T>;

  protected constructor(shape: number[], data: NDArrayData<T>, dtype: T) {
    // Sanity checks.
    util.assert(
        data.values != null || data.texture != null,
        'Either `values` or `texture` must be defined');

    util.assert(
        data.texture == null || (data.textureShapeRC != null),
        '`textureShape` must be defined when `texture` is defined');

    this.size = util.sizeFromShape(shape);

    if (data.values != null) {
      util.assert(
          this.size === data.values.length,
          'Constructing ndarray of shape (' + this.size + ') should match the' +
              ' length of values (' + data.values.length + ')');
    }

    this.shape = shape;

    if (data.textureType == null) {
      data.textureType = TextureType.DEFAULT;
    }
    this.ndarrayData = data;
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
  }

  /** Creates a ndarray of zeros with the specified shape. */
  static zeros<T extends keyof DataTypes = keyof DataTypes>(
      shape: number[], dtype?: T): NDArray<T> {
    const values = makeZerosTypedArray(util.sizeFromShape(shape), dtype);
    return NDArray.make(shape, {values}, dtype);
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
      shape: number[], data: NDArrayData<T>, dtype?: T): NDArray<T> {
    switch (shape.length) {
      case 0:
        return new Scalar(data, dtype);
      case 1:
        return new Array1D(data, dtype);
      case 2:
        return new Array2D(shape as [number, number], data, dtype);
      case 3:
        return new Array3D(shape as [number, number, number], data, dtype);
      case 4:
        return new Array4D(
            shape as [number, number, number, number], data, dtype);
      default:
        return new NDArray(shape, data, dtype);
    }
  }

  static fromPixels(
      pixels: ImageData|HTMLImageElement|HTMLCanvasElement|HTMLVideoElement,
      numChannels = 3): Array3D<'int32'> {
    if (numChannels > 4) {
      throw new Error(
          'Cannot construct NDArray with more than 4 channels from pixels.');
    }
    const shape: [number, number, number] =
        [pixels.height, pixels.width, numChannels];
    const textureShapeRC: [number, number] = [shape[0], shape[1]];
    const texture = TEXTURE_MANAGER.acquireTexture(textureShapeRC);
    const textureType = TextureType.RGBA_COLOR;

    GPGPU.uploadPixelDataToTexture(texture, pixels);

    return Array3D.make<'int32'>(
               shape, {texture, textureShapeRC, textureType}) as
        Array3D<'int32'>;
  }

  /** Reshapes the current ndarray into the provided shape. */
  reshape(newShape: number[]): NDArray<T> {
    newShape = util.inferFromImplicitShape(newShape, this.size);
    if (util.arraysEqual(this.shape, newShape)) {
      // No-op.
      return this;
    }

    util.assert(
        this.size === util.sizeFromShape(newShape),
        'new shape and old shape must have the same number of elements.');

    return NDArray.make(newShape, this.ndarrayData, this.dtype);
  }

  asScalar(): Scalar<T> {
    util.assert(this.size === 1, 'The array must have only 1 element.');
    return this.reshape([]);
  }

  as1D(): Array1D<T> {
    return this.reshape([this.size]) as Array1D<T>;
  }

  as2D(rows: number, columns: number): Array2D<T> {
    return this.reshape([rows, columns]) as Array2D<T>;
  }

  as3D(rows: number, columns: number, depth: number): Array3D<T> {
    return this.reshape([rows, columns, depth]) as Array3D<T>;
  }

  as4D(rows: number, columns: number, depth: number, depth2: number):
      Array4D<T> {
    return this.reshape([rows, columns, depth, depth2]) as Array4D<T>;
  }

  asType<G extends keyof DataTypes>(dtype: G): NDArray<G> {
    let newData: NDArrayData<T> = this.getData();
    if (newData.values != null) {
      newData = {values: toTypedArray(newData.values, dtype)};
    }
    return NDArray.make<G>(this.shape, newData, dtype);
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
    let index = locs[locs.length - 1];
    for (let i = 0; i < locs.length - 1; ++i) {
      index += this.strides[i] * locs[i];
    }
    this.getValues()[index] = value;
  }

  async val(...locs: number[]): Promise<number> {
    await this.data();
    return this.get(...locs);
  }

  locToIndex(locs: number[]): number {
    let index = locs[locs.length - 1];
    for (let i = 0; i < locs.length - 1; ++i) {
      index += this.strides[i] * locs[i];
    }
    return index;
  }

  indexToLoc(index: number): number[] {
    const locs: number[] = new Array(this.shape.length);
    for (let i = 0; i < locs.length - 1; ++i) {
      locs[i] = Math.floor(index / this.strides[i]);
      index -= locs[i] * this.strides[i];
    }
    locs[locs.length - 1] = index;
    return locs;
  }

  fill(value: number) {
    this.getValues().fill(value);
  }

  getData(): NDArrayData<T> {
    return this.ndarrayData;
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
  data(): Promise<DataTypes[T]> {
    return new Promise<DataTypes[T]>((resolve, reject) => {
      if (this.ndarrayData.values != null) {
        resolve(this.ndarrayData.values);
        return;
      }

      if (!ENV.get('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_ENABLED')) {
        resolve(this.getValues());
        return;
      }

      // Construct an empty query. We're just interested in getting a callback
      // when the GPU command queue has executed until this point in time.
      const queryFn = () => {};
      GPGPU.runQuery(queryFn).then(() => {
        resolve(this.getValues());
      });
    });
  }

  /**
   * Synchronously downloads the values from the NDArray. This blocks the UI
   * thread until the values are ready, which can cause performance issues.
   */
  dataSync(): DataTypes[T] {
    if (this.ndarrayData.values == null) {
      throwIfGPUNotInitialized();

      let values: Float32Array;
      if (this.ndarrayData.textureType === TextureType.DEFAULT) {
        values = GPGPU.downloadMatrixFromTexture(
            this.ndarrayData.texture, this.ndarrayData.textureShapeRC[0],
            this.ndarrayData.textureShapeRC[1]);
      } else {
        values = GPGPU.downloadMatrixFromRGBAColorTexture(
            this.ndarrayData.texture, this.ndarrayData.textureShapeRC[0],
            this.ndarrayData.textureShapeRC[1], this.shape[2]);
      }
      this.ndarrayData.values = convertFloat32ToDtype(values, this.dtype);
      this.disposeTexture();
    }
    return this.ndarrayData.values;
  }

  private uploadToGPU(preferredTexShape?: [number, number]) {
    throwIfGPUNotInitialized();
    this.ndarrayData.textureShapeRC =
        webgl_util.getTextureShapeFromLogicalShape(
            GPGPU.gl, this.shape, preferredTexShape);
    this.ndarrayData.texture =
        TEXTURE_MANAGER.acquireTexture(this.ndarrayData.textureShapeRC);
    this.ndarrayData.textureType = TextureType.DEFAULT;

    GPGPU.uploadMatrixToTexture(
        this.ndarrayData.texture, this.ndarrayData.textureShapeRC[0],
        // TODO(smilkov): Propagate the original typed array to gpgpu.
        this.ndarrayData.textureShapeRC[1],
        new Float32Array(this.ndarrayData.values));

    this.ndarrayData.values = null;
  }

  getTexture(preferredShapeRC?: [number, number]): WebGLTexture {
    if (this.ndarrayData.texture == null) {
      this.uploadToGPU(preferredShapeRC);
    }
    return this.ndarrayData.texture;
  }

  getTextureShapeRC(preferredShapeRC?: [number, number]): [number, number] {
    if (this.ndarrayData.textureShapeRC == null) {
      this.uploadToGPU(preferredShapeRC);
    }
    return this.ndarrayData.textureShapeRC;
  }

  dispose(): void {
    this.ndarrayData.values = null;
    this.shape = null;
    if (this.ndarrayData.texture != null) {
      this.disposeTexture();
    }
  }

  private disposeTexture() {
    throwIfGPUNotInitialized();
    TEXTURE_MANAGER.releaseTexture(
        this.ndarrayData.texture, this.ndarrayData.textureShapeRC);
    this.ndarrayData.texture = null;
    this.ndarrayData.textureShapeRC = null;
    this.ndarrayData.textureType = null;
  }

  inGPU(): boolean {
    return this.ndarrayData.texture != null;
  }

  equals(t: NDArray<T>): boolean {
    return this.dtype === t.dtype && util.arraysEqual(this.shape, t.shape) &&
        util.arraysEqual(this.getValues(), t.getValues());
  }

  static rand(shape: number[], randFunction: () => number): NDArray<'float32'> {
    const size = util.sizeFromShape(shape);
    const values = new Float32Array(size);
    for (let i = 0; i < size; i++) {
      values[i] = randFunction();
    }

    return NDArray.make(shape, {values});
  }

  static randNormal(shape: number[], mean = 0, stdDev = 1): NDArray<'float32'> {
    return NDArray.rand(shape, () => util.randGauss(mean, stdDev));
  }

  static randTruncatedNormal(shape: number[], mean = 0, stdDev = 1):
      NDArray<'float32'> {
    return NDArray.rand(shape, () => util.randGauss(mean, stdDev, true));
  }

  static randUniform(shape: number[], a: number, b: number):
      NDArray<'float32'> {
    return NDArray.rand(shape, () => util.randUniform(a, b));
  }
}

export class Scalar<T extends keyof DataTypes = keyof DataTypes> extends
    NDArray<T> {
  constructor(data: NDArrayData<T>, dtype: T) {
    if (data.texture != null) {
      data.textureShapeRC = [1, 1];
    }
    super([], data, dtype);
  }

  static new<T extends keyof DataTypes = keyof DataTypes>(
      value: number|boolean, dtype?: T) {
    const values = [value] as number[] | boolean[];
    return new Scalar({values: toTypedArray(values, dtype)}, dtype);
  }

  static ZERO = Scalar.new(0);
  static ONE = Scalar.new(1);
  static TWO = Scalar.new(2);
  static NEG_ONE = Scalar.new(-1);

  get(): number {
    return this.getValues()[0];
  }

  async val(): Promise<number> {
    await this.data();
    return this.get();
  }

  set(value: number) {
    this.getValues()[0] = value;
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

  constructor(data: NDArrayData<T>, dtype: T) {
    const shape = (data.values != null) ?
        [data.values.length] :
        [util.sizeFromShape(data.textureShapeRC)];
    super(shape, data, dtype);
  }

  static new<T extends keyof DataTypes = keyof DataTypes>(
      values: DataTypes[T]|number[]|boolean[], dtype?: T): Array1D<T> {
    if (!instanceofTypedArray(values)) {
      const inferredShape = util.inferShape(values as number[] | boolean[]);
      util.assert(
          inferredShape.length === 1,
          `Error constructing Array1D. Shape of values ${inferredShape} is ` +
              `not 1 dimensional.`);
    }
    return new Array1D({values: toTypedArray(values, dtype)}, dtype);
  }

  get(i: number): number {
    return this.getValues()[i];
  }

  set(value: number, i: number) {
    this.getValues()[i] = value;
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

  static zeros<T extends keyof DataTypes = keyof DataTypes>(
      shape: [number], dtype?: T): Array1D<T> {
    return NDArray.zeros(shape, dtype) as Array1D<T>;
  }

  static randNormal(shape: [number], mean = 0, stdDev = 1): Array1D<'float32'> {
    return NDArray.rand(shape, () => util.randGauss(mean, stdDev)) as
        Array1D<'float32'>;
  }

  static randTruncatedNormal(shape: [number], mean = 0, stdDev = 1):
      Array1D<'float32'> {
    return NDArray.rand(shape, () => util.randGauss(mean, stdDev, true)) as
        Array1D<'float32'>;
  }

  static randUniform(shape: [number], a: number, b: number):
      Array1D<'float32'> {
    return NDArray.rand(shape, () => util.randUniform(a, b)) as
        Array1D<'float32'>;
  }
}

export class Array2D<T extends keyof DataTypes = keyof DataTypes> extends
    NDArray<T> {
  shape: [number, number];

  private stride0: number;

  constructor(shape: [number, number], data: NDArrayData<T>, dtype: T) {
    util.assert(shape.length === 2, 'Shape should be of length 2');
    super(shape, data, dtype);
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
    return new Array2D(shape, {values: toTypedArray(values, dtype)}, dtype);
  }

  get(i: number, j: number) {
    return this.getValues()[this.stride0 * i + j];
  }

  set(value: number, i: number, j: number) {
    this.getValues()[this.stride0 * i + j] = value;
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

  static zeros<T extends keyof DataTypes = keyof DataTypes>(
      shape: [number, number], dtype?: T): Array2D<T> {
    return NDArray.zeros(shape, dtype) as Array2D<T>;
  }

  static randNormal(shape: [number, number], mean = 0, stdDev = 1):
      Array2D<'float32'> {
    return NDArray.rand(shape, () => util.randGauss(mean, stdDev)) as
        Array2D<'float32'>;
  }

  static randTruncatedNormal(shape: [number, number], mean = 0, stdDev = 1):
      Array2D<'float32'> {
    return NDArray.rand(shape, () => util.randGauss(mean, stdDev, true)) as
        Array2D<'float32'>;
  }

  static randUniform(shape: [number, number], a: number, b: number):
      Array2D<'float32'> {
    return NDArray.rand(shape, () => util.randUniform(a, b)) as
        Array2D<'float32'>;
  }
}

export class Array3D<T extends keyof DataTypes = keyof DataTypes> extends
    NDArray<T> {
  shape: [number, number, number];
  private stride0: number;
  private stride1: number;

  constructor(shape: [number, number, number], data: NDArrayData<T>, dtype: T) {
    util.assert(shape.length === 3, 'Shape should be of length 3');
    super(shape, data, dtype);
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
    return new Array3D(shape, {values: toTypedArray(values, dtype)}, dtype);
  }

  get(i: number, j: number, k: number) {
    return this.getValues()[this.stride0 * i + this.stride1 * j + k];
  }

  set(value: number, i: number, j: number, k: number) {
    this.getValues()[this.stride0 * i + this.stride1 * j + k] = value;
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

  static zeros<T extends keyof DataTypes = keyof DataTypes>(
      shape: [number, number, number], dtype?: T): Array3D<T> {
    return NDArray.zeros(shape, dtype) as Array3D<T>;
  }

  static randNormal(shape: [number, number, number], mean = 0, stdDev = 1):
      Array3D<'float32'> {
    return NDArray.rand(shape, () => util.randGauss(mean, stdDev)) as
        Array3D<'float32'>;
  }

  static randTruncatedNormal(
      shape: [number, number, number], mean = 0,
      stdDev = 1): Array3D<'float32'> {
    return NDArray.rand(shape, () => util.randGauss(mean, stdDev, true)) as
        Array3D<'float32'>;
  }

  static randUniform(shape: [number, number, number], a: number, b: number):
      Array3D<'float32'> {
    return NDArray.rand(shape, () => util.randUniform(a, b)) as
        Array3D<'float32'>;
  }
}

export class Array4D<T extends keyof DataTypes = keyof DataTypes> extends
    NDArray<T> {
  shape: [number, number, number, number];
  private stride0: number;
  private stride1: number;
  private stride2: number;

  constructor(
      shape: [number, number, number, number], data: NDArrayData<T>, dtype: T) {
    util.assert(shape.length === 4, 'Shape should be of length 4');
    super(shape, data, dtype);
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
    return new Array4D(shape, {values: toTypedArray(values, dtype)}, dtype);
  }

  get(i: number, j: number, k: number, l: number) {
    return this.getValues()
        [this.stride0 * i + this.stride1 * j + this.stride2 * k + l];
  }

  set(value: number, i: number, j: number, k: number, l: number) {
    this.getValues()
        [this.stride0 * i + this.stride1 * j + this.stride2 * k + l] = value;
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

  static zeros<T extends keyof DataTypes = keyof DataTypes>(
      shape: [number, number, number, number], dtype?: T): Array4D<T> {
    return NDArray.zeros(shape, dtype) as Array4D<T>;
  }

  static randNormal(
      shape: [number, number, number, number], mean = 0,
      stdDev = 1): Array4D<'float32'> {
    return NDArray.rand(shape, () => util.randGauss(mean, stdDev)) as
        Array4D<'float32'>;
  }

  static randTruncatedNormal(
      shape: [number, number, number, number], mean = 0,
      stdDev = 1): Array4D<'float32'> {
    return NDArray.rand(shape, () => util.randGauss(mean, stdDev, true)) as
        Array4D<'float32'>;
  }

  static randUniform(
      shape: [number, number, number, number], a: number,
      b: number): Array4D<'float32'> {
    return NDArray.rand(shape, () => util.randUniform(a, b)) as
        Array4D<'float32'>;
  }
}

function copyTypedArray<T extends keyof DataTypes>(
    array: DataTypes[T]|number[]|boolean[], dtype: T): DataTypes[T] {
  if (dtype == null || dtype === 'float32') {
    return new Float32Array(array as number[]);
  } else if (dtype === 'int32') {
    return new Int32Array(array as number[]);
  } else if (dtype === 'bool') {
    const bool = new Uint8Array(array.length);
    for (let i = 0; i < bool.length; ++i) {
      if (array[i]) {
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

function convertFloat32ToDtype<T extends keyof DataTypes>(
    values: Float32Array, dtype: T): DataTypes[T] {
  if (dtype === 'float32') {
    return values;
  } else if (dtype === 'int32' || dtype === 'bool') {
    const result = (dtype === 'int32') ? new Int32Array(values.length) :
                                         new Uint8Array(values.length);
    for (let i = 0; i < result.length; ++i) {
      result[i] = Math.round(values[i]);
    }
    return result;
  } else {
    throw new Error(`Unknown dtype ${dtype}`);
  }
}
