/* Copyright 2017 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

import * as util from '../util';

import {GPGPUContext} from './webgl/gpgpu_context';
import {TextureManager} from './webgl/texture_manager';
import * as webgl_util from './webgl/webgl_util';

// These global variables need to be initialized to null so that closure knows
// not to seal them.
/** @hidden */
export let GPGPU: GPGPUContext = null!;
/** @hidden */
export let TEXTURE_MANAGER: TextureManager = null!;

/** @hidden */
export interface NDArrayData {
  values?: Float32Array;
  texture?: WebGLTexture;
  /** [rows, columns] shape of the texture. */
  textureShapeRC?: [number, number];
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

export class NDArray {
  /** The shape of the ndarray. */
  shape: number[];
  /** Number of elements in the ndarray. */
  size: number;

  /**
   * Number of elements to skip in each dimension when indexing. See
   * https://docs.scipy.org/doc/numpy/reference/generated
   *     /numpy.ndarray.strides.html
   */
  protected strides: number[];

  private data: NDArrayData;

  protected constructor(shape: number[], data: NDArrayData) {
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
    this.data = data;
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
  static zeros(shape: number[]): NDArray {
    const values = new Float32Array(util.sizeFromShape(shape));
    return NDArray.make(shape, {values});
  }

  /** Creates a ndarray of zeros with the same shape as the specified ndarray.
   */
  static zerosLike<T extends NDArray>(another: T): T {
    return NDArray.zeros(another.shape) as T;
  }

  /** Creates a ndarray with the same values/shape as the specified ndarray. */
  static like<T extends NDArray>(another: T): T {
    const values = another.getValues();
    return NDArray.make<T>(another.shape, {values: new Float32Array(values)});
  }

  /**
   * Makes a new ndarray with the provided shape and values. Values should be in
   * a flat array.
   */
  static make<T extends NDArray>(shape: number[], data: NDArrayData): T {
    switch (shape.length) {
      case 0:
        return new Scalar(data) as T;
      case 1:
        // tslint:disable-next-line:no-any
        return new Array1D(data) as any;
      case 2:
        // tslint:disable-next-line:no-any
        return new Array2D(shape as [number, number], data) as any;
      case 3:
        // tslint:disable-next-line:no-any
        return new Array3D(shape as [number, number, number], data) as any;
      case 4:
        return new Array4D(
            // tslint:disable-next-line:no-any
            shape as [number, number, number, number], data) as any;
      default:
        // tslint:disable-next-line:no-any
        return new NDArray(shape, data) as any;
    }
  }

  /** Reshapes the current ndarray into the provided shape. */
  reshape<T extends NDArray>(newShape: number[]): T {
    if (util.arraysEqual(this.shape, newShape)) {
      // No-op.
      // tslint:disable-next-line:no-any
      return this as any;
    }

    util.assert(
        this.size === util.sizeFromShape(newShape),
        'new shape and old shape must have the same number of elements.');

    return NDArray.make<T>(newShape, this.data);
  }

  asScalar(): Scalar {
    util.assert(this.size === 1, 'The array must have only 1 element.');
    return this.reshape<Scalar>([]);
  }

  as1D(): Array1D {
    return this.reshape<Array1D>([this.size]);
  }

  as2D(rows: number, columns: number): Array2D {
    return this.reshape<Array2D>([rows, columns]);
  }

  as3D(rows: number, columns: number, depth: number): Array3D {
    return this.reshape<Array3D>([rows, columns, depth]);
  }

  as4D(rows: number, columns: number, depth: number, depth2: number): Array4D {
    return this.reshape<Array4D>([rows, columns, depth, depth2]);
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

  getData(): NDArrayData {
    return this.data;
  }

  getValues(): Float32Array {
    if (this.data.values == null) {
      throwIfGPUNotInitialized();
      this.data.values = GPGPU.downloadMatrixFromTexture(
          this.data.texture!, this.data.textureShapeRC![0],
          this.data.textureShapeRC![1]);
      this.disposeTexture();
    }
    return this.data.values;
  }

  private uploadToGPU(preferredTexShape?: [number, number]) {
    throwIfGPUNotInitialized();
    this.data.textureShapeRC = webgl_util.getTextureShapeFromLogicalShape(
        GPGPU.gl, this.shape, preferredTexShape);
    this.data.texture =
        TEXTURE_MANAGER.acquireTexture(this.data.textureShapeRC);

    GPGPU.uploadMatrixToTexture(
        this.data.texture, this.data.textureShapeRC[0],
        this.data.textureShapeRC[1], this.data.values!);

    this.data.values = null!;
  }

  getTexture(preferredShapeRC?: [number, number]): WebGLTexture {
    if (this.data.texture == null) {
      this.uploadToGPU(preferredShapeRC);
    }
    return this.data.texture!;
  }

  getTextureShapeRC(preferredShapeRC?: [number, number]): [number, number] {
    if (this.data.textureShapeRC == null) {
      this.uploadToGPU(preferredShapeRC);
    }
    return this.data.textureShapeRC!;
  }

  dispose(): void {
    this.data.values = null!;
    this.shape = null!;
    if (this.data.texture != null) {
      this.disposeTexture();
    }
  }

  private disposeTexture() {
    throwIfGPUNotInitialized();
    TEXTURE_MANAGER.releaseTexture(
        this.data.texture!, this.data.textureShapeRC!);
    this.data.texture = null!;
    this.data.textureShapeRC = null!;
  }

  inGPU(): boolean {
    return this.data.texture != null;
  }

  equals(t: NDArray): boolean {
    return util.arraysEqual(this.shape, t.shape) &&
        util.arraysEqual(this.getValues(), t.getValues());
  }

  static rand<T extends NDArray>(shape: number[], randFunction: () => number):
      T {
    const size = util.sizeFromShape(shape);
    const values = new Float32Array(size);
    for (let i = 0; i < size; i++) {
      values[i] = randFunction();
    }

    return NDArray.make<T>(shape, {values});
  }

  static randNormal<T extends NDArray>(shape: number[], mean = 0, stdDev = 1) {
    return NDArray.rand<T>(shape, () => util.randGauss(mean, stdDev));
  }

  static randTruncatedNormal<T extends NDArray>(
      shape: number[], mean = 0, stdDev = 1) {
    return NDArray.rand<T>(shape, () => util.randGauss(mean, stdDev, true));
  }

  static randUniform<T extends NDArray>(shape: number[], a: number, b: number) {
    return NDArray.rand<T>(shape, () => util.randUniform(a, b));
  }
}

export class Scalar extends NDArray {
  constructor(data: NDArrayData) {
    if (data.texture != null) {
      data.textureShapeRC = [1, 1];
    }
    super([], data);
  }

  static new(value: number) {
    return new Scalar({values: new Float32Array([value])});
  }

  static ZERO = Scalar.new(0);
  static ONE = Scalar.new(1);
  static TWO = Scalar.new(2);
  static NEG_ONE = Scalar.new(-1);

  get(): number {
    return this.getValues()[0];
  }

  set(value: number) {
    this.getValues()[0] = value;
  }

  add(value: number) {
    this.getValues()[0] += value;
  }
}

export class Array1D extends NDArray {
  shape: [number];

  constructor(data: NDArrayData) {
    const shape = (data.values != null) ?
        [data.values.length] :
        [util.sizeFromShape(data.textureShapeRC!)];
    super(shape, data);
  }

  static new(values: Float32Array|number[]) {
    if (!(values instanceof Float32Array)) {
      const inferredShape = util.inferShape(values);
      util.assert(
          inferredShape.length === 1,
          `Error constructing Array1D. Shape of values ${inferredShape} is ` +
              `not 1 dimensional.`);
    }
    return new Array1D({values: toTypedArray(values)});
  }

  get(i: number): number {
    return this.getValues()[i];
  }

  set(value: number, i: number) {
    this.getValues()[i] = value;
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

  static zeros(shape: [number]): Array1D {
    return NDArray.zeros(shape) as Array1D;
  }
}

export class Array2D extends NDArray {
  shape: [number, number];

  private stride0: number;

  constructor(shape: [number, number], data: NDArrayData) {
    util.assert(shape.length === 2, 'Shape should be of length 2');
    super(shape, data);
    this.stride0 = this.strides[0];
  }

  static new(
      shape: [number, number], values: Float32Array|number[]|number[][]) {
    if (!(values instanceof Float32Array)) {
      const inferredShape = util.inferShape(values);
      if (inferredShape.length > 1) {
        util.assertShapesMatch(
            shape, inferredShape,
            `Error when constructing Array2D. Shape of values ` +
                `${inferredShape} does not match the provided shape ` +
                `${shape}. `);
      }
    }
    return new Array2D(shape, {values: toTypedArray(values)});
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

  locToIndex(locs: [number, number]): number {
    return this.stride0 * locs[0] + locs[1];
  }

  indexToLoc(index: number): [number, number] {
    return [Math.floor(index / this.stride0), index % this.stride0];
  }

  static zeros(shape: [number, number]): Array2D {
    return NDArray.zeros(shape) as Array2D;
  }
}

export class Array3D extends NDArray {
  shape: [number, number, number];
  private stride0: number;
  private stride1: number;

  constructor(shape: [number, number, number], data: NDArrayData) {
    util.assert(shape.length === 3, 'Shape should be of length 3');
    super(shape, data);
    this.stride0 = this.strides[0];
    this.stride1 = this.strides[1];
  }

  static new(
      shape: [number, number, number],
      values: Float32Array|number[]|number[][][]) {
    if (!(values instanceof Float32Array)) {
      const inferredShape = util.inferShape(values);
      if (inferredShape.length > 1) {
        util.assertShapesMatch(
            shape, inferredShape,
            `Error when constructing Array3D. Shape of values ` +
                `${inferredShape} does not match the provided shape ` +
                `${shape}. `);
      }
    }
    return new Array3D(shape, {values: toTypedArray(values)});
  }

  get(i: number, j: number, k: number) {
    return this.getValues()[this.stride0 * i + this.stride1 * j + k];
  }

  set(value: number, i: number, j: number, k: number) {
    this.getValues()[this.stride0 * i + this.stride1 * j + k] = value;
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

  static zeros(shape: [number, number, number]): Array3D {
    return NDArray.zeros(shape) as Array3D;
  }
}

export class Array4D extends NDArray {
  shape: [number, number, number, number];
  private stride0: number;
  private stride1: number;
  private stride2: number;

  constructor(shape: [number, number, number, number], data: NDArrayData) {
    util.assert(shape.length === 4, 'Shape should be of length 4');
    super(shape, data);
    this.stride0 = this.strides[0];
    this.stride1 = this.strides[1];
    this.stride2 = this.strides[2];
  }

  static new(
      shape: [number, number, number, number],
      values: Float32Array|number[]|number[][][][]) {
    if (!(values instanceof Float32Array)) {
      const inferredShape = util.inferShape(values);
      if (inferredShape.length > 1) {
        util.assertShapesMatch(
            shape, inferredShape,
            `Error when constructing Array4D. Shape of values ` +
                `${inferredShape} does not match the provided shape ` +
                `${shape}. `);
      }
    }
    return new Array4D(shape, {values: toTypedArray(values)});
  }

  get(i: number, j: number, k: number, l: number) {
    return this.getValues()
        [this.stride0 * i + this.stride1 * j + this.stride2 * k + l];
  }

  set(value: number, i: number, j: number, k: number, l: number) {
    this.getValues()
        [this.stride0 * i + this.stride1 * j + this.stride2 * k + l] = value;
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

  static zeros(shape: [number, number, number, number]): Array4D {
    return NDArray.zeros(shape) as Array4D;
  }
}

type ArrayData = Float32Array|number[]|number[][]|number[][][]|number[][][][];

function toTypedArray(a: ArrayData): Float32Array {
  return (a instanceof Float32Array) ? a : new Float32Array(util.flatten(a));
}
