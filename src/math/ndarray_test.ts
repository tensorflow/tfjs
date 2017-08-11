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

import * as ndarray from './ndarray';
import {Array1D, Array2D, Array3D, Array4D, NDArray, Scalar} from './ndarray';
import {GPGPUContext} from './webgl/gpgpu_context';
import * as gpgpu_util from './webgl/gpgpu_util';
import {TextureManager} from './webgl/texture_manager';

describe('NDArray', () => {
  let gl: WebGLRenderingContext;
  let gpgpu: GPGPUContext;
  let textureManager: TextureManager;

  beforeEach(() => {
    gl = gpgpu_util.createWebGLContext();
    gpgpu = new GPGPUContext(gl);
    textureManager = new TextureManager(gpgpu);
    ndarray.initializeGPU(gpgpu, textureManager);
  });

  afterEach(() => {
    textureManager.dispose();
    gpgpu.dispose();
  });

  it('NDArrays of arbitrary size', () => {
    // [1, 2, 3]
    let t: NDArray = Array1D.new([1, 2, 3]);
    expect(t instanceof Array1D).toBe(true);
    expect(t.rank).toBe(1);
    expect(t.size).toBe(3);
    expect(t.getValues()).toEqual(new Float32Array([1, 2, 3]));
    expect(t.get(0)).toBe(1);
    expect(t.get(1)).toBe(2);
    expect(t.get(2)).toBe(3);
    // Out of bounds indexing.
    expect(t.get(4)).toBeUndefined();

    // [[1, 2, 3]]
    t = Array2D.new([1, 3], [1, 2, 3]);
    expect(t instanceof Array2D).toBe(true);
    expect(t.rank).toBe(2);
    expect(t.size).toBe(3);
    expect(t.get(0, 0)).toBe(1);
    expect(t.get(0, 1)).toBe(2);
    expect(t.get(0, 2)).toBe(3);
    // Out of bounds indexing.
    expect(t.get(4)).toBeUndefined();

    // [[1, 2, 3],
    //  [4, 5, 6]]
    t = Array2D.new([2, 3], [1, 2, 3, 4, 5, 6]);
    expect(t instanceof Array2D).toBe(true);
    expect(t.rank).toBe(2);
    expect(t.size).toBe(6);
    expect(t.get(0, 0)).toBe(1);
    expect(t.get(0, 1)).toBe(2);
    expect(t.get(0, 2)).toBe(3);
    expect(t.get(1, 0)).toBe(4);
    expect(t.get(1, 1)).toBe(5);
    expect(t.get(1, 2)).toBe(6);
    // Out of bounds indexing.
    expect(t.get(5, 3)).toBeUndefined();

    // Shape mismatch with the values.
    expect(() => Array2D.new([1, 2], [1])).toThrowError();
  });

  it('NDArrays of explicit size', () => {
    const t = Array1D.new([5, 3, 2]);
    expect(t.rank).toBe(1);
    expect(t.shape).toEqual([3]);
    expect(t.get(1)).toBe(3);

    expect(() => Array3D.new([1, 2, 3, 5], [
      1, 2
    ])).toThrowError('Shape should be of length 3');

    const t4 = Array4D.new([1, 2, 1, 2], [1, 2, 3, 4]);
    expect(t4.get(0, 0, 0, 0)).toBe(1);
    expect(t4.get(0, 0, 0, 1)).toBe(2);
    expect(t4.get(0, 1, 0, 0)).toBe(3);
    expect(t4.get(0, 1, 0, 1)).toBe(4);

    const t4Like = NDArray.like(t4);
    // Change t4.
    t4.set(10, 0, 0, 0, 1);
    expect(t4.get(0, 0, 0, 1)).toBe(10);
    // Make suree t4_like hasn't changed.
    expect(t4Like.get(0, 0, 0, 1)).toBe(2);

    // NDArray of zeros.
    const z = NDArray.zeros([3, 4, 2]) as Array3D;
    expect(z.rank).toBe(3);
    expect(z.size).toBe(24);
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 4; j++) {
        for (let k = 0; k < 2; k++) {
          expect(z.get(i, j, k)).toBe(0);
        }
      }
    }

    // Reshaping ndarrays.
    const a = Array2D.new([2, 3], [1, 2, 3, 4, 5, 6]);
    const b = a.reshape([3, 2, 1]);
    expect(a.get(1, 2)).toBe(6);

    // Modify the reshaped ndarray.
    b.set(10, 2, 1, 0);
    // Make sure the original ndarray is also modified.
    expect(a.get(1, 2)).toBe(10);
  });

  it('NDArray CPU --> GPU', () => {
    const a = Array2D.new([3, 2], [1, 2, 3, 4, 5, 6]);

    expect(a.inGPU()).toBe(false);

    expect(a.getValues()).toEqual(new Float32Array([1, 2, 3, 4, 5, 6]));

    expect(a.inGPU()).toBe(false);

    // Upload to GPU.
    expect(a.getTexture() != null).toBe(true);

    expect(a.inGPU()).toBe(true);
    a.dispose();
  });

  it('NDArray GPU --> CPU', () => {
    const texture = textureManager.acquireTexture([3, 2]);
    gpgpu.uploadMatrixToTexture(
        texture, 3, 2, new Float32Array([1, 2, 3, 4, 5, 6]));

    const a = new Array2D([3, 2], {texture, textureShapeRC: [3, 2]});
    expect(a.inGPU()).toBe(true);

    expect(a.getValues()).toEqual(new Float32Array([1, 2, 3, 4, 5, 6]));
    expect(a.inGPU()).toBe(false);
  });

  it('Scalar basic methods', () => {
    const a = Scalar.new(5);
    expect(a.get()).toBe(5);
    expect(a.getValues()).toEqual(new Float32Array([5]));
    expect(a.rank).toBe(0);
    expect(a.size).toBe(1);
    expect(a.shape).toEqual([]);
  });

  it('Scalar in GPU', () => {
    const texture = textureManager.acquireTexture([1, 1]);
    gpgpu.uploadMatrixToTexture(texture, 1, 1, new Float32Array([10]));

    const a = new Scalar({texture});
    expect(a.inGPU()).toBe(true);
    expect(a.getValues()).toEqual(new Float32Array([10]));
    expect(a.inGPU()).toBe(false);
  });

  it('Array1D in GPU', () => {
    const texture = textureManager.acquireTexture([1, 3]);
    gpgpu.uploadMatrixToTexture(texture, 1, 3, new Float32Array([10, 7, 3]));

    const a = new Array1D({texture, textureShapeRC: [1, 3]});
    expect(a.inGPU()).toBe(true);
    expect(a.getValues()).toEqual(new Float32Array([10, 7, 3]));
    expect(a.inGPU()).toBe(false);
  });

  it('Array1D in GPU, but incorrect c-tor (missing textureShape)', () => {
    const texture = textureManager.acquireTexture([1, 3]);
    gpgpu.uploadMatrixToTexture(texture, 1, 3, new Float32Array([10, 7, 3]));

    const f = () => {
      return new Array1D({texture});
    };

    expect(f).toThrowError();
    textureManager.releaseTexture(texture, [1, 3]);
  });

  it('NDArray.make() constructs a Scalar', () => {
    const a = NDArray.make([], {values: new Float32Array([3])});
    expect(a instanceof Scalar).toBe(true);
  });

  it('Array2D in GPU, reshaped to Array1D', () => {
    const texture = textureManager.acquireTexture([2, 2]);
    gpgpu.uploadMatrixToTexture(texture, 2, 2, new Float32Array([10, 7, 3, 5]));

    const a = new Array2D([2, 2], {texture, textureShapeRC: [2, 2]});
    const a1d = a.as1D();

    expect(a1d.getValues()).toEqual(new Float32Array([10, 7, 3, 5]));
  });

  it('Array1D in GPU, reshaped to Array2D', () => {
    const texture = textureManager.acquireTexture([1, 4]);
    gpgpu.uploadMatrixToTexture(texture, 1, 4, new Float32Array([10, 7, 3, 5]));

    const a = new Array1D({texture, textureShapeRC: [1, 4]});
    const a2d = a.as2D(2, 2);

    expect(a2d.getValues()).toEqual(new Float32Array([10, 7, 3, 5]));
  });

  it('Array2D in GPU with custom texture shape', () => {
    const texture = textureManager.acquireTexture([4, 1]);
    gpgpu.uploadMatrixToTexture(texture, 4, 1, new Float32Array([10, 7, 3, 5]));

    const a = new Array2D([2, 2], {texture, textureShapeRC: [4, 1]});

    expect(a.getValues()).toEqual(new Float32Array([10, 7, 3, 5]));
  });

  it('index2Loc Array1D', () => {
    const t = Array1D.zeros([3]);
    expect(t.indexToLoc(0)).toEqual([0]);
    expect(t.indexToLoc(1)).toEqual([1]);
    expect(t.indexToLoc(2)).toEqual([2]);
  });

  it('index2Loc Array2D', () => {
    const t = Array2D.zeros([3, 2]);
    expect(t.indexToLoc(0)).toEqual([0, 0]);
    expect(t.indexToLoc(1)).toEqual([0, 1]);
    expect(t.indexToLoc(2)).toEqual([1, 0]);
    expect(t.indexToLoc(3)).toEqual([1, 1]);
    expect(t.indexToLoc(4)).toEqual([2, 0]);
    expect(t.indexToLoc(5)).toEqual([2, 1]);
  });

  it('index2Loc Array3D', () => {
    const t = Array2D.zeros([3, 2, 2]);
    expect(t.indexToLoc(0)).toEqual([0, 0, 0]);
    expect(t.indexToLoc(1)).toEqual([0, 0, 1]);
    expect(t.indexToLoc(2)).toEqual([0, 1, 0]);
    expect(t.indexToLoc(3)).toEqual([0, 1, 1]);
    expect(t.indexToLoc(4)).toEqual([1, 0, 0]);
    expect(t.indexToLoc(5)).toEqual([1, 0, 1]);
    expect(t.indexToLoc(11)).toEqual([2, 1, 1]);
  });

  it('index2Loc NDArray 5D', () => {
    const values = new Float32Array([1, 2, 3, 4]);
    const t = NDArray.make([2, 1, 1, 1, 2], {values});
    expect(t.indexToLoc(0)).toEqual([0, 0, 0, 0, 0]);
    expect(t.indexToLoc(1)).toEqual([0, 0, 0, 0, 1]);
    expect(t.indexToLoc(2)).toEqual([1, 0, 0, 0, 0]);
    expect(t.indexToLoc(3)).toEqual([1, 0, 0, 0, 1]);
  });

  it('preferred texture shape, Scalar', () => {
    const t = Scalar.new(1);
    expect(t.getTextureShapeRC()).toEqual([1, 1]);
  });

  it('preferred texture shape, Array1D column vector', () => {
    const t = Array1D.zeros([4]);
    expect(t.getTextureShapeRC()).toEqual([4, 1]);
  });

  it('preferred texture shape, Array2D same shape', () => {
    const t = Array2D.zeros([5, 2]);
    expect(t.getTextureShapeRC()).toEqual([5, 2]);
  });

  it('preferred texture shape, Array3D depth strided along columns', () => {
    const t = Array3D.zeros([2, 2, 2]);
    expect(t.getTextureShapeRC()).toEqual([2, 4]);
  });

  it('preferred texture shape, Array4D is squareish', () => {
    const t = Array4D.zeros([8, 2, 4, 4]);
    expect(t.getTextureShapeRC()).toEqual([16, 16]);
  });
});  // Close describe.

describe('NDArray.new method', () => {
  it('Array1D.new() from number[]', () => {
    const a = Array1D.new([1, 2, 3]);
    expect(a.getValues()).toEqual(new Float32Array([1, 2, 3]));
  });

  it('Array1D.new() from number[][], shape mismatch', () => {
    // tslint:disable-next-line:no-any
    expect(() => Array1D.new([[1], [2], [3]] as any)).toThrowError();
  });

  it('Array2D.new() from number[][]', () => {
    const a = Array2D.new([2, 3], [[1, 2, 3], [4, 5, 6]]);
    expect(a.getValues()).toEqual(new Float32Array([1, 2, 3, 4, 5, 6]));
  });

  it('Array2D.new() from number[][], but shape does not match', () => {
    // Actual shape is [2, 3].
    expect(() => Array2D.new([3, 2], [[1, 2, 3], [4, 5, 6]])).toThrowError();
  });

  it('Array3D.new() from number[][][]', () => {
    const a = Array3D.new([2, 3, 1], [[[1], [2], [3]], [[4], [5], [6]]]);
    expect(a.getValues()).toEqual(new Float32Array([1, 2, 3, 4, 5, 6]));
  });

  it('Array3D.new() from number[][][], but shape does not match', () => {
    const values = [[[1], [2], [3]], [[4], [5], [6]]];
    // Actual shape is [2, 3, 1].
    expect(() => Array3D.new([3, 2, 1], values)).toThrowError();
  });

  it('Array4D.new() from number[][][][]', () => {
    const a = Array4D.new([2, 2, 1, 1], [[[[1]], [[2]]], [[[4]], [[5]]]]);
    expect(a.getValues()).toEqual(new Float32Array([1, 2, 4, 5]));
  });

  it('Array4D.new() from number[][][][], but shape does not match', () => {
    const f = () => {
      // Actual shape is [2, 2, 1, 1].
      Array4D.new([2, 1, 2, 1], [[[[1]], [[2]]], [[[4]], [[5]]]]);
    };
    expect(f).toThrowError();
  });
});
