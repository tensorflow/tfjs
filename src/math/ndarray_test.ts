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

import * as test_util from '../test_util';
import * as util from '../util';

import * as ndarray from './ndarray';
// tslint:disable-next-line:max-line-length
import {
  Array1D,
  Array2D,
  Array3D,
  Array4D,
  DType,
  NDArray,
  Scalar
} from './ndarray';
import {GPGPUContext} from './backends/webgl/gpgpu_context';
import * as gpgpu_util from './backends/webgl/gpgpu_util';
import {TextureManager} from './backends/webgl/texture_manager';

const FEATURES = [
  {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
  {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
  {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
];

let gl: WebGLRenderingContext;
let gpgpu: GPGPUContext;
let textureManager: TextureManager;

const customBeforeEach = () => {
  gl = gpgpu_util.createWebGLContext();
  gpgpu = new GPGPUContext(gl);
  textureManager = new TextureManager(gpgpu);
  ndarray.initializeGPU(gpgpu, textureManager);
};

const customAfterEach = () => {
  textureManager.dispose();
  gpgpu.dispose();
};

test_util.describeCustom('NDArray', () => {
  it('NDArrays of arbitrary size', () => {
    // [1, 2, 3]
    let t: NDArray = Array1D.new([1, 2, 3]);
    expect(t instanceof Array1D).toBe(true);
    expect(t.rank).toBe(1);
    expect(t.size).toBe(3);
    test_util.expectArraysClose(t.getValues(), new Float32Array([1, 2, 3]));
    // Out of bounds indexing.
    expect(t.get(4)).toBeUndefined();

    // [[1, 2, 3]]
    t = Array2D.new([1, 3], [1, 2, 3]);
    expect(t instanceof Array2D).toBe(true);
    expect(t.rank).toBe(2);
    expect(t.size).toBe(3);
    test_util.expectArraysClose(t.getValues(), new Float32Array([1, 2, 3]));
    // Out of bounds indexing.
    expect(t.get(4)).toBeUndefined();

    // [[1, 2, 3],
    //  [4, 5, 6]]
    t = Array2D.new([2, 3], [1, 2, 3, 4, 5, 6]);
    expect(t instanceof Array2D).toBe(true);
    expect(t.rank).toBe(2);
    expect(t.size).toBe(6);

    test_util.expectArraysClose(
        t.getValues(), new Float32Array([1, 2, 3, 4, 5, 6]));

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

  it('NDArray getValues CPU --> GPU', () => {
    const a = Array2D.new([3, 2], [1, 2, 3, 4, 5, 6]);

    expect(a.inGPU()).toBe(false);

    test_util.expectArraysClose(
        a.getValues(), new Float32Array([1, 2, 3, 4, 5, 6]));

    expect(a.inGPU()).toBe(false);

    // Upload to GPU.
    expect(a.getTexture() != null).toBe(true);

    expect(a.inGPU()).toBe(true);
    a.dispose();
  });

  it('NDArray getValues GPU --> CPU', () => {
    const texture = textureManager.acquireTexture([3, 2]);
    gpgpu.uploadMatrixToTexture(
        texture, 3, 2, new Float32Array([1, 2, 3, 4, 5, 6]));

    const a = Array2D.make([3, 2], {texture, textureShapeRC: [3, 2]});
    expect(a.inGPU()).toBe(true);

    test_util.expectArraysClose(
        a.getValues(), new Float32Array([1, 2, 3, 4, 5, 6]));
    expect(a.inGPU()).toBe(false);
  });

  it('NDArray getValuesAsync CPU --> GPU', (doneFn) => {
    const a = Array2D.new([3, 2], [1, 2, 3, 4, 5, 6]);

    expect(a.inGPU()).toBe(false);

    a.getValuesAsync().then(values => {
      test_util.expectArraysClose(values, new Float32Array([1, 2, 3, 4, 5, 6]));

      expect(a.inGPU()).toBe(false);

      // Upload to GPU.
      expect(a.getTexture() != null).toBe(true);

      expect(a.inGPU()).toBe(true);
      a.dispose();
      doneFn();
    });
  });

  it('NDArray getValuesAsync GPU --> CPU', () => {
    const texture = textureManager.acquireTexture([3, 2]);
    gpgpu.uploadMatrixToTexture(
        texture, 3, 2, new Float32Array([1, 2, 3, 4, 5, 6]));

    const a = Array2D.make([3, 2], {texture, textureShapeRC: [3, 2]});
    expect(a.inGPU()).toBe(true);

    a.getValuesAsync().then(values => {
      test_util.expectArraysClose(values, new Float32Array([1, 2, 3, 4, 5, 6]));
      expect(a.inGPU()).toBe(false);
    });
  });

  it('NDArray.data GPU --> CPU', async() => {
    const texture = textureManager.acquireTexture([3, 2]);
    gpgpu.uploadMatrixToTexture(
        texture, 3, 2, new Float32Array([1, 2, 3, 4, 5, 6]));

    const a = Array2D.make([3, 2], {texture, textureShapeRC: [3, 2]});
    expect(a.inGPU()).toBe(true);

    const values = await a.data();
    test_util.expectArraysClose(values, new Float32Array([1, 2, 3, 4, 5, 6]));
    expect(a.inGPU()).toBe(false);
  });

  it('NDArray.val() GPU --> CPU', async() => {
    const texture = textureManager.acquireTexture([3, 2]);
    gpgpu.uploadMatrixToTexture(
        texture, 3, 2, new Float32Array([1, 2, 3, 4, 5, 6]));

    const a = Array2D.make([3, 2], {texture, textureShapeRC: [3, 2]});
    expect(a.inGPU()).toBe(true);

    test_util.expectNumbersClose(1, await a.val(0));
    test_util.expectNumbersClose(2, await a.val(1));
    test_util.expectNumbersClose(3, await a.val(2));
    test_util.expectNumbersClose(4, await a.val(3));
    test_util.expectNumbersClose(5, await a.val(4));
    test_util.expectNumbersClose(6, await a.val(5));

    expect(a.inGPU()).toBe(false);
  });

  it('Scalar basic methods', () => {
    const a = Scalar.new(5);
    expect(a.get()).toBe(5);
    test_util.expectArraysClose(a.getValues(), new Float32Array([5]));
    expect(a.rank).toBe(0);
    expect(a.size).toBe(1);
    expect(a.shape).toEqual([]);
  });

  it('Scalar in GPU', () => {
    const texture = textureManager.acquireTexture([1, 1]);
    gpgpu.uploadMatrixToTexture(texture, 1, 1, new Float32Array([10]));

    const a = Scalar.make([], {texture});
    expect(a.inGPU()).toBe(true);
    test_util.expectArraysClose(a.getValues(), new Float32Array([10]));
    expect(a.inGPU()).toBe(false);
  });

  it('Array1D in GPU', () => {
    const texture = textureManager.acquireTexture([1, 3]);
    gpgpu.uploadMatrixToTexture(texture, 1, 3, new Float32Array([10, 7, 3]));

    const a = Array1D.make([3], {texture, textureShapeRC: [1, 3]});
    expect(a.inGPU()).toBe(true);
    test_util.expectArraysClose(a.getValues(), new Float32Array([10, 7, 3]));
    expect(a.inGPU()).toBe(false);
  });

  it('Array1D in GPU, but incorrect c-tor (missing textureShape)', () => {
    const texture = textureManager.acquireTexture([1, 3]);
    gpgpu.uploadMatrixToTexture(texture, 1, 3, new Float32Array([10, 7, 3]));

    const f = () => Array1D.make([3], {texture});

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

    const a = Array2D.make([2, 2], {texture, textureShapeRC: [2, 2]});
    const a1d = a.as1D();

    test_util.expectArraysClose(
        a1d.getValues(), new Float32Array([10, 7, 3, 5]));
  });

  it('Array1D in GPU, reshaped to Array2D', () => {
    const texture = textureManager.acquireTexture([1, 4]);
    gpgpu.uploadMatrixToTexture(texture, 1, 4, new Float32Array([10, 7, 3, 5]));

    const a = Array1D.make([4], {texture, textureShapeRC: [1, 4]});
    const a2d = a.as2D(2, 2);

    test_util.expectArraysClose(
        a2d.getValues(), new Float32Array([10, 7, 3, 5]));
  });

  it('Array2D in GPU with custom texture shape', () => {
    const texture = textureManager.acquireTexture([4, 1]);
    gpgpu.uploadMatrixToTexture(texture, 4, 1, new Float32Array([10, 7, 3, 5]));

    const a = Array2D.make([2, 2], {texture, textureShapeRC: [4, 1]});

    test_util.expectArraysClose(a.getValues(), new Float32Array([10, 7, 3, 5]));
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

  it('preferred texture shape, Array4D d1 and d2 strided along columns', () => {
    const t = Array4D.zeros([8, 2, 4, 4]);
    expect(t.getTextureShapeRC()).toEqual([8, 2 * 4 * 4]);
  });
}, FEATURES, customBeforeEach, customAfterEach);

test_util.describeCustom('NDArray.new', () => {
  it('Array1D.new() from number[]', () => {
    const a = Array1D.new([1, 2, 3]);
    test_util.expectArraysClose(a.getValues(), new Float32Array([1, 2, 3]));
  });

  it('Array1D.new() from number[][], shape mismatch', () => {
    // tslint:disable-next-line:no-any
    expect(() => Array1D.new([[1], [2], [3]] as any)).toThrowError();
  });

  it('Array2D.new() from number[][]', () => {
    const a = Array2D.new([2, 3], [[1, 2, 3], [4, 5, 6]]);
    test_util.expectArraysClose(
        a.getValues(), new Float32Array([1, 2, 3, 4, 5, 6]));
  });

  it('Array2D.new() from number[][], but shape does not match', () => {
    // Actual shape is [2, 3].
    expect(() => Array2D.new([3, 2], [[1, 2, 3], [4, 5, 6]])).toThrowError();
  });

  it('Array3D.new() from number[][][]', () => {
    const a = Array3D.new([2, 3, 1], [[[1], [2], [3]], [[4], [5], [6]]]);
    test_util.expectArraysClose(
        a.getValues(), new Float32Array([1, 2, 3, 4, 5, 6]));
  });

  it('Array3D.new() from number[][][], but shape does not match', () => {
    const values = [[[1], [2], [3]], [[4], [5], [6]]];
    // Actual shape is [2, 3, 1].
    expect(() => Array3D.new([3, 2, 1], values)).toThrowError();
  });

  it('Array4D.new() from number[][][][]', () => {
    const a = Array4D.new([2, 2, 1, 1], [[[[1]], [[2]]], [[[4]], [[5]]]]);
    test_util.expectArraysClose(a.getValues(), new Float32Array([1, 2, 4, 5]));
  });

  it('Array4D.new() from number[][][][], but shape does not match', () => {
    const f = () => {
      // Actual shape is [2, 2, 1, 1].
      Array4D.new([2, 1, 2, 1], [[[[1]], [[2]]], [[[4]], [[5]]]]);
    };
    expect(f).toThrowError();
  });
});

test_util.describeCustom('NDArray.zeros', () => {
  it('1D default dtype', () => {
    const a = Array1D.zeros([3]);
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([3]);
    expect(a.getValues()).toEqual(new Float32Array([0, 0, 0]));
  });

  it('1D float32 dtype', () => {
    const a = Array1D.zeros([3], 'float32');
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([3]);
    expect(a.getValues()).toEqual(new Float32Array([0, 0, 0]));
  });

  it('1D int32 dtype', () => {
    const a = Array1D.zeros([3], 'int32');
    expect(a.dtype).toBe('int32');
    expect(a.shape).toEqual([3]);
    expect(a.getValues()).toEqual(new Int32Array([0, 0, 0]));
  });

  it('1D bool dtype', () => {
    const a = Array1D.zeros([3], 'bool');
    expect(a.dtype).toBe('bool');
    expect(a.shape).toEqual([3]);
    expect(a.getValues()).toEqual(new Uint8Array([0, 0, 0]));
  });

  it('2D default dtype', () => {
    const a = Array2D.zeros([3, 2]);
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([3, 2]);
    expect(a.getValues()).toEqual(new Float32Array([0, 0, 0, 0, 0, 0]));
  });

  it('2D float32 dtype', () => {
    const a = Array2D.zeros([3, 2], 'float32');
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([3, 2]);
    expect(a.getValues()).toEqual(new Float32Array([0, 0, 0, 0, 0, 0]));
  });

  it('2D int32 dtype', () => {
    const a = Array2D.zeros([3, 2], 'int32');
    expect(a.dtype).toBe('int32');
    expect(a.shape).toEqual([3, 2]);
    expect(a.getValues()).toEqual(new Int32Array([0, 0, 0, 0, 0, 0]));
  });

  it('2D bool dtype', () => {
    const a = Array2D.zeros([3, 2], 'bool');
    expect(a.dtype).toBe('bool');
    expect(a.shape).toEqual([3, 2]);
    expect(a.getValues()).toEqual(new Uint8Array([0, 0, 0, 0, 0, 0]));
  });

  it('3D default dtype', () => {
    const a = Array3D.zeros([2, 2, 2]);
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([2, 2, 2]);
    expect(a.getValues()).toEqual(new Float32Array([0, 0, 0, 0, 0, 0, 0, 0]));
  });

  it('3D float32 dtype', () => {
    const a = Array3D.zeros([2, 2, 2], 'float32');
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([2, 2, 2]);
    expect(a.getValues()).toEqual(new Float32Array([0, 0, 0, 0, 0, 0, 0, 0]));
  });

  it('3D int32 dtype', () => {
    const a = Array3D.zeros([2, 2, 2], 'int32');
    expect(a.dtype).toBe('int32');
    expect(a.shape).toEqual([2, 2, 2]);
    expect(a.getValues()).toEqual(new Int32Array([0, 0, 0, 0, 0, 0, 0, 0]));
  });

  it('3D bool dtype', () => {
    const a = Array3D.zeros([2, 2, 2], 'bool');
    expect(a.dtype).toBe('bool');
    expect(a.shape).toEqual([2, 2, 2]);
    expect(a.getValues()).toEqual(new Uint8Array([0, 0, 0, 0, 0, 0, 0, 0]));
  });

  it('4D default dtype', () => {
    const a = Array4D.zeros([3, 2, 1, 1]);
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([3, 2, 1, 1]);
    expect(a.getValues()).toEqual(new Float32Array([0, 0, 0, 0, 0, 0]));
  });

  it('4D float32 dtype', () => {
    const a = Array4D.zeros([3, 2, 1, 1], 'float32');
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([3, 2, 1, 1]);
    expect(a.getValues()).toEqual(new Float32Array([0, 0, 0, 0, 0, 0]));
  });

  it('4D int32 dtype', () => {
    const a = Array4D.zeros([3, 2, 1, 1], 'int32');
    expect(a.dtype).toBe('int32');
    expect(a.shape).toEqual([3, 2, 1, 1]);
    expect(a.getValues()).toEqual(new Int32Array([0, 0, 0, 0, 0, 0]));
  });

  it('4D bool dtype', () => {
    const a = Array4D.zeros([3, 2, 1, 1], 'bool');
    expect(a.dtype).toBe('bool');
    expect(a.shape).toEqual([3, 2, 1, 1]);
    expect(a.getValues()).toEqual(new Uint8Array([0, 0, 0, 0, 0, 0]));
  });
});

test_util.describeCustom('NDArray.zerosLike', () => {
  it('1D default dtype', () => {
    const a = Array1D.new([1, 2, 3]);
    const b = NDArray.zerosLike(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([3]);
    expect(b.getValues()).toEqual(new Float32Array([0, 0, 0]));
  });

  it('1D float32 dtype', () => {
    const a = Array1D.new([1, 2, 3], 'float32');
    const b = NDArray.zerosLike(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([3]);
    expect(b.getValues()).toEqual(new Float32Array([0, 0, 0]));
  });

  it('1D int32 dtype', () => {
    const a = Array1D.new([1, 2, 3], 'int32');
    const b = NDArray.zerosLike(a);
    expect(b.dtype).toBe('int32');
    expect(b.shape).toEqual([3]);
    expect(b.getValues()).toEqual(new Int32Array([0, 0, 0]));
  });

  it('1D bool dtype', () => {
    const a = Array1D.new([1, 2, 3], 'bool');
    const b = NDArray.zerosLike(a);
    expect(b.dtype).toBe('bool');
    expect(b.shape).toEqual([3]);
    expect(b.getValues()).toEqual(new Uint8Array([0, 0, 0]));
  });

  it('2D default dtype', () => {
    const a = Array2D.new([2, 2], [1, 2, 3, 4]);
    const b = NDArray.zerosLike(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([2, 2]);
    expect(b.getValues()).toEqual(new Float32Array([0, 0, 0, 0]));
  });

  it('2D float32 dtype', () => {
    const a = Array2D.new([2, 2], [1, 2, 3, 4], 'float32');
    const b = NDArray.zerosLike(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([2, 2]);
    expect(b.getValues()).toEqual(new Float32Array([0, 0, 0, 0]));
  });

  it('2D int32 dtype', () => {
    const a = Array2D.new([2, 2], [1, 2, 3, 4], 'int32');
    const b = NDArray.zerosLike(a);
    expect(b.dtype).toBe('int32');
    expect(b.shape).toEqual([2, 2]);
    expect(b.getValues()).toEqual(new Int32Array([0, 0, 0, 0]));
  });

  it('2D bool dtype', () => {
    const a = Array2D.new([2, 2], [1, 2, 3, 4], 'bool');
    const b = NDArray.zerosLike(a);
    expect(b.dtype).toBe('bool');
    expect(b.shape).toEqual([2, 2]);
    expect(b.getValues()).toEqual(new Uint8Array([0, 0, 0, 0]));
  });

  it('3D default dtype', () => {
    const a = Array3D.new([2, 2, 1], [1, 2, 3, 4]);
    const b = NDArray.zerosLike(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([2, 2, 1]);
    expect(b.getValues()).toEqual(new Float32Array([0, 0, 0, 0]));
  });

  it('3D float32 dtype', () => {
    const a = Array3D.new([2, 2, 1], [1, 2, 3, 4], 'float32');
    const b = NDArray.zerosLike(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([2, 2, 1]);
    expect(b.getValues()).toEqual(new Float32Array([0, 0, 0, 0]));
  });

  it('3D int32 dtype', () => {
    const a = Array3D.new([2, 2, 1], [1, 2, 3, 4], 'int32');
    const b = NDArray.zerosLike(a);
    expect(b.dtype).toBe('int32');
    expect(b.shape).toEqual([2, 2, 1]);
    expect(b.getValues()).toEqual(new Int32Array([0, 0, 0, 0]));
  });

  it('3D bool dtype', () => {
    const a = Array3D.new([2, 2, 1], [1, 2, 3, 4], 'bool');
    const b = NDArray.zerosLike(a);
    expect(b.dtype).toBe('bool');
    expect(b.shape).toEqual([2, 2, 1]);
    expect(b.getValues()).toEqual(new Uint8Array([0, 0, 0, 0]));
  });

  it('4D default dtype', () => {
    const a = Array4D.new([2, 2, 1, 1], [1, 2, 3, 4]);
    const b = NDArray.zerosLike(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([2, 2, 1, 1]);
    expect(b.getValues()).toEqual(new Float32Array([0, 0, 0, 0]));
  });

  it('4D float32 dtype', () => {
    const a = Array4D.new([2, 2, 1, 1], [1, 2, 3, 4], 'float32');
    const b = NDArray.zerosLike(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([2, 2, 1, 1]);
    expect(b.getValues()).toEqual(new Float32Array([0, 0, 0, 0]));
  });

  it('4D int32 dtype', () => {
    const a = Array4D.new([2, 2, 1, 1], [1, 2, 3, 4], 'int32');
    const b = NDArray.zerosLike(a);
    expect(b.dtype).toBe('int32');
    expect(b.shape).toEqual([2, 2, 1, 1]);
    expect(b.getValues()).toEqual(new Int32Array([0, 0, 0, 0]));
  });

  it('4D bool dtype', () => {
    const a = Array4D.new([2, 2, 1, 1], [1, 2, 3, 4], 'bool');
    const b = NDArray.zerosLike(a);
    expect(b.dtype).toBe('bool');
    expect(b.shape).toEqual([2, 2, 1, 1]);
    expect(b.getValues()).toEqual(new Uint8Array([0, 0, 0, 0]));
  });
});

test_util.describeCustom('NDArray.like', () => {
  it('1D default dtype', () => {
    const a = Array1D.new([1, 2, 3]);
    const b = NDArray.like(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([3]);
    expect(b.getValues()).toEqual(new Float32Array([1, 2, 3]));
  });

  it('1D float32 dtype', () => {
    const a = Array1D.new([1, 2, 3], 'float32');
    const b = NDArray.like(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([3]);
    expect(b.getValues()).toEqual(new Float32Array([1, 2, 3]));
  });

  it('1D int32 dtype', () => {
    const a = Array1D.new([1, 2, 3], 'int32');
    const b = NDArray.like(a);
    expect(b.dtype).toBe('int32');
    expect(b.shape).toEqual([3]);
    expect(b.getValues()).toEqual(new Int32Array([1, 2, 3]));
  });

  it('1D bool dtype', () => {
    const a = Array1D.new([1, 2, 3], 'bool');
    const b = NDArray.like(a);
    expect(b.dtype).toBe('bool');
    expect(b.shape).toEqual([3]);
    expect(b.getValues()).toEqual(new Uint8Array([1, 1, 1]));
  });

  it('2D default dtype', () => {
    const a = Array2D.new([2, 2], [1, 2, 3, 4]);
    const b = NDArray.like(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([2, 2]);
    expect(b.getValues()).toEqual(new Float32Array([1, 2, 3, 4]));
  });

  it('2D float32 dtype', () => {
    const a = Array2D.new([2, 2], [1, 2, 3, 4], 'float32');
    const b = NDArray.like(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([2, 2]);
    expect(b.getValues()).toEqual(new Float32Array([1, 2, 3, 4]));
  });

  it('2D int32 dtype', () => {
    const a = Array2D.new([2, 2], [1, 2, 3, 4], 'int32');
    const b = NDArray.like(a);
    expect(b.dtype).toBe('int32');
    expect(b.shape).toEqual([2, 2]);
    expect(b.getValues()).toEqual(new Int32Array([1, 2, 3, 4]));
  });

  it('2D bool dtype', () => {
    const a = Array2D.new([2, 2], [1, 2, 3, 4], 'bool');
    const b = NDArray.like(a);
    expect(b.dtype).toBe('bool');
    expect(b.shape).toEqual([2, 2]);
    expect(b.getValues()).toEqual(new Uint8Array([1, 1, 1, 1]));
  });

  it('3D default dtype', () => {
    const a = Array3D.new([2, 2, 1], [1, 2, 3, 4]);
    const b = NDArray.like(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([2, 2, 1]);
    expect(b.getValues()).toEqual(new Float32Array([1, 2, 3, 4]));
  });

  it('3D float32 dtype', () => {
    const a = Array3D.new([2, 2, 1], [1, 2, 3, 4], 'float32');
    const b = NDArray.like(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([2, 2, 1]);
    expect(b.getValues()).toEqual(new Float32Array([1, 2, 3, 4]));
  });

  it('3D int32 dtype', () => {
    const a = Array3D.new([2, 2, 1], [1, 2, 3, 4], 'int32');
    const b = NDArray.like(a);
    expect(b.dtype).toBe('int32');
    expect(b.shape).toEqual([2, 2, 1]);
    expect(b.getValues()).toEqual(new Int32Array([1, 2, 3, 4]));
  });

  it('3D bool dtype', () => {
    const a = Array3D.new([2, 2, 1], [1, 2, 3, 4], 'bool');
    const b = NDArray.like(a);
    expect(b.dtype).toBe('bool');
    expect(b.shape).toEqual([2, 2, 1]);
    expect(b.getValues()).toEqual(new Uint8Array([1, 1, 1, 1]));
  });

  it('4D default dtype', () => {
    const a = Array4D.new([2, 2, 1, 1], [1, 2, 3, 4]);
    const b = NDArray.like(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([2, 2, 1, 1]);
    expect(b.getValues()).toEqual(new Float32Array([1, 2, 3, 4]));
  });

  it('4D float32 dtype', () => {
    const a = Array4D.new([2, 2, 1, 1], [1, 2, 3, 4], 'float32');
    const b = NDArray.like(a);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([2, 2, 1, 1]);
    expect(b.getValues()).toEqual(new Float32Array([1, 2, 3, 4]));
  });

  it('4D int32 dtype', () => {
    const a = Array4D.new([2, 2, 1, 1], [1, 2, 3, 4], 'int32');
    const b = NDArray.like(a);
    expect(b.dtype).toBe('int32');
    expect(b.shape).toEqual([2, 2, 1, 1]);
    expect(b.getValues()).toEqual(new Int32Array([1, 2, 3, 4]));
  });

  it('4D bool dtype', () => {
    const a = Array4D.new([2, 2, 1, 1], [1, 2, 3, 4], 'bool');
    const b = NDArray.like(a);
    expect(b.dtype).toBe('bool');
    expect(b.shape).toEqual([2, 2, 1, 1]);
    expect(b.getValues()).toEqual(new Uint8Array([1, 1, 1, 1]));
  });

});

test_util.describeCustom('Scalar.new', () => {
  it('default dtype', () => {
    const a = Scalar.new(3);
    expect(a.dtype).toBe('float32');
    expect(a.getValues()).toEqual(new Float32Array([3]));
  });

  it('float32 dtype', () => {
    const a = Scalar.new(3, 'float32');
    expect(a.dtype).toBe('float32');
    expect(a.getValues()).toEqual(new Float32Array([3]));
  });

  it('int32 dtype', () => {
    const a = Scalar.new(3, 'int32');
    expect(a.dtype).toBe('int32');
    expect(a.getValues()).toEqual(new Int32Array([3]));
  });

  it('int32 dtype, 3.9 => 3, like numpy', () => {
    const a = Scalar.new(3.9, 'int32');
    expect(a.dtype).toBe('int32');
    expect(a.getValues()).toEqual(new Int32Array([3]));
  });

  it('int32 dtype, -3.9 => -3, like numpy', () => {
    const a = Scalar.new(-3.9, 'int32');
    expect(a.dtype).toBe('int32');
    expect(a.getValues()).toEqual(new Int32Array([-3]));
  });

  it('bool dtype, 3 => true, like numpy', () => {
    const a = Scalar.new(3, 'bool');
    expect(a.dtype).toBe('bool');
    expect(a.get()).toBe(1);
  });

  it('bool dtype, -2 => true, like numpy', () => {
    const a = Scalar.new(-2, 'bool');
    expect(a.dtype).toBe('bool');
    expect(a.get()).toBe(1);
  });

  it('bool dtype, 0 => false, like numpy', () => {
    const a = Scalar.new(0, 'bool');
    expect(a.dtype).toBe('bool');
    expect(a.get()).toBe(0);
  });

  it('bool dtype from boolean', () => {
    const a = Scalar.new(false, 'bool');
    expect(a.get()).toBe(0);
    expect(a.dtype).toBe('bool');

    const b = Scalar.new(true, 'bool');
    expect(b.get()).toBe(1);
    expect(b.dtype).toBe('bool');
  });

  it('int32 dtype from boolean', () => {
    const a = Scalar.new(true, 'int32');
    expect(a.get()).toBe(1);
    expect(a.dtype).toBe('int32');
  });

  it('default dtype from boolean', () => {
    const a = Scalar.new(false);
    expect(a.get()).toBe(0);
    expect(a.dtype).toBe('float32');
  });
});

test_util.describeCustom('Array1D.new', () => {
  it('default dtype', () => {
    const a = Array1D.new([1, 2, 3]);
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([3]);
    expect(a.getValues()).toEqual(new Float32Array([1, 2, 3]));
  });

  it('float32 dtype', () => {
    const a = Array1D.new([1, 2, 3], 'float32');
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([3]);
    expect(a.getValues()).toEqual(new Float32Array([1, 2, 3]));
  });

  it('int32 dtype', () => {
    const a = Array1D.new([1, 2, 3], 'int32');
    expect(a.dtype).toBe('int32');
    expect(a.shape).toEqual([3]);
    expect(a.getValues()).toEqual(new Int32Array([1, 2, 3]));
  });

  it('int32 dtype, non-ints get floored, like numpy', () => {
    const a = Array1D.new([1.1, 2.5, 3.9], 'int32');
    expect(a.dtype).toBe('int32');
    expect(a.shape).toEqual([3]);
    expect(a.getValues()).toEqual(new Int32Array([1, 2, 3]));
  });

  it('int32 dtype, negative non-ints get ceiled, like numpy', () => {
    const a = Array1D.new([-1.1, -2.5, -3.9], 'int32');
    expect(a.dtype).toBe('int32');
    expect(a.shape).toEqual([3]);
    expect(a.getValues()).toEqual(new Int32Array([-1, -2, -3]));
  });

  it('bool dtype, !=0 is truthy, 0 is falsy, like numpy', () => {
    const a = Array1D.new([1, -2, 0, 3], 'bool');
    expect(a.dtype).toBe('bool');
    expect(a.shape).toEqual([4]);
    expect(a.get(0)).toBe(1);
    expect(a.get(1)).toBe(1);
    expect(a.get(2)).toBe(0);
    expect(a.get(3)).toBe(1);
  });

  it('default dtype from boolean[]', () => {
    const a = Array1D.new([false, false, true]);
    expect(a.dtype).toBe('float32');
    expect(a.getValues()).toEqual(new Float32Array([0, 0, 1]));
  });

  it('int32 dtype from boolean[]', () => {
    const a = Array1D.new([false, false, true], 'int32');
    expect(a.dtype).toBe('int32');
    expect(a.getValues()).toEqual(new Int32Array([0, 0, 1]));
  });

  it('bool dtype from boolean[]', () => {
    const a = Array1D.new([false, false, true], 'bool');
    expect(a.dtype).toBe('bool');
    expect(a.getValues()).toEqual(new Uint8Array([0, 0, 1]));
  });
});

test_util.describeCustom('Array2D.new', () => {
  it('default dtype', () => {
    const a = Array2D.new([2, 2], [1, 2, 3, 4]);
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([2, 2]);
    expect(a.getValues()).toEqual(new Float32Array([1, 2, 3, 4]));
  });

  it('float32 dtype', () => {
    const a = Array2D.new([2, 2], [1, 2, 3, 4]);
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([2, 2]);
    expect(a.getValues()).toEqual(new Float32Array([1, 2, 3, 4]));
  });

  it('int32 dtype', () => {
    const a = Array2D.new([2, 2], [[1, 2], [3, 4]], 'int32');
    expect(a.dtype).toBe('int32');
    expect(a.shape).toEqual([2, 2]);
    expect(a.getValues()).toEqual(new Int32Array([1, 2, 3, 4]));
  });

  it('int32 dtype, non-ints get floored, like numpy', () => {
    const a = Array2D.new([2, 2], [1.1, 2.5, 3.9, 4.0], 'int32');
    expect(a.dtype).toBe('int32');
    expect(a.shape).toEqual([2, 2]);
    expect(a.getValues()).toEqual(new Int32Array([1, 2, 3, 4]));
  });

  it('int32 dtype, negative non-ints get ceiled, like numpy', () => {
    const a = Array2D.new([2, 2], [-1.1, -2.5, -3.9, -4.0], 'int32');
    expect(a.dtype).toBe('int32');
    expect(a.shape).toEqual([2, 2]);
    expect(a.getValues()).toEqual(new Int32Array([-1, -2, -3, -4]));
  });

  it('bool dtype, !=0 is truthy, 0 is falsy, like numpy', () => {
    const a = Array2D.new([2, 2], [1, -2, 0, 3], 'bool');
    expect(a.dtype).toBe('bool');
    expect(a.shape).toEqual([2, 2]);
    expect(a.get(0, 0)).toBe(1);
    expect(a.get(0, 1)).toBe(1);
    expect(a.get(1, 0)).toBe(0);
    expect(a.get(1, 1)).toBe(1);
  });

  it('default dtype from boolean[]', () => {
    const a = Array2D.new([2, 2], [[false, false], [true, false]]);
    expect(a.dtype).toBe('float32');
    expect(a.getValues()).toEqual(new Float32Array([0, 0, 1, 0]));
  });

  it('int32 dtype from boolean[]', () => {
    const a = Array2D.new([2, 2], [[false, false], [true, false]], 'int32');
    expect(a.dtype).toBe('int32');
    expect(a.getValues()).toEqual(new Int32Array([0, 0, 1, 0]));
  });

  it('bool dtype from boolean[]', () => {
    const a = Array2D.new([2, 2], [[false, false], [true, false]], 'bool');
    expect(a.dtype).toBe('bool');
    expect(a.getValues()).toEqual(new Uint8Array([0, 0, 1, 0]));
  });
});

test_util.describeCustom('Array3D.new', () => {
  it('default dtype', () => {
    const a = Array3D.new([2, 2, 1], [1, 2, 3, 4]);
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([2, 2, 1]);
    expect(a.getValues()).toEqual(new Float32Array([1, 2, 3, 4]));
  });

  it('float32 dtype', () => {
    const a = Array3D.new([2, 2, 1], [1, 2, 3, 4]);
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([2, 2, 1]);
    expect(a.getValues()).toEqual(new Float32Array([1, 2, 3, 4]));
  });

  it('int32 dtype', () => {
    const a = Array3D.new([2, 2, 1], [[[1], [2]], [[3], [4]]], 'int32');
    expect(a.dtype).toBe('int32');
    expect(a.shape).toEqual([2, 2, 1]);
    expect(a.getValues()).toEqual(new Int32Array([1, 2, 3, 4]));
  });

  it('int32 dtype, non-ints get floored, like numpy', () => {
    const a = Array3D.new([2, 2, 1], [1.1, 2.5, 3.9, 4.0], 'int32');
    expect(a.dtype).toBe('int32');
    expect(a.shape).toEqual([2, 2, 1]);
    expect(a.getValues()).toEqual(new Int32Array([1, 2, 3, 4]));
  });

  it('int32 dtype, negative non-ints get ceiled, like numpy', () => {
    const a = Array3D.new([2, 2, 1], [-1.1, -2.5, -3.9, -4.0], 'int32');
    expect(a.dtype).toBe('int32');
    expect(a.shape).toEqual([2, 2, 1]);
    expect(a.getValues()).toEqual(new Int32Array([-1, -2, -3, -4]));
  });

  it('bool dtype, !=0 is truthy, 0 is falsy, like numpy', () => {
    const a = Array3D.new([2, 2, 1], [1, -2, 0, 3], 'bool');
    expect(a.dtype).toBe('bool');
    expect(a.shape).toEqual([2, 2, 1]);
    expect(a.get(0, 0, 0)).toBe(1);
    expect(a.get(0, 1, 0)).toBe(1);
    expect(a.get(1, 0, 0)).toBe(0);
    expect(a.get(1, 1, 0)).toBe(1);
  });

  it('default dtype from boolean[]', () => {
    const a = Array3D.new([2, 2, 1], [[[false], [false]], [[true], [false]]]);
    expect(a.dtype).toBe('float32');
    expect(a.getValues()).toEqual(new Float32Array([0, 0, 1, 0]));
  });

  it('int32 dtype from boolean[]', () => {
    const a = Array3D.new(
        [2, 2, 1], [[[false], [false]], [[true], [false]]], 'int32');
    expect(a.dtype).toBe('int32');
    expect(a.getValues()).toEqual(new Int32Array([0, 0, 1, 0]));
  });

  it('bool dtype from boolean[]', () => {
    const a =
        Array3D.new([2, 2, 1], [[[false], [false]], [[true], [false]]], 'bool');
    expect(a.dtype).toBe('bool');
    expect(a.getValues()).toEqual(new Uint8Array([0, 0, 1, 0]));
  });
});

test_util.describeCustom('Array4D.new', () => {
  it('default dtype', () => {
    const a = Array4D.new([2, 2, 1, 1], [1, 2, 3, 4]);
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([2, 2, 1, 1]);
    expect(a.getValues()).toEqual(new Float32Array([1, 2, 3, 4]));
  });

  it('float32 dtype', () => {
    const a = Array4D.new([2, 2, 1, 1], [1, 2, 3, 4]);
    expect(a.dtype).toBe('float32');
    expect(a.shape).toEqual([2, 2, 1, 1]);
    expect(a.getValues()).toEqual(new Float32Array([1, 2, 3, 4]));
  });

  it('int32 dtype', () => {
    const a =
        Array4D.new([2, 2, 1, 1], [[[[1]], [[2]]], [[[3]], [[4]]]], 'int32');
    expect(a.dtype).toBe('int32');
    expect(a.shape).toEqual([2, 2, 1, 1]);
    expect(a.getValues()).toEqual(new Int32Array([1, 2, 3, 4]));
  });

  it('int32 dtype, non-ints get floored, like numpy', () => {
    const a = Array4D.new([2, 2, 1, 1], [1.1, 2.5, 3.9, 4.0], 'int32');
    expect(a.dtype).toBe('int32');
    expect(a.shape).toEqual([2, 2, 1, 1]);
    expect(a.getValues()).toEqual(new Int32Array([1, 2, 3, 4]));
  });

  it('int32 dtype, negative non-ints get ceiled, like numpy', () => {
    const a = Array4D.new([2, 2, 1, 1], [-1.1, -2.5, -3.9, -4.0], 'int32');
    expect(a.dtype).toBe('int32');
    expect(a.shape).toEqual([2, 2, 1, 1]);
    expect(a.getValues()).toEqual(new Int32Array([-1, -2, -3, -4]));
  });

  it('bool dtype, !=0 is truthy, 0 is falsy, like numpy', () => {
    const a = Array4D.new([2, 2, 1, 1], [1, -2, 0, 3], 'bool');
    expect(a.dtype).toBe('bool');
    expect(a.shape).toEqual([2, 2, 1, 1]);
    expect(a.get(0, 0, 0, 0)).toBe(1);
    expect(a.get(0, 1, 0, 0)).toBe(1);
    expect(a.get(1, 0, 0, 0)).toBe(0);
    expect(a.get(1, 1, 0, 0)).toBe(1);
  });

  it('default dtype from boolean[]', () => {
    const a =
        Array4D.new([1, 2, 2, 1], [[[[false], [false]], [[true], [false]]]]);
    expect(a.dtype).toBe('float32');
    expect(a.getValues()).toEqual(new Float32Array([0, 0, 1, 0]));
  });

  it('int32 dtype from boolean[]', () => {
    const a = Array4D.new(
        [1, 2, 2, 1], [[[[false], [false]], [[true], [false]]]], 'int32');
    expect(a.dtype).toBe('int32');
    expect(a.getValues()).toEqual(new Int32Array([0, 0, 1, 0]));
  });

  it('bool dtype from boolean[]', () => {
    const a = Array4D.new(
        [1, 2, 2, 1], [[[[false], [false]], [[true], [false]]]], 'bool');
    expect(a.dtype).toBe('bool');
    expect(a.getValues()).toEqual(new Uint8Array([0, 0, 1, 0]));
  });
});

test_util.describeCustom('NDArray.reshape', () => {
  it('Scalar default dtype', () => {
    const a = Scalar.new(4);
    const b = a.reshape([1, 1]);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([1, 1]);
  });

  it('Scalar bool dtype', () => {
    const a = Scalar.new(4, 'bool');
    const b = a.reshape([1, 1, 1]);
    expect(b.dtype).toBe('bool');
    expect(b.shape).toEqual([1, 1, 1]);
  });

  it('Array1D default dtype', () => {
    const a = Array1D.new([1, 2, 3, 4]);
    const b = a.reshape([2, 2]);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([2, 2]);
  });

  it('Array1D int32 dtype', () => {
    const a = Array1D.new([1, 2, 3, 4], 'int32');
    const b = a.reshape([2, 2]);
    expect(b.dtype).toBe('int32');
    expect(b.shape).toEqual([2, 2]);
  });

  it('Array2D default dtype', () => {
    const a = Array2D.new([2, 3], [1, 2, 3, 4, 5, 6]);
    const b = a.reshape([6]);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([6]);
  });

  it('Array2D bool dtype', () => {
    const a = Array2D.new([2, 3], [1, 2, 3, 4, 5, 6], 'bool');
    const b = a.reshape([6]);
    expect(b.dtype).toBe('bool');
    expect(b.shape).toEqual([6]);
  });

  it('Array3D default dtype', () => {
    const a = Array3D.new([2, 3, 1], [1, 2, 3, 4, 5, 6]);
    const b = a.reshape([6]);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([6]);
  });

  it('Array3D bool dtype', () => {
    const a = Array3D.new([2, 3, 1], [1, 2, 3, 4, 5, 6], 'bool');
    const b = a.reshape([6]);
    expect(b.dtype).toBe('bool');
    expect(b.shape).toEqual([6]);
  });

  it('Array4D default dtype', () => {
    const a = Array4D.new([2, 3, 1, 1], [1, 2, 3, 4, 5, 6]);
    const b = a.reshape([2, 3]);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([2, 3]);
  });

  it('Array4D int32 dtype', () => {
    const a = Array4D.new([2, 3, 1, 1], [1, 2, 3, 4, 5, 6], 'int32');
    const b = a.reshape([3, 2]);
    expect(b.dtype).toBe('int32');
    expect(b.shape).toEqual([3, 2]);
  });
});

test_util.describeCustom('NDArray.asXD preserves dtype', () => {
  it('scalar -> 2d', () => {
    const a = Scalar.new(4, 'int32');
    const b = a.as2D(1, 1);
    expect(b.dtype).toBe('int32');
    expect(b.shape).toEqual([1, 1]);
  });

  it('1d -> 2d', () => {
    const a = Array1D.new([4, 2, 1], 'bool');
    const b = a.as2D(3, 1);
    expect(b.dtype).toBe('bool');
    expect(b.shape).toEqual([3, 1]);
  });

  it('2d -> 4d', () => {
    const a = Array2D.new([2, 2], [4, 2, 1, 3]);
    const b = a.as4D(1, 1, 2, 2);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([1, 1, 2, 2]);
  });

  it('3d -> 2d', () => {
    const a = Array3D.new([2, 2, 1], [4, 2, 1, 3], 'float32');
    const b = a.as2D(2, 2);
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([2, 2]);
  });

  it('4d -> 1d', () => {
    const a = Array4D.new([2, 2, 1, 1], [4, 2, 1, 3], 'bool');
    const b = a.as1D();
    expect(b.dtype).toBe('bool');
    expect(b.shape).toEqual([4]);
  });
});

test_util.describeCustom('NDArray.asType', () => {
  it('scalar bool -> int32', () => {
    const a = Scalar.new(true, 'bool').asType('int32');
    expect(a.dtype).toBe('int32');
    expect(a.get()).toBe(1);
  });

  it('array1d float32 -> int32', () => {
    const a = Array1D.new([1.1, 3.9, -2.9, 0]).asType('int32');
    expect(a.dtype).toBe('int32');
    expect(a.getValues()).toEqual(new Int32Array([1, 3, -2, 0]));
  });

  it('array2d float32 -> bool', () => {
    const a = Array2D.new([2, 2], [1.1, 3.9, -2.9, 0]).asType(DType.bool);
    expect(a.dtype).toBe('bool');
    expect(a.get(0, 0)).toBe(1);
    expect(a.get(0, 1)).toBe(1);
    expect(a.get(1, 0)).toBe(1);
    expect(a.get(1, 1)).toBe(0);
  });

  it('array3d bool -> float32', () => {
    const a = Array3D.new([2, 2, 1], [true, false, false, true], 'bool')
                  .asType('float32');
    expect(a.dtype).toBe('float32');
    expect(a.getValues()).toEqual(new Float32Array([1, 0, 0, 1]));
  });
});

test_util.describeCustom('NDArray CPU <--> GPU with dtype', () => {
  it('bool CPU -> GPU -> CPU', () => {
    const a = Array1D.new([1, 2, 0, 0, 5], 'bool');
    expect(a.inGPU()).toBe(false);
    expect(a.getValues()).toEqual(new Uint8Array([1, 1, 0, 0, 1]));

    // Upload to GPU.
    expect(a.getTexture() != null).toBe(true);

    expect(a.inGPU()).toBe(true);

    expect(a.getValues()).toEqual(new Uint8Array([1, 1, 0, 0, 1]));
    a.dispose();
  });

  it('bool GPU --> CPU', () => {
    const shape: [number, number] = [1, 5];
    const texture = textureManager.acquireTexture(shape);
    gpgpu.uploadMatrixToTexture(
        texture, shape[0], shape[1], new Float32Array([1, 1, 0, 0, 1]));

    const a = Array1D.make(shape, {texture, textureShapeRC: shape}, 'bool');
    expect(a.inGPU()).toBe(true);

    expect(a.getValues()).toEqual(new Uint8Array([1, 1, 0, 0, 1]));
    expect(a.inGPU()).toBe(false);
  });

  it('int32 CPU -> GPU -> CPU', () => {
    const a = Array1D.new([1, 2, 0, 0, 5], 'int32');
    expect(a.inGPU()).toBe(false);
    expect(a.getValues()).toEqual(new Int32Array([1, 2, 0, 0, 5]));

    // Upload to GPU.
    expect(a.getTexture() != null).toBe(true);

    expect(a.inGPU()).toBe(true);

    expect(a.getValues()).toEqual(new Int32Array([1, 2, 0, 0, 5]));
    a.dispose();
  });

  it('int32 GPU --> CPU', () => {
    const shape: [number, number] = [1, 5];
    const texture = textureManager.acquireTexture(shape);
    gpgpu.uploadMatrixToTexture(
        texture, shape[0], shape[1], new Float32Array([1, 5.003, 0, 0, 1.001]));

    const a = Array1D.make(shape, {texture, textureShapeRC: shape}, 'int32');
    expect(a.inGPU()).toBe(true);

    expect(a.getValues()).toEqual(new Int32Array([1, 5, 0, 0, 1]));
    expect(a.inGPU()).toBe(false);
  });
}, FEATURES, customBeforeEach, customAfterEach);

// NDArray.rand
test_util.describeCustom('NDArray.rand', () => {
  it('should return a random 1D float32 array', () => {
    const shape: [number] = [10];

    // Enusre defaults to float32 w/o type:
    let result = NDArray.rand(shape, () => util.randUniform(0, 2));
    expect(result.dtype).toBe('float32');
    test_util.expectValuesInRange(result.getValues(), 0, 2);

    result = NDArray.rand(shape, () => util.randUniform(0, 1.5));
    expect(result.dtype).toBe('float32');
    test_util.expectValuesInRange(result.getValues(), 0, 1.5);
  });

  it('should return a random 1D int32 array', () => {
    const shape: [number] = [10];
    const result = NDArray.rand(shape, () => util.randUniform(0, 2), 'int32');
    expect(result.dtype).toBe('int32');
    test_util.expectValuesInRange(result.getValues(), 0, 2);
  });

  it('should return a random 1D bool array', () => {
    const shape: [number] = [10];
    const result = NDArray.rand(shape, () => util.randUniform(0, 1), 'bool');
    expect(result.dtype).toBe('bool');
    test_util.expectValuesInRange(result.getValues(), 0, 1);
  });

  it('should return a random 2D float32 array', () => {
    const shape: [number] = [3, 4];

    // Enusre defaults to float32 w/o type:
    let result = NDArray.rand(shape, () => util.randUniform(0, 2.5));
    expect(result.dtype).toBe('float32');
    test_util.expectValuesInRange(result.getValues(), 0, 2.5);

    result = NDArray.rand(shape, () => util.randUniform(0, 1.5), 'float32');
    expect(result.dtype).toBe('float32');
    test_util.expectValuesInRange(result.getValues(), 0, 1.5);
  });

  it('should return a random 2D int32 array', () => {
    const shape: [number] = [3, 4];
    const result = NDArray.rand(shape, () => util.randUniform(0, 2), 'int32');
    expect(result.dtype).toBe('int32');
    test_util.expectValuesInRange(result.getValues(), 0, 2);
  });

  it('should return a random 2D bool array', () => {
    const shape: [number] = [3, 4];
    const result = NDArray.rand(shape, () => util.randUniform(0, 1), 'bool');
    expect(result.dtype).toBe('bool');
    test_util.expectValuesInRange(result.getValues(), 0, 1);
  });

  it('should return a random 3D float32 array', () => {
    const shape: [number] = [3, 4, 5];

    // Enusre defaults to float32 w/o type:
    let result = NDArray.rand(shape, () => util.randUniform(0, 2.5));
    expect(result.dtype).toBe('float32');
    test_util.expectValuesInRange(result.getValues(), 0, 2.5);

    result = NDArray.rand(shape, () => util.randUniform(0, 1.5), 'float32');
    expect(result.dtype).toBe('float32');
    test_util.expectValuesInRange(result.getValues(), 0, 1.5);
  });

  it('should return a random 3D int32 array', () => {
    const shape: [number] = [3, 4, 5];
    const result = NDArray.rand(shape, () => util.randUniform(0, 2), 'int32');
    expect(result.dtype).toBe('int32');
    test_util.expectValuesInRange(result.getValues(), 0, 2);
  });

  it('should return a random 3D bool array', () => {
    const shape: [number] = [3, 4, 5];
    const result = NDArray.rand(shape, () => util.randUniform(0, 1), 'bool');
    expect(result.dtype).toBe('bool');
    test_util.expectValuesInRange(result.getValues(), 0, 1);
  });

  it('should return a random 4D float32 array', () => {
    const shape: [number] = [3, 4, 5, 6];

    // Enusre defaults to float32 w/o type:
    let result = NDArray.rand(shape, () => util.randUniform(0, 2.5));
    expect(result.dtype).toBe('float32');
    test_util.expectValuesInRange(result.getValues(), 0, 2.5);

    result = NDArray.rand(shape, () => util.randUniform(0, 1.5));
    expect(result.dtype).toBe('float32');
    test_util.expectValuesInRange(result.getValues(), 0, 1.5);
  });

  it('should return a random 4D int32 array', () => {
    const shape: [number] = [3, 4, 5, 6];
    const result = NDArray.rand(shape, () => util.randUniform(0, 2), 'int32');
    expect(result.dtype).toBe('int32');
    test_util.expectValuesInRange(result.getValues(), 0, 2);
  });

  it('should return a random 4D bool array', () => {
    const shape: [number] = [3, 4, 5, 6];
    const result = NDArray.rand(shape, () => util.randUniform(0, 1), 'bool');
    expect(result.dtype).toBe('bool');
    test_util.expectValuesInRange(result.getValues(), 0, 1);
  });
});

// NDArray.randNormal
test_util.describeCustom('NDArray.randNormal', () => {
  const EPSILON_FLOAT32 = 0.05;
  const EPSILON_NONFLOAT = 0.10;

  it('should return a float32 1D of random normal values', () => {
    const SAMPLES = 10000;

    // Ensure defaults to float32.
    let result = NDArray.randNormal([SAMPLES], 0, 0.5);
    expect(result.dtype).toBe('float32');
    expect(result.shape).toEqual([SAMPLES]);
    test_util.jarqueBeraNormalityTest(result.getValues());
    test_util.expectArrayInMeanStdRange(
        result.getValues(), 0, 0.5, EPSILON_FLOAT32);

    result = NDArray.randNormal([SAMPLES], 0, 1.5, 'float32');
    expect(result.dtype).toBe('float32');
    expect(result.shape).toEqual([SAMPLES]);
    test_util.jarqueBeraNormalityTest(result.getValues());
    test_util.expectArrayInMeanStdRange(
        result.getValues(), 0, 1.5, EPSILON_FLOAT32);
  });

  it('should return a int32 1D of random normal values', () => {
    const SAMPLES = 1000;
    const result = NDArray.randNormal([SAMPLES], 0, 1, 'int32');
    expect(result.dtype).toBe('int32');
    expect(result.shape).toEqual([SAMPLES]);
    test_util.jarqueBeraNormalityTest(result.getValues());
    test_util.expectArrayInMeanStdRange(
        result.getValues(), 0, 1, EPSILON_NONFLOAT);
  });

  it('should return a float32 2D of random normal values', () => {
    const SAMPLES = 1000;

    // Ensure defaults to float32.
    let result = Array2D.randNormal([SAMPLES, SAMPLES], 0, 0.5);
    expect(result.dtype).toBe('float32');
    expect(result.shape).toEqual([SAMPLES, SAMPLES]);
    test_util.jarqueBeraNormalityTest(result.getValues());
    test_util.expectArrayInMeanStdRange(
        result.getValues(), 0, 0.5, EPSILON_FLOAT32);

    result = Array2D.randNormal([SAMPLES, SAMPLES], 0, 1.5, 'float32');
    expect(result.dtype).toBe('float32');
    expect(result.shape).toEqual([SAMPLES, SAMPLES]);
    test_util.jarqueBeraNormalityTest(result.getValues());
    test_util.expectArrayInMeanStdRange(
        result.getValues(), 0, 1.5, EPSILON_FLOAT32);
  });

  it('should return a int32 2D of random normal values', () => {
    const SAMPLES = 100;
    const result = Array2D.randNormal([SAMPLES, SAMPLES], 0, 2, 'int32');
    expect(result.dtype).toBe('int32');
    expect(result.shape).toEqual([SAMPLES, SAMPLES]);
    test_util.jarqueBeraNormalityTest(result.getValues());
    test_util.expectArrayInMeanStdRange(
        result.getValues(), 0, 2, EPSILON_NONFLOAT);
  });

  it('should return a float32 3D of random normal values', () => {
    const SAMPLES = 50;

    // Ensure defaults to float32.
    let result = Array3D.randNormal([SAMPLES, SAMPLES, SAMPLES], 0, 0.5);
    expect(result.dtype).toBe('float32');
    expect(result.shape).toEqual([SAMPLES, SAMPLES, SAMPLES]);
    test_util.jarqueBeraNormalityTest(result.getValues());
    test_util.expectArrayInMeanStdRange(
        result.getValues(), 0, 0.5, EPSILON_FLOAT32);

    result = Array3D.randNormal([SAMPLES, SAMPLES, SAMPLES], 0, 1.5, 'float32');
    expect(result.dtype).toBe('float32');
    expect(result.shape).toEqual([SAMPLES, SAMPLES, SAMPLES]);
    test_util.jarqueBeraNormalityTest(result.getValues());
    test_util.expectArrayInMeanStdRange(
        result.getValues(), 0, 1.5, EPSILON_FLOAT32);
  });

  it('should return a int32 3D of random normal values', () => {
    const SAMPLES = 50;
    const result =
        Array3D.randNormal([SAMPLES, SAMPLES, SAMPLES], 0, 2, 'int32');
    expect(result.dtype).toBe('int32');
    expect(result.shape).toEqual([SAMPLES, SAMPLES, SAMPLES]);
    test_util.jarqueBeraNormalityTest(result.getValues());
    test_util.expectArrayInMeanStdRange(
        result.getValues(), 0, 2, EPSILON_NONFLOAT);
  });

  it('should return a float32 4D of random normal values', () => {
    const SAMPLES = 25;

    // Ensure defaults to float32.
    let result =
        Array4D.randNormal([SAMPLES, SAMPLES, SAMPLES, SAMPLES], 0, 0.5);
    expect(result.dtype).toBe('float32');
    expect(result.shape).toEqual([SAMPLES, SAMPLES, SAMPLES, SAMPLES]);
    test_util.jarqueBeraNormalityTest(result.getValues());
    test_util.expectArrayInMeanStdRange(
        result.getValues(), 0, 0.5, EPSILON_FLOAT32);

    result = Array4D.randNormal([SAMPLES, SAMPLES, SAMPLES, SAMPLES], 0, 1.5);
    expect(result.dtype).toBe('float32');
    expect(result.shape).toEqual([SAMPLES, SAMPLES, SAMPLES, SAMPLES]);
    test_util.jarqueBeraNormalityTest(result.getValues());
    test_util.expectArrayInMeanStdRange(
        result.getValues(), 0, 1.5, EPSILON_FLOAT32);
  });

  it('should return a int32 4D of random normal values', () => {
    const SAMPLES = 25;

    const result =
        Array4D.randNormal([SAMPLES, SAMPLES, SAMPLES, SAMPLES], 0, 2, 'int32');
    expect(result.dtype).toBe('int32');
    expect(result.shape).toEqual([SAMPLES, SAMPLES, SAMPLES, SAMPLES]);
    test_util.jarqueBeraNormalityTest(result.getValues());
    test_util.expectArrayInMeanStdRange(
        result.getValues(), 0, 2, EPSILON_NONFLOAT);
  });
});

test_util.describeCustom('NDArray.randTruncatedNormal', () => {
  // Expect higher variances for truncated values.
  // TODO(kreeger): Consider a different gausiann method or seeding JS
  // Math.rand() for precision.
  const EPSILON_FLOAT32 = 0.50;
  const EPSILON_NONFLOAT = 0.55;

  it('should return a random 1D float32 array', () => {
    const shape: [number] = [2000];

    // Ensure defaults to float32 w/o type:
    let result = NDArray.randTruncatedNormal(shape, 0, 3.5);
    expect(result.dtype).toBe('float32');
    test_util.jarqueBeraNormalityTest(result.getValues());
    test_util.expectArrayInMeanStdRange(
        result.getValues(), 0, 3.5, EPSILON_FLOAT32);

    result = NDArray.randTruncatedNormal(shape, 0, 4.5, 'float32');
    expect(result.dtype).toBe('float32');
    test_util.jarqueBeraNormalityTest(result.getValues());
    test_util.expectArrayInMeanStdRange(
        result.getValues(), 0, 4.5, EPSILON_FLOAT32);
  });

  it('should return a randon 1D int32 array', () => {
    const shape: [number] = [1000];
    const result = NDArray.randTruncatedNormal(shape, 0, 5, 'int32');
    expect(result.dtype).toBe('int32');
    test_util.jarqueBeraNormalityTest(result.getValues());
    test_util.expectArrayInMeanStdRange(
        result.getValues(), 0, 5, EPSILON_NONFLOAT);
  });

  it('should return a 2D float32 array', () => {
    const shape: [number, number] = [50, 50];

    // Ensure defaults to float32 w/o type:
    let result = Array2D.randTruncatedNormal(shape, 0, 3.5);
    expect(result.dtype).toBe('float32');
    test_util.jarqueBeraNormalityTest(result.getValues());
    test_util.expectArrayInMeanStdRange(
        result.getValues(), 0, 3.5, EPSILON_FLOAT32);

    result = Array2D.randTruncatedNormal(shape, 0, 4.5, 'float32');
    expect(result.dtype).toBe('float32');
    test_util.jarqueBeraNormalityTest(result.getValues());
    test_util.expectArrayInMeanStdRange(
        result.getValues(), 0, 4.5, EPSILON_FLOAT32);
  });

  it('should return a 2D int32 array', () => {
    const shape: [number, number] = [100, 100];
    const result = Array2D.randTruncatedNormal(shape, 0, 6, 'int32');
    expect(result.dtype).toBe('int32');
    test_util.jarqueBeraNormalityTest(result.getValues());
    test_util.expectArrayInMeanStdRange(
        result.getValues(), 0, 6, EPSILON_NONFLOAT);
  });

  it('should return a 3D float32 array', () => {
    const shape: [number, number, number] = [10, 10, 10];

    // Ensure defaults to float32 w/o type:
    let result = Array3D.randTruncatedNormal(shape, 0, 3.5);
    expect(result.dtype).toBe('float32');
    test_util.jarqueBeraNormalityTest(result.getValues());
    test_util.expectArrayInMeanStdRange(
        result.getValues(), 0, 3.5, EPSILON_FLOAT32);

    result = Array3D.randTruncatedNormal(shape, 0, 4.5, 'float32');
    expect(result.dtype).toBe('float32');
    test_util.jarqueBeraNormalityTest(result.getValues());
    test_util.expectArrayInMeanStdRange(
        result.getValues(), 0, 4.5, EPSILON_FLOAT32);
  });

  it('should return a 3D int32 array', () => {
    const shape: [number, number, number] = [10, 10, 10];
    const result = Array3D.randTruncatedNormal(shape, 0, 5, 'int32');
    expect(result.dtype).toBe('int32');
    test_util.jarqueBeraNormalityTest(result.getValues());
    test_util.expectArrayInMeanStdRange(
        result.getValues(), 0, 5, EPSILON_NONFLOAT);
  });

  it('should return a 4D float32 array', () => {
    const shape: [number, number, number, number] = [10, 10, 10, 10];

    // Ensure defaults to float32 w/o type:
    let result = Array4D.randTruncatedNormal(shape, 0, 3.5);
    expect(result.dtype).toBe('float32');
    test_util.jarqueBeraNormalityTest(result.getValues());
    test_util.expectArrayInMeanStdRange(
        result.getValues(), 0, 3.5, EPSILON_FLOAT32);

    result = Array4D.randTruncatedNormal(shape, 0, 4.5, 'float32');
    expect(result.dtype).toBe('float32');
    test_util.jarqueBeraNormalityTest(result.getValues());
    test_util.expectArrayInMeanStdRange(
        result.getValues(), 0, 4.5, EPSILON_FLOAT32);
  });

  it('should return a 4D int32 array', () => {
    const shape: [number, number, number, number] = [10, 10, 10, 10];
    const result = Array4D.randTruncatedNormal(shape, 0, 5, 'int32');
    expect(result.dtype).toBe('int32');
    test_util.jarqueBeraNormalityTest(result.getValues());
    test_util.expectArrayInMeanStdRange(
        result.getValues(), 0, 5, EPSILON_NONFLOAT);
  });
});

// NDArray.randUniform()
test_util.describeCustom('NDArray.randUniform', () => {
  it('should return a random 1D float32 array', () => {
    const shape: [number] = [10];

    // Enusre defaults to float32 w/o type:
    let result = NDArray.randUniform(shape, 0, 2.5);
    expect(result.dtype).toBe('float32');
    test_util.expectValuesInRange(result.getValues(), 0, 2.5);

    result = NDArray.randUniform(shape, 0, 1.5, 'float32');
    expect(result.dtype).toBe('float32');
    test_util.expectValuesInRange(result.getValues(), 0, 1.5);
  });

  it('should return a random 1D int32 array', () => {
    const shape: [number] = [10];
    const result = NDArray.randUniform(shape, 0, 2, 'int32');
    expect(result.dtype).toBe('int32');
    test_util.expectValuesInRange(result.getValues(), 0, 2);
  });

  it('should return a random 1D bool array', () => {
    const shape: [number] = [10];
    const result = NDArray.randUniform(shape, 0, 1, 'bool');
    expect(result.dtype).toBe('bool');
    test_util.expectValuesInRange(result.getValues(), 0, 1);
  });

  it('should return a random 2D float32 array', () => {
    const shape: [number, number] = [3, 4];

    // Enusre defaults to float32 w/o type:
    let result = Array2D.randUniform(shape, 0, 2.5);
    expect(result.dtype).toBe('float32');
    test_util.expectValuesInRange(result.getValues(), 0, 2.5);

    result = Array2D.randUniform(shape, 0, 1.5, 'float32');
    expect(result.dtype).toBe('float32');
    test_util.expectValuesInRange(result.getValues(), 0, 1.5);
  });

  it('should return a random 2D int32 array', () => {
    const shape: [number, number] = [3, 4];
    const result = Array2D.randUniform(shape, 0, 2, 'int32');
    expect(result.dtype).toBe('int32');
    test_util.expectValuesInRange(result.getValues(), 0, 2);
  });

  it('should return a random 2D bool array', () => {
    const shape: [number, number] = [3, 4];
    const result = Array2D.randUniform(shape, 0, 1, 'bool');
    expect(result.dtype).toBe('bool');
    test_util.expectValuesInRange(result.getValues(), 0, 1);
  });

  it('should return a random 3D float32 array', () => {
    const shape: [number, number, number] = [3, 4, 5];

    // Enusre defaults to float32 w/o type:
    let result = Array3D.randUniform(shape, 0, 2.5);
    expect(result.dtype).toBe('float32');
    test_util.expectValuesInRange(result.getValues(), 0, 2.5);

    result = Array3D.randUniform(shape, 0, 1.5, 'float32');
    expect(result.dtype).toBe('float32');
    test_util.expectValuesInRange(result.getValues(), 0, 1.5);
  });

  it('should return a random 3D int32 array', () => {
    const shape: [number, number, number] = [3, 4, 5];
    const result = Array3D.randUniform(shape, 0, 2, 'int32');
    expect(result.dtype).toBe('int32');
    test_util.expectValuesInRange(result.getValues(), 0, 2);
  });

  it('should return a random 3D bool array', () => {
    const shape: [number, number, number] = [3, 4, 5];
    const result = Array3D.randUniform(shape, 0, 1, 'bool');
    expect(result.dtype).toBe('bool');
    test_util.expectValuesInRange(result.getValues(), 0, 1);
  });

  it('should return a random 4D float32 array', () => {
    const shape: [number, number, number, number] = [3, 4, 5, 6];

    // Enusre defaults to float32 w/o type:
    let result = Array4D.randUniform(shape, 0, 2.5);
    expect(result.dtype).toBe('float32');
    test_util.expectValuesInRange(result.getValues(), 0, 2.5);

    result = Array4D.randUniform(shape, 0, 1.5, 'float32');
    expect(result.dtype).toBe('float32');
    test_util.expectValuesInRange(result.getValues(), 0, 1.5);
  });

  it('should return a random 4D int32 array', () => {
    const shape: [number, number, number, number] = [3, 4, 5, 6];
    const result = Array4D.randUniform(shape, 0, 2, 'int32');
    expect(result.dtype).toBe('int32');
    test_util.expectValuesInRange(result.getValues(), 0, 2);
  });

  it('should return a random 4D bool array', () => {
    const shape: [number, number, number, number] = [3, 4, 5, 6];
    const result = Array4D.randUniform(shape, 0, 1, 'bool');
    expect(result.dtype).toBe('bool');
    test_util.expectValuesInRange(result.getValues(), 0, 1);
  });
});

// NDArray.fromPixels
{
  test_util.describeCustom(
      'NDArray.fromPixels',
      () => {
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

        it('ImageData 1x1x3', () => {
          const pixels = new ImageData(1, 1);
          pixels.data[0] = 0;
          pixels.data[1] = 80;
          pixels.data[2] = 160;
          pixels.data[3] = 240;

          const array = Array3D.fromPixels(pixels, 3);

          test_util.expectArraysClose(
              array.getValues(), new Float32Array([0, 80, 160]));
        });

        it('ImageData 1x1x4', () => {
          const pixels = new ImageData(1, 1);
          pixels.data[0] = 0;
          pixels.data[1] = 80;
          pixels.data[2] = 160;
          pixels.data[3] = 240;

          const array = Array3D.fromPixels(pixels, 4);

          test_util.expectArraysClose(
              array.getValues(), new Float32Array([0, 80, 160, 240]));
        });

        it('ImageData 2x2x3', () => {
          const pixels = new ImageData(2, 2);

          for (let i = 0; i < 8; i++) {
            pixels.data[i] = i * 2;
          }
          for (let i = 8; i < 16; i++) {
            pixels.data[i] = i * 2;
          }

          const array = Array3D.fromPixels(pixels, 3);

          test_util.expectArraysClose(
              array.getValues(),
              new Float32Array([0, 2, 4, 8, 10, 12, 16, 18, 20, 24, 26, 28]));
        });

        it('ImageData 2x2x4', () => {
          const pixels = new ImageData(2, 2);
          for (let i = 0; i < 8; i++) {
            pixels.data[i] = i * 2;
          }
          for (let i = 8; i < 16; i++) {
            pixels.data[i] = i * 2;
          }

          const array = Array3D.fromPixels(pixels, 4);

          test_util.expectArraysClose(
              array.getValues(),
              new Float32Array(
                  [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]));
        });
      },
      [
        {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
        {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
        {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
      ]);
}
