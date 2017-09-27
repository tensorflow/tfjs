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

import {NDArrayMathGPU} from './math_gpu';
import {Array1D, Array2D, Array3D, Array4D, Scalar} from './ndarray';
import * as webgl_util from './webgl/webgl_util';


describe('NDArrayMathGPU scope', () => {
  let math: NDArrayMathGPU;
  beforeEach(() => {
    math = new NDArrayMathGPU();
  });

  it('scope returns NDArray', () => {
    const a = Array1D.new([1, 2, 3]);
    let b = Array1D.new([0, 0, 0]);

    const numUsedTexturesBefore = math.getTextureManager().getNumUsedTextures();

    math.scope(() => {
      const result = math.scope(() => {
        b = math.add(a, b) as Array1D;
        b = math.add(a, b) as Array1D;
        b = math.add(a, b) as Array1D;
        return math.add(a, b);
      });

      // a, b, and result are new textures. All intermediates should be
      // disposed.
      expect(math.getTextureManager().getNumUsedTextures())
          .toEqual(numUsedTexturesBefore + 3);
      expect(result.getValues()).toEqual(new Float32Array([4, 8, 12]));
    });

    // a, b are new textures, result should be disposed.
    expect(math.getTextureManager().getNumUsedTextures())
        .toEqual(numUsedTexturesBefore + 2);
    a.dispose();
    b.dispose();
  });

  it('scope returns NDArray[]', () => {
    const a = Array1D.new([1, 2, 3]);
    const b = Array1D.new([0, -1, 1]);

    const numUsedTexturesBefore = math.getTextureManager().getNumUsedTextures();

    math.scope(() => {
      const result = math.scope(() => {
        math.add(a, b);
        return [math.add(a, b), math.sub(a, b)];
      });

      // a, b, and 2 results are new textures. All intermediates should be
      // disposed.
      expect(math.getTextureManager().getNumUsedTextures())
          .toEqual(numUsedTexturesBefore + 4);
      expect(result[0].getValues()).toEqual(new Float32Array([1, 1, 4]));
      expect(result[1].getValues()).toEqual(new Float32Array([1, 3, 2]));
    });

    // a, b are new textures, result should be disposed.
    expect(math.getTextureManager().getNumUsedTextures())
        .toEqual(numUsedTexturesBefore + 2);
    a.dispose();
    b.dispose();
  });

  it('basic scope usage without return', () => {
    const a = Array1D.new([1, 2, 3]);
    let b = Array1D.new([0, 0, 0]);

    const numUsedTexturesBefore = math.getTextureManager().getNumUsedTextures();

    math.scope(() => {
      b = math.add(a, b) as Array1D;
      b = math.add(a, b) as Array1D;
      b = math.add(a, b) as Array1D;
      math.add(a, b);
    });

    const numUsedTexturesAfter = math.getTextureManager().getNumUsedTextures();

    // original a and b, all intermediates should be disposed.
    expect(numUsedTexturesAfter).toEqual(numUsedTexturesBefore + 2);
  });

  it('nested scope usage', () => {
    const a = Array1D.new([1, 2, 3]);
    let b = Array1D.new([0, 0, 0]);

    const numUsedTexturesBefore = math.getTextureManager().getNumUsedTextures();

    math.scope(() => {
      const result = math.scope(() => {
        b = math.add(a, b) as Array1D;
        b = math.scope(() => {
          b = math.scope(() => {
            return math.add(a, b) as Array1D;
          });
          // a, original b, and two intermediate textures should be the only
          // textures.
          expect(math.getTextureManager().getNumUsedTextures())
              .toEqual(numUsedTexturesBefore + 4);

          math.scope(() => {
            math.add(a, b);
          });
          // All the intermediates should be cleaned up.
          expect(math.getTextureManager().getNumUsedTextures())
              .toEqual(numUsedTexturesBefore + 4);

          return math.add(a, b) as Array1D;
        });
        expect(math.getTextureManager().getNumUsedTextures())
            .toEqual(numUsedTexturesBefore + 4);

        return math.add(a, b) as Array1D;
      });

      // a, b, and result are new textures. All intermediates should be
      // disposed.
      expect(math.getTextureManager().getNumUsedTextures())
          .toEqual(numUsedTexturesBefore + 3);
      expect(result.getValues()).toEqual(new Float32Array([4, 8, 12]));
    });
    // a, b, are new textures, result should be disposed.
    expect(math.getTextureManager().getNumUsedTextures())
        .toEqual(numUsedTexturesBefore + 2);
  });
});

describe('NDArrayMathGPU clone', () => {
  let math: NDArrayMathGPU;
  beforeEach(() => {
    math = new NDArrayMathGPU();
    math.startScope();
  });

  afterEach(() => {
    math.endScope(null);
    math.dispose();
  });

  it('returns a ndarray with the same shape and value', () => {
    const a = Array2D.new([3, 3], [1, 2, 3, 4, 5, 6, 7, 8, 9]);
    const aPrime = math.clone(a);
    expect(aPrime.shape).toEqual(a.shape);
    expect(aPrime.getValues()).toEqual(a.getValues());
    a.dispose();
  });

  it('returns a ndarray with a different texture handle', () => {
    const a = Array2D.new([3, 3], [1, 2, 3, 4, 5, 6, 7, 8, 9]);
    const aPrime = math.clone(a);
    expect(a.inGPU()).toEqual(true);
    expect(aPrime.inGPU()).toEqual(true);
    expect(aPrime.getTexture()).not.toBe(a.getTexture());
    a.dispose();
  });
});

describe('NDArrayMathCPU slice1D', () => {
  let math: NDArrayMathGPU;
  beforeEach(() => {
    math = new NDArrayMathGPU();
    math.startScope();
  });

  afterEach(() => {
    math.endScope(null);
    math.dispose();
  });

  it('slices 1x1 into 1x1 (effectively a copy)', () => {
    const a = Array1D.new([5]);
    const result = math.slice1D(a, 0, 1);
    expect(result.shape).toEqual([1]);
    expect(result.get(0)).toBe(5);
  });

  it('slices 5x1 into shape 2x1 starting at 3', () => {
    const a = Array1D.new([1, 2, 3, 4, 5]);
    const result = math.slice1D(a, 3, 2);
    expect(result.shape).toEqual([2]);
    expect(result.getValues()).toEqual(new Float32Array([4, 5]));
  });

  it('slices 5x1 into shape 3x1 starting at 1', () => {
    const a = Array1D.new([1, 2, 3, 4, 5]);
    const result = math.slice1D(a, 1, 3);
    expect(result.shape).toEqual([3]);
    expect(result.getValues()).toEqual(new Float32Array([2, 3, 4]));
  });
});

describe('NDArrayMathGPU slice2D', () => {
  let math: NDArrayMathGPU;
  beforeEach(() => {
    math = new NDArrayMathGPU();
    math.startScope();
  });

  afterEach(() => {
    math.endScope(null);
    math.dispose();
  });

  it('slicing a 1x1 from a 1x1 returns a 1x1', () => {
    const a = Array2D.new([1, 1], [0]);
    const b = math.slice2D(a, [0, 0], [1, 1]);
    expect(b.shape).toEqual([1, 1]);
    a.dispose();
  });

  it('returns a ndarray of slice size', () => {
    const a = Array2D.zeros([100, 100]);
    const b = math.slice2D(a, [0, 0], [12, 34]);
    expect(b.shape).toEqual([12, 34]);
    a.dispose();
  });

  it('returns the upper-left submatrix when begin is [0, 0]', () => {
    const a = Array2D.randUniform([10, 10], -1, 1);
    const b = math.slice2D(a, [0, 0], [2, 2]);
    const aValues = a.getValues();
    const expected =
        new Float32Array([aValues[0], aValues[1], aValues[10], aValues[11]]);
    test_util.expectArraysClose(b.getValues(), expected);
    a.dispose();
  });

  it('returns the rectangle specified', () => {
    const a = Array2D.new([4, 3], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
    const b = math.slice2D(a, [1, 1], [3, 2]);
    const expected = new Float32Array([5, 6, 8, 9, 11, 12]);
    expect(b.getValues()).toEqual(expected);
    a.dispose();
  });

  it('throws when requesting out of bounds slice', () => {
    const a = Array2D.new([4, 3], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
    expect(() => math.slice2D(a, [1, 1], [10, 10])).toThrowError();
    a.dispose();
  });
});

describe('NDArrayMathCPU slice3D', () => {
  let math: NDArrayMathGPU;
  beforeEach(() => {
    math = new NDArrayMathGPU();
    math.startScope();
  });

  afterEach(() => {
    math.endScope(null);
    math.dispose();
  });

  it('slices 1x1x1 into shape 1x1x1 (effectively a copy)', () => {
    const a = Array3D.new([1, 1, 1], [[[5]]]);
    const result = math.slice3D(a, [0, 0, 0], [1, 1, 1]);
    expect(result.shape).toEqual([1, 1, 1]);
    expect(result.get(0, 0, 0)).toBe(5);
  });

  it('slices 2x2x2 array into 1x2x2 starting at [1, 0, 0]', () => {
    const a = Array3D.new([2, 2, 2], [1, 2, 3, 4, 5, 6, 7, 8]);
    const result = math.slice3D(a, [1, 0, 0], [1, 2, 2]);
    expect(result.shape).toEqual([1, 2, 2]);
    expect(result.getValues()).toEqual(new Float32Array([5, 6, 7, 8]));
  });

  it('slices 2x2x2 array into 2x1x1 starting at [0, 1, 1]', () => {
    const a = Array3D.new([2, 2, 2], [1, 2, 3, 4, 5, 6, 7, 8]);
    const result = math.slice3D(a, [0, 1, 1], [2, 1, 1]);
    expect(result.shape).toEqual([2, 1, 1]);
    expect(result.getValues()).toEqual(new Float32Array([4, 8]));
  });
});

describe('NDArrayMathCPU slice4D', () => {
  let math: NDArrayMathGPU;
  beforeEach(() => {
    math = new NDArrayMathGPU();
    math.startScope();
  });

  afterEach(() => {
    math.endScope(null);
    math.dispose();
  });

  it('slices 1x1x1x1 into shape 1x1x1x1 (effectively a copy)', () => {
    const a = Array4D.new([1, 1, 1, 1], [[[[5]]]]);
    const result = math.slice4D(a, [0, 0, 0, 0], [1, 1, 1, 1]);
    expect(result.shape).toEqual([1, 1, 1, 1]);
    expect(result.get(0, 0, 0, 0)).toBe(5);
  });

  it('slices 2x2x2x2 array into 1x2x2x2 starting at [1, 0, 0, 0]', () => {
    const a = Array4D.new(
        [2, 2, 2, 2], [1, 2, 3, 4, 5, 6, 7, 8, 11, 22, 33, 44, 55, 66, 77, 88]);
    const result = math.slice4D(a, [1, 0, 0, 0], [1, 2, 2, 2]);
    expect(result.shape).toEqual([1, 2, 2, 2]);
    expect(result.getValues()).toEqual(new Float32Array([
      11, 22, 33, 44, 55, 66, 77, 88
    ]));
  });

  it('slices 2x2x2x2 array into 2x1x1x1 starting at [0, 1, 1, 1]', () => {
    const a = Array4D.new(
        [2, 2, 2, 2], [1, 2, 3, 4, 5, 6, 7, 8, 11, 22, 33, 44, 55, 66, 77, 88]);
    const result = math.slice4D(a, [0, 1, 1, 1], [2, 1, 1, 1]);
    expect(result.shape).toEqual([2, 1, 1, 1]);
    expect(result.getValues()).toEqual(new Float32Array([8, 88]));
  });
});

describe('NDArrayMathGPU copy2D', () => {
  let math: NDArrayMathGPU;
  beforeEach(() => {
    math = new NDArrayMathGPU();
    math.startScope();
  });

  afterEach(() => {
    math.endScope(null);
    math.dispose();
  });

  it('throws an error if source and dest shapes have different areas', () => {
    const source = Array2D.zeros([100, 100]);
    const dest = Array2D.zeros([100, 100]);
    const sourceSize: [number, number] = [20, 20];
    const destSize: [number, number] = [5, 5];
    expect(
        () => math.copy2D(source, [0, 0], sourceSize, dest, [0, 0], destSize))
        .toThrowError();
    source.dispose();
    dest.dispose();
  });

  it('copies a src shape into a dst shape', () => {
    const source = Array2D.new([3, 4], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
    const dest = Array2D.zeros([6, 2]);
    math.copy2D(source, [1, 1], [2, 3], dest, [2, 0], [3, 2]);
    expect(dest.getValues()).toEqual(new Float32Array([
      0, 0, 0, 0, 6, 7, 8, 10, 11, 12, 0, 0
    ]));
    source.dispose();
    dest.dispose();
  });

  it('throws when requesting out of bounds source copy', () => {
    const source = Array2D.new([3, 4], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
    const dest = Array2D.zeros([6, 2]);

    expect(() => math.copy2D(source, [1, 1], [10, 10], dest, [2, 0], [
      3, 2
    ])).toThrowError();
    source.dispose();
    dest.dispose();
  });

  it('throws when requesting out of bounds dest copy', () => {
    const source = Array2D.new([3, 4], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
    const dest = Array2D.zeros([6, 2]);

    expect(() => math.copy2D(source, [1, 1], [2, 3], dest, [2, 0], [
      3, 10
    ])).toThrowError();
    source.dispose();
    dest.dispose();
  });
});

describe('NDArrayMathGPU scaledNDArrayAdd', () => {
  let math: NDArrayMathGPU;
  beforeEach(() => {
    math = new NDArrayMathGPU();
    math.startScope();
  });

  afterEach(() => {
    math.endScope(null);
    math.dispose();
  });

  it('with 2D ndarrays', () => {
    const a = Array2D.new([2, 3], [2, 4, 6, 8, 10, 12]);
    const b = Array2D.new([2, 3], [1, 2, 3, 4, 5, 6]);
    const c1 = Scalar.new(3);
    const c2 = Scalar.new(2);

    const expected = new Float32Array([8, 16, 24, 32, 40, 48]);
    const result = math.scaledArrayAdd<Array2D>(c1, a, c2, b);

    expect(result.shape).toEqual([2, 3]);
    expect(result.getValues()).toEqual(expected);

    a.dispose();
    b.dispose();
    c1.dispose();
    c2.dispose();
  });

  it('with 3D ndarrays', () => {
    const a = Array3D.new([2, 2, 2], [2, 4, 6, 8, 10, 12, 3, 5]);
    const b = Array3D.new([2, 2, 2], [1, 2, 3, 4, 5, 6, 7, 8]);
    const c1 = Scalar.new(3);
    const c2 = Scalar.new(2);

    const expected = new Float32Array([8, 16, 24, 32, 40, 48, 23, 31]);
    const result = math.scaledArrayAdd<Array3D>(c1, a, c2, b);

    expect(result.shape).toEqual([2, 2, 2]);
    expect(result.getValues()).toEqual(expected);

    a.dispose();
    b.dispose();
    c1.dispose();
    c2.dispose();
  });

  it('throws when passed non-scalars', () => {
    const a = Array2D.new([2, 3], [2, 4, 6, 8, 10, 12]);
    const b = Array2D.new([2, 3], [1, 2, 3, 4, 5, 6]);
    // tslint:disable-next-line:no-any
    const c1: any = Array1D.randNormal([10]);
    const c2 = Scalar.new(2);

    expect(() => math.scaledArrayAdd<Array2D>(c1 as Scalar, a, c2, b))
        .toThrowError();
    expect(() => math.scaledArrayAdd<Array2D>(c2, a, c1 as Scalar, b))
        .toThrowError();

    a.dispose();
    b.dispose();
    c1.dispose();
    c2.dispose();
  });

  it('throws when NDArrays are different shape', () => {
    const a = Array2D.new([2, 3], [2, 4, 6, 8, 10, 12]);
    const b = Array2D.new([2, 4], [1, 2, 3, 4, 5, 6, 7, 8]);
    const c1 = Scalar.new(3);
    const c2 = Scalar.new(2);

    expect(() => math.scaledArrayAdd<Array2D>(c1, a, c2, b)).toThrowError();

    a.dispose();
    b.dispose();
    c1.dispose();
    c2.dispose();
  });
});

describe('NDArrayMathGPU concat1D', () => {
  let math: NDArrayMathGPU;

  beforeEach(() => {
    math = new NDArrayMathGPU();
    math.startScope();
  });

  afterEach(() => {
    math.endScope(null);
    math.dispose();
  });

  it('3 + 5', () => {
    const a = Array1D.new([3]);
    const b = Array1D.new([5]);

    const result = math.concat1D(a, b);
    const expected = new Float32Array([3, 5]);
    test_util.expectArraysClose(result.getValues(), expected);
  });

  it('3 + [5,7]', () => {
    const a = Array1D.new([3]);
    const b = Array1D.new([5, 7]);

    const result = math.concat1D(a, b);
    const expected = new Float32Array([3, 5, 7]);
    test_util.expectArraysClose(result.getValues(), expected);
  });

  it('[3,5] + 7', () => {
    const a = Array1D.new([3, 5]);
    const b = Array1D.new([7]);

    const result = math.concat1D(a, b);
    const expected = new Float32Array([3, 5, 7]);
    test_util.expectArraysClose(result.getValues(), expected);
  });
});

describe('NDArrayMathGPU concat2D', () => {
  let math: NDArrayMathGPU;

  beforeEach(() => {
    math = new NDArrayMathGPU();
    math.startScope();
  });

  afterEach(() => {
    math.endScope(null);
    math.dispose();
  });

  it('[[3]] + [[5]], axis=0', () => {
    const axis = 0;
    const a = Array2D.new([1, 1], [3]);
    const b = Array2D.new([1, 1], [5]);

    const result = math.concat2D(a, b, axis);
    const expected = new Float32Array([3, 5]);

    expect(result.shape).toEqual([2, 1]);
    test_util.expectArraysClose(result.getValues(), expected);
  });

  it('[[3]] + [[5]], axis=1', () => {
    const axis = 1;
    const a = Array2D.new([1, 1], [3]);
    const b = Array2D.new([1, 1], [5]);

    const result = math.concat2D(a, b, axis);
    const expected = new Float32Array([3, 5]);

    expect(result.shape).toEqual([1, 2]);
    test_util.expectArraysClose(result.getValues(), expected);
  });

  it('[[1, 2], [3, 4]] + [[5, 6]], axis=0', () => {
    const axis = 0;
    const a = Array2D.new([2, 2], [[1, 2], [3, 4]]);
    const b = Array2D.new([1, 2], [[5, 6]]);

    const result = math.concat2D(a, b, axis);
    const expected = new Float32Array([1, 2, 3, 4, 5, 6]);

    expect(result.shape).toEqual([3, 2]);
    test_util.expectArraysClose(result.getValues(), expected);
  });

  it('[[1, 2], [3, 4]] + [[5, 6]], axis=1 throws error', () => {
    const axis = 1;
    const a = Array2D.new([2, 2], [[1, 2], [3, 4]]);
    const b = Array2D.new([1, 2], [[5, 6]]);

    expect(() => math.concat2D(a, b, axis)).toThrowError();
  });

  it('[[1, 2], [3, 4]] + [[5, 6], [7, 8]], axis=1', () => {
    const axis = 1;
    const a = Array2D.new([2, 2], [[1, 2], [3, 4]]);
    const b = Array2D.new([2, 2], [[5, 6], [7, 8]]);

    const result = math.concat2D(a, b, axis);
    const expected = new Float32Array([1, 2, 5, 6, 3, 4, 7, 8]);

    expect(result.shape).toEqual([2, 4]);
    test_util.expectArraysClose(result.getValues(), expected);
  });
});

describe('NDArrayMathGPU concat3D', () => {
  let math: NDArrayMathGPU;
  beforeEach(() => {
    math = new NDArrayMathGPU();
    math.startScope();
  });

  afterEach(() => {
    math.endScope(null);
    math.dispose();
  });

  it('concat axis=0', () => {
    const axis = 0;
    const x1 = Array3D.new([1, 2, 3], [1, 11, 111, 2, 22, 222]);
    const x2 = Array3D.new(
        [2, 2, 3], [5, 55, 555, 6, 66, 666, 7, 77, 777, 8, 88, 888]);
    const y = math.concat3D(x1, x2, axis);

    expect(y.shape).toEqual([3, 2, 3]);
    expect(y.getValues()).toEqual(new Float32Array([
      1, 11, 111, 2, 22, 222, 5, 55, 555, 6, 66, 666, 7, 77, 777, 8, 88, 888
    ]));
  });

  it('concat axis=1', () => {
    const axis = 1;
    const x1 = Array3D.new([2, 1, 3], [1, 11, 111, 3, 33, 333]);
    const x2 = Array3D.new(
        [2, 2, 3], [5, 55, 555, 6, 66, 666, 7, 77, 777, 8, 88, 888]);
    const result = math.concat3D(x1, x2, axis);

    expect(result.shape).toEqual([2, 3, 3]);
    expect(result.getValues()).toEqual(new Float32Array([
      1, 11, 111, 5, 55, 555, 6, 66, 666, 3, 33, 333, 7, 77, 777, 8, 88, 888
    ]));
  });

  it('concat axis=2', () => {
    const axis = 2;
    const x1 = Array3D.new([2, 2, 2], [1, 11, 2, 22, 3, 33, 4, 44]);
    const x2 = Array3D.new(
        [2, 2, 3], [5, 55, 555, 6, 66, 666, 7, 77, 777, 8, 88, 888]);
    const result = math.concat3D(x1, x2, axis);

    expect(result.shape).toEqual([2, 2, 5]);
    expect(result.getValues()).toEqual(new Float32Array([
      1, 11, 5, 55, 555, 2, 22, 6, 66, 666,
      3, 33, 7, 77, 777, 4, 44, 8, 88, 888
    ]));
  });

  it('concat throws when invalid non-axis shapes, axis=0', () => {
    const axis = 0;
    const x1 = Array3D.new([1, 1, 3], [1, 11, 111]);
    const x2 = Array3D.new(
        [2, 2, 3], [5, 55, 555, 6, 66, 666, 7, 77, 777, 8, 88, 888]);
    expect(() => math.concat3D(x1, x2, axis)).toThrowError();
  });

  it('concat throws when invalid non-axis shapes, axis=1', () => {
    const axis = 1;
    const x1 = Array3D.new([1, 1, 3], [1, 11, 111]);
    const x2 = Array3D.new(
        [2, 2, 3], [5, 55, 555, 6, 66, 666, 7, 77, 777, 8, 88, 888]);
    expect(() => math.concat3D(x1, x2, axis)).toThrowError();
  });

  it('concat throws when invalid non-axis shapes, axis=2', () => {
    const axis = 2;
    const x1 = Array3D.new([1, 2, 2], [1, 11, 2, 22]);
    const x2 = Array3D.new(
        [2, 2, 3], [5, 55, 555, 6, 66, 666, 7, 77, 777, 8, 88, 888]);
    expect(() => math.concat3D(x1, x2, axis)).toThrowError();
  });
});

describe('NDArrayMathGPU matMul', () => {
  let math: NDArrayMathGPU;
  beforeEach(() => {
    math = new NDArrayMathGPU();
    math.startScope();
  });

  afterEach(() => {
    math.endScope(null);
    math.dispose();
  });

  it('multiplies matrices', () => {
    const a = Array2D.new([2, 3], [1, 2, 3, 4, 5, 6]);
    const b = Array2D.new([3, 2], [0, 1, -3, 2, 2, 1]);
    const c = math.matMul(a, b);
    expect(c.shape).toEqual([2, 2]);
    expect(c.getValues()).toEqual(new Float32Array([0, 8, -3, 20]));

    a.dispose();
    b.dispose();
    c.dispose();
  });

  it('with implicit texture reshaping on GPU', () => {
    const a = Array2D.new([2, 3], [1, 2, 3, 4, 5, 6]);
    // Make the texture shape different than the logical shape on purpose.
    expect(a.getTextureShapeRC([6, 1])).toEqual([6, 1]);

    const b = Array2D.new([3, 2], [1, 3, 0, 1, 2, 0]);
    expect(b.getTextureShapeRC()).toEqual([3, 2]);

    // Matmul should do implicit texture reshape on ndarray A in order to
    // do the right logical multiplication.
    const result = math.matMul(a, b);
    expect(result.shape).toEqual([2, 2]);
    expect(result.getTextureShapeRC()).toEqual([2, 2]);
    expect(result.getValues()).toEqual(new Float32Array([7, 5, 16, 17]));
    a.dispose();
    b.dispose();
  });

  it('matmul throws when inner dimensions dont match', () => {
    const a = Array2D.new([2, 3], [1, 2, 3, 4, 5, 6]);
    const b = Array2D.new([4, 2], [0, 1, -3, 2, 2, 1, 2, 2]);
    expect(() => math.matMul(a, b)).toThrowError();

    a.dispose();
    b.dispose();
  });

  it('matmul throws when passed non matrices', () => {
    // tslint:disable-next-line:no-any
    const a: any =
        Array3D.new([2, 3, 2], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
    const b = Array2D.new([4, 2], [0, 1, -3, 2, 2, 1, 2, 2]);
    expect(() => math.matMul(a, b)).toThrowError();
    expect(() => math.matMul(b, a)).toThrowError();

    a.dispose();
    b.dispose();
  });

  it('Vector times matrix', () => {
    const v = Array1D.new([2, 3]);
    const matrix = Array2D.new([2, 2], [1, 2, 3, 4]);
    const result = math.vectorTimesMatrix(v, matrix);

    const expected = new Float32Array([11, 16]);
    expect(result.getValues()).toEqual(expected);
    v.dispose();
    matrix.dispose();
    result.dispose();
  });

  it('Vector times matrix with implicit reshape', () => {
    const v = Array1D.new([2, 3]);
    // Make the texture shape be column on purpose.
    expect(v.getTextureShapeRC([2, 1])).toEqual([2, 1]);

    const matrix = Array2D.new([2, 2], [1, 2, 3, 4]);
    const result = math.vectorTimesMatrix(v, matrix);

    const expected = new Float32Array([11, 16]);
    expect(result.getValues()).toEqual(expected);
    v.dispose();
    matrix.dispose();
  });

  it('Vector times matrix throws when not passed a vector', () => {
    // tslint:disable-next-line:no-any
    const v: any = Array2D.new([2, 2], [1, 2, 3, 4]);
    const matrix = Array2D.new([2, 2], [1, 2, 3, 4]);
    expect(() => math.vectorTimesMatrix(v, matrix)).toThrowError();
  });

  it('Vector times matrix throws when not passed a matrix', () => {
    const v = Array1D.new([2, 3]);
    // tslint:disable-next-line:no-any
    const matrix: any = Array3D.new([2, 2, 2], [1, 2, 3, 4, 5, 6, 7, 8]);
    expect(() => math.vectorTimesMatrix(v, matrix)).toThrowError();
  });

  it('Matrix times vector', () => {
    const matrix = Array2D.new([2, 2], [1, 2, 3, 4]);
    const v = Array1D.new([2, 3]);
    const result = math.matrixTimesVector(matrix, v);

    const expected = new Float32Array([8, 18]);
    expect(result.getValues()).toEqual(expected);
    matrix.dispose();
    v.dispose();
  });

  it('Matrix times vector, larger than max texture size', () => {
    const maxTexSize =
        webgl_util.queryMaxTextureSize(math.getGPGPUContext().gl);
    const matrix = Array2D.zeros([1, maxTexSize + 4]);
    matrix.fill(1);
    const v = Array1D.zeros([maxTexSize + 4]);
    v.fill(1);
    const result = math.matrixTimesVector(matrix, v);
    const expected = new Float32Array([maxTexSize + 4]);
    expect(result.getValues()).toEqual(expected);

    matrix.dispose();
    v.dispose();
  });

  it('Matrix * vector propagates NaNs', () => {
    const matrix = Array2D.new([2, 2], [1, 2, 3, 4]);
    const v = Array1D.new([2, NaN]);
    const result = math.matrixTimesVector(matrix, v);

    const expected = new Float32Array([NaN, NaN]);
    expect(result.getValues()).toEqual(expected);

    matrix.dispose();
    v.dispose();
  });

  it('Matrix times vector with implicit reshape', () => {
    const matrix = Array2D.new([2, 2], [1, 2, 3, 4]);
    const v = Array1D.new([2, 3]);
    // Make the texture shape be row on purpose.
    expect(v.getTextureShapeRC([1, 2])).toEqual([1, 2]);
    const result = math.matrixTimesVector(matrix, v);

    const expected = new Float32Array([8, 18]);
    expect(result.getValues()).toEqual(expected);
    matrix.dispose();
    v.dispose();
  });

  it('matrix times vector throws when not passed a vector', () => {
    // tslint:disable-next-line:no-any
    const v: any = Array2D.new([2, 2], [1, 2, 3, 4]);
    const matrix = Array2D.new([2, 2], [1, 2, 3, 4]);
    expect(() => math.matrixTimesVector(matrix, v)).toThrowError();
  });

  it('matrix times vector throws when not passed a matrix', () => {
    const v = Array1D.new([2, 3]);
    // tslint:disable-next-line:no-any
    const matrix: any = Array3D.new([2, 2, 2], [1, 2, 3, 4, 5, 6, 7, 8]);
    expect(() => math.matrixTimesVector(matrix, v)).toThrowError();
  });

  it('Dot product', () => {
    const v1 = Array1D.new([2, 3]);
    const v2 = Array1D.new([2, 1]);
    const result = math.dotProduct(v1, v2);

    expect(result.get()).toEqual(7);
    v1.dispose();
    v2.dispose();
    result.dispose();
  });

  it('Dot product propagates NaNs', () => {
    const v1 = Array1D.new([2, NaN]);
    const v2 = Array1D.new([2, 1]);
    const result = math.dotProduct(v1, v2);
    expect(result.get()).toEqual(NaN);

    v1.dispose();
    v2.dispose();
  });

  it('Dot product with implicit reshaping', () => {
    const v1 = Array1D.new([2, 3]);
    // Make the texture shape be column on purpose.
    expect(v1.getTextureShapeRC([2, 1])).toEqual([2, 1]);

    const v2 = Array1D.new([2, 1]);
    // Make the texture shape be row on purpose.
    expect(v2.getTextureShapeRC([1, 2])).toEqual([1, 2]);

    const result = math.dotProduct(v1, v2);
    expect(result.get()).toEqual(7);
    v1.dispose();
    v2.dispose();
  });

  it('Dot product throws when vectors are different size', () => {
    const v1 = Array1D.new([2, 3, 3]);
    const v2 = Array1D.new([2, 1]);
    expect(() => math.dotProduct(v1, v2)).toThrowError();
    expect(() => math.dotProduct(v2, v1)).toThrowError();

    v1.dispose();
    v2.dispose();
  });

  it('Dot product throws when passed non vectors', () => {
    // tslint:disable-next-line:no-any
    const v1: any = Array2D.new([2, 2], [1, 2, 3, 3]);
    const v2 = Array1D.new([2, 1]);
    expect(() => math.dotProduct(v1, v2)).toThrowError();
    expect(() => math.dotProduct(v2, v1)).toThrowError();

    v1.dispose();
    v2.dispose();
  });

  it('Outer product', () => {
    const v1 = Array1D.new([2, 3]);
    const v2 = Array1D.new([2, 1]);
    const result = math.outerProduct(v1, v2);

    const expected = new Float32Array([4, 2, 6, 3]);
    expect(result.shape).toEqual([2, 2]);
    expect(result.getValues()).toEqual(expected);
    v1.dispose();
    v2.dispose();
  });

  it('Outer product with implicit reshape', () => {
    const v1 = Array1D.new([2, 3]);
    // Make the texture shape be row on purpose.
    expect(v1.getTextureShapeRC([1, 2])).toEqual([1, 2]);

    const v2 = Array1D.new([2, 1]);
    // Make the texture shape be column on purpose.
    expect(v2.getTextureShapeRC([2, 1])).toEqual([2, 1]);

    const result = math.outerProduct(v1, v2);
    const expected = new Float32Array([4, 2, 6, 3]);
    expect(result.shape).toEqual([2, 2]);
    expect(result.getValues()).toEqual(expected);
    v1.dispose();
    v2.dispose();
  });
});

describe('NDArrayMathGPU element-wise mul/div', () => {
  let math: NDArrayMathGPU;
  beforeEach(() => {
    math = new NDArrayMathGPU();
    math.startScope();
  });

  afterEach(() => {
    math.endScope(null);
    math.dispose();
  });

  it('multiplies same-shaped ndarrays', () => {
    const a = Array2D.new([2, 2], [1, 2, -3, -4]);
    const b = Array2D.new([2, 2], [5, 3, 4, -7]);
    const expected = new Float32Array([5, 6, -12, 28]);
    const result = math.elementWiseMul(a, b);

    expect(result.shape).toEqual([2, 2]);
    expect(result.inGPU()).toBe(true);
    expect(result.getValues()).toEqual(expected);
    expect(result.inGPU()).toBe(false);

    a.dispose();
    b.dispose();
  });

  it('propagates NaNs', () => {
    const a = Array2D.new([2, 2], [1, 3, 4, 0]);
    const b = Array2D.new([2, 2], [NaN, 3, NaN, 3]);
    const result = math.elementWiseMul(a, b).getValues();
    expect(result).toEqual(new Float32Array([NaN, 9, NaN, 0]));

    a.dispose();
    b.dispose();
  });

  it('mul throws when passed ndarrays of different shapes', () => {
    const a = Array2D.new([2, 3], [1, 2, -3, -4, 5, 6]);
    const b = Array2D.new([2, 2], [5, 3, 4, -7]);
    expect(() => math.elementWiseMul(a, b)).toThrowError();
    expect(() => math.elementWiseMul(b, a)).toThrowError();

    a.dispose();
    b.dispose();
  });

  it('divide', () => {
    const a = Array2D.new([2, 3], [1, 2, 3, 4, 5, 6]);
    const c = Array2D.new([2, 3], [1, 2, 3, 4, 2, 5]);
    const r = math.divide(a, c);

    expect(r.get(0, 0)).toBeCloseTo(1);
    expect(r.get(0, 1)).toBeCloseTo(1);
    expect(r.get(0, 2)).toBeCloseTo(1);
    expect(r.get(1, 0)).toBeCloseTo(1);
    expect(r.get(1, 1)).toBeCloseTo(2.5);
    expect(r.get(1, 2)).toBeCloseTo(6 / 5);

    a.dispose();
    c.dispose();
  });

  it('divide propagates NaNs', () => {
    const a = Array2D.new([2, 1], [1, 2]);
    const c = Array2D.new([2, 1], [3, NaN]);
    const r = math.divide(a, c).getValues();
    expect(r[0]).toBeCloseTo(1 / 3);
    expect(r[1]).toEqual(NaN);

    a.dispose();
    c.dispose();
  });

  it('div throws when passed ndarrays of different shapes', () => {
    const a = Array2D.new([2, 3], [1, 2, -3, -4, 5, 6]);
    const b = Array2D.new([2, 2], [5, 3, 4, -7]);
    expect(() => math.divide(a, b)).toThrowError();
    expect(() => math.divide(b, a)).toThrowError();

    a.dispose();
    b.dispose();
  });

  it('scalar divided by array', () => {
    const c = Scalar.new(2);
    const a = Array2D.new([2, 3], [1, 2, 3, 4, 5, 6]);

    const r = math.scalarDividedByArray(c, a);

    expect(r.get(0, 0)).toBeCloseTo(2 / 1);
    expect(r.get(0, 1)).toBeCloseTo(2 / 2);
    expect(r.get(0, 2)).toBeCloseTo(2 / 3);
    expect(r.get(1, 0)).toBeCloseTo(2 / 4);
    expect(r.get(1, 1)).toBeCloseTo(2 / 5);
    expect(r.get(1, 2)).toBeCloseTo(2 / 6);

    a.dispose();
    c.dispose();
  });

  it('scalar divided by array propagates NaNs', () => {
    const c = Scalar.new(NaN);
    const a = Array2D.new([1, 3], [1, 2, 3]);
    const r = math.scalarDividedByArray(c, a).getValues();
    expect(r).toEqual(new Float32Array([NaN, NaN, NaN]));

    a.dispose();
    c.dispose();
  });

  it('scalar divided by array throws when passed non scalar', () => {
    // tslint:disable-next-line:no-any
    const c: any = Array1D.new([1, 2, 3]);
    const a = Array2D.new([2, 3], [1, 2, 3, 4, 5, 6]);

    expect(() => math.scalarDividedByArray(c, a)).toThrowError();

    a.dispose();
    c.dispose();
  });

  it('array divided by scalar', () => {
    const a = Array2D.new([2, 3], [1, 2, 3, 4, 5, 6]);
    const c = Scalar.new(2);

    const r = math.arrayDividedByScalar(a, c);

    expect(r.get(0, 0)).toBeCloseTo(1 / 2);
    expect(r.get(0, 1)).toBeCloseTo(2 / 2);
    expect(r.get(0, 2)).toBeCloseTo(3 / 2);
    expect(r.get(1, 0)).toBeCloseTo(4 / 2);
    expect(r.get(1, 1)).toBeCloseTo(5 / 2);
    expect(r.get(1, 2)).toBeCloseTo(6 / 2);

    a.dispose();
    c.dispose();
  });

  it('array divided by scalar propagates NaNs', () => {
    const a = Array2D.new([1, 3], [1, 2, NaN]);
    const c = Scalar.new(2);
    const r = math.arrayDividedByScalar(a, c).getValues();
    expect(r[0]).toBeCloseTo(1 / 2);
    expect(r[1]).toBeCloseTo(2 / 2);
    expect(r[2]).toEqual(NaN);

    a.dispose();
    c.dispose();
  });

  it('array divided by scalar throws when passed non scalar', () => {
    // tslint:disable-next-line:no-any
    const c: any = Array1D.new([1, 2, 3]);
    const a = Array2D.new([2, 3], [1, 2, 3, 4, 5, 6]);

    expect(() => math.arrayDividedByScalar(a, c)).toThrowError();

    a.dispose();
    c.dispose();
  });
});

describe('NDArrayMathGPU unary ops', () => {
  let math: NDArrayMathGPU;
  beforeEach(() => {
    math = new NDArrayMathGPU();
    math.startScope();
  });

  afterEach(() => {
    math.endScope(null);
    math.dispose();
  });

  it('relu', () => {
    const a = Array1D.new([1, -2, 0, 3, -0.1]);
    const result = math.relu(a);
    expect(result.getValues()).toEqual(new Float32Array([1, 0, 0, 3, 0]));

    a.dispose();
  });

  it('relu propagates NaNs', () => {
    const a = Array1D.new([1, -2, 0, 3, -0.1, NaN]);
    const result = math.relu(a);
    expect(result.getValues()).toEqual(new Float32Array([1, 0, 0, 3, 0, NaN]));
    a.dispose();
  });

  it('abs', () => {
    const a = Array1D.new([1, -2, 0, 3, -0.1]);
    const result = math.abs(a);
    expect(result.getValues()).toEqual(new Float32Array([1, 2, 0, 3, 0.1]));

    a.dispose();
  });

  it('abs propagates NaNs', () => {
    const a = Array1D.new([1, -2, 0, 3, -0.1, NaN]);
    const result = math.abs(a);
    expect(result.getValues()).toEqual(new Float32Array([
      1, 2, 0, 3, 0.1, NaN
    ]));
    a.dispose();
  });

  it('step with 1d ndarray', () => {
    const a = Array1D.new([1, -2, 0, 3, -0.1]);
    const result = math.step(a);
    expect(result.getValues()).toEqual(new Float32Array([1, 0, 0, 1, 0]));

    a.dispose();
  });

  it('step with 2d ndarray', () => {
    const a = Array2D.new([2, 2], [1, -5, -3, 4]);
    const result = math.step(a);

    expect(result.shape).toEqual([2, 2]);
    expect(result.getValues()).toEqual(new Float32Array([1, 0, 0, 1]));

    a.dispose();
  });

  it('step propagates NaNs', () => {
    const a = Array1D.new([1, -2, 0, 3, NaN]);
    const result = math.step(a);
    expect(result.getValues()).toEqual(new Float32Array([1, 0, 0, 1, NaN]));
    a.dispose();
  });

  it('neg', () => {
    const a = Array1D.new([1, -3, 2, 7, -4]);
    const result = math.neg(a);
    expect(result.getValues()).toEqual(new Float32Array([-1, 3, -2, -7, 4]));

    a.dispose();
  });

  it('neg propagate NaNs', () => {
    const a = Array1D.new([1, -3, 2, 7, NaN]);
    const expected = [-1, 3, -2, -7, NaN];
    expect(math.neg(a).getValues()).toEqual(new Float32Array(expected));
    a.dispose();
  });

  it('sigmoid', () => {
    const values = [1, -3, 2, 7, -4];
    const a = Array1D.new(values);
    const result = math.sigmoid(a);
    const expected = new Float32Array(a.size);
    for (let i = 0; i < a.size; i++) {
      expected[i] = 1 / (1 + Math.exp(-values[i]));
    }
    test_util.expectArraysClose(result.getValues(), expected);

    a.dispose();
  });

  it('sigmoid propagates NaNs', () => {
    const a = Array1D.new([3, NaN]);
    const res = math.sigmoid(a).getValues();
    test_util.expectArraysClose(
        res, new Float32Array([1 / (1 + Math.exp(-3)), NaN]));
    a.dispose();
  });

  it('sin', () => {
    const values = [1, -3, 2, 7, -4];
    const a = Array1D.new(values);
    const result = math.sin(a);
    const expected = new Float32Array(a.size);
    for (let i = 0; i < a.size; i++) {
      expected[i] = Math.sin(values[i]);
    }
    test_util.expectArraysClose(result.getValues(), expected);

    a.dispose();
  });

  it('sin propagates NaNs', () => {
    const a = Array1D.new([4, NaN, 0]);
    const res = math.sin(a).getValues();
    const expected = [Math.sin(4), NaN, Math.sin(0)];
    test_util.expectArraysClose(res, new Float32Array(expected));
    a.dispose();
  });

  it('cos', () => {
    const values = [1, -3, 2, 7, -4];
    const a = Array1D.new(values);
    const result = math.cos(a);
    const expected = new Float32Array(a.size);
    for (let i = 0; i < a.size; i++) {
      expected[i] = Math.cos(values[i]);
    }
    test_util.expectArraysClose(result.getValues(), expected);

    a.dispose();
  });

  it('cos propagates NaNs', () => {
    const a = Array1D.new([4, NaN, 0]);
    const res = math.cos(a).getValues();
    const expected = [Math.cos(4), NaN, Math.cos(0)];
    test_util.expectArraysClose(res, new Float32Array(expected));
    a.dispose();
  });

  it('tan', () => {
    const values = [1, -3, 2, 7, -4];
    const a = Array1D.new(values);
    const result = math.tan(a);
    const expected = new Float32Array(a.size);
    for (let i = 0; i < a.size; i++) {
      expected[i] = Math.tan(values[i]);
    }
    test_util.expectArraysClose(result.getValues(), expected, 1e-3);

    a.dispose();
  });

  it('tan propagates NaNs', () => {
    const a = Array1D.new([4, NaN, 0]);
    const res = math.tan(a).getValues();
    const expected = [Math.tan(4), NaN, Math.tan(0)];
    test_util.expectArraysClose(res, new Float32Array(expected));
    a.dispose();
  });

  it('asin', () => {
    const values = [1, -3, 2, 7, -4];
    const a = Array1D.new(values);
    const result = math.asin(a);
    const expected = new Float32Array(a.size);
    for (let i = 0; i < a.size; i++) {
      expected[i] = Math.asin(values[i]);
    }
    test_util.expectArraysClose(result.getValues(), expected, 1e-3);

    a.dispose();
  });

  it('asin propagates NaNs', () => {
    const a = Array1D.new([4, NaN, 0]);
    const res = math.asin(a).getValues();
    const expected = [Math.asin(4), NaN, Math.asin(0)];
    test_util.expectArraysClose(res, new Float32Array(expected));
    a.dispose();
  });

  it('acos', () => {
    const values = [1, -3, 2, 7, -4];
    const a = Array1D.new(values);
    const result = math.acos(a);
    const expected = new Float32Array(a.size);
    for (let i = 0; i < a.size; i++) {
      expected[i] = Math.acos(values[i]);
    }
    test_util.expectArraysClose(result.getValues(), expected, 1e-3);

    a.dispose();
  });

  it('acos propagates NaNs', () => {
    const a = Array1D.new([4, NaN, 0]);
    const res = math.acos(a).getValues();
    const expected = [Math.acos(4), NaN, Math.acos(0)];
    test_util.expectArraysClose(res, new Float32Array(expected));
    a.dispose();
  });

  it('atan', () => {
    const values = [1, -3, 2, 7, -4];
    const a = Array1D.new(values);
    const result = math.atan(a);
    const expected = new Float32Array(a.size);
    for (let i = 0; i < a.size; i++) {
      expected[i] = Math.atan(values[i]);
    }
    test_util.expectArraysClose(result.getValues(), expected, 1e-3);

    a.dispose();
  });

  it('atan propagates NaNs', () => {
    const a = Array1D.new([4, NaN, 0]);
    const res = math.atan(a).getValues();
    const expected = [Math.atan(4), NaN, Math.atan(0)];
    test_util.expectArraysClose(res, new Float32Array(expected));
    a.dispose();
  });

  it('sinh', () => {
    const values = [1, -3, 2, 7, -4];
    const a = Array1D.new(values);
    const result = math.sinh(a);
    const expected = new Float32Array(a.size);
    for (let i = 0; i < a.size; i++) {
      expected[i] = Math.sinh(values[i]);
    }
    test_util.expectArraysClose(result.getValues(), expected, 1e-3);

    a.dispose();
  });

  it('sinh propagates NaNs', () => {
    const a = Array1D.new([4, NaN, 0]);
    const res = math.sinh(a).getValues();
    const expected = [Math.sinh(4), NaN, Math.sinh(0)];
    test_util.expectArraysClose(res, new Float32Array(expected));
    a.dispose();
  });

  it('cosh', () => {
    const values = [1, -3, 2, 7, -4];
    const a = Array1D.new(values);
    const result = math.cosh(a);
    const expected = new Float32Array(a.size);
    for (let i = 0; i < a.size; i++) {
      expected[i] = Math.cosh(values[i]);
    }
    test_util.expectArraysClose(result.getValues(), expected, 1e-3);

    a.dispose();
  });

  it('cosh propagates NaNs', () => {
    const a = Array1D.new([4, NaN, 0]);
    const res = math.cosh(a).getValues();
    const expected = [Math.cosh(4), NaN, Math.cosh(0)];
    test_util.expectArraysClose(res, new Float32Array(expected));
    a.dispose();
  });

  it('tanh', () => {
    const values = [1, -3, 2, 7, -4];
    const a = Array1D.new(values);
    const result = math.tanh(a);
    const expected = new Float32Array(a.size);
    for (let i = 0; i < a.size; i++) {
      expected[i] = util.tanh(values[i]);
    }
    test_util.expectArraysClose(result.getValues(), expected);

    a.dispose();
  });

  it('tanh propagates NaNs', () => {
    const a = Array1D.new([4, NaN, 0]);
    const res = math.tanh(a).getValues();
    const expected = [util.tanh(4), NaN, util.tanh(0)];
    test_util.expectArraysClose(res, new Float32Array(expected));
    a.dispose();
  });
});

describe('NDArrayMathGPU min/max', () => {
  let math: NDArrayMathGPU;
  beforeEach(() => {
    math = new NDArrayMathGPU();
    math.startScope();
  });

  afterEach(() => {
    math.endScope(null);
    math.dispose();
  });

  it('max with one element dominating', () => {
    const a = Array1D.new([3, -1, 0, 100, -7, 2]);
    const r = math.max(a);

    expect(r.get()).toBe(100);

    a.dispose();
  });

  it('max with all elements being the same', () => {
    const a = Array1D.new([3, 3, 3]);
    const r = math.max(a);
    expect(r.get()).toBe(3);

    a.dispose();
  });

  it('max propagates NaNs', () => {
    expect(math.max(Array1D.new([3, NaN, 2])).get()).toEqual(NaN);
  });

  it('min Array1D', () => {
    const a = Array1D.new([3, -1, 0, 100, -7, 2]);
    expect(math.min(a).get()).toBe(-7);
    a.dispose();
  });

  it('min propagates NaNs', () => {
    const a = Array1D.new([3, NaN, 2]);
    expect(math.min(a).get()).toEqual(NaN);
    a.dispose();
  });
});

describe('NDArrayMathGPU scalar and element-wise', () => {
  let math: NDArrayMathGPU;
  beforeEach(() => {
    math = new NDArrayMathGPU();
    math.startScope();
  });

  afterEach(() => {
    math.endScope(null);
    math.dispose();
  });

  it('c + A', () => {
    const c = Scalar.new(5);
    const a = Array1D.new([1, 2, 3]);
    const result = math.scalarPlusArray(c, a);
    expect(result.getValues()).toEqual(new Float32Array([6, 7, 8]));

    a.dispose();
    c.dispose();
  });

  it('c + A propagates NaNs', () => {
    const c = Scalar.new(NaN);
    const a = Array1D.new([1, 2, 3]);
    const res = math.scalarPlusArray(c, a).getValues();
    expect(res).toEqual(new Float32Array([NaN, NaN, NaN]));
    a.dispose();
    c.dispose();
  });

  it('c + A throws when passed non scalar', () => {
    // tslint:disable-next-line:no-any
    const c: any = Array1D.new([1, 2, 3]);
    const a = Array1D.new([1, 2, 3]);
    expect(() => math.scalarPlusArray(c, a)).toThrowError();

    a.dispose();
    c.dispose();
  });

  it('c - A', () => {
    const c = Scalar.new(5);
    const a = Array1D.new([7, 2, 3]);
    const result = math.scalarMinusArray(c, a);
    expect(result.getValues()).toEqual(new Float32Array([-2, 3, 2]));

    a.dispose();
    c.dispose();
  });

  it('c - A throws when passed non scalar', () => {
    // tslint:disable-next-line:no-any
    const c: any = Array1D.new([1, 2, 3]);
    const a = Array1D.new([1, 2, 3]);
    expect(() => math.scalarMinusArray(c, a)).toThrowError();

    a.dispose();
    c.dispose();
  });

  it('A - c', () => {
    const a = Array1D.new([1, 2, -3]);
    const c = Scalar.new(5);
    const result = math.arrayMinusScalar(a, c);
    expect(result.getValues()).toEqual(new Float32Array([-4, -3, -8]));

    a.dispose();
    c.dispose();
    result.dispose();
  });

  it('A - c propagates NaNs', () => {
    const a = Array1D.new([1, NaN, 3]);
    const c = Scalar.new(5);
    const res = math.arrayMinusScalar(a, c).getValues();
    expect(res).toEqual(new Float32Array([-4, NaN, -2]));
    a.dispose();
    c.dispose();
  });

  it('A - c throws when passed non scalar', () => {
    // tslint:disable-next-line:no-any
    const c: any = Array1D.new([1, 2, 3]);
    const a = Array1D.new([1, 2, 3]);
    expect(() => math.arrayMinusScalar(a, c)).toThrowError();

    a.dispose();
    c.dispose();
  });

  it('A - B', () => {
    const a = Array1D.new([2, 5, 1]);
    const b = Array1D.new([4, 2, -1]);
    const expected = new Float32Array([-2, 3, 2]);
    const result = math.sub(a, b);

    expect(result.getValues()).toEqual(expected);

    a.dispose();
    b.dispose();
  });

  it('A - B propagates NaNs', () => {
    const a = Array1D.new([2, 5, 1]);
    const b = Array1D.new([4, NaN, -1]);
    const res = math.sub(a, b).getValues();
    expect(res).toEqual(new Float32Array([-2, NaN, 2]));

    a.dispose();
    b.dispose();
  });

  it('A - B throws when passed ndarrays with different shape', () => {
    const a = Array1D.new([2, 5, 1, 5]);
    const b = Array1D.new([4, 2, -1]);
    expect(() => math.sub(a, b)).toThrowError();
    expect(() => math.sub(b, a)).toThrowError();

    a.dispose();
    b.dispose();
  });

  it('A + B', () => {
    const a = Array1D.new([2, 5, 1]);
    const b = Array1D.new([4, 2, -1]);
    const expected = new Float32Array([6, 7, 0]);
    const result = math.add(a, b);

    expect(result.getValues()).toEqual(expected);

    a.dispose();
    b.dispose();
  });

  it('A + B propagates NaNs', () => {
    const a = Array1D.new([2, 5, NaN]);
    const b = Array1D.new([4, 2, -1]);
    const res = math.add(a, b).getValues();
    expect(res).toEqual(new Float32Array([6, 7, NaN]));

    a.dispose();
    b.dispose();
  });

  it('A + B throws when passed ndarrays with different shape', () => {
    const a = Array1D.new([2, 5, 1, 5]);
    const b = Array1D.new([4, 2, -1]);
    expect(() => math.add(a, b)).toThrowError();
    expect(() => math.add(b, a)).toThrowError();

    a.dispose();
    b.dispose();
  });
});

describe('NDArrayMathGPU scalarTimesNDArray', () => {
  let math: NDArrayMathGPU;
  beforeEach(() => {
    math = new NDArrayMathGPU();
    math.startScope();
  });

  afterEach(() => {
    math.endScope(null);
    math.dispose();
  });

  it('scalar times ndarray', () => {
    const a = Array2D.new([3, 2], [2, -5, 1, 1, 4, 0]);
    const c = Scalar.new(2);
    const expected = new Float32Array([4, -10, 2, 2, 8, 0]);
    const result = math.scalarTimesArray(c, a);

    expect(result.shape).toEqual([3, 2]);
    expect(result.getValues()).toEqual(expected);

    a.dispose();
    c.dispose();
  });

  it('scalar times ndarray throws when passed non-scalar', () => {
    const a = Array2D.new([3, 2], [2, -5, 1, 1, 4, 0]);
    // tslint:disable-next-line:no-any
    const c: any = Array1D.new([1, 2, 3, 4]);
    expect(() => math.scalarTimesArray(c, a)).toThrowError();

    a.dispose();
    c.dispose();
  });
});

describe('NDArrayMathGPU log/exp', () => {
  let math: NDArrayMathGPU;
  beforeEach(() => {
    math = new NDArrayMathGPU();
    math.startScope();
  });

  afterEach(() => {
    math.endScope(null);
    math.dispose();
  });

  it('exp', () => {
    const a = Array1D.new([1, 2, 0]);
    const r = math.exp(a);

    expect(r.get(0)).toBeCloseTo(Math.exp(1));
    expect(r.get(1)).toBeCloseTo(Math.exp(2));
    expect(r.get(2)).toBeCloseTo(1);

    a.dispose();
  });

  it('exp propagates NaNs', () => {
    const a = Array1D.new([1, NaN, 0]);
    const r = math.exp(a).getValues();
    expect(r).toEqual(new Float32Array([Math.exp(1), NaN, 1]));
    a.dispose();
  });

  it('log', () => {
    const a = Array1D.new([1, 2]);
    const r = math.log(a);

    expect(r.get(0)).toBeCloseTo(Math.log(1));
    expect(r.get(1)).toBeCloseTo(Math.log(2));

    a.dispose();
  });

  it('log propagates NaNs', () => {
    const a = Array1D.new([1, NaN]);
    const r = math.log(a).getValues();
    expect(r).toEqual(new Float32Array([Math.log(1), NaN]));
    a.dispose();
  });

  it('logSumExp', () => {
    const a = Array1D.new([1, 2, -3]);
    const result = math.logSumExp(a);
    expect(result.get())
        .toBeCloseTo(Math.log(Math.exp(1) + Math.exp(2) + Math.exp(-3)));

    a.dispose();
    result.dispose();
  });

  it('logSumExp propagates NaNs', () => {
    const a = Array1D.new([1, 2, NaN]);
    const result = math.logSumExp(a);
    expect(result.get()).toEqual(NaN);
    a.dispose();
  });
});

describe('NDArrayMathGPU sqrt', () => {
  let math: NDArrayMathGPU;
  beforeEach(() => {
    math = new NDArrayMathGPU();
    math.startScope();
  });

  afterEach(() => {
    math.endScope(null);
    math.dispose();
  });

  it('sqrt', () => {
    const a = Array1D.new([2, 4]);
    const r = math.sqrt(a);

    expect(r.get(0)).toBeCloseTo(Math.sqrt(2));
    expect(r.get(1)).toBeCloseTo(Math.sqrt(4));

    a.dispose();
  });

  it('sqrt propagates NaNs', () => {
    const a = Array1D.new([1, NaN]);
    const r = math.sqrt(a).getValues();
    expect(r).toEqual(new Float32Array([Math.sqrt(1), NaN]));
    a.dispose();
  });
});


describe('softmax', () => {
  let math: NDArrayMathGPU;

  beforeEach(() => {
    math = new NDArrayMathGPU();
    math.startScope();
  });

  afterEach(() => {
    math.endScope(null);
    math.dispose();
  });

  it('regular test', () => {
    const y = math.softmax(Array1D.new([2, 1, 3]));
    expect(y.get(0)).toBeCloseTo(0.24472847, 6);
    expect(y.get(1)).toBeCloseTo(0.09003057, 6);
    expect(y.get(2)).toBeCloseTo(0.66524095, 6);
    expect(y.get(0) + y.get(1) + y.get(2)).toBeCloseTo(1, 6);
  });

  it('overflow', () => {
    const y = math.softmax(Array1D.new([10000, 10000]));
    expect(y.get(0)).toBeCloseTo(0.5, 3);
    expect(y.get(1)).toBeCloseTo(0.5, 3);
  });

  it('underflow', () => {
    const y = math.softmax(Array1D.new([-10000, -10000]));
    expect(y.get(0)).toBeCloseTo(0.5, 3);
    expect(y.get(1)).toBeCloseTo(0.5, 3);
  });

  it('Huge difference between probabilities', () => {
    const y = math.softmax(Array1D.new([-10000, +10000]));
    expect(y.get(0)).toBeCloseTo(0.0, 6);
    expect(y.get(1)).toBeCloseTo(1, 6);
  });

  it('Propagates NaNs', () => {
    const a = Array1D.new([2, 1, NaN]);
    const y = math.softmax(a);
    expect(y.getValues()).toEqual(new Float32Array([NaN, NaN, NaN]));
    a.dispose();
  });
});

describe('NDArrayMathGPU sum', () => {
  let math: NDArrayMathGPU;
  beforeEach(() => {
    math = new NDArrayMathGPU();
    math.startScope();
  });

  afterEach(() => {
    math.endScope(null);
    math.dispose();
  });

  it('sum', () => {
    const a = Array2D.new([3, 2], [1, 2, 3, 0, 0, 1]);
    const result = math.sum(a);
    expect(result.get()).toBe(7);

    a.dispose();
  });

  it('propagates NaNs', () => {
    const a = Array2D.new([3, 2], [1, 2, 3, NaN, 0, 1]);
    expect(math.sum(a).get()).toEqual(NaN);
    a.dispose();
  });
});

describe('NDArrayMathGPU argmax', () => {
  let math: NDArrayMathGPU;
  beforeEach(() => {
    math = new NDArrayMathGPU();
    math.startScope();
  });

  afterEach(() => {
    math.endScope(null);
    math.dispose();
  });

  it('Array1D', () => {
    const a = Array1D.new([1, 0, 3, 2]);
    const result = math.argMax(a);
    expect(result.get()).toBe(2);

    a.dispose();
  });

  it('propagates NaNs', () => {
    const a = Array1D.new([5, 0, 3, NaN, 3]);
    expect(math.argMax(a).get()).toEqual(NaN);
    a.dispose();
  });
});

describe('NDArrayMathGPU argmin', () => {
  let math: NDArrayMathGPU;
  beforeEach(() => {
    math = new NDArrayMathGPU();
    math.startScope();
  });

  afterEach(() => {
    math.endScope(null);
    math.dispose();
  });

  it('argmin', () => {
    const a = Array1D.new([1, 0, 3, 2]);
    const result = math.argMin(a);
    expect(result.get()).toBe(1);

    a.dispose();
  });

  it('Arg min propagates NaNs', () => {
    const a = Array1D.new([5, 0, NaN, 7, 3]);
    expect(math.argMin(a).get()).toEqual(NaN);

    a.dispose();
  });
});

describe('NDArrayMathGPU argmax equals', () => {
  let math: NDArrayMathGPU;
  beforeEach(() => {
    math = new NDArrayMathGPU();
    math.startScope();
  });

  afterEach(() => {
    math.endScope(null);
    math.dispose();
  });

  it('equals', () => {
    const a = Array1D.new([5, 0, 3, 7, 3]);
    const b = Array1D.new([-100.3, -20.0, -10.0, -5, -100]);
    const result = math.argMaxEquals(a, b);
    expect(result.get()).toBe(1);
  });

  it('not equals', () => {
    const a = Array1D.new([5, 0, 3, 1, 3]);
    const b = Array1D.new([-100.3, -20.0, -10.0, -5, 0]);
    const result = math.argMaxEquals(a, b);
    expect(result.get()).toBe(0);
  });

  it('propagates NaNs', () => {
    const a = Array1D.new([0, 3, 1, 3]);
    const b = Array1D.new([NaN, -20.0, -10.0, -5]);
    const result = math.argMaxEquals(a, b);
    expect(result.get()).toEqual(NaN);
  });

  it('throws when given arrays of different shape', () => {
    const a = Array1D.new([5, 0, 3, 7, 3, 10]);
    const b = Array1D.new([-100.3, -20.0, -10.0, -5, -100]);
    expect(() => math.argMaxEquals(a, b)).toThrowError();
  });
});

describe('NDArrayMathGPU conv2d', () => {
  let math: NDArrayMathGPU;
  beforeEach(() => {
    math = new NDArrayMathGPU();
    math.startScope();
  });

  afterEach(() => {
    math.endScope(null);
    math.dispose();
  });

  it('input=2x2x1,d2=1,f=1,s=1,p=0', () => {
    const inputDepth = 1;
    const inputShape: [number, number, number] = [2, 2, inputDepth];
    const outputDepth = 1;
    const fSize = 1;
    const pad = 0;
    const stride = 1;

    const x = Array3D.new(inputShape, [1, 2, 3, 4]);
    const w = Array4D.new([fSize, fSize, inputDepth, outputDepth], [2]);
    const bias = Array1D.new([-1]);

    const result = math.conv2d(x, w, bias, stride, pad);
    const expected = new Float32Array([1, 3, 5, 7]);

    expect(result.inGPU()).toBe(true);
    expect(result.getValues()).toEqual(expected);
    x.dispose();
    w.dispose();
    bias.dispose();
  });

  it('input=2x2x1,d2=1,f=2,s=1,p=0', () => {
    const inputDepth = 1;
    const inputShape: [number, number, number] = [2, 2, inputDepth];
    const outputDepth = 1;
    const fSize = 2;
    const pad = 0;
    const stride = 1;

    const x = Array3D.new(inputShape, [1, 2, 3, 4]);
    const w =
        Array4D.new([fSize, fSize, inputDepth, outputDepth], [3, 1, 5, 0]);
    const bias = Array1D.new([-1]);

    const result = math.conv2d(x, w, bias, stride, pad);
    const expected = new Float32Array([19]);

    expect(result.inGPU()).toBe(true);
    expect(result.getValues()).toEqual(expected);

    x.dispose();
    w.dispose();
    bias.dispose();
  });

  it('throws when x is not rank 3', () => {
    const inputDepth = 1;
    const outputDepth = 1;
    const fSize = 2;
    const pad = 0;
    const stride = 1;

    // tslint:disable-next-line:no-any
    const x: any = Array2D.new([2, 2], [1, 2, 3, 4]);
    const w =
        Array4D.new([fSize, fSize, inputDepth, outputDepth], [3, 1, 5, 0]);
    const bias = Array1D.new([-1]);

    expect(() => math.conv2d(x, w, bias, stride, pad)).toThrowError();

    x.dispose();
    w.dispose();
    bias.dispose();
  });

  it('throws when weights is not rank 4', () => {
    const inputDepth = 1;
    const inputShape: [number, number, number] = [2, 2, inputDepth];
    const pad = 0;
    const stride = 1;

    const x = Array3D.new(inputShape, [1, 2, 3, 4]);
    // tslint:disable-next-line:no-any
    const w: any = Array3D.new([2, 2, 1], [3, 1, 5, 0]);
    const bias = Array1D.new([-1]);

    expect(() => math.conv2d(x, w, bias, stride, pad)).toThrowError();

    x.dispose();
    w.dispose();
    bias.dispose();
  });

  it('throws when biases is not rank 1', () => {
    const inputDepth = 1;
    const inputShape: [number, number, number] = [2, 2, inputDepth];
    const outputDepth = 1;
    const fSize = 2;
    const pad = 0;
    const stride = 1;

    const x = Array3D.new(inputShape, [1, 2, 3, 4]);
    const w =
        Array4D.new([fSize, fSize, inputDepth, outputDepth], [3, 1, 5, 0]);
    // tslint:disable-next-line:no-any
    const bias: any = Array2D.new([2, 2], [2, 2, 2, 2]);

    expect(() => math.conv2d(x, w, bias, stride, pad)).toThrowError();

    x.dispose();
    w.dispose();
    bias.dispose();
  });

  it('throws when x depth does not match weight depth', () => {
    const inputDepth = 1;
    const wrongInputDepth = 5;
    const inputShape: [number, number, number] = [2, 2, inputDepth];
    const outputDepth = 1;
    const fSize = 2;
    const pad = 0;
    const stride = 1;

    const x = Array3D.new(inputShape, [1, 2, 3, 4]);
    const w = Array4D.randNormal([fSize, fSize, wrongInputDepth, outputDepth]);
    const bias = Array1D.new([-1]);

    expect(() => math.conv2d(x, w, bias, stride, pad)).toThrowError();

    x.dispose();
    w.dispose();
    bias.dispose();
  });
});

describe('NDArrayMathGPU conv2dTranspose', () => {
  let math: NDArrayMathGPU;
  beforeEach(() => {
    math = new NDArrayMathGPU();
    math.startScope();
  });

  afterEach(() => {
    math.endScope(null);
    math.dispose();
  });

  it('input=2x2x1,d2=1,f=2,s=1,p=0', () => {
    const origInputDepth = 1;
    const origOutputDepth = 1;
    const inputShape: [number, number, number] = [1, 1, origOutputDepth];
    const fSize = 2;
    const origPad = 0;
    const origStride = 1;

    const x = Array3D.new(inputShape, [2]);
    const w = Array4D.new(
        [fSize, fSize, origInputDepth, origOutputDepth], [3, 1, 5, 0]);

    const result = math.conv2dTranspose(x, w, [2, 2, 1], origStride, origPad);
    const expected = new Float32Array([6, 2, 10, 0]);

    expect(result.inGPU()).toBe(true);
    expect(result.shape).toEqual([2, 2, 1]);
    expect(result.getValues()).toEqual(expected);

    x.dispose();
    w.dispose();
  });

  it('throws when x is not rank 3', () => {
    const origInputDepth = 1;
    const origOutputDepth = 1;
    const fSize = 2;
    const origPad = 0;
    const origStride = 1;

    // tslint:disable-next-line:no-any
    const x: any = Array2D.new([2, 1], [2, 2]);
    const w = Array4D.new(
        [fSize, fSize, origInputDepth, origOutputDepth], [3, 1, 5, 0]);

    expect(() => math.conv2dTranspose(x, w, [2, 2, 1], origStride, origPad))
        .toThrowError();

    x.dispose();
    w.dispose();
  });

  it('throws when weights is not rank 4', () => {
    const origInputDepth = 1;
    const origOutputDepth = 1;
    const inputShape: [number, number, number] = [1, 1, origOutputDepth];
    const fSize = 2;
    const origPad = 0;
    const origStride = 1;

    const x = Array3D.new(inputShape, [2]);
    // tslint:disable-next-line:no-any
    const w: any = Array3D.new([fSize, fSize, origInputDepth], [3, 1, 5, 0]);

    expect(() => math.conv2dTranspose(x, w, [2, 2, 1], origStride, origPad))
        .toThrowError();

    x.dispose();
    w.dispose();
  });

  it('throws when x depth does not match weights original output depth', () => {
    const origInputDepth = 1;
    const origOutputDepth = 2;
    const wrongOrigOutputDepth = 3;
    const inputShape: [number, number, number] = [1, 1, origOutputDepth];
    const fSize = 2;
    const origPad = 0;
    const origStride = 1;

    const x = Array3D.new(inputShape, [2, 2]);
    const w = Array4D.randNormal(
        [fSize, fSize, origInputDepth, wrongOrigOutputDepth]);

    expect(() => math.conv2dTranspose(x, w, [2, 2, 2], origStride, origPad))
        .toThrowError();

    x.dispose();
    w.dispose();
  });
});

describe('NDArrayMathGPU conv2dDerWeights', () => {
  let math: NDArrayMathGPU;
  beforeEach(() => {
    math = new NDArrayMathGPU();
    math.startScope();
  });

  afterEach(() => {
    math.endScope(null);
    math.dispose();
  });

  it('conv2dDerWeights input=3x3x1,d2=1,f=2,s=1,p=0', () => {
    const inputDepth = 1;
    const outputDepth = 1;
    const inputShape: [number, number, number] = [3, 3, inputDepth];
    const fSize = 2;
    const stride = 1;
    const pad = 0;

    const weightsShape: [number, number, number, number] =
        [fSize, fSize, inputDepth, outputDepth];

    const x = Array3D.new(inputShape, [1, 2, 3, 4, 5, 6, 7, 8, 9]);
    const dy = Array3D.new([2, 2, 1], [3, 1, 2, 0]);

    const result = math.conv2dDerFilter(x, dy, weightsShape, stride, pad);
    const expected = new Float32Array([13, 19, 31, 37]);

    expect(result.inGPU()).toBe(true);
    expect(result.shape).toEqual(weightsShape);
    expect(result.getValues()).toEqual(expected);

    x.dispose();
    dy.dispose();
  });
});

describe('NDArrayMathGPU conv2dDerWeights', () => {
  let math: NDArrayMathGPU;
  beforeEach(() => {
    math = new NDArrayMathGPU();
    math.startScope();
  });

  afterEach(() => {
    math.endScope(null);
    math.dispose();
  });

  it('conv2dDerBias dy=2x2x2', () => {
    const outputDepth = 2;
    const dyShape: [number, number, number] = [2, 2, outputDepth];
    const dy = Array3D.new(dyShape, [1, 2, 3, 4, 5, 6, 7, 8]);

    const result = math.conv2dDerBias(dy);
    const expected = new Float32Array([16, 20]);

    expect(result.inGPU()).toBe(true);
    expect(result.shape).toEqual([outputDepth]);
    expect(result.getValues()).toEqual(expected);

    dy.dispose();
  });
});

describe('NDArrayMathGPU maxPool', () => {
  let math: NDArrayMathGPU;
  beforeEach(() => {
    math = new NDArrayMathGPU();
    math.startScope();
  });

  afterEach(() => {
    math.endScope(null);
    math.dispose();
  });

  it('3x3x2 in, 2x2 filter, 1 stride', () => {
    // Feed forward.
    const a = Array3D.new(
        [3, 3, 2],
        [1, 99, 2, 88, 3, 77, 4, 66, 5, 55, 6, 44, 7, 33, 9, 22, 8, 11]);
    const result = math.maxPool(a, 2, 1, 0);

    expect(result.inGPU()).toBe(true);
    expect(result.shape).toEqual([2, 2, 2]);
    expect(result.getValues()).toEqual(new Float32Array([
      5, 99, 6, 88, 9, 66, 9, 55
    ]));
    a.dispose();
  });

  it('3x3x1 in, 2x2 filter, 1 stride, propagates NaNs', () => {
    const a = Array3D.new([3, 3, 1], [1, 2, 3, 4, 5, 6, 7, NaN, 9]);
    const result = math.maxPool(a, 2, 1, 0);

    expect(result.shape).toEqual([2, 2, 1]);
    expect(result.getValues()).toEqual(new Float32Array([5, 6, NaN, NaN]));
    a.dispose();
  });

  it('4x4x1 in, 2x2 filter, 2 stride', () => {
    // Feed forward.
    const a = Array3D.new(
        [4, 4, 1], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
    const result = math.maxPool(a, 2, 2, 0);

    expect(result.inGPU()).toBe(true);
    expect(result.shape).toEqual([2, 2, 1]);
    expect(result.getValues()).toEqual(new Float32Array([5, 7, 13, 15]));

    a.dispose();
  });

  it('throws when x is not rank 3', () => {
    // tslint:disable-next-line:no-any
    const a: any = Array2D.new([3, 3], [1, 2, 3, 4, 5, 6, 7, 8, 9]);
    expect(() => math.maxPool(a, 2, 1, 0)).toThrowError();

    a.dispose();
  });
});

describe('NDArrayMathGPU maxPoolBackprop', () => {
  let math: NDArrayMathGPU;
  beforeEach(() => {
    math = new NDArrayMathGPU();
    math.startScope();
  });

  afterEach(() => {
    math.endScope(null);
    math.dispose();
  });

  it('x=2x2x1, f=2, s=2, pad=1', () => {
    const dy = Array3D.new([2, 2, 1], [1, 2, 3, 4]);
    const maxPositions = Array3D.new([2, 2, 1], [3, 2, 1, 0]);
    const expected = new Float32Array([1, 2, 3, 4]);
    const dx = math.maxPoolBackprop(dy, maxPositions, 2, 2, 1);

    expect(dx.inGPU()).toBe(true);
    expect(dx.getValues()).toEqual(expected);

    dy.dispose();
    maxPositions.dispose();
    dx.dispose();
  });

  // Max pool depth > 1.
  it('x=3x3x2, f=2, s=1, no duplicate max value', () => {
    const dy = Array3D.new([2, 2, 2], [1, 44, 2, 33, 3, 22, 4, 11]);
    const x = Array3D.new(
        [3, 3, 2],
        [1, 99, 2, 55, 3, 66, 4, 66, 5, 88, 6, 44, 7, 99, 8, 55, 9, 100]);
    const expected = new Float32Array(
        [0, 44, 0, 0, 0, 0, 0, 0, 1, 33, 2, 0, 0, 22, 3, 0, 4, 11]);
    const dx = math.maxPoolBackprop(dy, x, 2, 1, 0);

    expect(dx.inGPU()).toBe(true);
    expect(dx.getValues()).toEqual(expected);

    dy.dispose();
    x.dispose();
    dx.dispose();
  });

  it('x=3x3x2, f=2, s=1 duplicate max value', () => {
    const dy = Array3D.new([2, 2, 2], [1, 44, 2, 33, 3, 22, 4, 11]);
    const x = Array3D.new(
        [3, 3, 2], [0, 1, 0, 3, 0, 2, 0, 1, 5, 2, 0, 1, 0, 1, 0, 1, 0, 5]);
    const expected = new Float32Array(
        [0, 0, 0, 77, 0, 0, 0, 0, 10, 22, 0, 0, 0, 0, 0, 0, 0, 11]);
    const dx = math.maxPoolBackprop(dy, x, 2, 1, 0);

    expect(dx.inGPU()).toBe(true);
    expect(dx.getValues()).toEqual(expected);

    dy.dispose();
    x.dispose();
    dx.dispose();
  });
});

describe('NDArrayMathGPU resizeBilinear', () => {
  let math: NDArrayMathGPU;
  beforeEach(() => {
    math = new NDArrayMathGPU();
    math.startScope();
  });

  afterEach(() => {
    math.endScope(null);
    math.dispose();
  });

  it('simple alignCorners=false', () => {
    const input = Array3D.new([2, 2, 1], [2, 2, 4, 4]);
    const output = math.resizeBilinear3D(input, [3, 3], false);

    test_util.expectArraysClose(
        output.getValues(),
        new Float32Array([2, 2, 2, 10 / 3, 10 / 3, 10 / 3, 4, 4, 4]));
    input.dispose();
  });

  it('simple alignCorners=true', () => {
    const input = Array3D.new([2, 2, 1], [2, 2, 4, 4]);
    const output = math.resizeBilinear3D(input, [3, 3], true);

    test_util.expectArraysClose(
        output.getValues(), new Float32Array([2, 2, 2, 3, 3, 3, 4, 4, 4]));
    input.dispose();
  });

  it('matches tensorflow w/ random numbers alignCorners=false', () => {
    const input = Array3D.new([2, 3, 2], [
      1.19074044, 0.91373104, 2.01611669, -0.52270832, 0.38725395, 1.30809779,
      0.61835143, 3.49600659, 2.09230986, 0.56473997, 0.03823943, 1.19864896
    ]);
    const output = math.resizeBilinear3D(input, [4, 5], false);

    test_util.expectArraysClose(
        output.getValues(), new Float32Array([
          1.19074047,  0.91373104, 1.68596613, 0.05186744, 1.69034398,
          -0.15654698, 0.7130264,  0.94193673, 0.38725394, 1.30809784,
          0.9045459,   2.20486879, 1.59434628, 0.89455694, 1.68591988,
          0.26748738,  0.58103991, 1.00690198, 0.21274668, 1.25337338,
          0.6183514,   3.49600649, 1.50272655, 1.73724651, 1.68149579,
          0.69152176,  0.44905344, 1.07186723, 0.03823943, 1.19864893,
          0.6183514,   3.49600649, 1.50272655, 1.73724651, 1.68149579,
          0.69152176,  0.44905344, 1.07186723, 0.03823943, 1.19864893
        ]));
    input.dispose();
  });

  it('matches tensorflow w/ random numbers alignCorners=true', () => {
    const input = Array3D.new([2, 3, 2], [
      1.56324531, 2.13817752, 1.44398421, 1.07632684, 0.59306785, -0.36970865,
      1.62451879, 1.8367334, 1.13944798, 2.01993218, 2.01919952, 2.67524054
    ]);
    const output = math.resizeBilinear3D(input, [4, 5], true);

    test_util.expectArraysClose(
        output.getValues(), new Float32Array([
          1.5632453,  2.13817763, 1.50361478, 1.60725224, 1.44398427,
          1.07632685, 1.01852608, 0.35330909, 0.59306782, -0.36970866,
          1.58366978, 2.03769612, 1.46307099, 1.71427906, 1.3424722,
          1.39086199, 1.20545864, 1.01806819, 1.06844509, 0.6452744,
          1.60409427, 1.93721485, 1.42252707, 1.82130599, 1.24096,
          1.70539713, 1.3923912,  1.68282723, 1.54382229, 1.66025746,
          1.62451875, 1.83673346, 1.38198328, 1.92833281, 1.13944793,
          2.01993227, 1.57932377, 2.34758639, 2.01919961, 2.67524052
        ]));

    input.dispose();
  });
});

describe('NDArrayMathGPU batchNorm', () => {
  let math: NDArrayMathGPU;
  beforeEach(() => {
    math = new NDArrayMathGPU();
    math.startScope();
  });

  afterEach(() => {
    math.endScope(null);
    math.startScope();
  });

  it('simple batchnorm, no offset or scale, 2x1x2', () => {
    const x = Array3D.new([2, 1, 2], new Float32Array([2, 100, 4, 400]));
    const mean = Array1D.new([1, 2]);
    const variance = Array1D.new([2, 3]);
    const varianceEpsilon = .001;

    const result = math.batchNormalization3D(
        x, mean, variance, varianceEpsilon, undefined, undefined);

    test_util.expectArraysClose(
        result.getValues(), new Float32Array([
          (x.get(0, 0, 0) - mean.get(0)) * 1 /
              Math.sqrt(variance.get(0) + varianceEpsilon),
          (x.get(0, 0, 1) - mean.get(1)) * 1 /
              Math.sqrt(variance.get(1) + varianceEpsilon),
          (x.get(1, 0, 0) - mean.get(0)) * 1 /
              Math.sqrt(variance.get(0) + varianceEpsilon),
          (x.get(1, 0, 1) - mean.get(1)) * 1 /
              Math.sqrt(variance.get(1) + varianceEpsilon)
        ]));
    x.dispose();
    mean.dispose();
    variance.dispose();
  });

  it('simple batchnorm, no offset, 2x1x2', () => {
    const x = Array3D.new([2, 1, 2], new Float32Array([2, 100, 4, 400]));
    const mean = Array1D.new([1, 2]);
    const variance = Array1D.new([2, 3]);
    const scale = Array1D.new([4, 5]);
    const varianceEpsilon = .001;

    const result = math.batchNormalization3D(
        x, mean, variance, varianceEpsilon, scale, undefined);

    test_util.expectArraysClose(
        result.getValues(), new Float32Array([
          (x.get(0, 0, 0) - mean.get(0)) * scale.get(0) /
              Math.sqrt(variance.get(0) + varianceEpsilon),
          (x.get(0, 0, 1) - mean.get(1)) * scale.get(1) /
              Math.sqrt(variance.get(1) + varianceEpsilon),
          (x.get(1, 0, 0) - mean.get(0)) * scale.get(0) /
              Math.sqrt(variance.get(0) + varianceEpsilon),
          (x.get(1, 0, 1) - mean.get(1)) * scale.get(1) /
              Math.sqrt(variance.get(1) + varianceEpsilon)
        ]));
    x.dispose();
    mean.dispose();
    variance.dispose();
    scale.dispose();
  });

  it('simple batchnorm, no scale, 2x1x2', () => {
    const x = Array3D.new([2, 1, 2], new Float32Array([2, 100, 4, 400]));
    const mean = Array1D.new([1, 2]);
    const variance = Array1D.new([2, 3]);
    const offset = Array1D.new([4, 5]);

    const varianceEpsilon = .001;

    const result = math.batchNormalization3D(
        x, mean, variance, varianceEpsilon, undefined, offset);

    test_util.expectArraysClose(
        result.getValues(), new Float32Array([
          offset.get(0) +
              (x.get(0, 0, 0) - mean.get(0)) * 1 /
                  Math.sqrt(variance.get(0) + varianceEpsilon),
          offset.get(1) +
              (x.get(0, 0, 1) - mean.get(1)) * 1 /
                  Math.sqrt(variance.get(1) + varianceEpsilon),
          offset.get(0) +
              (x.get(1, 0, 0) - mean.get(0)) * 1 /
                  Math.sqrt(variance.get(0) + varianceEpsilon),
          offset.get(1) +
              (x.get(1, 0, 1) - mean.get(1)) * 1 /
                  Math.sqrt(variance.get(1) + varianceEpsilon)
        ]));
    x.dispose();
    mean.dispose();
    variance.dispose();
    offset.dispose();
  });

  it('simple batchnorm, 2x1x2', () => {
    const x = Array3D.new([2, 1, 2], new Float32Array([2, 100, 4, 400]));
    const mean = Array1D.new([1, 2]);
    const variance = Array1D.new([2, 3]);
    const offset = Array1D.new([3, 4]);
    const scale = Array1D.new([4, 5]);

    const varianceEpsilon = .001;

    const result = math.batchNormalization3D(
        x, mean, variance, varianceEpsilon, scale, offset);

    test_util.expectArraysClose(
        result.getValues(), new Float32Array([
          offset.get(0) +
              (x.get(0, 0, 0) - mean.get(0)) * scale.get(0) /
                  Math.sqrt(variance.get(0) + varianceEpsilon),
          offset.get(1) +
              (x.get(0, 0, 1) - mean.get(1)) * scale.get(1) /
                  Math.sqrt(variance.get(1) + varianceEpsilon),
          offset.get(0) +
              (x.get(1, 0, 0) - mean.get(0)) * scale.get(0) /
                  Math.sqrt(variance.get(0) + varianceEpsilon),
          offset.get(1) +
              (x.get(1, 0, 1) - mean.get(1)) * scale.get(1) /
                  Math.sqrt(variance.get(1) + varianceEpsilon)
        ]));
    x.dispose();
    mean.dispose();
    variance.dispose();
    scale.dispose();
    offset.dispose();
  });

  it('batchnorm matches tensorflow, 2x3x3', () => {
    const x =
        Array3D.new([2, 3, 3], new Float32Array([
                      0.49955603, 0.04158615, -1.09440524, 2.03854165,
                      -0.61578344, 2.87533573, 1.18105987, 0.807462, 1.87888837,
                      2.26563962, -0.37040935, 1.35848753, -0.75347094,
                      0.15683117, 0.91925946, 0.34121279, 0.92717143, 1.89683965
                    ]));
    const mean = Array1D.new([0.39745062, -0.48062894, 0.4847822]);
    const variance = Array1D.new([0.32375343, 0.67117643, 1.08334653]);
    const offset = Array1D.new([0.69398749, -1.29056387, 0.9429723]);
    const scale = Array1D.new([-0.5607271, 0.9878457, 0.25181573]);
    const varianceEpsilon = .001;

    const result = math.batchNormalization3D(
        x, mean, variance, varianceEpsilon, scale, offset);

    test_util.expectArraysClose(
        result.getValues(), new Float32Array([
          0.59352049, -0.66135202, 0.5610874, -0.92077015, -1.45341019,
          1.52106473, -0.07704776, 0.26144429, 1.28010017, -1.14422404,
          -1.15776136, 1.15425493, 1.82644104, -0.52249442, 1.04803919,
          0.74932291, 0.40568101, 1.2844412
        ]));
    x.dispose();
    mean.dispose();
    variance.dispose();
    scale.dispose();
    offset.dispose();
  });
});

describe('NDArrayMathGPU debug mode', () => {
  let math: NDArrayMathGPU;

  beforeEach(() => {
    math = new NDArrayMathGPU();
    math.startScope();
  });

  afterEach(() => {
    math.endScope(null);
  });

  it('debug mode does not error when no nans', () => {
    math.enableDebugMode();
    const a = Array1D.new([2, -1, 0, 3]);
    const res = math.relu(a);
    expect(res.getValues()).toEqual(new Float32Array([2, 0, 0, 3]));
  });

  it('debug mode errors when there are nans', () => {
    math.enableDebugMode();
    const a = Array1D.new([2, NaN]);
    const f = () => math.relu(a);
    expect(f).toThrowError();
  });

  it('no errors where there are nans, and debug mode is disabled', () => {
    const a = Array1D.new([2, NaN]);
    const res = math.relu(a);
    expect(res.getValues()).toEqual(new Float32Array([2, NaN]));
  });
});

describe('LSTMCell', () => {
  let math: NDArrayMathGPU;
  beforeEach(() => {
    math = new NDArrayMathGPU();
    math.startScope();
  });

  afterEach(() => {
    math.endScope(null);
    math.startScope();
  });

  it('Batch size must be 1 for MultiRNNCell', () => {
    const lstmKernel1 = Array2D.zeros([3, 4]);
    const lstmBias1 = Array1D.zeros([4]);
    const lstmKernel2 = Array2D.zeros([2, 4]);
    const lstmBias2 = Array1D.zeros([4]);

    const forgetBias = Scalar.new(1.0);
    const lstm1 =
        math.basicLSTMCell.bind(math, forgetBias, lstmKernel1, lstmBias1);
    const lstm2 =
        math.basicLSTMCell.bind(math, forgetBias, lstmKernel2, lstmBias2);

    const c = [
      Array2D.zeros([1, lstmBias1.shape[0] / 4]),
      Array2D.zeros([1, lstmBias2.shape[0] / 4])
    ];
    const h = [
      Array2D.zeros([1, lstmBias1.shape[0] / 4]),
      Array2D.zeros([1, lstmBias2.shape[0] / 4])
    ];

    const onehot = Array2D.zeros([2, 2]);
    onehot.set(1.0, 1, 0);
    const output = () => math.multiRNNCell([lstm1, lstm2], onehot, c, h);
    expect(output).toThrowError();
  });

  it('Batch size must be 1 for basicLSTMCell', () => {
    const lstmKernel = Array2D.zeros([3, 4]);
    const lstmBias = Array1D.zeros([4]);

    const forgetBias = Scalar.new(1.0);

    const c = Array2D.zeros([1, lstmBias.shape[0] / 4]);
    const h = Array2D.zeros([1, lstmBias.shape[0] / 4]);

    const onehot = Array2D.zeros([2, 2]);
    onehot.set(1.0, 1, 0);
    const output = () =>
        math.basicLSTMCell(forgetBias, lstmKernel, lstmBias, onehot, c, h);
    expect(output).toThrowError();
  });

  it('MultiRNNCell with 2 BasicLSTMCells', () => {
    const lstmKernel1 = Array2D.new(
        [3, 4], new Float32Array([
          0.26242125034332275, -0.8787832260131836, 0.781475305557251,
          1.337337851524353, 0.6180247068405151, -0.2760246992111206,
          -0.11299663782119751, -0.46332040429115295, -0.1765323281288147,
          0.6807947158813477, -0.8326982855796814, 0.6732975244522095
        ]));
    const lstmBias1 = Array1D.new(new Float32Array(
        [1.090713620185852, -0.8282332420349121, 0, 1.0889357328414917]));
    const lstmKernel2 = Array2D.new(
        [2, 4], new Float32Array([
          -1.893059492111206, -1.0185645818710327, -0.6270437240600586,
          -2.1829540729522705, -0.4583775997161865, -0.5454602241516113,
          -0.3114445209503174, 0.8450229167938232
        ]));
    const lstmBias2 = Array1D.new(new Float32Array(
        [0.9906240105628967, 0.6248329877853394, 0, 1.0224634408950806]));

    const forgetBias = Scalar.new(1.0);
    const lstm1 =
        math.basicLSTMCell.bind(math, forgetBias, lstmKernel1, lstmBias1);
    const lstm2 =
        math.basicLSTMCell.bind(math, forgetBias, lstmKernel2, lstmBias2);

    const c = [
      Array2D.zeros([1, lstmBias1.shape[0] / 4]),
      Array2D.zeros([1, lstmBias2.shape[0] / 4])
    ];
    const h = [
      Array2D.zeros([1, lstmBias1.shape[0] / 4]),
      Array2D.zeros([1, lstmBias2.shape[0] / 4])
    ];

    const onehot = Array2D.zeros([1, 2]);
    onehot.set(1.0, 0, 0);

    const output = math.multiRNNCell([lstm1, lstm2], onehot, c, h);

    test_util.expectArraysClose(
        output[0][0].getValues(), new Float32Array([-0.7440074682235718]));
    test_util.expectArraysClose(
        output[0][1].getValues(), new Float32Array([0.7460772395133972]));
    test_util.expectArraysClose(
        output[1][0].getValues(), new Float32Array([-0.5802832245826721]));
    test_util.expectArraysClose(
        output[1][1].getValues(), new Float32Array([0.5745711922645569]));
  });
});
