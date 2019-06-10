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

import {describeWithFlags} from '../../jasmine_util';
import {expectArraysClose} from '../../test_util';
import {WEBGL_ENVS} from './backend_webgl_test_registry';
import * as tex_util from './tex_util';

describe('tex_util getUnpackedMatrixTextureShapeWidthHeight', () => {
  it('[1x1] => [1x1]', () => {
    expect(tex_util.getUnpackedMatrixTextureShapeWidthHeight(1, 1)).toEqual([
      1, 1
    ]);
  });

  it('[MxN] => [NxM]', () => {
    expect(tex_util.getUnpackedMatrixTextureShapeWidthHeight(123, 456))
        .toEqual([456, 123]);
  });
});

describe('tex_util getPackedMatrixTextureShapeWidthHeight', () => {
  it('[1x1] => [1x1]', () => {
    const shape = tex_util.getPackedMatrixTextureShapeWidthHeight(1, 1);
    expect(shape).toEqual([1, 1]);
  });

  it('[1x2] => [1x1]', () => {
    const shape = tex_util.getPackedMatrixTextureShapeWidthHeight(1, 2);
    expect(shape).toEqual([1, 1]);
  });

  it('[2x1] => [1x1]', () => {
    const shape = tex_util.getPackedMatrixTextureShapeWidthHeight(2, 1);
    expect(shape).toEqual([1, 1]);
  });

  it('[2x2] => [1x1]', () => {
    const shape = tex_util.getPackedMatrixTextureShapeWidthHeight(2, 2);
    expect(shape).toEqual([1, 1]);
  });

  it('[3x3] => [2x2]', () => {
    const shape = tex_util.getPackedMatrixTextureShapeWidthHeight(3, 3);
    expect(shape).toEqual([2, 2]);
  });

  it('[4x3] => [2x2]', () => {
    const shape = tex_util.getPackedMatrixTextureShapeWidthHeight(4, 3);
    expect(shape).toEqual([2, 2]);
  });

  it('[3x4] => [2x2]', () => {
    const shape = tex_util.getPackedMatrixTextureShapeWidthHeight(3, 4);
    expect(shape).toEqual([2, 2]);
  });

  it('[4x4] => [2x2]', () => {
    const shape = tex_util.getPackedMatrixTextureShapeWidthHeight(4, 4);
    expect(shape).toEqual([2, 2]);
  });

  it('[1024x1024] => [512x512]', () => {
    const shape = tex_util.getPackedMatrixTextureShapeWidthHeight(1024, 1024);
    expect(shape).toEqual([512, 512]);
  });

  it('[MxN] => [ceil(N/2)xceil(M/2)]', () => {
    const M = 123;
    const N = 5013;
    const shape = tex_util.getPackedMatrixTextureShapeWidthHeight(M, N);
    expect(shape).toEqual([Math.ceil(N / 2), Math.ceil(M / 2)]);
  });
});

describeWithFlags('tex_util decodeMatrixFromUnpackedArray', WEBGL_ENVS, () => {
  it('1x1 writes the only matrix array value to the first element', () => {
    const unpackedRGBA = new Float32Array([1, 0, 0, 0]);
    const matrix = new Float32Array(1);
    tex_util.decodeMatrixFromUnpackedArray(unpackedRGBA, matrix, 4);
    expect(matrix.length).toEqual(1);
    expect(matrix[0]).toEqual(1);
  });

  it('1x2 writes the second texel R component to the second element', () => {
    const unpackedRGBA = new Float32Array([1, 0, 0, 0, 2, 0, 0, 0]);
    const matrix = new Float32Array(2);
    tex_util.decodeMatrixFromUnpackedArray(unpackedRGBA, matrix, 4);
    expect(matrix.length).toEqual(2);
    expectArraysClose(matrix, new Float32Array([1, 2]));
  });
});

describeWithFlags('tex_util decodeMatrixFromPackedRGBA', WEBGL_ENVS, () => {
  it('1x1 matrix only loads R component from only texel', () => {
    const packedRGBA = new Float32Array([1, 0, 0, 0]);
    const matrix = new Float32Array(1);
    tex_util.decodeMatrixFromPackedRGBA(packedRGBA, 1, 1, 1, matrix);
    expect(matrix[0]).toEqual(1);
  });

  it('1x2 matrix loads RG from only texel', () => {
    const packedRGBA = new Float32Array([1, 2, 0, 0]);
    const matrix = new Float32Array(2);
    tex_util.decodeMatrixFromPackedRGBA(packedRGBA, 1, 1, 2, matrix);
    expectArraysClose(matrix, new Float32Array([1, 2]));
  });

  it('2x1 matrix loads RB from only texel', () => {
    const packedRGBA = new Float32Array([1, 0, 2, 0]);
    const matrix = new Float32Array(2);
    tex_util.decodeMatrixFromPackedRGBA(packedRGBA, 1, 2, 1, matrix);
    expectArraysClose(matrix, new Float32Array([1, 2]));
  });

  it('2x2 matrix loads RGBA from only texel', () => {
    const packedRGBA = new Float32Array([1, 2, 3, 4]);
    const matrix = new Float32Array(4);
    tex_util.decodeMatrixFromPackedRGBA(packedRGBA, 1, 2, 2, matrix);
    expectArraysClose(matrix, new Float32Array([1, 2, 3, 4]));
  });

  it('4x3 final column only reads RB from edge texels', () => {
    /*
      1  2  4  5 | 3  0  6  0     1  2  3
      -----------+----------- =>  4  5  6
      7  8 10 11 | 9  0 12  0     7  8  9
                                 10 11 12
     */
    const packedRGBA =
        new Float32Array([1, 2, 4, 5, 3, 0, 6, 0, 7, 8, 10, 11, 9, 0, 12, 0]);
    const matrix = new Float32Array(12);
    tex_util.decodeMatrixFromPackedRGBA(packedRGBA, 1, 4, 3, matrix);
    expectArraysClose(
        matrix, new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]));
  });

  it('3x4 final row only reads RG from edge texels', () => {
    /*
       1  2 5 6 |  3  4 7 8    1  2  3  4
       ---------+---------- => 5  6  7  8
       9 10 0 0 | 11 12 0 0    9 10 11 12
     */
    const packedRGBA =
        new Float32Array([1, 2, 5, 6, 3, 4, 7, 8, 9, 10, 0, 0, 11, 12, 0, 0]);
    const matrix = new Float32Array(12);
    tex_util.decodeMatrixFromPackedRGBA(packedRGBA, 1, 3, 4, matrix);
    expectArraysClose(
        matrix, new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]));
  });

  it('3x3 bottom-right only reads R from corner texel', () => {
    /*
      1 2 4 5 | 3 0 6 0    1 2 3
      --------+-------- => 4 5 6
      7 8 0 0 | 9 0 0 0    7 8 9
     */
    const packedRGBA =
        new Float32Array([1, 2, 4, 5, 3, 0, 6, 0, 7, 8, 0, 0, 9, 0, 0, 0]);
    const matrix = new Float32Array(9);
    tex_util.decodeMatrixFromPackedRGBA(packedRGBA, 1, 3, 3, matrix);
    expectArraysClose(matrix, new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9]));
  });

  it('2x3x4 bottom row in each batch only reads RG', () => {
    const packedRGBA = new Float32Array([
      1,  2,  5,  6,  3,  4,  7,  8,  9,  10, 0, 0, 11, 12, 0, 0,
      13, 14, 17, 18, 15, 16, 19, 20, 21, 22, 0, 0, 23, 24, 0, 0
    ]);
    const matrix = new Float32Array(24);
    tex_util.decodeMatrixFromPackedRGBA(packedRGBA, 2, 3, 4, matrix);
    expectArraysClose(matrix, new Float32Array([
                        1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24
                      ]));
  });

  it('2x4x3 final column in each batch only reads RB', () => {
    const packedRGBA = new Float32Array([
      1,  2,  4,  5,  3,  0, 6,  0, 7,  8,  10, 11, 9,  0, 12, 0,
      13, 14, 16, 17, 15, 0, 18, 0, 19, 20, 22, 23, 21, 0, 24, 0
    ]);
    const matrix = new Float32Array(24);
    tex_util.decodeMatrixFromPackedRGBA(packedRGBA, 2, 4, 3, matrix);
    expectArraysClose(matrix, new Float32Array([
                        1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24
                      ]));
  });

  it('2x3x3 bottom right texel in each batch only reads R', () => {
    const packedRGBA = new Float32Array([
      1,  2,  4,  5,  3,  0, 6,  0, 7,  8,  0, 0, 9,  0, 0, 0,
      10, 11, 13, 14, 12, 0, 15, 0, 16, 17, 0, 0, 18, 0, 0, 0
    ]);
    const matrix = new Float32Array(18);
    tex_util.decodeMatrixFromPackedRGBA(packedRGBA, 2, 3, 3, matrix);
    expectArraysClose(
        matrix,
        new Float32Array(
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]));
  });

  it('4D (2x3x3x4) is properly decoded', () => {
    const packedRGBA = new Float32Array([
      1,  2,  5,  6,  3,  4,  7,  8,  9,  10, 0, 0, 11, 12, 0, 0,
      13, 14, 17, 18, 15, 16, 19, 20, 21, 22, 0, 0, 23, 24, 0, 0,
      25, 26, 29, 30, 27, 28, 31, 32, 33, 34, 0, 0, 35, 36, 0, 0,
      37, 38, 41, 42, 39, 40, 43, 44, 45, 46, 0, 0, 47, 48, 0, 0,
      49, 50, 53, 54, 51, 52, 55, 56, 57, 58, 0, 0, 59, 60, 0, 0,
      61, 62, 65, 66, 63, 64, 67, 68, 69, 70, 0, 0, 71, 72, 0, 0
    ]);
    const matrix = new Float32Array(72);
    tex_util.decodeMatrixFromPackedRGBA(packedRGBA, 6, 3, 4, matrix);
    expectArraysClose(
        matrix, new Float32Array([
          1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
          16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
          31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
          46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
          61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72
        ]));
  });
});
