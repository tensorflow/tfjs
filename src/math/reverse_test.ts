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
import {MathTests} from '../test_util';

import {Array1D, Array2D, Array3D, Array4D} from './ndarray';

// math.reverse1D
{
  const tests: MathTests = it => {
    it('reverse a 1D array', math => {
      const input = Array1D.new([1, 2, 3, 4, 5]);
      const result = math.reverse1D(input);
      expect(result.shape).toEqual(input.shape);
      test_util.expectArraysClose(result, [5, 4, 3, 2, 1]);
    });
  };

  test_util.describeMathCPU('reverse1D', [tests]);
  test_util.describeMathGPU('reverse1D', [tests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}

// math.reverse2D
{
  const tests: MathTests = it => {
    it('reverse a 2D array at axis [0]', math => {
      const axis = [0];
      const a = Array2D.new([2, 3], [1, 2, 3, 4, 5, 6]);
      const result = math.reverse2D(a, axis);

      expect(result.shape).toEqual(a.shape);
      test_util.expectArraysClose(result, [4, 5, 6, 1, 2, 3]);
    });

    it('reverse a 2D array at axis [1]', math => {
      const axis = [1];
      const a = Array2D.new([2, 3], [1, 2, 3, 4, 5, 6]);
      const result = math.reverse2D(a, axis);

      expect(result.shape).toEqual(a.shape);
      test_util.expectArraysClose(result, [3, 2, 1, 6, 5, 4]);
    });

    it('throws error with invalid input', math => {
      // tslint:disable-next-line:no-any
      const x: any = Array1D.new([1, 20, 300, 4]);
      expect(() => math.reverse2D(x, [0])).toThrowError();
    });

    it('throws error with invalid axis param', math => {
      const x = Array2D.new([1, 4], [1, 20, 300, 4]);
      expect(() => math.reverse2D(x, [2])).toThrowError();
      expect(() => math.reverse2D(x, [-3])).toThrowError();
    });

    it('throws error with non integer axis param', math => {
      const x = Array2D.new([1, 4], [1, 20, 300, 4]);
      expect(() => math.reverse2D(x, [0.5])).toThrowError();
    });
  };

  test_util.describeMathCPU('reverse2D', [tests]);
  test_util.describeMathGPU('reverse2D', [tests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}

// math.reverse3D
{
  const tests: MathTests = it => {
    // [
    //   [
    //     [0,  1,  2,  3],
    //     [4,  5,  6,  7],
    //     [8,  9,  10, 11]
    //   ],
    //   [
    //     [12, 13, 14, 15],
    //     [16, 17, 18, 19],
    //     [20, 21, 22, 23]
    //   ]
    // ]
    const shape: [number, number, number] = [2, 3, 4];
    const data = [
      0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
      12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23
    ];

    it('reverse a 3D array at axis [0]', math => {
      const input = Array3D.new(shape, data);
      const result = math.reverse3D(input, [0]);
      expect(result.shape).toEqual(input.shape);
      test_util.expectArraysClose(result, [
        12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11
      ]);
    });

    it('reverse a 3D array at axis [1]', math => {
      const input = Array3D.new(shape, data);
      const result = math.reverse3D(input, [1]);
      expect(result.shape).toEqual(input.shape);
      test_util.expectArraysClose(result, [
        8,  9,  10, 11, 4,  5,  6,  7,  0,  1,  2,  3,
        20, 21, 22, 23, 16, 17, 18, 19, 12, 13, 14, 15
      ]);
    });

    it('reverse a 3D array at axis [2]', math => {
      const input = Array3D.new(shape, data);
      const result = math.reverse3D(input, [2]);
      expect(result.shape).toEqual(input.shape);
      test_util.expectArraysClose(result, [
        3,  2,  1,  0,  7,  6,  5,  4,  11, 10, 9,  8,
        15, 14, 13, 12, 19, 18, 17, 16, 23, 22, 21, 20
      ]);
    });

    it('reverse a 3D array at axis [0, 1]', math => {
      const input = Array3D.new(shape, data);
      const result = math.reverse3D(input, [0, 1]);
      expect(result.shape).toEqual(input.shape);
      test_util.expectArraysClose(result, [
        20, 21, 22, 23, 16, 17, 18, 19, 12, 13, 14, 15,
        8,  9,  10, 11, 4,  5,  6,  7,  0,  1,  2,  3
      ]);
    });

    it('reverse a 3D array at axis [0, 2]', math => {
      const input = Array3D.new(shape, data);
      const result = math.reverse3D(input, [0, 2]);
      expect(result.shape).toEqual(input.shape);
      test_util.expectArraysClose(result, [
        15, 14, 13, 12, 19, 18, 17, 16, 23, 22, 21, 20,
        3,  2,  1,  0,  7,  6,  5,  4,  11, 10, 9,  8
      ]);
    });

    it('reverse a 3D array at axis [1, 2]', math => {
      const input = Array3D.new(shape, data);
      const result = math.reverse3D(input, [1, 2]);
      expect(result.shape).toEqual(input.shape);
      test_util.expectArraysClose(result, [
        11, 10, 9,  8,  7,  6,  5,  4,  3,  2,  1,  0,
        23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12
      ]);
    });

    it('throws error with invalid input', math => {
      // tslint:disable-next-line:no-any
      const x: any = Array2D.new([1, 4], [1, 20, 300, 4]);
      expect(() => math.reverse3D(x, [1])).toThrowError();
    });

    it('throws error with invalid axis param', math => {
      const x = Array3D.new([1, 1, 4], [1, 20, 300, 4]);
      expect(() => math.reverse3D(x, [3])).toThrowError();
      expect(() => math.reverse3D(x, [-4])).toThrowError();
    });

    it('throws error with non integer axis param', math => {
      const x = Array3D.new([1, 1, 4], [1, 20, 300, 4]);
      expect(() => math.reverse3D(x, [0.5])).toThrowError();
    });
  };

  test_util.describeMathCPU('reverse3D', [tests]);
  test_util.describeMathGPU('reverse3D', [tests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}

// math.reverse4D
{
  const tests: MathTests = it => {
    // [
    //   [
    //     [
    //       [0,  1,  2,  3],
    //       [4,  5,  6,  7],
    //       [8,  9,  10, 11]
    //     ],
    //     [
    //       [12, 13, 14, 15],
    //       [16, 17, 18, 19],
    //       [20, 21, 22, 23]
    //     ]
    //   ],
    //   [
    //     [
    //       [24, 25, 26, 27],
    //       [28, 29, 30, 31],
    //       [32, 33, 34, 35]
    //     ],
    //     [
    //       [36, 37, 38, 39],
    //       [40, 41, 42, 43],
    //       [44, 45, 46, 47]
    //     ]
    //   ],
    //   [
    //     [
    //       [48, 49, 50, 51],
    //       [52, 53, 54, 55],
    //       [56, 57, 58, 59]
    //     ],
    //     [
    //       [60, 61, 62, 63],
    //       [64, 65, 66, 67],
    //       [68, 69, 70, 71]
    //     ]
    //   ]
    // ]
    const shape: [number, number, number, number] = [3, 2, 3, 4];
    const data = [
      0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17,
      18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
      36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53,
      54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71
    ];

    it('reverse a 4D array at axis [0]', math => {
      const input = Array4D.new(shape, data);
      const result = math.reverse4D(input, [0]);
      expect(result.shape).toEqual(input.shape);
      test_util.expectArraysClose(result, [
        48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62,
        63, 64, 65, 66, 67, 68, 69, 70, 71, 24, 25, 26, 27, 28, 29,
        30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
        45, 46, 47, 0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
        12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23
      ]);
    });

    it('reverse a 4D array at axis [1]', math => {
      const input = Array4D.new(shape, data);
      const result = math.reverse4D(input, [1]);
      expect(result.shape).toEqual(input.shape);
      test_util.expectArraysClose(result, [
        12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 0,  1,  2,
        3,  4,  5,  6,  7,  8,  9,  10, 11, 36, 37, 38, 39, 40, 41,
        42, 43, 44, 45, 46, 47, 24, 25, 26, 27, 28, 29, 30, 31, 32,
        33, 34, 35, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71,
        48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59
      ]);
    });

    it('reverse a 4D array at axis [2]', math => {
      const input = Array4D.new(shape, data);
      const result = math.reverse4D(input, [2]);
      expect(result.shape).toEqual(input.shape);
      test_util.expectArraysClose(result, [
        8,  9,  10, 11, 4,  5,  6,  7,  0,  1,  2,  3,  20, 21, 22,
        23, 16, 17, 18, 19, 12, 13, 14, 15, 32, 33, 34, 35, 28, 29,
        30, 31, 24, 25, 26, 27, 44, 45, 46, 47, 40, 41, 42, 43, 36,
        37, 38, 39, 56, 57, 58, 59, 52, 53, 54, 55, 48, 49, 50, 51,
        68, 69, 70, 71, 64, 65, 66, 67, 60, 61, 62, 63
      ]);
    });

    it('reverse a 4D array at axis [3]', math => {
      const input = Array4D.new(shape, data);
      const result = math.reverse4D(input, [3]);
      expect(result.shape).toEqual(input.shape);
      test_util.expectArraysClose(result, [
        3,  2,  1,  0,  7,  6,  5,  4,  11, 10, 9,  8,  15, 14, 13,
        12, 19, 18, 17, 16, 23, 22, 21, 20, 27, 26, 25, 24, 31, 30,
        29, 28, 35, 34, 33, 32, 39, 38, 37, 36, 43, 42, 41, 40, 47,
        46, 45, 44, 51, 50, 49, 48, 55, 54, 53, 52, 59, 58, 57, 56,
        63, 62, 61, 60, 67, 66, 65, 64, 71, 70, 69, 68
      ]);
    });

    it('reverse a 4D array at axis [0, 2]', math => {
      const input = Array4D.new(shape, data);
      const result = math.reverse4D(input, [0, 2]);
      expect(result.shape).toEqual(input.shape);
      test_util.expectArraysClose(result, [
        56, 57, 58, 59, 52, 53, 54, 55, 48, 49, 50, 51, 68, 69, 70,
        71, 64, 65, 66, 67, 60, 61, 62, 63, 32, 33, 34, 35, 28, 29,
        30, 31, 24, 25, 26, 27, 44, 45, 46, 47, 40, 41, 42, 43, 36,
        37, 38, 39, 8,  9,  10, 11, 4,  5,  6,  7,  0,  1,  2,  3,
        20, 21, 22, 23, 16, 17, 18, 19, 12, 13, 14, 15
      ]);
    });

    it('reverse a 4D array at axis [1, 3]', math => {
      const input = Array4D.new(shape, data);
      const result = math.reverse4D(input, [1, 3]);
      expect(result.shape).toEqual(input.shape);
      test_util.expectArraysClose(result, [
        15, 14, 13, 12, 19, 18, 17, 16, 23, 22, 21, 20, 3,  2,  1,
        0,  7,  6,  5,  4,  11, 10, 9,  8,  39, 38, 37, 36, 43, 42,
        41, 40, 47, 46, 45, 44, 27, 26, 25, 24, 31, 30, 29, 28, 35,
        34, 33, 32, 63, 62, 61, 60, 67, 66, 65, 64, 71, 70, 69, 68,
        51, 50, 49, 48, 55, 54, 53, 52, 59, 58, 57, 56
      ]);
    });

    it('throws error with invalid input', math => {
      // tslint:disable-next-line:no-any
      const x: any = Array3D.new([1, 1, 4], [1, 20, 300, 4]);
      expect(() => math.reverse4D(x, [1])).toThrowError();
    });

    it('throws error with invalid axis param', math => {
      const x = Array4D.new([1, 1, 1, 4], [1, 20, 300, 4]);
      expect(() => math.reverse4D(x, [4])).toThrowError();
      expect(() => math.reverse4D(x, [-5])).toThrowError();
    });

    it('throws error with non integer axis param', math => {
      const x = Array4D.new([1, 1, 1, 4], [1, 20, 300, 4]);
      expect(() => math.reverse4D(x, [0.5])).toThrowError();
    });
  };

  test_util.describeMathCPU('reverse4D', [tests]);
  test_util.describeMathGPU('reverse4D', [tests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}
