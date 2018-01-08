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

import {Array2D, Array3D, Array4D} from './ndarray';

const sqArr = (arr: number[]) => arr.map(d => d*d);
const sumArr = (arr: number[]) => arr.reduce((prev, curr) => prev + curr, 0);

// tslint:disable-next-line:no-any
const flatten = (arr: any): number[] => {
  // tslint:disable-next-line:no-any
  return arr.reduce((prev: any, curr: any) => {
    return prev.concat(Array.isArray(curr) ? flatten(curr) : curr);
  }, []);
};

// math.localResponseNormalization3D
{
  const tests: MathTests = it => {

    it('throws error with invalid input', math => {
      // tslint:disable-next-line:no-any
      const x: any = Array2D.new([1, 4], new Float32Array([1, 20, 300, 4]));
      const radius = 3;

      expect(() => math.localResponseNormalization3D(x, radius))
        .toThrowError();
    });

    it('throws error with invalid radius', math => {
      const x = Array3D.new([1, 1, 4], new Float32Array([1, 20, 300, 4]));
      const radius = 0.5;

      expect(() => math.localResponseNormalization3D(x, radius))
        .toThrowError();
    });

    it('computes simple normalization across channels', math => {
      const x = Array3D.new([1, 1, 4], new Float32Array([1, 20, 300, 4]));
      const radius = 1;
      const bias = 1;
      const alpha = 1;
      const beta = 0.5;

      const result = math.localResponseNormalization3D(x, radius, bias, alpha,
        beta);

      const f = (...vals: number[]) =>
        Math.pow(bias + alpha * sumArr(sqArr(vals)), -beta);

      test_util.expectArraysClose(
        result,
        [
          x.get(0, 0, 0) * f(x.get(0, 0, 0), x.get(0, 0, 1)),
          x.get(0, 0, 1) * f(x.get(0, 0, 0), x.get(0, 0, 1), x.get(0, 0, 2)),
          x.get(0, 0, 2) * f(x.get(0, 0, 1), x.get(0, 0, 2), x.get(0, 0, 3)),
          x.get(0, 0, 3) * f(x.get(0, 0, 2), x.get(0, 0, 3)),
        ]);
    });

    it('uses beta = 1.0 to test GPU optimization', math => {
      const x = Array3D.new([1, 1, 4], new Float32Array([1, 20, 300, 4]));
      const radius = 1;
      const bias = 1;
      const alpha = 1;
      const beta = 1.0;

      const result = math.localResponseNormalization3D(x, radius, bias, alpha,
        beta);

      const f = (...vals: number[]) =>
        Math.pow(bias + alpha * sumArr(sqArr(vals)), -beta);

      test_util.expectArraysClose(
        result,
        [
          x.get(0, 0, 0) * f(x.get(0, 0, 0), x.get(0, 0, 1)),
          x.get(0, 0, 1) * f(x.get(0, 0, 0), x.get(0, 0, 1), x.get(0, 0, 2)),
          x.get(0, 0, 2) * f(x.get(0, 0, 1), x.get(0, 0, 2), x.get(0, 0, 3)),
          x.get(0, 0, 3) * f(x.get(0, 0, 2), x.get(0, 0, 3)),
        ]);
    });

    it('uses beta = 0.75 to test GPU optimization', math => {
      const x = Array3D.new([1, 1, 4], new Float32Array([1, 20, 300, 4]));
      const radius = 1;
      const bias = 1;
      const alpha = 1;
      const beta = 0.75;

      const result = math.localResponseNormalization3D(x, radius, bias, alpha,
        beta);

      const f = (...vals: number[]) =>
        Math.pow(bias + alpha * sumArr(sqArr(vals)), -beta);

      test_util.expectArraysClose(
        result,
        [
          x.get(0, 0, 0) * f(x.get(0, 0, 0), x.get(0, 0, 1)),
          x.get(0, 0, 1) * f(x.get(0, 0, 0), x.get(0, 0, 1), x.get(0, 0, 2)),
          x.get(0, 0, 2) * f(x.get(0, 0, 1), x.get(0, 0, 2), x.get(0, 0, 3)),
          x.get(0, 0, 3) * f(x.get(0, 0, 2), x.get(0, 0, 3)),
        ]);
    });

    it('computes complex normalization across channels', math => {
      const x = Array3D.new([2, 2, 4], new Float32Array([
        1, 20, 300, 4, 5, 15, 24, 200, 1, 20, 300, 4, 5, 15, 24, 200
      ]));
      const radius = 1;
      const bias = 1;
      const alpha = 1;
      const beta = 0.5;
      const normRegion = "acrossChannels";

      const result = math.localResponseNormalization3D(x, radius, bias, alpha,
        beta, normRegion);

      const f = (...vals: number[]) =>
        Math.pow(bias + alpha * sumArr(sqArr(vals)), -beta);

      // 1       | 2       | 3       | 4
      // ------- | ------- | ------- | -------
      // o x . . | x o x . | . x o x | . . x o

      test_util.expectArraysClose(
          result,
          [
            // 1 - 4
            x.get(0, 0, 0) * f(x.get(0, 0, 0), x.get(0, 0, 1)),
            x.get(0, 0, 1) * f(x.get(0, 0, 0), x.get(0, 0, 1), x.get(0, 0, 2)),
            x.get(0, 0, 2) * f(x.get(0, 0, 1), x.get(0, 0, 2), x.get(0, 0, 3)),
            x.get(0, 0, 3) * f(x.get(0, 0, 2), x.get(0, 0, 3)),

            // 1 - 4
            x.get(0, 1, 0) * f(x.get(0, 1, 0), x.get(0, 1, 1)),
            x.get(0, 1, 1) * f(x.get(0, 1, 0), x.get(0, 1, 1), x.get(0, 1, 2)),
            x.get(0, 1, 2) * f(x.get(0, 1, 1), x.get(0, 1, 2), x.get(0, 1, 3)),
            x.get(0, 1, 3) * f(x.get(0, 1, 2), x.get(0, 1, 3)),

            // 1 - 4
            x.get(1, 0, 0) * f(x.get(1, 0, 0), x.get(1, 0, 1)),
            x.get(1, 0, 1) * f(x.get(1, 0, 0), x.get(1, 0, 1), x.get(1, 0, 2)),
            x.get(1, 0, 2) * f(x.get(1, 0, 1), x.get(1, 0, 2), x.get(1, 0, 3)),
            x.get(1, 0, 3) * f(x.get(1, 0, 2), x.get(1, 0, 3)),

            // 1 - 4
            x.get(1, 1, 0) * f(x.get(1, 1, 0), x.get(1, 1, 1)),
            x.get(1, 1, 1) * f(x.get(1, 1, 0), x.get(1, 1, 1), x.get(1, 1, 2)),
            x.get(1, 1, 2) * f(x.get(1, 1, 1), x.get(1, 1, 2), x.get(1, 1, 3)),
            x.get(1, 1, 3) * f(x.get(1, 1, 2), x.get(1, 1, 3)),
          ]);
    });

    it('computes simple normalization within channel', math => {
      const x = Array3D.new([2, 2, 1], new Float32Array([1, 20, 300, 4]));
      const radius = 1;
      const bias = 1;
      const alpha = 1;
      const beta = 0.5;
      const normRegion = "withinChannel";

      const result = math.localResponseNormalization3D(x, radius, bias, alpha,
        beta, normRegion);

      const f = (...vals: number[]) =>
        Math.pow(bias + alpha * sumArr(sqArr(vals)), -beta);

      const multip =
        f(x.get(0, 0, 0), x.get(1, 0, 0), x.get(0, 1, 0), x.get(1, 1, 0));

      test_util.expectArraysClose(
        result,
        [
          x.get(0, 0, 0) * multip,
          x.get(0, 1, 0) * multip,
          x.get(1, 0, 0) * multip,
          x.get(1, 1, 0) * multip,
        ]);
    });

    it('computes complex normalization within channel', math => {
      const x = Array3D.new([3, 3, 2], new Float32Array([
        1, 20, 300, 4, 23, 25, 13, 156, 123, 5, 15, 24, 200, 12, 12, 13, 21, 3
      ]));
      const radius = 1;
      const bias = 1;
      const alpha = 1;
      const beta = 0.5;
      const normRegion = "withinChannel";

      const result = math.localResponseNormalization3D(x, radius, bias, alpha,
        beta, normRegion);

      const f = (...vals: number[]) =>
        Math.pow(bias + alpha * sumArr(sqArr(vals)), -beta);

      // Easier to read using these vars
      const d0 = 0;
      const d1 = 1;

      // 1     | 2     | 3     | 4     | 5     | 6     | 7     | 8     | 9
      // ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | -----
      // o x . | x o x | . x o | x x . | x x x | . x x | . . . | . . . | . . .
      // x x . | x x x | . x x | o x . | x o x | . x o | x x . | x x x | . x x
      // . . . | . . . | . . . | x x . | x x x | . x x | o x . | x o x | . x o

      test_util.expectArraysClose(
          result,
          [
            // 1
            x.get(0, 0, d0) * f(
              x.get(0, 0, d0), x.get(1, 0, d0),
              x.get(0, 1, d0), x.get(1, 1, d0)),

            x.get(0, 0, d1) * f(
              x.get(0, 0, d1), x.get(1, 0, d1),
              x.get(0, 1, d1), x.get(1, 1, d1)),

            // 4
            x.get(0, 1, d0) * f(
              x.get(0, 0, d0), x.get(1, 0, d0),
              x.get(0, 1, d0), x.get(1, 1, d0),
              x.get(0, 2, d0), x.get(1, 2, d0)),
            x.get(0, 1, d1) * f(
              x.get(0, 0, d1), x.get(1, 0, d1),
              x.get(0, 1, d1), x.get(1, 1, d1),
              x.get(0, 2, d1), x.get(1, 2, d1)),

            // 7
            x.get(0, 2, d0) * f(
              x.get(0, 1, d0), x.get(1, 1, d0),
              x.get(0, 2, d0), x.get(1, 2, d0)),
            x.get(0, 2, d1) * f(
              x.get(0, 1, d1), x.get(1, 1, d1),
              x.get(0, 2, d1), x.get(1, 2, d1)),

            // 2
            x.get(1, 0, d0) * f(
              x.get(0, 0, d0), x.get(1, 0, d0), x.get(2, 0, d0),
              x.get(0, 1, d0), x.get(1, 1, d0), x.get(2, 1, d0)),
            x.get(1, 0, d1) * f(
              x.get(0, 0, d1), x.get(1, 0, d1), x.get(2, 0, d1),
              x.get(0, 1, d1), x.get(1, 1, d1), x.get(2, 1, d1)),

            // 5
            x.get(1, 1, d0) * f(
              x.get(0, 0, d0), x.get(1, 0, d0), x.get(2, 0, d0),
              x.get(0, 1, d0), x.get(1, 1, d0), x.get(2, 1, d0),
              x.get(0, 2, d0), x.get(1, 2, d0), x.get(2, 2, d0)),
            x.get(1, 1, d1) * f(
              x.get(0, 0, d1), x.get(1, 0, d1), x.get(2, 0, d1),
              x.get(0, 1, d1), x.get(1, 1, d1), x.get(2, 1, d1),
              x.get(0, 2, d1), x.get(1, 2, d1), x.get(2, 2, d1)),

            // 8
            x.get(1, 2, d0) * f(
              x.get(0, 1, d0), x.get(1, 1, d0), x.get(2, 1, d0),
              x.get(0, 2, d0), x.get(1, 2, d0), x.get(2, 2, d0)),
             x.get(1, 2, d1) * f(
              x.get(0, 1, d1), x.get(1, 1, d1), x.get(2, 1, d1),
              x.get(0, 2, d1), x.get(1, 2, d1), x.get(2, 2, d1)),

            // 3
            x.get(2, 0, d0) * f(
              x.get(1, 0, d0), x.get(2, 0, d0),
              x.get(1, 1, d0), x.get(2, 1, d0)),
            x.get(2, 0, d1) * f(
              x.get(1, 0, d1), x.get(2, 0, d1),
              x.get(1, 1, d1), x.get(2, 1, d1)),

            // 6
            x.get(2, 1, d0) * f(
              x.get(1, 0, d0), x.get(2, 0, d0),
              x.get(1, 1, d0), x.get(2, 1, d0),
              x.get(1, 2, d0), x.get(2, 2, d0)),
            x.get(2, 1, d1) * f(
              x.get(1, 0, d1), x.get(2, 0, d1),
              x.get(1, 1, d1), x.get(2, 1, d1),
              x.get(1, 2, d1), x.get(2, 2, d1)),

            // 9
            x.get(2, 2, d0) * f(
              x.get(1, 1, d0), x.get(2, 1, d0),
              x.get(1, 2, d0), x.get(2, 2, d0)),
            x.get(2, 2, d1) * f(
              x.get(1, 1, d1), x.get(2, 1, d1),
              x.get(1, 2, d1), x.get(2, 2, d1)),
          ]);
    });

    it('yields same result as tensorflow', math => {

      // t = tf.random_uniform([1, 3, 3, 8])
      // l = tf.nn.lrn(t, depth_radius=2)
      // print(tf.Session().run([t, l]))

      const input = [
        [[ 0.95782757,  0.12892687,  0.63624668,  0.70160735,  0.77376258,
            0.54166114,  0.71172535,  0.65087497],
          [ 0.91872108,  0.38846886,  0.37847793,  0.50477624,  0.42154622,
            0.43310916,  0.36253822,  0.07576156],
          [ 0.48662257,  0.4154036 ,  0.81704032,  0.91660416,  0.87671542,
            0.64215934,  0.29933751,  0.90671134]],

        [[ 0.6208992 ,  0.60847163,  0.41475761,  0.2127713 ,  0.65306914,
            0.13923979,  0.32003641,  0.28183973],
          [ 0.04751575,  0.26870155,  0.45150304,  0.58678186,  0.99118924,
            0.58878231,  0.30913198,  0.18836617],
          [ 0.16166461,  0.56322742,  0.67908955,  0.2269547 ,  0.38491273,
            0.97113752,  0.51210916,  0.69430435]],

        [[ 0.06625497,  0.13011181,  0.59202921,  0.88871598,  0.6366322 ,
            0.47911358,  0.96530843,  0.74259472],
          [ 0.62660718,  0.0445286 ,  0.18430257,  0.76863647,  0.87511849,
            0.53588808,  0.27980685,  0.30281997],
          [ 0.73987067,  0.91034842,  0.26241004,  0.72832751,  0.78974342,
            0.50751543,  0.05434644,  0.8231523 ]]
      ];

      const expected = [
        [[ 0.62630326,  0.07662392,  0.34354961,  0.41885775,  0.42621866,
            0.29751951,  0.42365381,  0.4364861 ],
          [ 0.62828875,  0.251122  ,  0.23605582,  0.36483878,  0.30624411,
            0.32672295,  0.29576892,  0.06582346],
          [ 0.3376624 ,  0.24321821,  0.42558169,  0.46646208,  0.45103404,
            0.32380751,  0.17021206,  0.59476018]],

        [[ 0.44719055,  0.43318295,  0.26775005,  0.14921051,  0.49148726,
            0.10764983,  0.25084552,  0.25714993],
          [ 0.04202608,  0.21094096,  0.27973703,  0.34166718,  0.57487047,
            0.35158369,  0.19708875,  0.15495601],
          [ 0.12034657,  0.41341963,  0.47968671,  0.13278878,  0.22735766,
            0.57154536,  0.30411762,  0.42352781]],

        [[ 0.05656794,  0.08849642,  0.36951816,  0.53186077,  0.33065733,
            0.24236222,  0.54666328,  0.45085984],
          [ 0.52425432,  0.03133496,  0.11043368,  0.46954039,  0.5271349 ,
            0.31946796,  0.1876673 ,  0.25085902],
          [ 0.47316891,  0.5277527 ,  0.13831842,  0.40036613,  0.50113004,
            0.28860986,  0.03395459,  0.59127772]]
      ];

      const x = Array3D.new([3, 3, 8], new Float32Array(flatten(input)));
      const radius = 2;
      const bias = 1;
      const alpha = 1;
      const beta = 0.5;

      const result = math.localResponseNormalization3D(x, radius, bias,
        alpha, beta);

      test_util.expectArraysClose(result, flatten(expected));
    });
  };

  test_util.describeMathCPU('localResponseNormalization3D', [tests]);
  test_util.describeMathGPU('localResponseNormalization3D', [tests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}

// math.localResponseNormalization4D
{
  const tests: MathTests = it => {

    it('throws error with invalid input', math => {
      // tslint:disable-next-line:no-any
      const x: any = Array3D.new([1, 1, 4], new Float32Array([1, 20, 300, 4]));
      const radius = 3;

      expect(() => math.localResponseNormalization4D(x, radius))
        .toThrowError();
    });

    it('throws error with invalid radius', math => {
      const x = Array4D.new([1, 1, 1, 4], new Float32Array([1, 20, 300, 4]));
      const radius = 0.5;

      expect(() => math.localResponseNormalization4D(x, radius))
        .toThrowError();
    });

    it('computes simple normalization across channels', math => {
      const x = Array4D.new([2, 1, 1, 4],
        new Float32Array([1, 20, 300, 4, 1, 20, 300, 4]));
      const radius = 1;
      const bias = 1;
      const alpha = 1;
      const beta = 0.5;

      const result = math.localResponseNormalization4D(x, radius, bias, alpha,
        beta);

      const f = (...vals: number[]) =>
        Math.pow(bias + alpha * sumArr(sqArr(vals)), -beta);

      // Easier to read using these vars
      const b0 = 0;
      const b1 = 1;

      test_util.expectArraysClose(
        result,
        [
          x.get(b0, 0, 0, 0) * f(x.get(b0, 0, 0, 0), x.get(b0, 0, 0, 1)),
          x.get(b0, 0, 0, 1) *
            f(x.get(b0, 0, 0, 0),x.get(b0, 0, 0, 1), x.get(b0, 0, 0, 2)),
          x.get(b0, 0, 0, 2) *
            f(x.get(b0, 0, 0, 1), x.get(b0, 0, 0, 2), x.get(b0, 0, 0, 3)),
          x.get(b0, 0, 0, 3) * f(x.get(b0, 0, 0, 2), x.get(b0, 0, 0, 3)),

          x.get(b1, 0, 0, 0) * f(x.get(b1, 0, 0, 0), x.get(b1, 0, 0, 1)),
          x.get(b1, 0, 0, 1) *
            f(x.get(b1, 0, 0, 0), x.get(b1, 0, 0, 1), x.get(b1, 0, 0, 2)),
          x.get(b1, 0, 0, 2) *
            f(x.get(b1, 0, 0, 1), x.get(b1, 0, 0, 2), x.get(b1, 0, 0, 3)),
          x.get(b1, 0, 0, 3) * f(x.get(b1, 0, 0, 2), x.get(b1, 0, 0, 3)),
        ]);
    });

    it('computes simple normalization within channel', math => {
      const x = Array4D.new([2, 2, 2, 1],
        new Float32Array([1, 20, 50, 4, 1, 20, 50, 4]));
      const radius = 1;
      const bias = 1;
      const alpha = 1;
      const beta = 0.5;
      const normRegion = "withinChannel";

      const result = math.localResponseNormalization4D(x, radius, bias, alpha,
        beta, normRegion);

      const f = (...vals: number[]) =>
        Math.pow(bias + alpha * sumArr(sqArr(vals)), -beta);

      const multip = (b: number, d: number) => f(
            x.get(b, 0, 0, d), x.get(b, 1, 0, d),
            x.get(b, 0, 1, d), x.get(b, 1, 1, d));

      // Easier to read using these vars
      const b0 = 0;
      const b1 = 1;

      test_util.expectArraysClose(
        result,
        [
          x.get(b0, 0, 0, 0) * multip(b0, 0),
          x.get(b0, 0, 1, 0) * multip(b0, 0),
          x.get(b0, 1, 0, 0) * multip(b0, 0),
          x.get(b0, 1, 1, 0) * multip(b0, 0),
          x.get(b1, 0, 0, 0) * multip(b1, 0),
          x.get(b1, 0, 1, 0) * multip(b1, 0),
          x.get(b1, 1, 0, 0) * multip(b1, 0),
          x.get(b1, 1, 1, 0) * multip(b1, 0),
        ]);
    });

    it('yields same result as tensorflow', math => {

      // t = tf.random_uniform([2, 3, 3, 8])
      // l = tf.nn.lrn(t, depth_radius=2)
      // print(tf.Session().run([t, l]))

      const input = [[
        [[ 0.5659827 ,  0.57000327,  0.75555623,  0.89843333,  0.55120194,
            0.53531718,  0.56402838,  0.95481384],
          [ 0.57334661,  0.65172958,  0.75794137,  0.80764937,  0.376616  ,
            0.92726362,  0.36422753,  0.60535395],
          [ 0.82404268,  0.01054764,  0.4649173 ,  0.91637003,  0.82287347,
            0.043468  ,  0.44953859,  0.92056584]],

        [[ 0.68583369,  0.52534163,  0.53325927,  0.39608097,  0.9337523 ,
            0.37397444,  0.81212556,  0.5697    ],
          [ 0.34278774,  0.57656682,  0.2356832 ,  0.02636456,  0.49111438,
            0.17981696,  0.65398049,  0.70132935],
          [ 0.14241767,  0.68376505,  0.65419888,  0.69369483,  0.21489143,
            0.46235347,  0.0559243 ,  0.60612857]],

        [[ 0.59678483,  0.09368539,  0.3017447 ,  0.36870825,  0.68145788,
            0.52048779,  0.46136606,  0.94114387],
          [ 0.3156569 ,  0.75275254,  0.31970251,  0.3154043 ,  0.61088014,
            0.13359487,  0.99048364,  0.33625424],
          [ 0.82103574,  0.52066624,  0.63629258,  0.42294252,  0.93214262,
            0.57041013,  0.66087878,  0.7019999 ]]],

        [[[ 0.21894431,  0.43085241,  0.79883206,  0.19462204,  0.68623316,
            0.08703053,  0.82380795,  0.85634673],
          [ 0.45011401,  0.70312083,  0.86319792,  0.83205295,  0.67109787,
            0.82081223,  0.46556532,  0.46408331],
          [ 0.07028461,  0.0038743 ,  0.44619524,  0.0611403 ,  0.96373355,
            0.80561554,  0.42428243,  0.46897113]],

        [[ 0.21006894,  0.48764861,  0.36842632,  0.23030031,  0.69685507,
            0.31707478,  0.68662715,  0.0639503 ],
          [ 0.53940296,  0.50777435,  0.12625301,  0.12324154,  0.89205229,
            0.69380629,  0.33191144,  0.81000078],
          [ 0.52650976,  0.71220326,  0.07246161,  0.08874547,  0.42528927,
            0.36320579,  0.54055619,  0.79342318]],

        [[ 0.75916636,  0.74499428,  0.76877356,  0.87210917,  0.93040991,
            0.49491942,  0.70801985,  0.14901721],
          [ 0.27037835,  0.89302075,  0.69147241,  0.23044991,  0.98916364,
            0.60161841,  0.63691151,  0.56759977],
          [ 0.56307781,  0.92782414,  0.25880754,  0.98518133,  0.04097319,
            0.24640906,  0.54566145,  0.99261606]]
      ]];

      const expected = [[
        [[ 0.38019636,  0.32782161,  0.414222  ,  0.49507114,  0.3040463 ,
            0.28107059,  0.33586296,  0.60191077],
          [ 0.37577698,  0.37752095,  0.42895618,  0.4225589 ,  0.2054275 ,
            0.52219951,  0.23032214,  0.39414096],
          [ 0.59856331,  0.00637784,  0.25168711,  0.5541048 ,  0.48015645,
            0.02301128,  0.27214608,  0.6427291 ]],

        [[ 0.48127589,  0.35518789,  0.30486941,  0.23976389,  0.52926594,
            0.21061926,  0.46920502,  0.39090639],
          [ 0.27937523,  0.46979892,  0.17829391,  0.02044933,  0.37045884,
            0.12140442,  0.44160855,  0.50198948],
          [ 0.10289387,  0.44164398,  0.41853485,  0.42720893,  0.14580171,
            0.31817055,  0.043797  ,  0.48155668]],

        [[ 0.49458414,  0.07425242,  0.21042404,  0.26262277,  0.46205613,
            0.30202535,  0.27406475,  0.61140078],
          [ 0.23736385,  0.55076694,  0.2135559 ,  0.21463785,  0.38077739,
            0.08309806,  0.62830603,  0.23137885],
          [ 0.5355776 ,  0.32740855,  0.3451882 ,  0.24221195,  0.51988536,
            0.31387195,  0.37391993,  0.46748781]]],

        [[[ 0.16003507,  0.31178808,  0.51775187,  0.12722474,  0.40769571,
            0.05085804,  0.48455271,  0.5505302 ],
          [ 0.2880325 ,  0.39714804,  0.45591024,  0.4131493 ,  0.34525412,
            0.4554069 ,  0.29119283,  0.31980222],
          [ 0.0640529 ,  0.00352532,  0.3052578 ,  0.03666528,  0.56009793,
            0.46656418,  0.24587312,  0.32762629]],

        [[ 0.17643087,  0.40210918,  0.2634095 ,  0.16233148,  0.4649446 ,
            0.21803913,  0.47819966,  0.05093931],
          [ 0.43121469,  0.403974  ,  0.08191212,  0.07693455,  0.57362044,
            0.39671475,  0.19025819,  0.54028469],
          [ 0.39356521,  0.53120333,  0.05151648,  0.06554616,  0.33433318,
            0.2425479 ,  0.36161765,  0.5536595 ]],

        [[ 0.46011236,  0.39919043,  0.36865807,  0.43511948,  0.46734285,
            0.26861796,  0.43624333,  0.11205748],
          [ 0.17642327,  0.57622254,  0.37609601,  0.12030836,  0.54640025,
            0.34052721,  0.36361033,  0.3926385 ],
          [ 0.37581176,  0.51741964,  0.14429154,  0.57254595,  0.02646073,
            0.13531584,  0.35629693,  0.64837402]]
      ]];

      const x = Array4D.new([2, 3, 3, 8], new Float32Array(flatten(input)));
      const radius = 2;

      const result = math.localResponseNormalization4D(x, radius);

      test_util.expectArraysClose(result, flatten(expected));
    });
  };

  test_util.describeMathCPU('localResponseNormalization4D', [tests]);
  test_util.describeMathGPU('localResponseNormalization4D', [tests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}